"""Agent tools — actions the overlord LLM can invoke.

Each tool is a function that takes the current context (config, state, etc.)
and returns a result dict. The agent calls these by name; the orchestrator
dispatches and records the outcome.

Key design: start_search launches evoloop as a background process. The agent
ticks every N minutes regardless. get_search_status reads evoloop's live
experiment DB from disk to check progress without blocking.
"""

from __future__ import annotations

import inspect
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.lifecycle import run_deploy, run_monitor, run_setup
from orchestrator.state import AgentState


# ──────────────────────────────────────────────
# Tool definitions (used to build the LLM prompt)
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "run_setup",
        "description": (
            "Run the subnet's setup script to validate prerequisites "
            "(data source access, dependencies, eval harness). "
            "Use this before starting a search, or if you suspect something is broken."
        ),
        "parameters": {},
    },
    {
        "name": "start_search",
        "description": (
            "Start the model search (evoloop or other strategy) as a background process. "
            "Returns immediately — use get_search_status to check progress. "
            "Only call this if no search is currently running."
        ),
        "parameters": {
            "max_experiments": {
                "type": "integer",
                "description": "Maximum experiments to run (0 = unlimited). Optional.",
                "required": False,
            },
        },
    },
    {
        "name": "stop_search",
        "description": (
            "Stop the currently running search. Use when: search has converged "
            "(stale for too long), time budget exceeded, or you want to deploy "
            "the best result so far. Sends SIGINT for a graceful shutdown."
        ),
        "parameters": {},
    },
    {
        "name": "get_search_status",
        "description": (
            "Check the current search progress by reading the live experiment "
            "database: experiments run, best metric, staleness, whether the "
            "process is still alive. Non-blocking — returns immediately."
        ),
        "parameters": {},
    },
    {
        "name": "deploy",
        "description": (
            "Deploy the best model/artifact from the search to production. "
            "Runs the subnet's deploy script. Only call after search has "
            "produced results (check get_search_status first)."
        ),
        "parameters": {
            "artifact_path": {
                "type": "string",
                "description": "Path to the artifact to deploy. If empty, uses the best from last search.",
                "required": False,
            },
        },
    },
    {
        "name": "check_live_performance",
        "description": (
            "Run the subnet's monitor script to check live miner performance. "
            "Returns health status, metrics (emission share, rank), and whether "
            "re-evolution is recommended."
        ),
        "parameters": {},
    },
    {
        "name": "wait",
        "description": (
            "Do nothing for now. Use when the search is running and making "
            "progress, or when monitoring and everything looks healthy. "
            "You MUST provide a reason explaining why waiting is the right call."
        ),
        "parameters": {
            "reason": {
                "type": "string",
                "description": "Why you're choosing to wait.",
                "required": True,
            },
        },
    },
]


# ──────────────────────────────────────────────
# Process management helpers
# ──────────────────────────────────────────────

def _pid_is_alive(pid: int | None) -> bool:
    """Check if a process with the given PID is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = check existence, don't actually kill
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it


def _read_experiment_db(config: SubnetConfig) -> list[dict]:
    """Read evoloop's experiment database from disk.

    evoloop writes experiments as JSON files or a single db.json.
    We check multiple possible locations.
    """
    evoloop_dir = Path(
        config.strategy.config.get(
            "evoloop_dir",
            os.environ.get("EVOLOOP_DIR", ""),
        )
    )

    search_paths = [
        evoloop_dir / "experiments" if evoloop_dir.name else None,
        evoloop_dir / "db.json" if evoloop_dir.name else None,
        config.workspace_dir / "experiments",
        config.workspace_dir / "db.json",
    ]

    experiments = []
    for search_path in search_paths:
        if search_path is None or not search_path.exists():
            continue

        if search_path.is_dir():
            for exp_file in sorted(search_path.glob("*.json")):
                try:
                    with open(exp_file) as f:
                        experiments.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    continue
        elif search_path.is_file():
            try:
                with open(search_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    experiments.extend(data)
                elif isinstance(data, dict) and "experiments" in data:
                    experiments.extend(data["experiments"])
            except (json.JSONDecodeError, OSError):
                pass

        if experiments:
            break

    return experiments


def _find_best_experiment(experiments: list[dict]) -> dict | None:
    """Find the best experiment by primary objective (lower is better)."""
    best = None
    for exp in experiments:
        objectives = exp.get("objectives", {})
        if not objectives:
            continue
        if best is None:
            best = exp
            continue
        best_obj = best.get("objectives", {})
        primary_key = next(iter(objectives))
        if primary_key in best_obj:
            if objectives[primary_key] < best_obj.get(primary_key, float("inf")):
                best = exp
    return best


# ──────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────

def tool_run_setup(config: SubnetConfig, state: AgentState) -> dict[str, Any]:
    """Run the subnet's setup/prerequisite checks."""
    state.phase = "setup"
    state.phase_started_at = time.time()

    success = run_setup(config.subnet_dir, config.workspace_dir)

    if success:
        state.consecutive_errors = 0
        state.last_error = None
        return {"success": True, "message": "Setup checks passed."}
    else:
        state.consecutive_errors += 1
        state.last_error = "Setup failed"
        return {"success": False, "message": "Setup checks failed. Check logs above."}


def tool_start_search(
    config: SubnetConfig, state: AgentState, max_experiments: int = 0
) -> dict[str, Any]:
    """Launch the search strategy as a background process. Returns immediately."""
    # Check if a search is already running
    if state.search_running and _pid_is_alive(state.search_pid):
        return {
            "success": False,
            "message": (
                f"Search is already running (PID {state.search_pid}). "
                "Use get_search_status or stop_search first."
            ),
        }

    # Build the evoloop launch command
    from orchestrator.strategies import get_strategy
    strategy = get_strategy(config)

    if not strategy.setup():
        state.consecutive_errors += 1
        state.last_error = "Strategy setup failed"
        return {"success": False, "message": "Strategy setup failed."}

    # Build subprocess command and env
    proc = _launch_search_process(config, strategy, max_experiments)
    if proc is None:
        state.consecutive_errors += 1
        state.last_error = "Failed to launch search process"
        return {"success": False, "message": "Failed to launch search process."}

    # Update state — search is now running in background
    state.phase = "searching"
    state.phase_started_at = time.time()
    state.search_running = True
    state.search_pid = proc.pid
    state.search_started_at = time.time()
    state.experiments_run = 0
    state.experiments_at_last_improvement = 0
    state.stale_count = 0
    state.consecutive_errors = 0
    state.last_error = None

    return {
        "success": True,
        "message": f"Search launched in background (PID {proc.pid}).",
        "pid": proc.pid,
    }


def _launch_search_process(
    config: SubnetConfig, strategy: Any, max_experiments: int
) -> subprocess.Popen | None:
    """Launch evoloop (or other strategy) as a background subprocess."""
    sc = config.strategy.config
    evoloop_dir = Path(
        sc.get("evoloop_dir", os.environ.get("EVOLOOP_DIR", ""))
    )

    # Determine command
    evoloop_cli = shutil.which("evoloop")
    source_task_dir = config.subnet_dir / sc.get("task_dir", "evoloop_task/")
    task_yaml = source_task_dir / "task.yaml"

    effective_max = max_experiments or config.convergence.max_experiments or sc.get("max_experiments", 0)

    if evoloop_cli:
        cmd = [evoloop_cli, "--task", str(task_yaml), "--backend", sc.get("backend", "basilica")]
        if effective_max > 0:
            cmd.extend(["--max-experiments", str(effective_max)])
        cwd = None
    elif evoloop_dir.name and evoloop_dir.exists():
        # Copy task files into evoloop's tasks/ dir
        target_task_dir = evoloop_dir / "tasks" / config.name
        target_task_dir.mkdir(parents=True, exist_ok=True)
        for src_file in source_task_dir.iterdir():
            if src_file.is_file():
                shutil.copy2(src_file, target_task_dir / src_file.name)

        cmd = ["python", "loop.py"]
        task_yaml = target_task_dir / "task.yaml"
        cwd = str(evoloop_dir)
    else:
        print("[search] ERROR: No evoloop CLI or EVOLOOP_DIR found.")
        return None

    env = {
        **os.environ,
        "EVOLOOP_TASK": str(task_yaml),
        "EVOLOOP_RUNNER_BACKEND": sc.get("backend", "basilica"),
        "EVOLOOP_TIME_BUDGET": str(sc.get("time_budget", 600)),
        "EVOLOOP_LLM_PROVIDER": sc.get("llm_provider", "openai"),
        "EVOLOOP_LLM_MODEL": sc.get("llm_model", "gpt-4.1"),
        "EVOLOOP_LLM_MODEL_STRONG": sc.get("llm_model_strong", "o3"),
        "EVOLOOP_BASILICA_GPU_MODELS": sc.get("gpu", "A4000"),
        "EVOLOOP_BASILICA_GPU_COUNT": str(sc.get("gpu_count", 1)),
    }

    if effective_max > 0:
        env["EVOLOOP_MAX_EXPERIMENTS"] = str(effective_max)

    # Write logs to workspace
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.workspace_dir / "search.log"

    print(f"[search] Launching: {' '.join(cmd)}")
    print(f"[search] Log: {log_path}")
    print(f"[search] PID will be reported after launch.")

    try:
        log_file = open(log_path, "a")
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Search started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write(f"{'='*60}\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from parent — survives orchestrator restart
        )
        # Don't close log_file — the subprocess holds it.
        # It will be closed when the process exits.
        return proc
    except Exception as e:
        print(f"[search] Failed to launch: {e}")
        return None


def tool_stop_search(config: SubnetConfig, state: AgentState) -> dict[str, Any]:
    """Stop the running search process gracefully."""
    pid = state.search_pid

    if pid is None or not _pid_is_alive(pid):
        state.search_running = False
        state.search_pid = None
        return {
            "success": True,
            "message": f"No search process running. Best metric: {state.best_metric}",
        }

    print(f"[search] Sending SIGINT to PID {pid}...")
    try:
        # SIGINT lets evoloop finish its current experiment and save state
        os.kill(pid, signal.SIGINT)

        # Wait up to 60s for graceful shutdown
        for _ in range(12):
            time.sleep(5)
            if not _pid_is_alive(pid):
                break
        else:
            # Still alive after 60s — force kill
            print(f"[search] PID {pid} didn't stop gracefully. Sending SIGKILL.")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)

    except ProcessLookupError:
        pass  # Already dead
    except Exception as e:
        return {"success": False, "message": f"Failed to stop PID {pid}: {e}"}

    state.search_running = False
    state.search_pid = None

    # Read final results
    experiments = _read_experiment_db(config)
    best = _find_best_experiment(experiments)
    if best:
        objectives = best.get("objectives", {})
        if objectives:
            primary_key = next(iter(objectives))
            state.best_metric = objectives[primary_key]
        artifact = best.get("artifact_path")
        if artifact:
            state.best_artifact = artifact
        state.experiments_run = len(experiments)

        # Save result for deploy
        from orchestrator.strategies.base import StrategyResult
        result = StrategyResult(
            success=True,
            best_artifact=Path(artifact) if artifact else None,
            metrics=objectives,
            experiments_run=len(experiments),
            summary=f"Stopped after {len(experiments)} experiments. Best: {objectives}",
        )
        result_path = config.workspace_dir / "strategy_result.json"
        result.save(result_path)

    return {
        "success": True,
        "message": (
            f"Search stopped (PID {pid}). "
            f"{state.experiments_run} experiments. "
            f"Best metric: {state.best_metric}"
        ),
    }


def tool_get_search_status(config: SubnetConfig, state: AgentState) -> dict[str, Any]:
    """Check live search progress — reads experiment DB from disk. Non-blocking."""
    pid = state.search_pid
    process_alive = _pid_is_alive(pid)

    # If state says running but process is dead, the search finished
    if state.search_running and not process_alive:
        state.search_running = False
        state.search_pid = None

    # Read live experiment database
    experiments = _read_experiment_db(config)
    current_count = len(experiments)
    prev_count = state.experiments_run

    # Track the best metric and detect staleness
    prev_best = state.best_metric
    best = _find_best_experiment(experiments)

    if best:
        objectives = best.get("objectives", {})
        if objectives:
            primary_key = next(iter(objectives))
            current_best = objectives[primary_key]

            # Check for improvement
            improved = (
                prev_best is None
                or current_best < prev_best
            )

            if improved:
                state.best_metric = current_best
                state.experiments_at_last_improvement = current_count
                state.stale_count = 0
                artifact = best.get("artifact_path")
                if artifact:
                    state.best_artifact = artifact
            else:
                state.stale_count = current_count - state.experiments_at_last_improvement

    state.experiments_run = current_count

    # Build status
    status: dict[str, Any] = {
        "process_alive": process_alive,
        "search_finished": state.search_running is False and current_count > 0,
        "experiments_run": current_count,
        "new_since_last_check": current_count - prev_count,
        "best_metric": state.best_metric,
        "stale_count": state.stale_count,
    }

    if state.search_started_at:
        elapsed = time.time() - state.search_started_at
        status["elapsed"] = f"{elapsed / 3600:.1f}h" if elapsed > 3600 else f"{elapsed / 60:.0f}m"

    # Check recent log output
    log_path = config.workspace_dir / "search.log"
    if log_path.exists():
        try:
            with open(log_path) as f:
                lines = f.readlines()
            # Last 5 non-empty lines
            recent = [l.rstrip() for l in lines[-10:] if l.strip()][-5:]
            if recent:
                status["recent_log"] = recent
        except OSError:
            pass

    # Build a human-readable summary
    parts = []
    if process_alive:
        parts.append("RUNNING")
    elif current_count > 0:
        parts.append("FINISHED")
    else:
        parts.append("NOT STARTED")

    parts.append(f"{current_count} experiments")
    if state.best_metric is not None:
        parts.append(f"best={state.best_metric:.4f}")
    if state.stale_count > 0:
        parts.append(f"stale for {state.stale_count}")

    status["message"] = ", ".join(parts)

    # If search finished, save the result for deploy
    if not process_alive and current_count > 0 and best:
        from orchestrator.strategies.base import StrategyResult
        objectives = best.get("objectives", {})
        artifact = best.get("artifact_path")
        result = StrategyResult(
            success=True,
            best_artifact=Path(artifact) if artifact else None,
            metrics=objectives,
            experiments_run=current_count,
            summary=f"Completed {current_count} experiments. Best: {objectives}",
        )
        result_path = config.workspace_dir / "strategy_result.json"
        result.save(result_path)

    return status


def tool_deploy(
    config: SubnetConfig, state: AgentState, artifact_path: str = ""
) -> dict[str, Any]:
    """Deploy the best artifact to production."""
    state.phase = "deploying"
    state.phase_started_at = time.time()

    # Resolve artifact path
    deploy_artifact = None
    if artifact_path:
        deploy_artifact = Path(artifact_path)
    elif state.best_artifact:
        deploy_artifact = Path(state.best_artifact)
    else:
        # Try loading from last search result
        result_path = config.workspace_dir / "strategy_result.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    data = json.load(f)
                if data.get("best_artifact"):
                    deploy_artifact = Path(data["best_artifact"])
            except (json.JSONDecodeError, OSError):
                pass

    result = run_deploy(config.subnet_dir, deploy_artifact, config.workspace_dir)

    if result.success:
        state.deployed = True
        state.deployed_at = time.time()
        state.deployed_model_id = result.metadata.get("model_id")
        state.phase = "monitoring"
        state.phase_started_at = time.time()
        state.consecutive_errors = 0
        state.last_error = None
        return {
            "success": True,
            "message": result.message,
            "model_id": state.deployed_model_id,
        }
    else:
        state.consecutive_errors += 1
        state.last_error = result.message
        return {"success": False, "message": result.message}


def tool_check_live_performance(
    config: SubnetConfig, state: AgentState
) -> dict[str, Any]:
    """Check live miner performance via the subnet's monitor script."""
    result = run_monitor(config.subnet_dir, config.workspace_dir)

    state.last_monitor_at = time.time()
    state.last_monitor_healthy = result.healthy

    if result.metrics:
        state.last_emission_share = result.metrics.get("emission_share")
        state.last_rank = result.metrics.get("rank")

    return {
        "healthy": result.healthy,
        "metrics": result.metrics,
        "should_re_evolve": result.should_re_evolve,
        "message": result.message,
    }


def tool_wait(
    config: SubnetConfig, state: AgentState, reason: str = ""
) -> dict[str, Any]:
    """Explicitly do nothing. The reason is logged."""
    return {"success": True, "message": f"Waiting: {reason}"}


# ──────────────────────────────────────────────
# Tool dispatcher
# ──────────────────────────────────────────────

TOOL_MAP = {
    "run_setup": tool_run_setup,
    "start_search": tool_start_search,
    "stop_search": tool_stop_search,
    "get_search_status": tool_get_search_status,
    "deploy": tool_deploy,
    "check_live_performance": tool_check_live_performance,
    "wait": tool_wait,
}


def execute_tool(
    tool_name: str,
    config: SubnetConfig,
    state: AgentState,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch a tool call from the agent."""
    if tool_name not in TOOL_MAP:
        return {"success": False, "message": f"Unknown tool: {tool_name}"}

    fn = TOOL_MAP[tool_name]
    params = params or {}

    try:
        sig = inspect.signature(fn)
        valid_params = {
            k: v for k, v in params.items()
            if k in sig.parameters and k not in ("config", "state")
        }
        return fn(config=config, state=state, **valid_params)
    except Exception as e:
        state.consecutive_errors += 1
        state.last_error = str(e)
        return {"success": False, "message": f"Tool error: {e}"}
