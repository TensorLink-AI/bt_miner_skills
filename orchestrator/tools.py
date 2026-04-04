"""Agent tools — actions the overlord LLM can invoke.

Every tool is non-blocking. Long-running operations (setup, search, deploy,
monitor) launch as background subprocesses via _launch_background_task().
Each writes a result JSON when done. On the next tick the agent re-calls the
same tool, which detects the finished task and reads the result.

Pattern for each async tool:
  1. If already running → check PID alive → if dead, read result file
  2. If not running → launch subprocess, store PID, return "launched"
"""

from __future__ import annotations

import inspect
import json
import os
import signal
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.state import AgentState


# ──────────────────────────────────────────────
# Tool definitions (used to build the LLM prompt)
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "run_setup",
        "description": (
            "Validate prerequisites (data sources, dependencies, eval harness). "
            "Non-blocking: launches in the background and reports result on next call. "
            "Use before starting a search, or if something seems broken."
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
            "Deploy the best model/artifact to production. Non-blocking: "
            "launches deploy script in background, reports result on next call. "
            "Only call after search has produced results."
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
            "Check live miner performance via the subnet's monitor script. "
            "Non-blocking: launches in background, reports result on next call. "
            "Returns health status, metrics (emission share, rank), and trends."
        ),
        "parameters": {},
    },
    {
        "name": "wait",
        "description": (
            "Do nothing for now. Use when background tasks are running and "
            "making progress, or when monitoring and everything looks healthy. "
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
# Generic background task infrastructure
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


def _result_path_for(config: SubnetConfig, task_name: str) -> Path:
    """Convention-based result file: workspace/<task_name>_result.json."""
    return config.workspace_dir / f"{task_name}_result.json"


def _log_path_for(config: SubnetConfig, task_name: str) -> Path:
    """Convention-based log file: workspace/<task_name>.log."""
    return config.workspace_dir / f"{task_name}.log"


def _launch_background_task(
    cmd: list[str],
    cwd: str | None,
    env: dict[str, str],
    result_path: Path,
    log_path: Path,
    capture_stdout: bool = True,
) -> subprocess.Popen | None:
    """Launch a subprocess that writes its result to a JSON file when done.

    Uses a tiny Python wrapper that:
    1. Runs the actual command
    2. Captures stdout/stderr
    3. Writes {exit_code, stdout, stderr} to result_path
    """
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale result from a previous run
    if result_path.exists():
        result_path.unlink()

    if capture_stdout:
        # Wrap in a Python script that captures output and writes result JSON
        wrapper = textwrap.dedent(f"""\
            import subprocess, json, sys
            try:
                r = subprocess.run(
                    {cmd!r},
                    cwd={str(cwd) if cwd else None!r},
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                result = {{"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}}
            except subprocess.TimeoutExpired:
                result = {{"exit_code": -1, "stdout": "", "stderr": "Timed out after 600s"}}
            except Exception as e:
                result = {{"exit_code": -1, "stdout": "", "stderr": str(e)}}
            with open({str(result_path)!r}, "w") as f:
                json.dump(result, f, indent=2)
        """)
        launch_cmd = [sys.executable, "-c", wrapper]
    else:
        # For search: stream stdout to log, no wrapper needed
        launch_cmd = cmd

    try:
        log_file = open(log_path, "a")
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Task started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write(f"{'='*60}\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            launch_cmd,
            cwd=cwd if not capture_stdout else None,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc
    except Exception as e:
        print(f"[bg-task] Failed to launch: {e}")
        return None


def _read_task_result(result_path: Path) -> dict | None:
    """Read a background task's result JSON. Returns None if not ready."""
    if not result_path.exists():
        return None
    try:
        with open(result_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ──────────────────────────────────────────────
# Search-specific helpers (evoloop experiment DB)
# ──────────────────────────────────────────────

def _read_experiment_db(config: SubnetConfig) -> list[dict]:
    """Read evoloop's experiment database from disk."""
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
# Tool implementations — all non-blocking
# ──────────────────────────────────────────────

def tool_run_setup(config: SubnetConfig, state: AgentState) -> dict[str, Any]:
    """Non-blocking setup: launch or read result."""

    # If setup is already running, check if it finished
    if state.setup_running:
        if _pid_is_alive(state.setup_pid):
            return {"success": True, "message": f"Setup still running (PID {state.setup_pid}). Will check next tick."}

        # Process finished — read result
        state.setup_running = False
        state.setup_pid = None

        result = _read_task_result(_result_path_for(config, "setup"))
        if result is None:
            state.consecutive_errors += 1
            state.last_error = "Setup finished but no result file found"
            return {"success": False, "message": "Setup finished but no result file found."}

        if result["exit_code"] == 0:
            state.consecutive_errors = 0
            state.last_error = None
            return {"success": True, "message": "Setup checks passed."}
        else:
            state.consecutive_errors += 1
            stderr_snippet = result.get("stderr", "")[:300]
            state.last_error = f"Setup failed (exit {result['exit_code']}): {stderr_snippet}"
            return {"success": False, "message": state.last_error}

    # Launch setup in background
    setup_script = config.subnet_dir / "setup.py"
    if not setup_script.exists():
        return {"success": True, "message": "No setup.py found, skipping."}

    state.phase = "setup"
    state.phase_started_at = time.time()
    config.workspace_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "WORKSPACE_DIR": str(config.workspace_dir),
        "SUBNET_DIR": str(config.subnet_dir),
    }

    proc = _launch_background_task(
        cmd=[sys.executable, str(setup_script)],
        cwd=str(config.subnet_dir),
        env=env,
        result_path=_result_path_for(config, "setup"),
        log_path=_log_path_for(config, "setup"),
    )

    if proc is None:
        state.consecutive_errors += 1
        state.last_error = "Failed to launch setup process"
        return {"success": False, "message": "Failed to launch setup process."}

    state.setup_running = True
    state.setup_pid = proc.pid
    print(f"[setup] Launched in background (PID {proc.pid})")

    return {"success": True, "message": f"Setup launched in background (PID {proc.pid})."}


def tool_start_search(
    config: SubnetConfig, state: AgentState, max_experiments: int = 0
) -> dict[str, Any]:
    """Launch the search strategy as a background process. Returns immediately."""
    if state.search_running and _pid_is_alive(state.search_pid):
        return {
            "success": False,
            "message": (
                f"Search is already running (PID {state.search_pid}). "
                "Use get_search_status or stop_search first."
            ),
        }

    from orchestrator.strategies import get_strategy
    strategy = get_strategy(config)

    if not strategy.setup():
        state.consecutive_errors += 1
        state.last_error = "Strategy setup failed"
        return {"success": False, "message": "Strategy setup failed."}

    proc = _launch_search_process(config, strategy, max_experiments)
    if proc is None:
        state.consecutive_errors += 1
        state.last_error = "Failed to launch search process"
        return {"success": False, "message": "Failed to launch search process."}

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
    """Launch evoloop (or other strategy) as a background subprocess.

    Search is special: it streams to a log file (no wrapper) because it's
    long-running and we read experiment DB from disk instead.
    """
    sc = config.strategy.config
    evoloop_dir = Path(
        sc.get("evoloop_dir", os.environ.get("EVOLOOP_DIR", ""))
    )

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

    log_path = _log_path_for(config, "search")

    print(f"[search] Launching: {' '.join(cmd)}")
    print(f"[search] Log: {log_path}")

    # Search uses direct streaming (no wrapper) — we read experiment DB from disk
    return _launch_background_task(
        cmd=cmd,
        cwd=cwd,
        env=env,
        result_path=_result_path_for(config, "search"),  # not used for search
        log_path=log_path,
        capture_stdout=False,
    )


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
        os.kill(pid, signal.SIGINT)

        # Wait up to 60s for graceful shutdown
        for _ in range(12):
            time.sleep(5)
            if not _pid_is_alive(pid):
                break
        else:
            print(f"[search] PID {pid} didn't stop gracefully. Sending SIGKILL.")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)

    except ProcessLookupError:
        pass
    except Exception as e:
        return {"success": False, "message": f"Failed to stop PID {pid}: {e}"}

    state.search_running = False
    state.search_pid = None

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

        from orchestrator.strategies.base import StrategyResult
        result = StrategyResult(
            success=True,
            best_artifact=Path(artifact) if artifact else None,
            metrics=objectives,
            experiments_run=len(experiments),
            summary=f"Stopped after {len(experiments)} experiments. Best: {objectives}",
        )
        result.save(config.workspace_dir / "strategy_result.json")

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

    if state.search_running and not process_alive:
        state.search_running = False
        state.search_pid = None

    experiments = _read_experiment_db(config)
    current_count = len(experiments)
    prev_count = state.experiments_run

    prev_best = state.best_metric
    best = _find_best_experiment(experiments)

    if best:
        objectives = best.get("objectives", {})
        if objectives:
            primary_key = next(iter(objectives))
            current_best = objectives[primary_key]

            improved = prev_best is None or current_best < prev_best

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

    log_path = _log_path_for(config, "search")
    if log_path.exists():
        try:
            with open(log_path) as f:
                lines = f.readlines()
            recent = [l.rstrip() for l in lines[-10:] if l.strip()][-5:]
            if recent:
                status["recent_log"] = recent
        except OSError:
            pass

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
        result.save(config.workspace_dir / "strategy_result.json")

    return status


def tool_deploy(
    config: SubnetConfig, state: AgentState, artifact_path: str = ""
) -> dict[str, Any]:
    """Non-blocking deploy: launch or read result."""

    # If deploy is already running, check if it finished
    if state.deploy_running:
        if _pid_is_alive(state.deploy_pid):
            return {"success": True, "message": f"Deploy still running (PID {state.deploy_pid}). Will check next tick."}

        # Process finished — read result
        state.deploy_running = False
        state.deploy_pid = None

        result = _read_task_result(_result_path_for(config, "deploy"))
        if result is None:
            state.consecutive_errors += 1
            state.last_error = "Deploy finished but no result file found"
            return {"success": False, "message": "Deploy finished but no result file found."}

        if result["exit_code"] == 0:
            # Try to parse metadata from stdout (last line = JSON)
            metadata = {}
            stdout_lines = result.get("stdout", "").strip().split("\n")
            if stdout_lines:
                try:
                    metadata = json.loads(stdout_lines[-1])
                except json.JSONDecodeError:
                    pass

            state.deployed = True
            state.deployed_at = time.time()
            state.deployed_model_id = metadata.get("model_id")
            state.phase = "monitoring"
            state.phase_started_at = time.time()
            state.consecutive_errors = 0
            state.last_error = None
            return {
                "success": True,
                "message": "Deployment complete.",
                "model_id": state.deployed_model_id,
            }
        else:
            stderr_snippet = result.get("stderr", "")[:300]
            state.consecutive_errors += 1
            state.last_error = f"Deploy failed (exit {result['exit_code']}): {stderr_snippet}"
            return {"success": False, "message": state.last_error}

    # Resolve artifact path
    deploy_artifact = ""
    if artifact_path:
        deploy_artifact = artifact_path
    elif state.best_artifact:
        deploy_artifact = state.best_artifact
    else:
        result_file = config.workspace_dir / "strategy_result.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if data.get("best_artifact"):
                    deploy_artifact = data["best_artifact"]
            except (json.JSONDecodeError, OSError):
                pass

    # Find deploy script
    deploy_script = config.subnet_dir / "deploy.py"
    if not deploy_script.exists():
        deploy_script = config.subnet_dir / "deploy.sh"
    if not deploy_script.exists():
        return {"success": False, "message": f"No deploy script found in {config.subnet_dir}"}

    cmd = (
        [sys.executable, str(deploy_script), deploy_artifact]
        if deploy_script.suffix == ".py"
        else ["bash", str(deploy_script), deploy_artifact]
    )

    state.phase = "deploying"
    state.phase_started_at = time.time()

    env = {
        **os.environ,
        "WORKSPACE_DIR": str(config.workspace_dir),
        "ARTIFACT_PATH": deploy_artifact,
    }

    proc = _launch_background_task(
        cmd=cmd,
        cwd=str(config.subnet_dir),
        env=env,
        result_path=_result_path_for(config, "deploy"),
        log_path=_log_path_for(config, "deploy"),
    )

    if proc is None:
        state.consecutive_errors += 1
        state.last_error = "Failed to launch deploy process"
        return {"success": False, "message": "Failed to launch deploy process."}

    state.deploy_running = True
    state.deploy_pid = proc.pid
    print(f"[deploy] Launched in background (PID {proc.pid})")

    return {"success": True, "message": f"Deploy launched in background (PID {proc.pid})."}


def tool_check_live_performance(
    config: SubnetConfig, state: AgentState
) -> dict[str, Any]:
    """Non-blocking monitor: launch or read result."""

    # If monitor is already running, check if it finished
    if state.monitor_running:
        if _pid_is_alive(state.monitor_pid):
            return {"success": True, "message": f"Monitor still running (PID {state.monitor_pid}). Will check next tick."}

        # Process finished — read result
        state.monitor_running = False
        state.monitor_pid = None

        result = _read_task_result(_result_path_for(config, "monitor"))
        if result is None:
            return {"success": False, "message": "Monitor finished but no result file found."}

        # Parse monitor JSON output from stdout
        monitor_data = {}
        stdout_lines = result.get("stdout", "").strip().split("\n")
        if stdout_lines:
            try:
                monitor_data = json.loads(stdout_lines[-1])
            except json.JSONDecodeError:
                pass

        healthy = monitor_data.get("healthy", result["exit_code"] == 0)
        metrics = monitor_data.get("metrics", {})
        should_re_evolve = monitor_data.get("should_re_evolve", False)

        state.last_monitor_at = time.time()
        state.last_monitor_healthy = healthy
        if metrics:
            state.last_emission_share = metrics.get("emission_share")
            state.last_rank = metrics.get("rank")
            state.add_monitor_snapshot(metrics)

        # Build trend info
        trend_info = _build_trend_info(state.monitor_history)

        return {
            "healthy": healthy,
            "metrics": metrics,
            "should_re_evolve": should_re_evolve,
            "trend": trend_info,
            "message": monitor_data.get("message", f"exit_code={result['exit_code']}"),
        }

    # Launch monitor in background
    monitor_script = config.subnet_dir / "monitor.py"
    if not monitor_script.exists():
        return {"healthy": True, "message": "No monitor.py — skipping."}

    env = {
        **os.environ,
        "WORKSPACE_DIR": str(config.workspace_dir),
    }

    proc = _launch_background_task(
        cmd=[sys.executable, str(monitor_script)],
        cwd=str(config.subnet_dir),
        env=env,
        result_path=_result_path_for(config, "monitor"),
        log_path=_log_path_for(config, "monitor"),
    )

    if proc is None:
        return {"success": False, "message": "Failed to launch monitor process."}

    state.monitor_running = True
    state.monitor_pid = proc.pid
    print(f"[monitor] Launched in background (PID {proc.pid})")

    return {"success": True, "message": f"Monitor launched in background (PID {proc.pid})."}


def _build_trend_info(history: list[dict]) -> dict[str, Any]:
    """Analyze monitor history to detect trends."""
    if len(history) < 2:
        return {}

    trend: dict[str, Any] = {}

    # Emission share trend
    emissions = [(h["timestamp"], h.get("emission_share")) for h in history if h.get("emission_share") is not None]
    if len(emissions) >= 2:
        latest = emissions[-1][1]
        oldest = emissions[0][1]
        hours_span = (emissions[-1][0] - emissions[0][0]) / 3600

        if hours_span > 0:
            direction = "rising" if latest > oldest else "declining" if latest < oldest else "stable"
            trend["emission_share"] = {
                "current": latest,
                "oldest": oldest,
                "direction": direction,
                "span_hours": round(hours_span, 1),
            }

    # Rank trend
    ranks = [(h["timestamp"], h.get("rank")) for h in history if h.get("rank") is not None]
    if len(ranks) >= 2:
        latest_rank = ranks[-1][1]
        oldest_rank = ranks[0][1]
        direction = "improving" if latest_rank < oldest_rank else "declining" if latest_rank > oldest_rank else "stable"
        trend["rank"] = {
            "current": latest_rank,
            "oldest": oldest_rank,
            "direction": direction,
        }

    return trend


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
