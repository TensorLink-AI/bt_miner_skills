"""Agent tools — actions the overlord LLM can invoke.

Each tool is a function that takes the current context (config, state, etc.)
and returns a result dict. The agent calls these by name; the orchestrator
dispatches and records the outcome.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.lifecycle import run_deploy, run_monitor, run_setup
from orchestrator.state import AgentState

# Track background processes
_search_process: subprocess.Popen | None = None


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
            "Start the model search (evoloop or other strategy). "
            "This launches a background process that runs experiments. "
            "Use get_search_status to check progress. "
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
            "the best result so far."
        ),
        "parameters": {},
    },
    {
        "name": "get_search_status",
        "description": (
            "Check the current search progress: experiments run, best metric, "
            "staleness, whether it's still running. Use this frequently to "
            "decide whether to keep searching or stop."
        ),
        "parameters": {},
    },
    {
        "name": "deploy",
        "description": (
            "Deploy the best model/artifact from the search to production. "
            "Runs the subnet's deploy script. Only call after a successful search."
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
            "Do nothing for now. Use when the current action is still in progress "
            "and needs more time, or when monitoring and everything looks healthy. "
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
    """Launch the search strategy as a background process."""
    global _search_process

    if _search_process is not None and _search_process.poll() is None:
        return {
            "success": False,
            "message": "Search is already running. Use get_search_status or stop_search first.",
        }

    from orchestrator.strategies import get_strategy

    strategy = get_strategy(config)

    # Override max_experiments if provided
    if max_experiments > 0:
        strategy.max_experiments = max_experiments

    if not strategy.setup():
        state.consecutive_errors += 1
        state.last_error = "Strategy setup failed"
        return {"success": False, "message": "Strategy setup failed."}

    # For subprocess-based strategies (evoloop), we can run in background
    # For now, we run synchronously but update state before/after
    state.phase = "searching"
    state.phase_started_at = time.time()
    state.search_running = True
    state.search_started_at = time.time()
    state.experiments_run = 0
    state.stale_count = 0

    result = strategy.run()

    # Update state with results
    state.search_running = False
    state.experiments_run = result.experiments_run

    if result.success:
        state.best_artifact = str(result.best_artifact) if result.best_artifact else None
        # Extract primary metric
        if result.metrics:
            primary_key = next(iter(result.metrics))
            state.best_metric = result.metrics[primary_key]
        state.consecutive_errors = 0
        state.last_error = None

        # Save result for deploy
        result_path = config.workspace_dir / "strategy_result.json"
        result.save(result_path)

        return {
            "success": True,
            "message": result.summary,
            "experiments_run": result.experiments_run,
            "best_metric": state.best_metric,
            "best_artifact": state.best_artifact,
        }
    else:
        state.consecutive_errors += 1
        state.last_error = result.summary
        return {
            "success": False,
            "message": result.summary,
            "experiments_run": result.experiments_run,
        }


def tool_stop_search(config: SubnetConfig, state: AgentState) -> dict[str, Any]:
    """Stop any running search process."""
    global _search_process

    if _search_process is not None and _search_process.poll() is None:
        _search_process.send_signal(signal.SIGINT)
        try:
            _search_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _search_process.kill()
        _search_process = None

    state.search_running = False
    return {
        "success": True,
        "message": f"Search stopped. Best metric so far: {state.best_metric}",
    }


def tool_get_search_status(config: SubnetConfig, state: AgentState) -> dict[str, Any]:
    """Check current search progress by reading the experiment DB."""
    # Try to read evoloop's experiment database
    workspace = config.workspace_dir
    result_path = workspace / "strategy_result.json"

    status: dict[str, Any] = {
        "search_running": state.search_running,
        "experiments_run": state.experiments_run,
        "best_metric": state.best_metric,
        "stale_count": state.stale_count,
    }

    if state.search_started_at:
        elapsed = time.time() - state.search_started_at
        status["elapsed_time"] = f"{elapsed / 3600:.1f}h" if elapsed > 3600 else f"{elapsed / 60:.0f}m"

    if result_path.exists():
        try:
            with open(result_path) as f:
                data = json.load(f)
            status["last_result"] = {
                "success": data.get("success"),
                "experiments_run": data.get("experiments_run"),
                "metrics": data.get("metrics"),
                "summary": data.get("summary"),
            }
        except (json.JSONDecodeError, OSError):
            pass

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
        # Filter params to only those the function accepts
        import inspect
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
