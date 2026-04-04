"""Status snapshot builder — gathers all context for the agent's decision.

Builds a structured text summary of the current situation that the LLM
reads to decide what action to take next.
"""

from __future__ import annotations

import time
from orchestrator.config import SubnetConfig
from orchestrator.state import AgentState


def build_snapshot(config: SubnetConfig, state: AgentState) -> str:
    """Build a status snapshot for the agent to reason over."""
    now = time.time()
    lines = []

    lines.append(f"Subnet: {config.name} (netuid {config.netuid}, {config.network})")
    lines.append(f"Strategy: {config.strategy.type}")
    lines.append(f"Current phase: {state.phase} ({state.time_in_phase_str()} in this phase)")
    lines.append("")

    # --- Search status ---
    lines.append("== Search ==")
    if state.search_running:
        elapsed = now - state.search_started_at if state.search_started_at else 0
        elapsed_str = f"{elapsed / 3600:.1f}h" if elapsed > 3600 else f"{elapsed / 60:.0f}m"
        pid_info = f", PID {state.search_pid}" if state.search_pid else ""
        lines.append(f"  Status: RUNNING (started {elapsed_str} ago{pid_info})")
    elif state.experiments_run > 0 and state.phase == "searching":
        lines.append(f"  Status: FINISHED (search process exited)")
    else:
        lines.append(f"  Status: not running")

    lines.append(f"  Experiments run: {state.experiments_run}")

    if state.best_metric is not None:
        lines.append(f"  Best metric: {state.best_metric}")
    else:
        lines.append(f"  Best metric: none yet")

    if state.stale_count > 0:
        lines.append(f"  Stale count: {state.stale_count} experiments since last improvement")

    if state.best_artifact:
        lines.append(f"  Best artifact: {state.best_artifact}")

    lines.append("")

    # --- Deployment status ---
    lines.append("== Deployment ==")
    if state.deployed:
        deployed_ago = now - state.deployed_at if state.deployed_at else 0
        deployed_str = f"{deployed_ago / 3600:.1f}h" if deployed_ago > 3600 else f"{deployed_ago / 60:.0f}m"
        lines.append(f"  Status: DEPLOYED ({deployed_str} ago)")
        if state.deployed_model_id:
            lines.append(f"  Model: {state.deployed_model_id}")
    else:
        lines.append(f"  Status: not deployed")

    lines.append("")

    # --- Live performance ---
    lines.append("== Live Performance ==")
    if state.last_monitor_at:
        monitor_ago = now - state.last_monitor_at
        monitor_str = f"{monitor_ago / 60:.0f}m" if monitor_ago < 3600 else f"{monitor_ago / 3600:.1f}h"
        lines.append(f"  Last checked: {monitor_str} ago")
        lines.append(f"  Healthy: {state.last_monitor_healthy}")
        if state.last_emission_share is not None:
            lines.append(f"  Emission share: {state.last_emission_share:.4f}")
        if state.last_rank is not None:
            lines.append(f"  Rank: {state.last_rank}")
    else:
        lines.append(f"  Not yet checked")

    lines.append("")

    # --- Goals (from config) ---
    lines.append("== Goals ==")
    comp = config.competitiveness
    lines.append(f"  Beat baseline by: {comp.baseline_improvement:.0%}")
    lines.append(f"  Beat median on: {comp.min_asset_coverage:.0%} of assets")
    lines.append(f"  Stable for: {comp.consistency_window_hours}h before production")

    conv = config.convergence
    lines.append(f"  Stop search if stale for: {conv.stale_threshold} experiments")
    lines.append(f"  Minimum experiments: {conv.min_experiments}")
    if conv.max_experiments:
        lines.append(f"  Hard ceiling: {conv.max_experiments} experiments")

    lines.append("")

    # --- Errors ---
    if state.consecutive_errors > 0:
        lines.append("== Errors ==")
        lines.append(f"  Consecutive errors: {state.consecutive_errors}")
        if state.last_error:
            lines.append(f"  Last error: {state.last_error}")
        lines.append("")

    # --- Recent decisions ---
    if state.decision_log:
        lines.append("== Recent Decisions ==")
        for entry in state.decision_log[-5:]:
            lines.append(f"  [{entry.get('time_str', '?')}] {entry['action']}: {entry['reasoning']}")
            if entry.get("result"):
                lines.append(f"    Result: {entry['result']}")
        lines.append("")

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are the overlord agent managing a Bittensor subnet miner. Your job is to \
look at the current situation and decide what to do next.

You have these tools available:
- run_setup: Validate prerequisites (data sources, dependencies, eval harness)
- start_search: Launch model search (evoloop). Only if no search is running.
- stop_search: Stop the current search
- get_search_status: Check search progress (experiments, best metric, staleness)
- deploy: Deploy the best model to production
- check_live_performance: Check live metrics via the subnet's monitor
- wait: Explicitly do nothing (must provide a reason)

Decision principles:
1. Don't rush. If search is running and improving, let it run.
2. If search is stale (no improvement for many experiments), stop and deploy best-so-far.
3. Always run setup before the first search.
4. After deploying, monitor. If performance degrades, re-search.
5. If something is broken (errors, failed setup), diagnose before retrying.
6. You can only call ONE tool per tick. Choose the most important action.

Respond with a JSON object:
{
    "reasoning": "Brief explanation of what you see and why you're choosing this action",
    "tool": "tool_name",
    "params": {}
}
"""
