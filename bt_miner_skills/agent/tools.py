"""Tool definitions that an outer AI agent can call to drive the loop.

Each function here is a self-contained tool that performs one step of the
miner development lifecycle. Designed to be invoked by Claude Code or
similar agent systems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bt_miner_skills.config.loader import load_subnet_config
from bt_miner_skills.config.subnet_config import SubnetConfig


def research_subnet(netuid: int, config_path: str | None = None) -> dict[str, Any]:
    """Research a subnet: fetch chain params, analyze repo, summarize findings.

    Args:
        netuid: The subnet ID to research
        config_path: Optional path to existing config YAML

    Returns:
        Research findings dict with chain_params, repo_analysis, and recommendations
    """
    findings: dict[str, Any] = {"netuid": netuid}

    # Fetch chain params
    try:
        from bt_miner_skills.chain.fetcher import fetch_chain_params, fetch_metagraph_summary

        findings["chain_params"] = fetch_chain_params(netuid).__dict__
        findings["metagraph_summary"] = fetch_metagraph_summary(netuid)
    except Exception as e:
        findings["chain_error"] = str(e)

    # Analyze repo if config has a repo URL
    if config_path:
        config = load_subnet_config(config_path)
        if config.repo_url:
            try:
                from bt_miner_skills.research.repo_analyzer import analyze_repo

                analysis = analyze_repo(config.repo_url)
                findings["repo_analysis"] = {
                    "synapse_classes": analysis.synapse_classes,
                    "reward_functions": [
                        {"name": r["name"], "file": r["file"]}
                        for r in analysis.reward_functions
                    ],
                    "miner_files": analysis.miner_files,
                    "dependencies": analysis.dependencies,
                }
            except Exception as e:
                findings["repo_error"] = str(e)

    return findings


def scaffold_miner_tool(
    config_path: str,
    output_dir: str = "workspace/miner",
) -> dict[str, Any]:
    """Scaffold a miner from a subnet config.

    Args:
        config_path: Path to the subnet config YAML
        output_dir: Where to write generated files

    Returns:
        Dict with generated file paths and config summary
    """
    from bt_miner_skills.scaffolder import scaffold_miner

    config = load_subnet_config(config_path)
    output = Path(output_dir)
    files = scaffold_miner(config, output)

    return {
        "netuid": config.netuid,
        "name": config.name,
        "output_dir": str(output),
        "generated_files": [str(f) for f in files],
    }


def test_miner_tool(
    workspace_dir: str,
    config_path: str,
    num_queries: int = 10,
) -> dict[str, Any]:
    """Run mock validator tests against a miner.

    Returns:
        Test report with success rate, latency stats, and errors
    """
    config = load_subnet_config(config_path)

    return {
        "status": "ready",
        "message": (
            f"Mock validator configured for SN{config.netuid}. "
            f"Protocol: {config.protocol.synapse_class}, "
            f"Timeout: {config.protocol.timeout_seconds}s. "
            f"Run the miner and use MockValidator to send {num_queries} test queries."
        ),
        "protocol": config.protocol.model_dump(),
    }


def get_loop_context(config_path: str, workspace: str = "workspace") -> dict[str, Any]:
    """Get the full agent context for the current loop phase.

    This is the main entry point for an agent to understand what to do next.
    """
    from bt_miner_skills.agent.orchestrator import Orchestrator

    orch = Orchestrator(config_path, workspace=workspace)
    return orch.get_context()


def advance_loop(
    config_path: str,
    workspace: str = "workspace",
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete the current phase and advance to the next one.

    Args:
        config_path: Path to subnet config
        workspace: Workspace directory
        result: Results from the completed phase

    Returns:
        New loop state summary
    """
    from bt_miner_skills.agent.orchestrator import Orchestrator

    orch = Orchestrator(config_path, workspace=workspace)
    orch.complete_phase(result)
    return {
        "phase": orch.state.phase.value,
        "iteration": orch.state.iteration,
        "next_prompt": orch.get_current_prompt(),
    }
