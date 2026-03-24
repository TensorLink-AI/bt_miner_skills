"""CLI entry point for bt_miner_skills."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_scaffold(args):
    """Scaffold a miner from a subnet config."""
    from bt_miner_skills.config.loader import load_subnet_config
    from bt_miner_skills.scaffolder import scaffold_miner

    config = load_subnet_config(args.config)
    output = Path(args.output)
    files = scaffold_miner(config, output)
    print(f"Scaffolded miner for SN{config.netuid} ({config.name}):")
    for f in files:
        print(f"  {f}")


def cmd_show_config(args):
    """Show a parsed subnet config."""
    from bt_miner_skills.config.loader import load_subnet_config

    config = load_subnet_config(args.config)
    print(json.dumps(config.model_dump(), indent=2, default=str))


def cmd_list_skills(args):
    """List available skills."""
    from bt_miner_skills.skills.registry import list_skills

    skills = list_skills()
    for skill in skills:
        print(f"  [{skill.category}] {skill.name}: {skill.description}")
        if skill.dependencies:
            print(f"    deps: {', '.join(skill.dependencies)}")


def cmd_loop_status(args):
    """Show the current agent loop status for a subnet."""
    from bt_miner_skills.agent.loop import LoopState

    workspace = Path(args.workspace)
    state = LoopState.load(workspace, args.netuid)
    print(f"Subnet: SN{state.netuid}")
    print(f"Phase:  {state.phase.value}")
    print(f"Iter:   {state.iteration}")
    print(f"Errors: {len(state.errors)}")
    if state.history:
        print(f"History: {len(state.history)} entries")
        last = state.history[-1]
        print(f"  Last: {last['phase']} (iter {last['iteration']})")


def cmd_loop_context(args):
    """Get the full agent context for the current loop phase."""
    from bt_miner_skills.agent.orchestrator import Orchestrator

    orch = Orchestrator(args.config, workspace=args.workspace)
    context = orch.get_context()
    print(json.dumps(context, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="bt_miner_skills - Agent-driven Bittensor miner builder"
    )
    sub = parser.add_subparsers(dest="command")

    # scaffold
    p = sub.add_parser("scaffold", help="Scaffold a miner from config")
    p.add_argument("config", help="Path to subnet config YAML")
    p.add_argument("-o", "--output", default="workspace/miner", help="Output directory")

    # show-config
    p = sub.add_parser("show-config", help="Show parsed subnet config")
    p.add_argument("config", help="Path to subnet config YAML")

    # list-skills
    sub.add_parser("list-skills", help="List available skills")

    # loop-status
    p = sub.add_parser("loop-status", help="Show agent loop status")
    p.add_argument("netuid", type=int, help="Subnet ID")
    p.add_argument("-w", "--workspace", default="workspace", help="Workspace directory")

    # loop-context
    p = sub.add_parser("loop-context", help="Get agent context for current phase")
    p.add_argument("config", help="Path to subnet config YAML")
    p.add_argument("-w", "--workspace", default="workspace", help="Workspace directory")

    args = parser.parse_args()

    commands = {
        "scaffold": cmd_scaffold,
        "show-config": cmd_show_config,
        "list-skills": cmd_list_skills,
        "loop-status": cmd_loop_status,
        "loop-context": cmd_loop_context,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
