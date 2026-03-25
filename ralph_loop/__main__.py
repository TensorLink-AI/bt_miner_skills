"""Entry point: python -m ralph_loop"""

import argparse
import logging
import sys

from ralph_loop.config import CHUTES_API_KEY, CHUTES_MODEL, WORKSPACE_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ralph Loop — autonomous Bittensor miner builder using Chutes AI"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--list-skills", action="store_true", help="List discovered skills and exit"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current state for all skills"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset state for the targeted skill(s) and start fresh",
    )
    parser.add_argument(
        "--subnet", "-s", type=str, default=None,
        help="Target a specific subnet by netuid (e.g. 50), name (e.g. synth), "
             "or package name. Without this, all discovered skills run.",
    )
    parser.add_argument(
        "--share-knowledge", action="store_true",
        help="Enable knowledge sharing via the-commons. Ralph will download "
             "the skill from github.com/TensorLink-AI/the-commons and use it "
             "to log experiments and learn from prior attempts.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from ralph_loop.skill_discovery import discover_skills
    from ralph_loop.state import load_state, reset_state

    if args.list_skills:
        skills = discover_skills(filter_subnet=args.subnet)
        for s in skills:
            sn_label = f"SN{s.netuid}" if s.netuid else "no netuid"
            print(f"  {s.name} ({sn_label}) — {s.path}")
            if s.subnet_name:
                print(f"    Subnet: {s.subnet_name}")
            print(f"    References: {list(s.references.keys())}")
        if not skills:
            print("  No skill packages found.")
        return

    if args.reset:
        skills = discover_skills(filter_subnet=args.subnet)
        for s in skills:
            reset_state(s.name)
            print(f"  Reset state for: {s.name}")
        if not skills:
            print("  No skill packages found to reset.")
        return

    if args.status:
        skills = discover_skills(filter_subnet=args.subnet)
        for s in skills:
            state = load_state(s.name)
            sn_label = f"SN{s.netuid}" if s.netuid else "no netuid"
            print(f"  {s.name} ({sn_label}):")
            print(f"    Iterations: {state.iteration_count}")
            print(f"    Files written: {len(state.files_written)}")
            print(f"    Commands run: {len(state.commands_run)}")
            print(f"    Workspace: {state.workspace_dir or '(not started)'}")
            if state.requested_references:
                print(f"    Pending refs: {state.requested_references}")
        return

    # Validate config before starting
    if not CHUTES_API_KEY:
        print("Error: CHUTES_API_KEY environment variable is required.", file=sys.stderr)
        print("  export CHUTES_API_KEY=your_key_here", file=sys.stderr)
        print("  Get your key at https://chutes.ai", file=sys.stderr)
        sys.exit(1)

    print(f"Ralph Loop starting (model={CHUTES_MODEL})")
    print(f"Workspace root: {WORKSPACE_ROOT}")
    if args.subnet:
        print(f"Targeting subnet: {args.subnet}")
    if args.share_knowledge:
        from ralph_loop.config import COMMONS_API_TOKEN, COMMONS_URL
        print("Knowledge sharing: ON (the-commons)")
        if COMMONS_URL:
            print(f"  Commons URL: {COMMONS_URL}")
        if not COMMONS_API_TOKEN:
            print(
                "  Warning: COMMONS_API_TOKEN not set in .env — "
                "auth may be required by the remote server.",
                file=sys.stderr,
            )

    from ralph_loop.loop import run_loop
    run_loop(filter_subnet=args.subnet, share_knowledge=args.share_knowledge)


if __name__ == "__main__":
    main()
