"""Entry point: python -m ralph_loop"""

import argparse
import logging
import sys

from ralph_loop.config import CHUTES_API_KEY, CHUTES_MODEL, LOOP_INTERVAL_SECONDS


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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from ralph_loop.skill_discovery import discover_skills
    from ralph_loop.state import load_state

    if args.list_skills:
        skills = discover_skills()
        for s in skills:
            print(f"  {s.name} ({s.path})")
            print(f"    References: {list(s.references.keys())}")
        if not skills:
            print("  No skill packages found.")
        return

    if args.status:
        skills = discover_skills()
        for s in skills:
            state = load_state(s.name)
            print(f"  {s.name}:")
            print(f"    Current phase: {state.current_phase}")
            print(f"    Iterations: {state.iteration_count}")
            print(f"    Phase status: {state.phase_status}")
        return

    # Validate config before starting
    if not CHUTES_API_KEY:
        print("Error: CHUTES_API_KEY environment variable is required.", file=sys.stderr)
        print("  export CHUTES_API_KEY=your_key_here", file=sys.stderr)
        print("  Get your key at https://chutes.ai", file=sys.stderr)
        sys.exit(1)

    print(f"Ralph Loop starting (model={CHUTES_MODEL}, interval={LOOP_INTERVAL_SECONDS}s)")

    from ralph_loop.loop import run_loop
    run_loop()


if __name__ == "__main__":
    main()
