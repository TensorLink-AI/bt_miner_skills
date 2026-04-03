"""Main orchestrator — generic lifecycle engine for Bittensor subnet miners.

The orchestrator knows the lifecycle but not the technique:
  1. Setup — prepare prerequisites (subnet-specific)
  2. Search — iterate toward a good solution (strategy-specific)
  3. Deploy — put it into production (subnet-specific)
  4. Monitor — watch live performance, re-search if degrading

It does NOT know about CRPS, price forecasting, LLM serving, or any other
subnet-specific domain. That knowledge lives in:
  - subnets/<name>/setup.py, deploy.py, monitor.py
  - The strategy plugin (evoloop, config_search, etc.)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from orchestrator.config import SubnetConfig, load_subnet_config
from orchestrator.lifecycle import run_deploy, run_monitor, run_setup
from orchestrator.strategies import get_strategy


def discover_subnets(subnets_dir: Path) -> list[Path]:
    """Find all subnet directories (those containing subnet.yaml)."""
    if not subnets_dir.exists():
        return []
    return sorted(
        d for d in subnets_dir.iterdir()
        if d.is_dir() and (d / "subnet.yaml").exists()
    )


def run_subnet(
    config: SubnetConfig,
    phase: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Run the full lifecycle (or a specific phase) for a subnet."""
    print(f"\n{'=' * 60}")
    print(f"  Subnet: {config.name} (netuid {config.netuid}, {config.network})")
    print(f"  Strategy: {config.strategy.type}")
    print(f"  Workspace: {config.workspace_dir}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("[orchestrator] Dry run — would execute the above.")
        return True

    phases_to_run = (
        [phase] if phase else ["setup", "search", "deploy", "monitor"]
    )

    # --- SETUP ---
    if "setup" in phases_to_run:
        print("\n--- Phase: Setup ---")
        if not run_setup(config.subnet_dir, config.workspace_dir):
            print("[orchestrator] Setup failed. Stopping.")
            return False

    # --- SEARCH ---
    if "search" in phases_to_run:
        print("\n--- Phase: Search ---")
        strategy = get_strategy(config)

        if not strategy.setup():
            print(f"[orchestrator] Strategy setup failed ({strategy.name}). Stopping.")
            return False

        result = strategy.run()

        # Save result
        result_path = config.workspace_dir / "strategy_result.json"
        result.save(result_path)
        print(f"[orchestrator] Strategy result saved to: {result_path}")

        if not result.success:
            print(f"[orchestrator] Strategy failed: {result.summary}")
            return False

        print(f"[orchestrator] Strategy succeeded: {result.summary}")

        if result.best_artifact:
            print(f"[orchestrator] Best artifact: {result.best_artifact}")

    # --- DEPLOY ---
    if "deploy" in phases_to_run:
        print("\n--- Phase: Deploy ---")

        # Load best artifact from strategy result
        import json
        result_path = config.workspace_dir / "strategy_result.json"
        artifact_path = None
        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            if result_data.get("best_artifact"):
                artifact_path = Path(result_data["best_artifact"])

        deploy_result = run_deploy(
            config.subnet_dir, artifact_path, config.workspace_dir
        )

        if not deploy_result.success:
            print(f"[orchestrator] Deploy failed: {deploy_result.message}")
            return False

        print(f"[orchestrator] Deploy succeeded: {deploy_result.message}")

    # --- MONITOR ---
    if "monitor" in phases_to_run:
        print("\n--- Phase: Monitor ---")
        check_interval = config.monitor.check_interval_minutes * 60

        if phase == "monitor":
            # Continuous monitoring loop
            print(f"[orchestrator] Monitoring every {config.monitor.check_interval_minutes}m...")
            print("[orchestrator] Press Ctrl+C to stop.\n")

            try:
                while True:
                    monitor_result = run_monitor(config.subnet_dir, config.workspace_dir)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    status = "HEALTHY" if monitor_result.healthy else "UNHEALTHY"
                    print(f"[{timestamp}] {status}: {monitor_result.message}")

                    if monitor_result.metrics:
                        for k, v in monitor_result.metrics.items():
                            print(f"  {k}: {v}")

                    if monitor_result.should_re_evolve:
                        print("[orchestrator] Re-evolution triggered!")
                        # Re-run search + deploy
                        run_subnet(config, phase="search")
                        run_subnet(config, phase="deploy")

                    time.sleep(check_interval)
            except KeyboardInterrupt:
                print("\n[orchestrator] Monitoring stopped.")
        else:
            # Single check as part of full lifecycle
            monitor_result = run_monitor(config.subnet_dir, config.workspace_dir)
            if monitor_result.healthy:
                print(f"[orchestrator] Monitor: {monitor_result.message}")
            else:
                print(f"[orchestrator] Monitor WARNING: {monitor_result.message}")

    print(f"\n[orchestrator] Done with {config.name}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subnet miner orchestrator — generic lifecycle engine",
    )
    parser.add_argument(
        "--subnet",
        action="append",
        default=[],
        help="Subnet name(s) to run (from subnets/ directory). Can specify multiple.",
    )
    parser.add_argument(
        "--subnets-dir",
        type=Path,
        default=Path(__file__).parent.parent / "subnets",
        help="Directory containing subnet packages (default: ./subnets)",
    )
    parser.add_argument(
        "--phase",
        choices=["setup", "search", "deploy", "monitor"],
        help="Run only a specific phase (default: full lifecycle).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered subnets and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing.",
    )

    args = parser.parse_args()

    # Discover available subnets
    subnet_dirs = discover_subnets(args.subnets_dir)

    if args.list:
        if not subnet_dirs:
            print(f"No subnets found in {args.subnets_dir}")
            sys.exit(0)
        print("Available subnets:")
        for d in subnet_dirs:
            config = load_subnet_config(d)
            print(f"  {config.name} (netuid {config.netuid}, {config.network}) "
                  f"[strategy: {config.strategy.type}]")
        sys.exit(0)

    # Select subnets to run
    if args.subnet:
        selected = []
        for name in args.subnet:
            matching = [d for d in subnet_dirs if d.name == name]
            if not matching:
                print(f"Subnet '{name}' not found in {args.subnets_dir}")
                print(f"Available: {[d.name for d in subnet_dirs]}")
                sys.exit(1)
            selected.extend(matching)
    else:
        selected = subnet_dirs

    if not selected:
        print(f"No subnets found in {args.subnets_dir}")
        print("Create a subnet package with subnet.yaml in the subnets/ directory.")
        sys.exit(1)

    # Run each subnet
    for subnet_dir in selected:
        config = load_subnet_config(subnet_dir)
        success = run_subnet(config, phase=args.phase, dry_run=args.dry_run)
        if not success:
            print(f"\n[orchestrator] FAILED: {config.name}")
            sys.exit(1)


if __name__ == "__main__":
    main()
