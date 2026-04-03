"""Custom strategy — runs a user-provided script as the search loop.

Full escape hatch for subnets that don't fit the other patterns.
The script is expected to:
  1. Do whatever iteration it needs
  2. Write results to a JSON file at a known path
  3. Exit 0 on success
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.strategies.base import Strategy, StrategyResult


class CustomStrategy(Strategy):
    """Run a user-provided script as the iteration strategy."""

    def __init__(self, config: SubnetConfig) -> None:
        super().__init__(config)
        self.strategy_config = config.strategy.config

        self.script = self.strategy_config.get("script", "iterate.py")
        self.results_file = self.strategy_config.get("results_file", "results.json")
        self.timeout = self.strategy_config.get("timeout", 3600)

    def setup(self) -> bool:
        script_path = self.config.subnet_dir / self.script
        if not script_path.exists():
            print(f"[custom] ERROR: script not found: {script_path}")
            return False

        print(f"[custom] Script: {script_path}")
        return True

    def run(self) -> StrategyResult:
        script_path = self.config.subnet_dir / self.script
        results_path = self.config.workspace_dir / self.results_file

        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)

        print(f"[custom] Running: {script_path}")

        try:
            result = subprocess.run(
                ["python", str(script_path)],
                cwd=str(self.config.subnet_dir),
                env={
                    **__import__("os").environ,
                    "WORKSPACE_DIR": str(self.config.workspace_dir),
                    "RESULTS_FILE": str(results_path),
                    "SUBNET_NAME": self.config.name,
                    "SUBNET_NETUID": str(self.config.netuid),
                },
                capture_output=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return StrategyResult(
                success=False,
                summary=f"Script timed out after {self.timeout}s.",
            )
        except Exception as e:
            return StrategyResult(success=False, summary=f"Script failed: {e}")

        if result.returncode != 0:
            return StrategyResult(
                success=False,
                summary=f"Script exited with code {result.returncode}.",
            )

        # Read results file
        if results_path.exists():
            try:
                with open(results_path) as f:
                    results = json.load(f)

                return StrategyResult(
                    success=results.get("success", True),
                    best_artifact=Path(results["artifact"]) if "artifact" in results else None,
                    metrics=results.get("metrics", {}),
                    experiments_run=results.get("experiments_run", 1),
                    summary=results.get("summary", "Custom strategy completed."),
                )
            except (json.JSONDecodeError, OSError) as e:
                return StrategyResult(
                    success=True,
                    summary=f"Script succeeded but results file unreadable: {e}",
                )

        return StrategyResult(
            success=True,
            summary="Script completed. No results file written.",
        )

    def get_status(self) -> dict[str, Any]:
        results_path = self.config.workspace_dir / self.results_file
        return {
            "strategy": "custom",
            "has_results": results_path.exists() if self.config.workspace_dir else False,
        }
