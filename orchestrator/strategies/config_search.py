"""Config search strategy — grid/random search over a configuration space.

Useful for subnets where the miner is an existing tool/model and you just
need to find the best parameters (not train a new model).
"""

from __future__ import annotations

import itertools
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.strategies.base import Strategy, StrategyResult


class ConfigSearchStrategy(Strategy):
    """Search over a defined configuration space by running a script with each config."""

    def __init__(self, config: SubnetConfig) -> None:
        super().__init__(config)
        self.strategy_config = config.strategy.config

        self.search_script = self.strategy_config.get("search_script", "search.py")
        self.search_space = self.strategy_config.get("search_space", {})
        self.mode = self.strategy_config.get("mode", "grid")  # grid | random
        self.max_trials = self.strategy_config.get(
            "max_trials", config.convergence.max_experiments or 100
        )
        self.metric_key = self.strategy_config.get("metric_key", "score")
        self.minimize = self.strategy_config.get("minimize", True)
        self.timeout = self.strategy_config.get("timeout", 300)

        self._results: list[dict[str, Any]] = []

    def setup(self) -> bool:
        script_path = self.config.subnet_dir / self.search_script
        if not script_path.exists():
            print(f"[config_search] ERROR: search script not found: {script_path}")
            return False

        if not self.search_space:
            print("[config_search] ERROR: no search_space defined in strategy config")
            return False

        print(f"[config_search] Search space: {len(self.search_space)} parameters")
        print(f"[config_search] Mode: {self.mode}, max trials: {self.max_trials}")
        return True

    def run(self) -> StrategyResult:
        configs = self._generate_configs()
        script_path = self.config.subnet_dir / self.search_script

        print(f"[config_search] Running {len(configs)} configurations...")

        best_metric = float("inf") if self.minimize else float("-inf")
        best_config = None
        best_output = None

        for i, trial_config in enumerate(configs):
            print(f"[config_search] Trial {i + 1}/{len(configs)}: {trial_config}")

            try:
                result = subprocess.run(
                    ["python", str(script_path), json.dumps(trial_config)],
                    cwd=str(self.config.subnet_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                # Parse metric from stdout (expect JSON on last line)
                output_lines = result.stdout.strip().split("\n")
                if output_lines:
                    try:
                        output = json.loads(output_lines[-1])
                        metric = output.get(self.metric_key)

                        self._results.append(
                            {"config": trial_config, "metric": metric, "output": output}
                        )

                        if metric is not None:
                            is_better = (
                                metric < best_metric
                                if self.minimize
                                else metric > best_metric
                            )
                            if is_better:
                                best_metric = metric
                                best_config = trial_config
                                best_output = output
                                print(
                                    f"[config_search] New best: {self.metric_key}={metric}"
                                )
                    except json.JSONDecodeError:
                        print(f"[config_search] Could not parse output: {output_lines[-1]}")

            except subprocess.TimeoutExpired:
                print(f"[config_search] Trial {i + 1} timed out after {self.timeout}s")
            except Exception as e:
                print(f"[config_search] Trial {i + 1} failed: {e}")

        if best_config is None:
            return StrategyResult(
                success=False,
                experiments_run=len(self._results),
                summary="No valid results from any trial.",
            )

        # Save best config as artifact
        artifact_path = self.config.workspace_dir / "best_config.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump({"config": best_config, "output": best_output}, f, indent=2)

        return StrategyResult(
            success=True,
            best_artifact=artifact_path,
            metrics={self.metric_key: best_metric, **(best_output or {})},
            experiments_run=len(self._results),
            summary=f"Best config: {best_config} ({self.metric_key}={best_metric})",
        )

    def _generate_configs(self) -> list[dict[str, Any]]:
        """Generate trial configurations from the search space."""
        if self.mode == "grid":
            keys = list(self.search_space.keys())
            values = [self.search_space[k] for k in keys]
            configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
            if len(configs) > self.max_trials:
                random.shuffle(configs)
                configs = configs[: self.max_trials]
            return configs

        # Random search
        configs = []
        for _ in range(self.max_trials):
            config = {}
            for key, values in self.search_space.items():
                config[key] = random.choice(values)
            configs.append(config)
        return configs

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": "config_search",
            "trials_completed": len(self._results),
            "max_trials": self.max_trials,
        }
