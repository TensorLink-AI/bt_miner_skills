"""Model selection strategy — benchmark existing models without training.

Useful for subnets where the task is to pick/serve the best existing model
(e.g., LLM serving, image generation) rather than train one from scratch.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.strategies.base import Strategy, StrategyResult


class ModelSelectionStrategy(Strategy):
    """Evaluate a list of candidate models and select the best one."""

    def __init__(self, config: SubnetConfig) -> None:
        super().__init__(config)
        self.strategy_config = config.strategy.config

        self.candidates_file = self.strategy_config.get("candidates_file", "candidates.yaml")
        self.eval_script = self.strategy_config.get("eval_script", "evaluate.py")
        self.metric_key = self.strategy_config.get("metric_key", "score")
        self.minimize = self.strategy_config.get("minimize", True)
        self.timeout = self.strategy_config.get("timeout", 600)

        self._results: list[dict[str, Any]] = []

    def setup(self) -> bool:
        eval_path = self.config.subnet_dir / self.eval_script
        candidates_path = self.config.subnet_dir / self.candidates_file

        if not eval_path.exists():
            print(f"[model_selection] ERROR: eval script not found: {eval_path}")
            return False

        if not candidates_path.exists():
            print(f"[model_selection] ERROR: candidates file not found: {candidates_path}")
            return False

        print(f"[model_selection] Eval script: {eval_path}")
        print(f"[model_selection] Candidates: {candidates_path}")
        return True

    def run(self) -> StrategyResult:
        import yaml

        candidates_path = self.config.subnet_dir / self.candidates_file
        with open(candidates_path) as f:
            candidates = yaml.safe_load(f)

        if not candidates or "models" not in candidates:
            return StrategyResult(
                success=False,
                summary="No models found in candidates file.",
            )

        models = candidates["models"]
        eval_script = self.config.subnet_dir / self.eval_script

        print(f"[model_selection] Evaluating {len(models)} candidates...")

        best_metric = float("inf") if self.minimize else float("-inf")
        best_model = None
        best_output = None

        for i, model in enumerate(models):
            model_id = model.get("id", model.get("name", f"model_{i}"))
            print(f"[model_selection] Evaluating {model_id} ({i + 1}/{len(models)})")

            try:
                result = subprocess.run(
                    ["python", str(eval_script), json.dumps(model)],
                    cwd=str(self.config.subnet_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                output_lines = result.stdout.strip().split("\n")
                if output_lines:
                    try:
                        output = json.loads(output_lines[-1])
                        metric = output.get(self.metric_key)

                        self._results.append(
                            {"model": model, "metric": metric, "output": output}
                        )

                        if metric is not None:
                            is_better = (
                                metric < best_metric
                                if self.minimize
                                else metric > best_metric
                            )
                            if is_better:
                                best_metric = metric
                                best_model = model
                                best_output = output
                                print(
                                    f"[model_selection] New best: {model_id} "
                                    f"({self.metric_key}={metric})"
                                )
                    except json.JSONDecodeError:
                        print(f"[model_selection] Could not parse output from {model_id}")

            except subprocess.TimeoutExpired:
                print(f"[model_selection] {model_id} timed out")
            except Exception as e:
                print(f"[model_selection] {model_id} failed: {e}")

        if best_model is None:
            return StrategyResult(
                success=False,
                experiments_run=len(self._results),
                summary="No valid results from any candidate.",
            )

        artifact_path = self.config.workspace_dir / "best_model.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump({"model": best_model, "output": best_output}, f, indent=2)

        return StrategyResult(
            success=True,
            best_artifact=artifact_path,
            metrics={self.metric_key: best_metric, **(best_output or {})},
            experiments_run=len(self._results),
            summary=f"Best model: {best_model.get('id', 'unknown')} ({self.metric_key}={best_metric})",
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": "model_selection",
            "candidates_evaluated": len(self._results),
        }
