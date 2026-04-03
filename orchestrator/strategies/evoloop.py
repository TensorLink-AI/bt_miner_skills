"""Evoloop strategy — evolutionary model search via the evoloop engine.

Invokes evoloop as a subprocess, monitors its experiment database for
convergence, and returns the best model artifact.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.strategies.base import Strategy, StrategyResult


class EvoloopStrategy(Strategy):
    """Runs evoloop's evolutionary loop to search for the best model."""

    def __init__(self, config: SubnetConfig) -> None:
        super().__init__(config)
        self.strategy_config = config.strategy.config

        # Where evoloop lives (configurable via env or strategy config)
        self.evoloop_dir = Path(
            self.strategy_config.get(
                "evoloop_dir",
                os.environ.get("EVOLOOP_DIR", ""),
            )
        )
        # Task dir within the subnet package
        self.task_dir = self.strategy_config.get("task_dir", "evoloop_task/")
        self.backend = self.strategy_config.get("backend", "basilica")
        self.gpu = self.strategy_config.get("gpu", "A4000")
        self.gpu_count = self.strategy_config.get("gpu_count", 1)
        self.time_budget = self.strategy_config.get("time_budget", 600)
        self.llm_provider = self.strategy_config.get("llm_provider", "openai")
        self.llm_model = self.strategy_config.get("llm_model", "gpt-4.1")
        self.llm_model_strong = self.strategy_config.get("llm_model_strong", "o3")

        # Convergence from top-level config
        self.max_experiments = (
            config.convergence.max_experiments
            or self.strategy_config.get("max_experiments", 0)
        )
        self.stale_threshold = config.convergence.stale_threshold
        self.min_experiments = config.convergence.min_experiments

        # Internal state
        self._process: subprocess.Popen | None = None
        self._experiment_db_path: Path | None = None

    def setup(self) -> bool:
        """Validate evoloop is available and provision the task directory."""
        if not self.evoloop_dir or not self.evoloop_dir.exists():
            print(f"[evoloop] ERROR: evoloop_dir not found: {self.evoloop_dir}")
            print("[evoloop] Set EVOLOOP_DIR env var or strategy.config.evoloop_dir")
            return False

        loop_py = self.evoloop_dir / "loop.py"
        if not loop_py.exists():
            print(f"[evoloop] ERROR: loop.py not found in {self.evoloop_dir}")
            return False

        # Copy task files from subnet package into evoloop's tasks/ directory
        source_task_dir = self.config.subnet_dir / self.task_dir
        if not source_task_dir.exists():
            print(f"[evoloop] ERROR: task dir not found: {source_task_dir}")
            return False

        target_task_dir = self.evoloop_dir / "tasks" / self.config.name
        target_task_dir.mkdir(parents=True, exist_ok=True)

        # Copy all task files (task.yaml, train.py, prepare.py, etc.)
        for src_file in source_task_dir.iterdir():
            if src_file.is_file():
                dst_file = target_task_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                print(f"[evoloop] Provisioned: {src_file.name} -> {dst_file}")

        # Locate experiment DB (evoloop writes JSON files to experiments/)
        self._experiment_db_path = self.evoloop_dir / "experiments"

        print(f"[evoloop] Setup complete. Task: {target_task_dir}")
        return True

    def run(self) -> StrategyResult:
        """Launch evoloop and monitor until convergence or budget exhaustion."""
        task_yaml = self.evoloop_dir / "tasks" / self.config.name / "task.yaml"

        env = {
            **os.environ,
            "EVOLOOP_TASK": str(task_yaml),
            "EVOLOOP_RUNNER_BACKEND": self.backend,
            "EVOLOOP_TIME_BUDGET": str(self.time_budget),
            "EVOLOOP_LLM_PROVIDER": self.llm_provider,
            "EVOLOOP_LLM_MODEL": self.llm_model,
            "EVOLOOP_LLM_MODEL_STRONG": self.llm_model_strong,
            "EVOLOOP_BASILICA_GPU_MODELS": self.gpu,
            "EVOLOOP_BASILICA_GPU_COUNT": str(self.gpu_count),
        }

        if self.max_experiments > 0:
            env["EVOLOOP_MAX_EXPERIMENTS"] = str(self.max_experiments)

        print(f"[evoloop] Launching: EVOLOOP_TASK={task_yaml}")
        print(f"[evoloop] Backend: {self.backend}, GPU: {self.gpu}")
        print(f"[evoloop] Max experiments: {self.max_experiments or 'unlimited'}")

        try:
            result = subprocess.run(
                ["python", "loop.py"],
                cwd=str(self.evoloop_dir),
                env=env,
                capture_output=False,  # let output stream to terminal
            )
        except KeyboardInterrupt:
            print("[evoloop] Interrupted by user.")
        except Exception as e:
            print(f"[evoloop] Error: {e}")
            return StrategyResult(success=False, summary=f"evoloop failed: {e}")

        # Read results from experiment database
        return self._collect_results()

    def _collect_results(self) -> StrategyResult:
        """Parse evoloop's experiment database and return the best result."""
        if not self._experiment_db_path or not self._experiment_db_path.exists():
            return StrategyResult(
                success=False, summary="No experiment database found."
            )

        # evoloop stores experiments as JSON files in experiments/
        experiments = []
        for exp_file in sorted(self._experiment_db_path.glob("*.json")):
            try:
                with open(exp_file) as f:
                    experiments.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

        if not experiments:
            return StrategyResult(
                success=False,
                summary="No experiments found in database.",
            )

        # Find the best experiment (evoloop tracks Pareto front;
        # we pick the one with the best primary objective)
        best = None
        for exp in experiments:
            objectives = exp.get("objectives", {})
            if not objectives:
                continue
            if best is None:
                best = exp
            else:
                # Compare primary objective (first one in objectives dict)
                best_obj = best.get("objectives", {})
                primary_key = next(iter(objectives))
                if primary_key in best_obj:
                    # Lower is better for most objectives (CRPS, loss, etc.)
                    if objectives[primary_key] < best_obj.get(primary_key, float("inf")):
                        best = exp

        if best is None:
            return StrategyResult(
                success=False,
                experiments_run=len(experiments),
                summary="No experiments with valid objectives found.",
            )

        # Locate the best model artifact
        best_artifact = None
        artifact_path = best.get("artifact_path")
        if artifact_path:
            best_artifact = Path(artifact_path)
            if not best_artifact.is_absolute():
                best_artifact = self.evoloop_dir / best_artifact

        return StrategyResult(
            success=True,
            best_artifact=best_artifact,
            metrics=best.get("objectives", {}),
            experiments_run=len(experiments),
            summary=(
                f"Best of {len(experiments)} experiments. "
                f"Metrics: {best.get('objectives', {})}"
            ),
        )

    def get_status(self) -> dict[str, Any]:
        """Return current evoloop progress."""
        status: dict[str, Any] = {"strategy": "evoloop", "running": False}

        if self._experiment_db_path and self._experiment_db_path.exists():
            exp_count = len(list(self._experiment_db_path.glob("*.json")))
            status["experiments_run"] = exp_count
            status["running"] = self._process is not None and self._process.poll() is None
        return status
