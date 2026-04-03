"""Evoloop strategy — evolutionary model search via the evoloop engine.

Supports two modes:
1. Programmatic API (preferred) — imports evoloop directly when pip-installed
2. Subprocess fallback — shells out to `evoloop` CLI or `python loop.py`

Install evoloop: pip install git+https://github.com/TensorLink-AI/evoloop.git
Or for development: pip install -e /path/to/evoloop
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig
from orchestrator.strategies.base import Strategy, StrategyResult

# Try to import evoloop's programmatic API
try:
    from evoloop import run_loop as _evoloop_run_loop
    from evoloop.database import ExperimentDB as _ExperimentDB

    _HAS_EVOLOOP = True
except ImportError:
    _HAS_EVOLOOP = False


class EvoloopStrategy(Strategy):
    """Runs evoloop's evolutionary loop to search for the best model.

    When evoloop is pip-installed, uses the programmatic API directly.
    Otherwise falls back to subprocess invocation via EVOLOOP_DIR.
    """

    def __init__(self, config: SubnetConfig) -> None:
        super().__init__(config)
        self.strategy_config = config.strategy.config

        # Task dir within the subnet package (always needed)
        self.task_dir = self.strategy_config.get("task_dir", "evoloop_task/")

        # Evoloop execution settings
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

        # Subprocess fallback: where evoloop is cloned
        self._evoloop_dir = Path(
            self.strategy_config.get(
                "evoloop_dir",
                os.environ.get("EVOLOOP_DIR", ""),
            )
        )

        # Resolved task path (set during setup)
        self._task_path: Path | None = None
        self._use_api = _HAS_EVOLOOP

    def setup(self) -> bool:
        """Validate evoloop is available and locate task files."""
        source_task_dir = self.config.subnet_dir / self.task_dir
        if not source_task_dir.exists():
            print(f"[evoloop] ERROR: task dir not found: {source_task_dir}")
            return False

        # Check for task.yaml
        task_yaml = source_task_dir / "task.yaml"
        if not task_yaml.exists():
            print(f"[evoloop] ERROR: task.yaml not found in {source_task_dir}")
            return False

        if self._use_api:
            # Programmatic mode: point directly at the task files in-place
            self._task_path = task_yaml
            print(f"[evoloop] Using programmatic API (evoloop is pip-installed)")
            print(f"[evoloop] Task: {task_yaml}")
            return True

        # Subprocess fallback: need EVOLOOP_DIR, copy task files into it
        return self._setup_subprocess(source_task_dir)

    def _setup_subprocess(self, source_task_dir: Path) -> bool:
        """Setup for subprocess mode — validate EVOLOOP_DIR and provision tasks."""
        # Check if evoloop CLI is available even without the import
        evoloop_cli = shutil.which("evoloop")
        if evoloop_cli:
            self._task_path = source_task_dir / "task.yaml"
            print(f"[evoloop] Using CLI: {evoloop_cli}")
            print(f"[evoloop] Task: {self._task_path}")
            return True

        # Fall back to EVOLOOP_DIR + python loop.py
        if not self._evoloop_dir or not self._evoloop_dir.exists():
            print(f"[evoloop] ERROR: evoloop not installed and EVOLOOP_DIR not found.")
            print(f"[evoloop] Install: pip install git+https://github.com/TensorLink-AI/evoloop.git")
            print(f"[evoloop] Or set EVOLOOP_DIR to the evoloop repository path.")
            return False

        loop_py = self._evoloop_dir / "loop.py"
        if not loop_py.exists():
            print(f"[evoloop] ERROR: loop.py not found in {self._evoloop_dir}")
            return False

        # Copy task files into evoloop's tasks/ directory
        target_task_dir = self._evoloop_dir / "tasks" / self.config.name
        target_task_dir.mkdir(parents=True, exist_ok=True)

        for src_file in source_task_dir.iterdir():
            if src_file.is_file():
                dst_file = target_task_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                print(f"[evoloop] Provisioned: {src_file.name} -> {dst_file}")

        self._task_path = target_task_dir / "task.yaml"
        print(f"[evoloop] Using subprocess: EVOLOOP_DIR={self._evoloop_dir}")
        print(f"[evoloop] Task: {self._task_path}")
        return True

    def run(self) -> StrategyResult:
        """Execute evoloop — programmatic API or subprocess fallback."""
        if self._use_api:
            return self._run_api()
        return self._run_subprocess()

    def _run_api(self) -> StrategyResult:
        """Run evoloop via its Python API (preferred path)."""
        print(f"[evoloop] Starting evolutionary search (programmatic API)")
        print(f"[evoloop] Backend: {self.backend}, GPU: {self.gpu}")
        print(f"[evoloop] Max experiments: {self.max_experiments or 'unlimited'}")

        try:
            result = _evoloop_run_loop(
                task=str(self._task_path),
                backend=self.backend,
                max_experiments=self.max_experiments or None,
                time_budget=self.time_budget,
                gpu=self.gpu,
                gpu_count=self.gpu_count,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                llm_model_strong=self.llm_model_strong,
            )
        except KeyboardInterrupt:
            print("[evoloop] Interrupted by user.")
            return StrategyResult(success=False, summary="Interrupted by user.")
        except Exception as e:
            print(f"[evoloop] Error: {e}")
            return StrategyResult(success=False, summary=f"evoloop failed: {e}")

        # Convert evoloop result to StrategyResult
        if result is None:
            return StrategyResult(success=False, summary="evoloop returned no result.")

        best = result.best_experiment
        if best is None:
            return StrategyResult(
                success=False,
                experiments_run=len(result.pareto_front) if result.pareto_front else 0,
                summary="No successful experiments.",
            )

        # Extract artifact path from the best experiment
        best_artifact = None
        artifact_path = getattr(best, "artifact_path", None) or best.get("artifact_path") if isinstance(best, dict) else None
        if artifact_path:
            best_artifact = Path(artifact_path)

        # Extract metrics
        metrics = {}
        if isinstance(best, dict):
            metrics = best.get("objectives", {})
        elif hasattr(best, "objectives"):
            metrics = best.objectives if isinstance(best.objectives, dict) else {}

        pareto_size = len(result.pareto_front) if result.pareto_front else 0
        db_size = len(result.db) if result.db else 0
        total_experiments = db_size or pareto_size

        return StrategyResult(
            success=True,
            best_artifact=best_artifact,
            metrics=metrics,
            experiments_run=total_experiments,
            summary=(
                f"Best of {total_experiments} experiments "
                f"(Pareto front: {pareto_size}). "
                f"Metrics: {metrics}"
            ),
        )

    def _run_subprocess(self) -> StrategyResult:
        """Run evoloop via CLI or subprocess (fallback)."""
        evoloop_cli = shutil.which("evoloop")

        if evoloop_cli:
            cmd = [
                evoloop_cli,
                "--task", str(self._task_path),
                "--backend", self.backend,
            ]
            if self.max_experiments > 0:
                cmd.extend(["--max-experiments", str(self.max_experiments)])
            cwd = None
        else:
            cmd = ["python", "loop.py"]
            cwd = str(self._evoloop_dir)

        env = {
            **os.environ,
            "EVOLOOP_TASK": str(self._task_path),
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

        print(f"[evoloop] Launching: {' '.join(cmd)}")
        print(f"[evoloop] Backend: {self.backend}, GPU: {self.gpu}")
        print(f"[evoloop] Max experiments: {self.max_experiments or 'unlimited'}")

        try:
            subprocess.run(cmd, cwd=cwd, env=env, capture_output=False)
        except KeyboardInterrupt:
            print("[evoloop] Interrupted by user.")
        except Exception as e:
            print(f"[evoloop] Error: {e}")
            return StrategyResult(success=False, summary=f"evoloop failed: {e}")

        return self._collect_results_from_files()

    def _collect_results_from_files(self) -> StrategyResult:
        """Parse evoloop's experiment database from disk (subprocess mode)."""
        # Look for experiment DB in evoloop_dir or workspace
        search_paths = [
            self._evoloop_dir / "experiments" if self._evoloop_dir else None,
            self._evoloop_dir / "db.json" if self._evoloop_dir else None,
            self.config.workspace_dir / "experiments",
        ]

        experiments = []
        for search_path in search_paths:
            if search_path is None or not search_path.exists():
                continue

            if search_path.is_dir():
                for exp_file in sorted(search_path.glob("*.json")):
                    try:
                        with open(exp_file) as f:
                            experiments.append(json.load(f))
                    except (json.JSONDecodeError, OSError):
                        continue
            elif search_path.is_file():
                try:
                    with open(search_path) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        experiments.extend(data)
                    elif isinstance(data, dict) and "experiments" in data:
                        experiments.extend(data["experiments"])
                except (json.JSONDecodeError, OSError):
                    pass

            if experiments:
                break

        if not experiments:
            return StrategyResult(
                success=False, summary="No experiment database found after run."
            )

        # Find best by primary objective (lower is better)
        best = None
        for exp in experiments:
            objectives = exp.get("objectives", {})
            if not objectives:
                continue
            if best is None:
                best = exp
                continue

            best_obj = best.get("objectives", {})
            primary_key = next(iter(objectives))
            if primary_key in best_obj:
                if objectives[primary_key] < best_obj.get(primary_key, float("inf")):
                    best = exp

        if best is None:
            return StrategyResult(
                success=False,
                experiments_run=len(experiments),
                summary="No experiments with valid objectives.",
            )

        best_artifact = None
        artifact_path = best.get("artifact_path")
        if artifact_path:
            best_artifact = Path(artifact_path)
            if not best_artifact.is_absolute() and self._evoloop_dir:
                best_artifact = self._evoloop_dir / best_artifact

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
        status: dict[str, Any] = {
            "strategy": "evoloop",
            "mode": "api" if self._use_api else "subprocess",
        }

        if self._use_api and _HAS_EVOLOOP:
            # Could query evoloop's DB directly here
            status["evoloop_installed"] = True
        else:
            status["evoloop_installed"] = False
            status["evoloop_dir"] = str(self._evoloop_dir) if self._evoloop_dir else None

        return status
