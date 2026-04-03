"""Base strategy interface — all strategies implement this."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.config import SubnetConfig


@dataclass
class StrategyResult:
    """Structured output from a strategy run."""

    success: bool
    best_artifact: Path | None = None  # path to best model / config / output
    metrics: dict[str, Any] = field(default_factory=dict)
    experiments_run: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "best_artifact": str(self.best_artifact) if self.best_artifact else None,
            "metrics": self.metrics,
            "experiments_run": self.experiments_run,
            "summary": self.summary,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Strategy(ABC):
    """Base class for all search/iteration strategies.

    A strategy knows how to iterate toward a goal. It does NOT know about
    Bittensor, deployment, or monitoring — those are the orchestrator's job.
    """

    def __init__(self, config: SubnetConfig) -> None:
        self.config = config

    @abstractmethod
    def setup(self) -> bool:
        """Prepare whatever this strategy needs (install deps, validate paths, etc.).

        Returns True if setup succeeded and run() can proceed.
        """

    @abstractmethod
    def run(self) -> StrategyResult:
        """Execute the strategy's iteration loop.

        This is the main entry point. It should run until convergence,
        budget exhaustion, or competitiveness gates are met.
        """

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Return current progress (for the orchestrator to inspect mid-run)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
