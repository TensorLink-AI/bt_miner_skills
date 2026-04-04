"""Persistent state for the orchestrator agent.

Tracks what the agent has done, what's running, and what it decided last time.
Survives process restarts — the agent picks up where it left off.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentState:
    """Everything the agent needs to know about its history and current situation."""

    # Current status
    phase: str = "idle"  # idle, setup, searching, deploying, monitoring
    phase_started_at: float = 0.0  # unix timestamp

    # Search state
    search_running: bool = False
    search_pid: int | None = None  # PID of background search process
    search_started_at: float = 0.0
    experiments_run: int = 0
    experiments_at_last_improvement: int = 0
    best_metric: float | None = None
    best_artifact: str | None = None
    stale_count: int = 0  # experiments since last improvement

    # Deployment state
    deployed: bool = False
    deployed_at: float = 0.0
    deployed_model_id: str | None = None

    # Monitoring state
    last_monitor_at: float = 0.0
    last_monitor_healthy: bool | None = None
    last_emission_share: float | None = None
    last_rank: int | None = None

    # Agent decision history (last N decisions for context)
    decision_log: list[dict[str, Any]] = field(default_factory=list)

    # Error tracking
    consecutive_errors: int = 0
    last_error: str | None = None

    def log_decision(self, action: str, reasoning: str, result: str = "") -> None:
        """Record a decision the agent made."""
        entry = {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "reasoning": reasoning,
            "result": result,
        }
        self.decision_log.append(entry)
        # Keep last 50 decisions
        if len(self.decision_log) > 50:
            self.decision_log = self.decision_log[-50:]

    def time_in_phase(self) -> float:
        """Seconds spent in the current phase."""
        if self.phase_started_at == 0:
            return 0.0
        return time.time() - self.phase_started_at

    def time_in_phase_str(self) -> str:
        """Human-readable time in current phase."""
        secs = self.time_in_phase()
        if secs < 60:
            return f"{secs:.0f}s"
        if secs < 3600:
            return f"{secs / 60:.0f}m"
        return f"{secs / 3600:.1f}h"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "phase_started_at": self.phase_started_at,
            "search_running": self.search_running,
            "search_pid": self.search_pid,
            "search_started_at": self.search_started_at,
            "experiments_run": self.experiments_run,
            "experiments_at_last_improvement": self.experiments_at_last_improvement,
            "best_metric": self.best_metric,
            "best_artifact": self.best_artifact,
            "stale_count": self.stale_count,
            "deployed": self.deployed,
            "deployed_at": self.deployed_at,
            "deployed_model_id": self.deployed_model_id,
            "last_monitor_at": self.last_monitor_at,
            "last_monitor_healthy": self.last_monitor_healthy,
            "last_emission_share": self.last_emission_share,
            "last_rank": self.last_rank,
            "decision_log": self.decision_log,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        state = cls()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


class StateStore:
    """Persists agent state to disk as JSON."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, subnet_name: str) -> Path:
        return self.state_dir / f"{subnet_name}.json"

    def load(self, subnet_name: str) -> AgentState:
        path = self._path(subnet_name)
        if not path.exists():
            return AgentState()
        try:
            with open(path) as f:
                return AgentState.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError):
            return AgentState()

    def save(self, subnet_name: str, state: AgentState) -> None:
        path = self._path(subnet_name)
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
