"""Orchestrator that drives the agent loop for a subnet.

This is the main entry point that coordinates the loop phases. It's designed
to be called by an outer agent (e.g., Claude Code) that executes each phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bt_miner_skills.config.loader import load_subnet_config
from bt_miner_skills.config.subnet_config import SubnetConfig

from .loop import LoopState, Phase, get_phase_prompt


class Orchestrator:
    """Drives the agent loop for building a miner on a specific subnet."""

    def __init__(
        self,
        config_path: str | Path,
        workspace: str | Path = "workspace",
    ):
        self.config = load_subnet_config(config_path)
        self.workspace = Path(workspace)
        self.state = LoopState.load(self.workspace, self.config.netuid)

    @property
    def netuid(self) -> int:
        return self.config.netuid

    def get_current_prompt(self) -> str:
        """Get the agent prompt for the current phase."""
        return get_phase_prompt(self.state, self.config)

    def get_context(self) -> dict[str, Any]:
        """Get full context for the agent, including config and state."""
        return {
            "config": self.config.model_dump(),
            "state": {
                "phase": self.state.phase.value,
                "iteration": self.state.iteration,
                "artifacts": self.state.artifacts,
                "history": self.state.history[-5:],  # Last 5 entries
                "errors": self.state.errors[-10:],
            },
            "prompt": self.get_current_prompt(),
        }

    def complete_phase(self, result: dict[str, Any] | None = None):
        """Mark the current phase as complete and advance."""
        self.state.advance(result)
        self.state.save()

    def record_error(self, error: str):
        """Record an error in the current phase."""
        self.state.errors.append(f"[{self.state.phase.value}] {error}")
        self.state.save()

    def set_artifact(self, key: str, value: Any):
        """Store an artifact from the current phase."""
        self.state.artifacts[key] = value
        self.state.save()

    def reset_to_phase(self, phase: Phase):
        """Reset the loop to a specific phase (e.g., after a failure)."""
        self.state.phase = phase
        self.state.save()

    def setup_workspace(self):
        """Create the workspace directory structure."""
        dirs = [
            self.state.miner_dir,
            self.state.test_dir,
            self.state.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
