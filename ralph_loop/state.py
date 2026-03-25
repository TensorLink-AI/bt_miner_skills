"""Persistent state tracking for the Ralph loop."""

import json
import os
import time
from dataclasses import dataclass, field, asdict

from ralph_loop.config import STATE_DIR


@dataclass
class PhaseState:
    """Tracks progress through a skill's phases."""

    current_phase: int = 1
    phase_status: dict[str, str] = field(default_factory=dict)  # phase_num -> "pending"|"in_progress"|"done"|"failed"
    conversation_history: list[dict] = field(default_factory=list)
    last_updated: float = 0.0
    iteration_count: int = 0


def _state_path(skill_name: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"{skill_name}.json")


def load_state(skill_name: str) -> PhaseState:
    """Load state for a skill, or return fresh state."""
    path = _state_path(skill_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return PhaseState(**data)
    return PhaseState()


def save_state(skill_name: str, state: PhaseState) -> None:
    """Persist state to disk."""
    state.last_updated = time.time()
    path = _state_path(skill_name)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)
