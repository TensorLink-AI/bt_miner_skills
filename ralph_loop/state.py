"""Persistent state tracking for the Ralph loop."""

import json
import os
import time
from dataclasses import dataclass, field, asdict

from ralph_loop.config import STATE_DIR


@dataclass
class LoopState:
    """Tracks progress for a skill — adaptive, no rigid phases."""

    iteration_count: int = 0
    conversation_history: list[dict] = field(default_factory=list)
    last_execution_output: str = ""
    workspace_snapshot: str = ""
    files_written: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    requested_references: list[str] = field(default_factory=list)
    workspace_dir: str = ""
    last_updated: float = 0.0


def _state_path(skill_name: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"{skill_name}.json")


def load_state(skill_name: str) -> LoopState:
    """Load state for a skill, or return fresh state."""
    path = _state_path(skill_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        # Filter to only known fields for forward compat
        known = {f.name for f in LoopState.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return LoopState(**filtered)
    return LoopState()


def save_state(skill_name: str, state: LoopState) -> None:
    """Persist state to disk."""
    state.last_updated = time.time()
    path = _state_path(skill_name)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)


def reset_state(skill_name: str) -> None:
    """Delete persisted state for a skill."""
    path = _state_path(skill_name)
    if os.path.exists(path):
        os.remove(path)
