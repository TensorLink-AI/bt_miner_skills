"""Persistent state tracking for the Ralph loop."""

import json
import os
import time
from dataclasses import dataclass, field, asdict

from ralph_loop.config import STATE_DIR


@dataclass
class BacktestResult:
    """A single backtest evidence record."""

    iteration: int = 0
    timestamp: float = 0.0
    model_name: str = ""
    crps_scores: dict = field(default_factory=dict)  # asset -> score
    crps_overall: float = 0.0
    baseline_comparison: dict = field(default_factory=dict)  # baseline_name -> score
    synth_api_compared: bool = False
    live_crps: dict = field(default_factory=dict)  # from Synth API comparison
    assets_evaluated: list[str] = field(default_factory=list)
    intervals_evaluated: list[str] = field(default_factory=list)
    raw_output: str = ""  # the execution output that produced this evidence


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

    # Backtest evidence tracking
    backtest_results: list[dict] = field(default_factory=list)
    best_crps_overall: float = 0.0
    has_validated_emulator: bool = False
    has_backtest_scores: bool = False
    has_baseline_comparison: bool = False
    has_synth_api_check: bool = False
    deployment_ready: bool = False


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
