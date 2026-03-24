"""The core agent loop: research -> build -> test -> deploy -> monitor -> improve.

This module defines the phases of the agent loop and the state machine that
drives iterative miner development. Each phase produces artifacts that feed
into the next phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Phase(str, Enum):
    """Phases of the agent loop."""

    RESEARCH = "research"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    IMPROVE = "improve"


@dataclass
class LoopState:
    """Tracks the current state of an agent loop for a subnet."""

    netuid: int
    phase: Phase = Phase.RESEARCH
    iteration: int = 0
    workspace: Path = field(default_factory=lambda: Path("workspace"))
    artifacts: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def miner_dir(self) -> Path:
        return self.workspace / f"sn{self.netuid}" / "miner"

    @property
    def test_dir(self) -> Path:
        return self.workspace / f"sn{self.netuid}" / "tests"

    @property
    def logs_dir(self) -> Path:
        return self.workspace / f"sn{self.netuid}" / "logs"

    def advance(self, result: dict[str, Any] | None = None):
        """Move to the next phase, recording results."""
        self.history.append(
            {
                "phase": self.phase.value,
                "iteration": self.iteration,
                "result": result or {},
            }
        )
        phases = list(Phase)
        idx = phases.index(self.phase)
        if idx == len(phases) - 1:
            # Loop back to BUILD (skip research on subsequent iterations)
            self.phase = Phase.BUILD
            self.iteration += 1
        else:
            self.phase = phases[idx + 1]

    def save(self):
        """Persist loop state to disk."""
        import json

        state_file = self.workspace / f"sn{self.netuid}" / "loop_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(
                {
                    "netuid": self.netuid,
                    "phase": self.phase.value,
                    "iteration": self.iteration,
                    "artifacts": self.artifacts,
                    "history": self.history,
                    "errors": self.errors,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, workspace: Path, netuid: int) -> LoopState:
        """Load loop state from disk."""
        import json

        state_file = workspace / f"sn{netuid}" / "loop_state.json"
        if not state_file.exists():
            return cls(netuid=netuid, workspace=workspace)
        with open(state_file) as f:
            data = json.load(f)
        state = cls(
            netuid=data["netuid"],
            phase=Phase(data["phase"]),
            iteration=data["iteration"],
            workspace=workspace,
            artifacts=data.get("artifacts", {}),
            history=data.get("history", []),
            errors=data.get("errors", []),
        )
        return state


# --- Phase Descriptions (for agent prompting) ---

PHASE_PROMPTS: dict[Phase, str] = {
    Phase.RESEARCH: """## Research Phase
You are researching subnet {netuid} ({name}).

Your goals:
1. Read the subnet config to understand what miners must do
2. If a repo_url is provided, study the validator and miner code
3. Understand the scoring mechanism and what makes a miner competitive
4. Identify the Synapse protocol (request/response format)
5. Check the metagraph to understand the competitive landscape
6. Document your findings in the loop state artifacts

Output a research report with:
- Task description (what miners do)
- Protocol details (synapse fields, timeouts)
- Scoring criteria (how responses are ranked)
- Competitive landscape (how many miners, top performer stats)
- Strategy recommendations (how to build a winning miner)
""",
    Phase.BUILD: """## Build Phase
You are building a miner for subnet {netuid} ({name}). Iteration {iteration}.

Using the research from the previous phase, build a miner that:
1. Implements the correct Synapse protocol
2. Handles the task the subnet expects
3. Follows the scoring criteria to maximize incentive
4. Is well-structured and deployable

Use the miner template as a starting point and customize it.
Write the miner code to: {miner_dir}

If this is iteration > 0, review the monitoring data and improve on the previous version.
""",
    Phase.TEST: """## Test Phase
You are testing the miner for subnet {netuid} ({name}).

Run the test suite:
1. Unit tests for the core logic
2. Protocol compliance tests (does it respond correctly to synapses?)
3. Performance benchmarks (latency, throughput)
4. Edge case handling

Fix any failures before advancing.
""",
    Phase.DEPLOY: """## Deploy Phase
You are deploying the miner for subnet {netuid} ({name}).

Steps:
1. Verify the miner passes all tests
2. Set up the runtime environment (dependencies, models, env vars)
3. Register on the subnet (if not already registered)
4. Start the miner process
5. Verify it's receiving and responding to queries
""",
    Phase.MONITOR: """## Monitor Phase
You are monitoring the miner on subnet {netuid} ({name}).

Check:
1. Is the miner running and responding?
2. What is our current incentive/rank?
3. Are there any errors in the logs?
4. How do we compare to top miners?
5. Collect performance metrics for the improve phase
""",
    Phase.IMPROVE: """## Improve Phase
You are improving the miner for subnet {netuid} ({name}). Iteration {iteration}.

Based on monitoring data:
1. Identify why top miners are outperforming us
2. Analyze scoring patterns to find optimization opportunities
3. Plan specific improvements (better model, faster inference, etc.)
4. Document what to change in the next build iteration

This will feed back into the BUILD phase.
""",
}


def get_phase_prompt(state: LoopState, config: Any) -> str:
    """Generate the prompt for the current phase."""
    return PHASE_PROMPTS[state.phase].format(
        netuid=state.netuid,
        name=getattr(config, "name", f"SN{state.netuid}"),
        iteration=state.iteration,
        miner_dir=state.miner_dir,
    )
