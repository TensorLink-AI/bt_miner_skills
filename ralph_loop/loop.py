"""The main Ralph loop — iterates through skill phases using LLM inference.

The loop has two modes:
  1. BUILD (phases 1-8): Work through the phased plan to get the miner operational.
  2. MAINTAIN (forever): Continuously retrain on fresh data, re-score against the
     live network, search for better models, and hot-swap improvements into production.

The loop never stops. Competitive mining requires constant adaptation.
"""

import logging
import time

from ralph_loop.config import LOOP_INTERVAL_SECONDS, MAX_ITERATIONS
from ralph_loop.llm import chat
from ralph_loop.skill_discovery import SkillPackage, discover_skills
from ralph_loop.state import PhaseState, load_state, save_state

logger = logging.getLogger(__name__)

BUILD_SYSTEM_PROMPT = """\
You are Ralph, an autonomous agent that builds competitive Bittensor subnet miners.

You are given a miner skill package that describes what a particular subnet expects,
how miners are scored, and a phased plan to build and deploy a miner.

Your job is to work through the phases one at a time. For each phase:
1. Understand the goal and gate criteria
2. Plan the concrete steps (files to create, code to write, commands to run)
3. Output the code / commands needed for this phase
4. Evaluate whether the gate criteria are met
5. If the gate passes, move to the next phase. If not, diagnose and fix.

You have access to the skill documentation and reference files. Use them.

IMPORTANT:
- Output concrete, runnable Python code and shell commands
- Be methodical — complete one phase fully before moving on
- Track which phase you're on and what's left to do
- When you need external data (APIs, prices), include the fetch code
"""

MAINTAIN_SYSTEM_PROMPT = """\
You are Ralph, an autonomous agent that continuously improves a live Bittensor subnet miner.

The miner has been built and deployed. Your job now is to keep it competitive FOREVER.
Each iteration you should pick ONE of these actions based on what will help most right now:

1. **RETRAIN** — Fetch the latest market data and retrain the model on a fresh window.
   Models trained on stale data drift. Retrain at least daily.

2. **SCORE CHECK** — Query the live network (Synth API or equivalent) to see current
   CRPS scores, ranking, and emission share. Compare against your last known scores.

3. **MODEL SEARCH** — Try a new hyperparameter config or architecture variant.
   Run walk-forward eval and compare against the current production model.

4. **DIAGNOSE** — If scores have degraded, investigate. Check per-asset and per-interval
   breakdown. Identify which assets or time horizons are weak. Fix them.

5. **HOT-SWAP** — If a better model is found, promote it to production via the model
   registry. The miner picks it up automatically.

6. **DATA REFRESH** — Re-fetch historical data, check for source issues (API changes,
   gaps, new assets), and rebuild features.

Pick the highest-impact action. Output concrete code and commands. Be concise.

IMPORTANT: You are in a continuous loop. Do not try to do everything at once.
Each iteration = one focused action. The loop runs forever.
"""


def build_phase_prompt(skill: SkillPackage, state: PhaseState) -> list[dict]:
    """Build the conversation messages for the BUILD mode (phases 1-8)."""
    messages = [{"role": "system", "content": BUILD_SYSTEM_PROMPT}]

    skill_context = (
        f"# Skill: {skill.name}\n\n"
        f"## Skill Overview\n{skill.skill_doc}\n\n"
        f"## Agent Prompt (full plan)\n{skill.agent_prompt}\n"
    )
    messages.append({"role": "user", "content": skill_context})

    ref_map = _references_for_phase(state.current_phase)
    for ref_name in ref_map:
        if ref_name in skill.references:
            messages.append({
                "role": "user",
                "content": f"## Reference: {ref_name}\n{skill.references[ref_name]}",
            })

    recent = state.conversation_history[-10:]
    messages.extend(recent)

    phase_instruction = (
        f"You are currently on **Phase {state.current_phase}**.\n"
        f"Iteration: {state.iteration_count + 1}\n\n"
        f"Work on this phase. Output the code and commands needed. "
        f"If you believe the phase gate is satisfied, say 'PHASE_COMPLETE' "
        f"and explain why. If you need more iterations, say 'CONTINUE' "
        f"and describe what's left."
    )
    messages.append({"role": "user", "content": phase_instruction})

    return messages


def build_maintain_prompt(skill: SkillPackage, state: PhaseState) -> list[dict]:
    """Build the conversation messages for MAINTAIN mode (continuous improvement)."""
    messages = [{"role": "system", "content": MAINTAIN_SYSTEM_PROMPT}]

    # Shorter context — the model is already built, just need key references
    skill_context = (
        f"# Skill: {skill.name}\n\n"
        f"## Skill Overview\n{skill.skill_doc}\n"
    )
    messages.append({"role": "user", "content": skill_context})

    # Include deployment + scoring references for maintain mode
    for ref_name in ["deployment.md", "synth_api.md", "leaderboard.md"]:
        if ref_name in skill.references:
            messages.append({
                "role": "user",
                "content": f"## Reference: {ref_name}\n{skill.references[ref_name]}",
            })

    # Replay recent maintain conversation (last 6 turns — keep it focused)
    recent = state.conversation_history[-6:]
    messages.extend(recent)

    maintain_instruction = (
        f"**MAINTAIN MODE** — Cycle {state.maintain_cycle + 1} "
        f"(total iterations: {state.iteration_count + 1})\n\n"
        f"The miner is deployed and running. Pick the single highest-impact "
        f"action from the maintain playbook (RETRAIN, SCORE CHECK, MODEL SEARCH, "
        f"DIAGNOSE, HOT-SWAP, or DATA REFRESH) and execute it.\n\n"
        f"Output the action name, then the concrete code/commands."
    )
    messages.append({"role": "user", "content": maintain_instruction})

    return messages


def _references_for_phase(phase: int) -> list[str]:
    """Map phase numbers to relevant reference files."""
    mapping = {
        1: ["architecture.md"],
        2: ["data_pipeline.md"],
        3: ["models.md"],
        4: ["validator_emulator.md"],
        5: ["data_pipeline.md", "models.md"],
        6: ["leaderboard.md"],
        7: ["validator_emulator.md", "synth_api.md"],
        8: ["deployment.md", "synth_api.md"],
    }
    return mapping.get(phase, [])


def run_skill_iteration(skill: SkillPackage, state: PhaseState) -> PhaseState:
    """Run one iteration of the loop for a single skill."""
    if state.mode == "build":
        logger.info(
            "=== %s | BUILD Phase %d | Iteration %d ===",
            skill.name, state.current_phase, state.iteration_count + 1,
        )
        messages = build_phase_prompt(skill, state)
    else:
        logger.info(
            "=== %s | MAINTAIN Cycle %d | Iteration %d ===",
            skill.name, state.maintain_cycle + 1, state.iteration_count + 1,
        )
        messages = build_maintain_prompt(skill, state)

    response = chat(messages)
    logger.info("LLM response:\n%s", response[:500])

    # Update conversation history
    state.conversation_history.append({"role": "assistant", "content": response})
    state.iteration_count += 1

    if state.mode == "build":
        # Check for phase completion
        if "PHASE_COMPLETE" in response:
            logger.info("Phase %d marked complete!", state.current_phase)
            state.phase_status[str(state.current_phase)] = "done"
            state.current_phase += 1

            # Transition to maintain mode after phase 8
            if state.current_phase > 8:
                logger.info(
                    "All build phases complete for %s — entering MAINTAIN mode.",
                    skill.name,
                )
                state.mode = "maintain"
                # Clear build conversation to start maintain with a fresh context
                state.conversation_history = []
            else:
                logger.info("Advancing to Phase %d", state.current_phase)
        else:
            state.phase_status[str(state.current_phase)] = "in_progress"
    else:
        state.maintain_cycle += 1

    return state


def run_loop(filter_subnet: str | None = None) -> None:
    """Main ralph loop — discover skills and iterate forever.

    The loop NEVER exits on its own (unless MAX_ITERATIONS is set).
    Competitive mining requires continuous adaptation.
    """
    skills = discover_skills(filter_subnet=filter_subnet)
    if not skills:
        logger.error("No skill packages found in the repo!")
        return

    logger.info("Discovered %d skill(s): %s", len(skills), [s.name for s in skills])

    iteration = 0
    while MAX_ITERATIONS == 0 or iteration < MAX_ITERATIONS:
        for skill in skills:
            state = load_state(skill.name)
            state = run_skill_iteration(skill, state)
            save_state(skill.name, state)

        iteration += 1
        logger.info("Sleeping %d seconds before next iteration...", LOOP_INTERVAL_SECONDS)
        time.sleep(LOOP_INTERVAL_SECONDS)
