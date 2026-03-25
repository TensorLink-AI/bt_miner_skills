"""The main Ralph loop — iterates through skill phases using LLM inference."""

import logging
import time

from ralph_loop.config import LOOP_INTERVAL_SECONDS, MAX_ITERATIONS
from ralph_loop.llm import chat
from ralph_loop.skill_discovery import SkillPackage, discover_skills
from ralph_loop.state import PhaseState, load_state, save_state

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
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


def build_phase_prompt(skill: SkillPackage, state: PhaseState) -> list[dict]:
    """Build the conversation messages for the current phase."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include skill context (summarized to save tokens)
    skill_context = (
        f"# Skill: {skill.name}\n\n"
        f"## Skill Overview\n{skill.skill_doc}\n\n"
        f"## Agent Prompt (full plan)\n{skill.agent_prompt}\n"
    )
    messages.append({"role": "user", "content": skill_context})

    # Include relevant reference files for the current phase
    ref_map = _references_for_phase(state.current_phase)
    for ref_name in ref_map:
        if ref_name in skill.references:
            messages.append({
                "role": "user",
                "content": f"## Reference: {ref_name}\n{skill.references[ref_name]}",
            })

    # Replay recent conversation history (last 10 turns to stay in context)
    recent = state.conversation_history[-10:]
    messages.extend(recent)

    # Add the phase instruction
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
    logger.info(
        "=== %s | Phase %d | Iteration %d ===",
        skill.name, state.current_phase, state.iteration_count + 1,
    )

    messages = build_phase_prompt(skill, state)
    response = chat(messages)

    # Log the response
    logger.info("LLM response:\n%s", response[:500])

    # Update conversation history
    state.conversation_history.append({"role": "assistant", "content": response})
    state.iteration_count += 1

    # Check for phase completion
    if "PHASE_COMPLETE" in response:
        logger.info("Phase %d marked complete!", state.current_phase)
        state.phase_status[str(state.current_phase)] = "done"
        state.current_phase += 1
        logger.info("Advancing to Phase %d", state.current_phase)
    else:
        state.phase_status[str(state.current_phase)] = "in_progress"

    return state


def run_loop(filter_subnet: str | None = None) -> None:
    """Main ralph loop — discover skills and iterate through their phases."""
    skills = discover_skills(filter_subnet=filter_subnet)
    if not skills:
        logger.error("No skill packages found in the repo!")
        return

    logger.info("Discovered %d skill(s): %s", len(skills), [s.name for s in skills])

    iteration = 0
    while MAX_ITERATIONS == 0 or iteration < MAX_ITERATIONS:
        for skill in skills:
            state = load_state(skill.name)

            # Skip completed skills (all 8 phases done)
            if state.current_phase > 8:
                logger.info("Skill %s is complete, skipping.", skill.name)
                continue

            state = run_skill_iteration(skill, state)
            save_state(skill.name, state)

        iteration += 1

        # Check if all skills are done
        all_done = all(load_state(s.name).current_phase > 8 for s in skills)
        if all_done:
            logger.info("All skills complete!")
            break

        logger.info("Sleeping %d seconds before next iteration...", LOOP_INTERVAL_SECONDS)
        time.sleep(LOOP_INTERVAL_SECONDS)
