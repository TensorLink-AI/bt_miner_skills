"""The main Ralph loop — adaptive autonomous agent with code execution.

No rigid phases. The LLM reads the skill docs, sees the workspace state and
execution output, and decides what to work on each iteration.
"""

import logging
import os
import re

from ralph_loop.config import MAX_ITERATIONS, WORKSPACE_ROOT
from ralph_loop.executor import execute_response, get_workspace_snapshot
from ralph_loop.llm import chat
from ralph_loop.skill_discovery import SkillPackage, discover_skills
from ralph_loop.state import LoopState, load_state, save_state

logger = logging.getLogger(__name__)

# Load the system prompt from the external markdown file
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "RALPH_PROMPT.md")


def _load_system_prompt() -> str:
    with open(_PROMPT_PATH, "r") as f:
        return f.read()


def build_prompt(skill: SkillPackage, state: LoopState) -> list[dict]:
    """Build conversation messages for the next LLM iteration."""
    system_prompt = _load_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # Skill documentation
    skill_context = (
        f"# Skill Package: {skill.name}\n\n"
        f"## Skill Overview (SKILL.md)\n{skill.skill_doc}\n\n"
        f"## Agent Prompt (AGENT_PROMPT.md)\n{skill.agent_prompt}\n"
    )
    messages.append({"role": "user", "content": skill_context})

    # Include requested references, or a default set on first iteration
    refs_to_include = state.requested_references if state.requested_references else []
    if state.iteration_count == 0:
        # First iteration: include architecture overview
        refs_to_include = ["architecture.md"]

    for ref_name in refs_to_include:
        if ref_name in skill.references:
            messages.append({
                "role": "user",
                "content": f"## Reference: {ref_name}\n{skill.references[ref_name]}",
            })

    available_refs = list(skill.references.keys())
    messages.append({
        "role": "user",
        "content": f"**Available reference files** (request with NEED_REF): {', '.join(available_refs)}",
    })

    # Recent conversation history (keep last 10 turns to manage context)
    recent = state.conversation_history[-10:]
    messages.extend(recent)

    # Current state context
    workspace_dir = state.workspace_dir or os.path.join(WORKSPACE_ROOT, skill.name)
    snapshot = get_workspace_snapshot(workspace_dir)

    state_context_parts = [
        f"## Current State — Iteration {state.iteration_count + 1}",
        snapshot,
    ]

    if state.last_execution_output:
        state_context_parts.append(state.last_execution_output)

    if state.files_written:
        recent_files = state.files_written[-20:]
        state_context_parts.append(f"**Files written so far:** {', '.join(recent_files)}")

    state_context_parts.append(
        "\nDecide what to do next. Follow the response format (STATUS, DECISION, code blocks, RESULT_CHECK)."
    )

    messages.append({"role": "user", "content": "\n\n".join(state_context_parts)})

    return messages


def _parse_requested_refs(response: str) -> list[str]:
    """Extract NEED_REF: filename lines from the LLM response."""
    refs = []
    for match in re.finditer(r"NEED_REF:\s*(\S+)", response):
        refs.append(match.group(1))
    return refs


def run_skill_iteration(skill: SkillPackage, state: LoopState) -> LoopState:
    """Run one iteration of the loop for a single skill."""
    workspace_dir = state.workspace_dir or os.path.join(WORKSPACE_ROOT, skill.name)
    state.workspace_dir = workspace_dir

    logger.info(
        "=== %s | Iteration %d ===",
        skill.name, state.iteration_count + 1,
    )

    messages = build_prompt(skill, state)
    response = chat(messages)
    logger.info("LLM response (%d chars):\n%s", len(response), response[:500])

    # Execute code blocks from the response
    report = execute_response(response, workspace_dir)
    logger.info(
        "Execution: %d steps, %d files written, %d commands",
        len(report.results), len(report.files_written), len(report.commands_run),
    )

    # Update state
    state.conversation_history.append({"role": "assistant", "content": response})
    state.iteration_count += 1
    state.last_execution_output = report.format_for_llm()
    state.workspace_snapshot = get_workspace_snapshot(workspace_dir)
    state.files_written.extend(report.files_written)
    state.commands_run.extend(report.commands_run)
    state.requested_references = _parse_requested_refs(response)

    return state


def run_loop(filter_subnet: str | None = None) -> None:
    """Main ralph loop — discover skills and iterate forever."""
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
