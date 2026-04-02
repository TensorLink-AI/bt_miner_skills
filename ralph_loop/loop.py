"""The main Ralph loop — adaptive autonomous agent with code execution.

No rigid phases. The LLM reads the skill docs, sees the workspace state and
execution output, and decides what to work on each iteration.
"""

import logging
import os
import re

from ralph_loop.basilica import BASILICA_TOOLS, execute_tool_call
from ralph_loop.config import MAX_ITERATIONS, WORKSPACE_ROOT, COMMONS_URL
from ralph_loop.evidence import (
    build_evidence_gate_warning,
    build_evidence_summary,
    extract_evidence,
)
from ralph_loop.executor import execute_response, get_workspace_snapshot
from ralph_loop.llm import chat_with_tools, LLMResponse
from ralph_loop.skill_discovery import SkillPackage, discover_skills
from ralph_loop.state import LoopState, load_state, save_state

logger = logging.getLogger(__name__)

# Load the system prompt from the external markdown file
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "RALPH_PROMPT.md")

_COMMONS_REPO = "https://github.com/TensorLink-AI/the-commons"


def _build_knowledge_prompt() -> str:
    """Build the knowledge-sharing prompt, injecting connection details."""
    parts = [
        "## Knowledge Sharing — the-commons\n",
        "Knowledge sharing is **enabled** for this session. You have access to",
        "**the-commons**, a shared experiment log where agents record what they tried,",
        "what worked, and what failed.\n",
        "**Setup (first iteration only, if not already done):**",
        f"1. Clone the repo into your workspace: `git clone {_COMMONS_REPO}`",
        "2. Read `the-commons/skill/SKILL.md` to learn the available tools and how to use them.",
        "3. Install its dependencies: `pip install -e the-commons` or `pip install mcp`\n",
    ]

    if COMMONS_URL:
        parts.append(f"**Remote SSE server URL:** `{COMMONS_URL}`")
        parts.append(
            "Connect to this URL using the MCP SSE transport. "
            "The server uses agent identification (pass your agent name in tool calls), "
            "not bearer tokens.\n"
        )

    parts.extend([
        "**Every iteration:**",
        "- Before starting new work, query the commons for prior experiments on your task",
        "  (use the `search`, `best`, and `failures` tools).",
        "- After completing work, log your attempt (use the `log` tool) with what you",
        "  tried, what happened, and an optional score.",
        "- Rate useful entries from others when you encounter them.\n",
        "The commons is a remote MCP server — read the skill doc for connection details",
        "and tool signatures. Use it to avoid repeating known failures and to build on",
        "what already works.",
    ])
    return "\n".join(parts)


def _load_system_prompt() -> str:
    with open(_PROMPT_PATH, "r") as f:
        return f.read()


def build_prompt(
    skill: SkillPackage,
    state: LoopState,
    share_knowledge: bool = False,
) -> list[dict]:
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

    # Knowledge sharing prompt (if toggled on)
    if share_knowledge:
        messages.append({"role": "user", "content": _build_knowledge_prompt()})

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

    # Inject backtest evidence summary — shows what's been proven vs missing
    evidence_summary = build_evidence_summary(state)
    if evidence_summary:
        state_context_parts.append(evidence_summary)

    # Inject evidence gate warning if agent claimed readiness without proof
    gate_warning = build_evidence_gate_warning(state)
    if gate_warning:
        state_context_parts.append(gate_warning)

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


def _handle_tool_calls(
    llm_response: LLMResponse,
    workspace_dir: str,
) -> str:
    """Execute any tool calls from the LLM response and format results."""
    if not llm_response.has_tool_calls:
        return ""

    parts = ["## Tool Call Results\n"]
    for tc in llm_response.tool_calls:
        tool_name = tc["name"]
        arguments = tc["arguments"]
        logger.info("Executing tool call: %s(%s)", tool_name, list(arguments.keys()))

        result = execute_tool_call(tool_name, arguments, workspace_dir)
        parts.append(f"### {tool_name}")
        parts.append(f"Arguments: {arguments}")
        parts.append(f"Result:\n```json\n{result}\n```\n")

    return "\n".join(parts)


def run_skill_iteration(
    skill: SkillPackage,
    state: LoopState,
    share_knowledge: bool = False,
) -> LoopState:
    """Run one iteration of the loop for a single skill."""
    workspace_dir = state.workspace_dir or os.path.join(WORKSPACE_ROOT, skill.name)
    state.workspace_dir = workspace_dir

    logger.info(
        "=== %s | Iteration %d ===",
        skill.name, state.iteration_count + 1,
    )

    messages = build_prompt(skill, state, share_knowledge=share_knowledge)
    llm_response = chat_with_tools(messages, tools=BASILICA_TOOLS)

    response = llm_response.content
    logger.info("LLM response (%d chars, %d tool calls):\n%s",
                len(response), len(llm_response.tool_calls), response[:500])

    # Handle any Basilica tool calls
    tool_output = _handle_tool_calls(llm_response, workspace_dir)

    # Execute code blocks from the text response
    report = execute_response(response, workspace_dir)
    logger.info(
        "Execution: %d steps, %d files written, %d commands",
        len(report.results), len(report.files_written), len(report.commands_run),
    )

    # Update state
    state.conversation_history.append({"role": "assistant", "content": response})
    state.iteration_count += 1

    # Combine code block output and tool call output
    exec_output = report.format_for_llm()
    if tool_output:
        exec_output = f"{exec_output}\n\n{tool_output}"
    state.last_execution_output = exec_output
    state.workspace_snapshot = get_workspace_snapshot(workspace_dir)
    state.files_written.extend(report.files_written)
    state.commands_run.extend(report.commands_run)
    state.requested_references = _parse_requested_refs(response)

    # Extract and track backtest evidence from this iteration
    evidence = extract_evidence(
        execution_output=exec_output,
        llm_response=response,
        iteration=state.iteration_count,
    )
    state.backtest_results.append(evidence)

    # Update cumulative evidence flags
    if evidence.get("has_real_scores"):
        state.has_backtest_scores = True
        crps_vals = evidence.get("crps_scores_found", [])
        if crps_vals:
            best = min(crps_vals)
            if state.best_crps_overall == 0.0 or best < state.best_crps_overall:
                state.best_crps_overall = best
    if evidence.get("baseline_scores"):
        state.has_baseline_comparison = True
    if evidence.get("synth_api_compared"):
        state.has_synth_api_check = True
    if evidence.get("emulator_validated"):
        state.has_validated_emulator = True

    # Deployment is only ready when all evidence gates are met
    state.deployment_ready = (
        state.has_backtest_scores
        and state.has_baseline_comparison
        and state.has_synth_api_check
        and state.has_validated_emulator
    )

    if evidence.get("claims_ready_without_evidence"):
        logger.warning(
            "EVIDENCE GATE: Agent claims readiness at iteration %d "
            "but no backtest scores found in execution output!",
            state.iteration_count,
        )

    return state


def run_loop(
    filter_subnet: str | None = None,
    share_knowledge: bool = False,
) -> None:
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
            state = run_skill_iteration(
                skill, state, share_knowledge=share_knowledge,
            )
            save_state(skill.name, state)

        iteration += 1
