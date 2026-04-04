"""Orchestrator — an LLM agent that manages subnet miner lifecycle.

Instead of a rigid phase-based pipeline, the orchestrator:
1. Gathers a status snapshot (what's running, metrics, errors, history)
2. Shows it to an LLM agent
3. The agent picks from a fixed set of tools (start_search, deploy, monitor, etc.)
4. The orchestrator executes the tool and records the decision
5. Repeat on a tick interval

The agent sees the full picture and makes judgment calls — when to keep
searching, when to deploy, when to re-evolve. No hardcoded state machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from orchestrator.config import SubnetConfig, load_subnet_config
from orchestrator.snapshot import SYSTEM_PROMPT, build_snapshot
from orchestrator.state import AgentState, StateStore
from orchestrator.tools import TOOL_DEFINITIONS, execute_tool


def discover_subnets(subnets_dir: Path) -> list[Path]:
    """Find all subnet directories (those containing subnet.yaml)."""
    if not subnets_dir.exists():
        return []
    return sorted(
        d for d in subnets_dir.iterdir()
        if d.is_dir() and (d / "subnet.yaml").exists()
    )


def call_llm(system_prompt: str, user_message: str) -> dict:
    """Call the LLM to get a decision.

    Supports OpenAI-compatible APIs (configurable via env vars).
    Falls back to a simple heuristic agent if no LLM is available.
    """
    api_key = os.environ.get("ORCHESTRATOR_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("ORCHESTRATOR_MODEL", "gpt-4.1-mini")
    base_url = os.environ.get("ORCHESTRATOR_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        # No LLM available — use heuristic fallback
        return heuristic_decide(user_message)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except ImportError:
        print("[agent] openai package not installed. Using heuristic fallback.")
        return heuristic_decide(user_message)
    except json.JSONDecodeError as e:
        print(f"[agent] Failed to parse LLM response as JSON: {e}")
        print(f"[agent] Raw response: {content[:500]}")
        return heuristic_decide(user_message)
    except Exception as e:
        print(f"[agent] LLM call failed: {e}")
        return heuristic_decide(user_message)


def heuristic_decide(snapshot: str) -> dict:
    """Simple rule-based fallback when no LLM is available.

    Follows the obvious path: setup → search → deploy → monitor → re-search.
    """
    # Parse key facts from the snapshot text
    phase = "idle"
    search_running = False
    deployed = False
    experiments_run = 0
    best_metric = None
    last_monitor = False
    stale_count = 0
    errors = 0

    for line in snapshot.split("\n"):
        line = line.strip()
        if line.startswith("Current phase:"):
            phase = line.split(":")[1].strip().split()[0]
        elif "Status: RUNNING" in line:
            search_running = True
        elif "Status: DEPLOYED" in line:
            deployed = True
        elif line.startswith("Experiments run:"):
            try:
                experiments_run = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("Best metric:") and "none" not in line:
            try:
                best_metric = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("Stale count:"):
            try:
                stale_count = int(line.split(":")[1].strip().split()[0])
            except ValueError:
                pass
        elif line.startswith("Consecutive errors:"):
            try:
                errors = int(line.split(":")[1].strip())
            except ValueError:
                pass

    # Also parse stale threshold from goals section
    stale_threshold = 15  # default
    for line in snapshot.split("\n"):
        line = line.strip()
        if line.startswith("Stop search if stale for:"):
            try:
                stale_threshold = int(line.split(":")[1].strip().split()[0])
            except ValueError:
                pass
        elif "Search finished" in line or "FINISHED" in line:
            search_running = False

    # Decision logic
    if errors >= 3:
        return {
            "reasoning": f"{errors} consecutive errors. Re-running setup to validate prerequisites.",
            "tool": "run_setup",
            "params": {},
        }

    if phase == "idle":
        return {
            "reasoning": "Just starting. Need to validate prerequisites before searching.",
            "tool": "run_setup",
            "params": {},
        }

    if phase == "setup":
        return {
            "reasoning": "Setup complete. Starting model search.",
            "tool": "start_search",
            "params": {},
        }

    if phase == "searching":
        if search_running:
            # Search is running in background — check progress
            if stale_count >= stale_threshold:
                return {
                    "reasoning": (
                        f"Search stale for {stale_count} experiments "
                        f"(threshold: {stale_threshold}). Stopping to deploy best so far."
                    ),
                    "tool": "stop_search",
                    "params": {},
                }
            return {
                "reasoning": (
                    f"Search running in background ({experiments_run} experiments, "
                    f"stale: {stale_count}/{stale_threshold}). Checking live progress."
                ),
                "tool": "get_search_status",
                "params": {},
            }
        else:
            # Search process finished (or was never started)
            if best_metric is not None:
                return {
                    "reasoning": f"Search finished with best metric {best_metric}. Deploying.",
                    "tool": "deploy",
                    "params": {},
                }
            elif experiments_run > 0:
                return {
                    "reasoning": "Search finished but no good results. Re-running setup and retrying.",
                    "tool": "run_setup",
                    "params": {},
                }
            else:
                return {
                    "reasoning": "No search running and no results. Starting search.",
                    "tool": "start_search",
                    "params": {},
                }

    if phase == "deploying":
        return {
            "reasoning": "Deployment phase — deploying best artifact.",
            "tool": "deploy",
            "params": {},
        }

    if phase == "monitoring":
        return {
            "reasoning": "Deployed. Checking live performance.",
            "tool": "check_live_performance",
            "params": {},
        }

    return {
        "reasoning": f"Unknown phase '{phase}'. Running setup to reset.",
        "tool": "run_setup",
        "params": {},
    }


def run_agent_tick(
    config: SubnetConfig,
    state: AgentState,
) -> None:
    """Run one tick of the agent loop."""
    # Build status snapshot
    snapshot = build_snapshot(config, state)

    print(f"\n{'─' * 60}")
    print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Agent tick — {config.name}")
    print(f"{'─' * 60}")
    print(snapshot)

    # Ask the agent what to do
    tool_list = "\n".join(
        f"  - {t['name']}: {t['description']}" for t in TOOL_DEFINITIONS
    )
    prompt = f"Current status:\n\n{snapshot}\n\nAvailable tools:\n{tool_list}\n\nWhat should I do next?"

    decision = call_llm(SYSTEM_PROMPT, prompt)

    reasoning = decision.get("reasoning", "No reasoning provided.")
    tool_name = decision.get("tool", "wait")
    params = decision.get("params", {})

    print(f"\n[agent] Decision: {tool_name}")
    print(f"[agent] Reasoning: {reasoning}")
    if params:
        print(f"[agent] Params: {params}")

    # Execute the tool
    result = execute_tool(tool_name, config, state, params)

    result_str = result.get("message", json.dumps(result)[:200])
    print(f"[agent] Result: {result_str}")

    # Log the decision
    state.log_decision(
        action=tool_name,
        reasoning=reasoning,
        result=result_str,
    )


def run_agent(
    config: SubnetConfig,
    state_store: StateStore,
    tick_interval: int = 300,
    max_ticks: int = 0,
    single_tick: bool = False,
) -> None:
    """Run the agent loop for a subnet."""
    state = state_store.load(config.name)

    print(f"\n{'=' * 60}")
    print(f"  Orchestrator Agent — {config.name}")
    print(f"  netuid {config.netuid} | {config.network} | strategy: {config.strategy.type}")
    print(f"  Tick interval: {tick_interval}s")
    print(f"{'=' * 60}")

    tick_count = 0

    try:
        while True:
            run_agent_tick(config, state)
            state_store.save(config.name, state)
            tick_count += 1

            if single_tick:
                break

            if max_ticks > 0 and tick_count >= max_ticks:
                print(f"\n[agent] Reached max ticks ({max_ticks}). Stopping.")
                break

            print(f"\n[agent] Next tick in {tick_interval}s... (Ctrl+C to stop)")
            time.sleep(tick_interval)

    except KeyboardInterrupt:
        print(f"\n[agent] Interrupted. Saving state...")
        state_store.save(config.name, state)
        print(f"[agent] State saved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subnet miner orchestrator — LLM agent managing miner lifecycle",
    )
    parser.add_argument(
        "--subnet",
        action="append",
        default=[],
        help="Subnet name(s) to manage (from subnets/ directory).",
    )
    parser.add_argument(
        "--subnets-dir",
        type=Path,
        default=Path(__file__).parent.parent / "subnets",
        help="Directory containing subnet packages (default: ./subnets)",
    )
    parser.add_argument(
        "--tick-interval",
        type=int,
        default=300,
        help="Seconds between agent ticks (default: 300 = 5min).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="Maximum ticks to run (0 = unlimited).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single tick and exit (useful for testing).",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(__file__).parent.parent / ".orchestrator_state",
        help="Directory for persistent agent state.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered subnets and exit.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current agent state for all subnets and exit.",
    )

    args = parser.parse_args()
    subnets_dir = args.subnets_dir
    state_store = StateStore(args.state_dir)

    # Discover available subnets
    subnet_dirs = discover_subnets(subnets_dir)

    if args.list:
        if not subnet_dirs:
            print(f"No subnets found in {subnets_dir}")
            sys.exit(0)
        print("Available subnets:")
        for d in subnet_dirs:
            config = load_subnet_config(d)
            print(
                f"  {config.name} (netuid {config.netuid}, {config.network}) "
                f"[strategy: {config.strategy.type}]"
            )
        sys.exit(0)

    if args.status:
        print("Agent state:")
        for d in subnet_dirs:
            config = load_subnet_config(d)
            state = state_store.load(config.name)
            print(f"\n  {config.name}:")
            print(f"    Phase: {state.phase} ({state.time_in_phase_str()})")
            print(f"    Experiments: {state.experiments_run}")
            print(f"    Best metric: {state.best_metric}")
            print(f"    Deployed: {state.deployed}")
            if state.decision_log:
                last = state.decision_log[-1]
                print(f"    Last decision: [{last.get('time_str')}] {last['action']}")
        sys.exit(0)

    # Select subnets
    if args.subnet:
        selected = []
        for name in args.subnet:
            matching = [d for d in subnet_dirs if d.name == name]
            if not matching:
                print(f"Subnet '{name}' not found in {subnets_dir}")
                print(f"Available: {[d.name for d in subnet_dirs]}")
                sys.exit(1)
            selected.extend(matching)
    else:
        selected = subnet_dirs

    if not selected:
        print(f"No subnets found in {subnets_dir}")
        sys.exit(1)

    # Run agent for each subnet
    for subnet_dir in selected:
        config = load_subnet_config(subnet_dir)
        run_agent(
            config,
            state_store,
            tick_interval=args.tick_interval,
            max_ticks=args.max_ticks,
            single_tick=args.once,
        )


if __name__ == "__main__":
    main()
