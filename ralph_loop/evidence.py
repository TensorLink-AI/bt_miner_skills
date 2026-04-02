"""Backtest evidence extraction and validation.

Parses execution output to detect actual backtest results — CRPS scores,
baseline comparisons, Synth API checks — so the loop can gate deployment
on real evidence rather than trusting the agent's self-assessment.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Known asset tickers in the Synth Subnet
EXPECTED_ASSETS = {"BTC", "ETH", "SOL", "SUI", "PEPE", "SPY", "QQQ", "NVDA", "AAPL"}

# Scoring interval names
EXPECTED_INTERVALS = {"5min", "30min", "3hr", "absolute", "1hr", "24hr"}

# Patterns that indicate real CRPS scores in output
_CRPS_PATTERNS = [
    # "crps_sum: 123.45"
    re.compile(r"crps_sum[\s:=]+(\d+\.?\d*)", re.IGNORECASE),
    # "overall_crps: 123.45" or "Overall CRPS: 123.45"
    re.compile(r"overall[\s_]crps[\s:=]+(\d+\.?\d*)", re.IGNORECASE),
    # "prompt_score: 0.85"
    re.compile(r"prompt_score[\s:=]+(\d+\.?\d*)", re.IGNORECASE),
    # Generic "CRPS: 123.45" or "crps = 123.45" — must not be preceded by
    # a word char (avoids double-matching overall_crps, crps_sum, etc.)
    re.compile(r"(?<![a-zA-Z_])(?:crps|CRPS)[\s:=]+(\d+\.?\d*)", re.IGNORECASE),
]

# Patterns for per-asset CRPS: "BTC CRPS: 123.45" or "CRPS[BTC]: 123.45"
_ASSET_CRPS_PATTERN = re.compile(
    r"(?:(" + "|".join(EXPECTED_ASSETS) + r")[\s_]*(?:crps|CRPS)[\s:=]+(\d+\.?\d*)"
    r"|(?:crps|CRPS)\s*[\[\(]\s*(" + "|".join(EXPECTED_ASSETS) + r")\s*[\]\)][\s:=]*(\d+\.?\d*))",
    re.IGNORECASE,
)

# Baseline comparison patterns
_BASELINE_PATTERNS = [
    re.compile(r"(?:gbm|GBM|geometric.brownian)[\s_]*(?:baseline[\s_]*)?(?:crps|CRPS|score)[\s:=]+(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:historical.sim|hist_sim|historical_simulation)[\s_]*(?:baseline[\s_]*)?(?:crps|CRPS|score)[\s:=]+(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"baseline[\s_]*(?:crps|CRPS|score)[\s:=]+(\d+\.?\d*)", re.IGNORECASE),
]

# Synth API comparison patterns
_SYNTH_API_PATTERNS = [
    re.compile(r"synth[\s_]*api", re.IGNORECASE),
    re.compile(r"live[\s_]*(?:crps|score|network)", re.IGNORECASE),
    re.compile(r"network[\s_]*(?:median|score|crps)", re.IGNORECASE),
    re.compile(r"/validation/scores", re.IGNORECASE),
    re.compile(r"/v2/leaderboard", re.IGNORECASE),
]

# Emulator validation patterns
_EMULATOR_VALIDATION_PATTERNS = [
    re.compile(r"emulator[\s_]*(?:valid|match|correct|verified)", re.IGNORECASE),
    re.compile(r"(?:live|network)\s+(?:vs|versus|compared)\s+(?:emulator|backtest)", re.IGNORECASE),
    re.compile(r"same\s+order\s+of\s+magnitude", re.IGNORECASE),
]

# Patterns that suggest the agent THINKS it's ready but may not have evidence
_FALSE_CONFIDENCE_PATTERNS = [
    re.compile(r"ready\s+(?:for|to)\s+deploy", re.IGNORECASE),
    re.compile(r"deployment\s+ready", re.IGNORECASE),
    re.compile(r"model\s+is\s+(?:ready|complete|done)", re.IGNORECASE),
    re.compile(r"pipeline\s+is\s+(?:complete|ready|done)", re.IGNORECASE),
    re.compile(r"all\s+(?:checks|tests)\s+pass", re.IGNORECASE),
]


def extract_crps_scores(text: str) -> list[float]:
    """Extract all CRPS score values from text."""
    scores = []
    for pattern in _CRPS_PATTERNS:
        for match in pattern.finditer(text):
            try:
                scores.append(float(match.group(1)))
            except (ValueError, IndexError):
                continue
    return scores


def extract_asset_scores(text: str) -> dict[str, float]:
    """Extract per-asset CRPS scores from text."""
    asset_scores = {}
    for match in _ASSET_CRPS_PATTERN.finditer(text):
        # Pattern has two capture groups depending on format
        asset = match.group(1) or match.group(3)
        score = match.group(2) or match.group(4)
        if asset and score:
            try:
                asset_scores[asset.upper()] = float(score)
            except ValueError:
                continue
    return asset_scores


def extract_baseline_scores(text: str) -> dict[str, float]:
    """Extract baseline comparison scores from text."""
    baselines = {}
    for pattern in _BASELINE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                # Use the pattern's name hint as key
                if "gbm" in pattern.pattern.lower():
                    baselines["GBM"] = float(match.group(1))
                elif "hist" in pattern.pattern.lower():
                    baselines["historical_sim"] = float(match.group(1))
                else:
                    baselines["baseline"] = float(match.group(1))
            except (ValueError, IndexError):
                continue
    return baselines


def has_synth_api_evidence(text: str) -> bool:
    """Check if the text contains evidence of Synth API comparison."""
    matches = sum(1 for p in _SYNTH_API_PATTERNS if p.search(text))
    # Require at least 2 patterns to match (reduces false positives)
    return matches >= 2


def has_emulator_validation(text: str) -> bool:
    """Check if the text shows the emulator was validated against live data."""
    return any(p.search(text) for p in _EMULATOR_VALIDATION_PATTERNS)


def has_false_confidence(text: str) -> bool:
    """Detect if the agent claims readiness without evidence."""
    return any(p.search(text) for p in _FALSE_CONFIDENCE_PATTERNS)


def extract_evidence(execution_output: str, llm_response: str, iteration: int) -> dict:
    """Extract all backtest evidence from an iteration's output.

    Returns a dict suitable for appending to LoopState.backtest_results.
    """
    import time

    combined = f"{execution_output}\n{llm_response}"

    crps_scores = extract_crps_scores(execution_output)
    asset_scores = extract_asset_scores(execution_output)
    baseline_scores = extract_baseline_scores(execution_output)
    synth_checked = has_synth_api_evidence(combined)
    emulator_valid = has_emulator_validation(combined)
    claims_ready = has_false_confidence(llm_response)

    has_real_scores = len(crps_scores) > 0 or len(asset_scores) > 0

    evidence = {
        "iteration": iteration,
        "timestamp": time.time(),
        "crps_scores_found": crps_scores[:10],  # cap to avoid bloat
        "asset_scores": asset_scores,
        "baseline_scores": baseline_scores,
        "synth_api_compared": synth_checked,
        "emulator_validated": emulator_valid,
        "has_real_scores": has_real_scores,
        "claims_ready": claims_ready,
        "claims_ready_without_evidence": claims_ready and not has_real_scores,
    }

    if has_real_scores:
        logger.info(
            "Backtest evidence found at iteration %d: %d CRPS scores, %d assets",
            iteration, len(crps_scores), len(asset_scores),
        )
    elif claims_ready:
        logger.warning(
            "Agent claims readiness at iteration %d but NO backtest scores found!",
            iteration,
        )

    return evidence


def build_evidence_summary(state) -> str:
    """Build a human-readable summary of all backtest evidence collected so far.

    Used to inject into the prompt so the agent knows what's been proven.
    """
    if not state.backtest_results:
        return ""

    parts = ["## Backtest Evidence Tracker\n"]

    total_with_scores = sum(1 for r in state.backtest_results if r.get("has_real_scores"))
    total_iterations = state.iteration_count

    parts.append(f"**Iterations with actual backtest scores: {total_with_scores}/{total_iterations}**\n")

    if not state.has_backtest_scores:
        parts.append(
            "**WARNING: No actual CRPS scores have been detected in any execution output yet.**\n"
            "You must RUN your validator emulator on real data and produce numeric CRPS scores\n"
            "before your pipeline can be considered functional. Writing code is not enough —\n"
            "execute it and show the scores.\n"
        )

    if state.has_backtest_scores and not state.has_baseline_comparison:
        parts.append(
            "**MISSING: Baseline comparison.** You have CRPS scores but haven't compared\n"
            "against GBM or historical simulation baselines. Do this before claiming the\n"
            "model is competitive.\n"
        )

    if state.has_backtest_scores and not state.has_synth_api_check:
        parts.append(
            "**MISSING: Synth API cross-check.** Your emulator scores haven't been verified\n"
            "against live network scores. Fetch from /validation/scores/latest and compare.\n"
        )

    # Show best scores found
    all_crps = []
    for r in state.backtest_results:
        all_crps.extend(r.get("crps_scores_found", []))

    if all_crps:
        parts.append(f"**Best CRPS seen:** {min(all_crps):.4f}")
        parts.append(f"**Latest CRPS values:** {all_crps[-5:]}")

    # Show per-asset coverage
    all_assets = set()
    for r in state.backtest_results:
        all_assets.update(r.get("asset_scores", {}).keys())

    if all_assets:
        missing = EXPECTED_ASSETS - all_assets
        parts.append(f"\n**Assets evaluated:** {', '.join(sorted(all_assets))}")
        if missing:
            parts.append(f"**Assets MISSING:** {', '.join(sorted(missing))}")
    else:
        parts.append(f"\n**Assets evaluated:** NONE — no per-asset scores detected")

    # Evidence flags
    flags = []
    flags.append(f"Emulator validated: {'YES' if state.has_validated_emulator else 'NO'}")
    flags.append(f"Backtest scores: {'YES' if state.has_backtest_scores else 'NO'}")
    flags.append(f"Baseline comparison: {'YES' if state.has_baseline_comparison else 'NO'}")
    flags.append(f"Synth API check: {'YES' if state.has_synth_api_check else 'NO'}")
    flags.append(f"Deployment ready: {'YES' if state.deployment_ready else 'NO'}")
    parts.append("\n**Evidence checklist:**\n" + "\n".join(f"  - {f}" for f in flags))

    return "\n".join(parts)


def build_evidence_gate_warning(state) -> str:
    """Build a stern warning if the agent claims readiness without evidence.

    Injected into the prompt when the agent's last response claimed readiness
    but no backtest evidence exists.
    """
    if not state.backtest_results:
        return ""

    latest = state.backtest_results[-1]
    if not latest.get("claims_ready_without_evidence"):
        return ""

    return (
        "\n## EVIDENCE GATE — ACTION REQUIRED\n\n"
        "**You claimed deployment readiness but NO actual backtest scores were found**\n"
        "**in your execution output.** This is not acceptable.\n\n"
        "Before you can claim readiness, you MUST have ALL of the following in your\n"
        "execution output (not just in code you wrote — actually EXECUTED with real numbers):\n\n"
        "1. CRPS scores from your validator emulator on real price data\n"
        "2. Per-asset CRPS breakdown (all 9 assets)\n"
        "3. Comparison against GBM and historical simulation baselines\n"
        "4. Cross-check against Synth API live network scores\n\n"
        "**Do NOT proceed to deployment. Go back and run actual backtests.**\n"
    )
