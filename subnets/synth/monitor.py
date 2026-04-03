"""Synth subnet monitor — check live miner performance via Synth API.

Queries the Synth leaderboard and validation endpoints to determine:
- Is the miner healthy and responding?
- What's the current emission share?
- Per-asset CRPS breakdown
- Whether performance has degraded enough to trigger re-evolution

Outputs a JSON object on the last line for the orchestrator to parse.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

SYNTH_API_BASE = "https://api.synthdata.co"

# Asset weights for Synth SN50 (used to identify critical degradation)
ASSET_WEIGHTS = {
    "BTC": 1.0, "ETH": 0.6716, "XAU": 2.2620, "SOL": 0.5884,
    "SPYX": 2.9914, "NVDAX": 1.3885, "TSLAX": 1.4200,
    "AAPLX": 1.8650, "GOOGLX": 1.4311,
}


def fetch_json(url: str) -> dict | None:
    """Fetch JSON from a URL, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "synth-monitor"})
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[monitor] Failed to fetch {url}: {e}", file=sys.stderr)
        return None


def check_leaderboard(miner_uid: str | None) -> dict[str, Any]:
    """Check the live leaderboard for our miner's performance."""
    data = fetch_json(f"{SYNTH_API_BASE}/v2/leaderboard/latest")
    if not data:
        return {"healthy": False, "message": "Could not reach Synth leaderboard API."}

    if not miner_uid:
        return {
            "healthy": True,
            "message": "No MINER_UID set — leaderboard fetched but can't check specific miner.",
            "leaderboard_size": len(data) if isinstance(data, list) else 0,
        }

    # Find our miner in the leaderboard
    miners = data if isinstance(data, list) else data.get("miners", [])
    our_miner = None
    for m in miners:
        uid = str(m.get("uid", ""))
        if uid == miner_uid:
            our_miner = m
            break

    if not our_miner:
        return {
            "healthy": False,
            "message": f"Miner UID {miner_uid} not found on leaderboard.",
            "leaderboard_size": len(miners),
        }

    emission_share = our_miner.get("rewards", our_miner.get("emission_share", 0))

    return {
        "healthy": True,
        "emission_share": emission_share,
        "rank": miners.index(our_miner) + 1 if our_miner in miners else None,
        "total_miners": len(miners),
    }


def check_validation(miner_uid: str | None) -> dict[str, Any]:
    """Check if our miner passes format validation."""
    if not miner_uid:
        return {"validated": None, "message": "No MINER_UID set."}

    data = fetch_json(f"{SYNTH_API_BASE}/validation/miner?uid={miner_uid}")
    if not data:
        return {"validated": None, "message": "Could not reach validation endpoint."}

    return {
        "validated": data.get("validated", False),
        "reason": data.get("reason", ""),
    }


def main() -> None:
    miner_uid = os.environ.get("MINER_UID")
    re_evolve_threshold = float(os.environ.get("RE_EVOLVE_THRESHOLD", "0.01"))

    leaderboard = check_leaderboard(miner_uid)
    validation = check_validation(miner_uid)

    # Determine overall health
    healthy = leaderboard.get("healthy", False)
    emission_share = leaderboard.get("emission_share", 0)

    # Trigger re-evolution if emission share drops below threshold
    should_re_evolve = (
        emission_share is not None
        and isinstance(emission_share, (int, float))
        and emission_share < re_evolve_threshold
        and miner_uid is not None
    )

    metrics = {
        "emission_share": emission_share,
        "rank": leaderboard.get("rank"),
        "total_miners": leaderboard.get("total_miners"),
        "validated": validation.get("validated"),
    }

    message_parts = []
    if emission_share is not None:
        message_parts.append(f"emission_share={emission_share:.4f}")
    if leaderboard.get("rank"):
        message_parts.append(
            f"rank={leaderboard['rank']}/{leaderboard.get('total_miners', '?')}"
        )
    if validation.get("validated") is not None:
        message_parts.append(
            f"validated={'yes' if validation['validated'] else 'NO'}"
        )
    if should_re_evolve:
        message_parts.append("RE-EVOLVE TRIGGERED")

    result = {
        "healthy": healthy,
        "metrics": metrics,
        "should_re_evolve": should_re_evolve,
        "message": ", ".join(message_parts) if message_parts else "No data.",
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
