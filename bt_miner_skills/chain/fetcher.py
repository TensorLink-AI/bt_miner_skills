"""Fetch on-chain data for a subnet and populate ChainParams."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from bt_miner_skills.config.subnet_config import ChainParams


def fetch_chain_params(netuid: int) -> ChainParams:
    """Fetch chain parameters for a subnet from subtensor.

    Uses the bittensor CLI or direct subtensor RPC to pull live data.
    Falls back to defaults if the chain is unreachable.
    """
    try:
        import bittensor as bt

        sub = bt.subtensor()
        params = sub.get_subnet_hyperparameters(netuid)
        return ChainParams(
            netuid=netuid,
            tempo=params.tempo,
            immunity_period=params.immunity_period,
            max_allowed_validators=params.max_allowed_validators,
            min_allowed_weights=params.min_allowed_weights,
            max_weight_limit=params.max_weight_limit,
            difficulty=params.difficulty,
            subnetwork_n=sub.subnetwork_n(netuid),
            max_n=sub.max_n(netuid),
            kappa=params.kappa,
            rho=params.rho,
        )
    except Exception as e:
        print(f"Warning: Could not fetch chain params for netuid {netuid}: {e}")
        return ChainParams(netuid=netuid)


def fetch_metagraph_summary(netuid: int) -> dict[str, Any]:
    """Fetch a summary of the metagraph for competitive analysis.

    Returns stats about current miners: count, avg stake, top incentives, etc.
    """
    try:
        import bittensor as bt

        meta = bt.metagraph(netuid)
        miners = [
            {
                "uid": uid,
                "incentive": float(meta.incentive[uid]),
                "stake": float(meta.stake[uid]),
                "trust": float(meta.trust[uid]),
            }
            for uid in range(meta.n)
            if not meta.validator_permit[uid]
        ]
        miners.sort(key=lambda m: m["incentive"], reverse=True)
        return {
            "total_neurons": int(meta.n),
            "total_miners": len(miners),
            "top_10_miners": miners[:10],
            "avg_incentive": sum(m["incentive"] for m in miners) / max(len(miners), 1),
            "avg_stake": sum(m["stake"] for m in miners) / max(len(miners), 1),
        }
    except Exception as e:
        print(f"Warning: Could not fetch metagraph for netuid {netuid}: {e}")
        return {"error": str(e)}
