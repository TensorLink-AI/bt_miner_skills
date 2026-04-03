"""Synth evaluation harness — frozen file for evoloop.

THIS FILE IS NOT MUTATED BY EVOLOOP. It provides:
1. Data loading (fetch OHLCV, compute causal features)
2. Walk-forward split generation
3. CRPS scoring (replicates the Synth validator exactly)
4. Metric reporting in the format evoloop expects

evoloop calls: python prepare.py
Which imports train.py, runs train_and_evaluate(), and reports results.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────────────────────────
# CRPS Scoring (replicates Synth validator)
# ──────────────────────────────────────────────

def crps_ensemble(predictions: np.ndarray, observed: float) -> float:
    """Calculate CRPS for an ensemble of predictions.

    CRPS = (1/N) * sum(|y_n - x|) - (1/(2*N^2)) * sum(sum(|y_n - y_m|))

    Args:
        predictions: array of shape (n_paths,) — ensemble predictions
        observed: scalar — the actual observed value

    Returns:
        CRPS score (lower is better)
    """
    n = len(predictions)
    if n == 0:
        return float("inf")

    # Term 1: MAE
    mae = np.mean(np.abs(predictions - observed))

    # Term 2: ensemble spread
    sorted_preds = np.sort(predictions)
    # Efficient computation: sum |y_i - y_j| = 2 * sum_i (2*i - n) * y_i / n^2
    indices = np.arange(n)
    spread = 2.0 * np.sum((2 * indices - n + 1) * sorted_preds) / (n * n)

    return mae - 0.5 * abs(spread)


def prices_to_bps(prices: np.ndarray, start_price: float) -> np.ndarray:
    """Convert price paths to basis point changes from start price.

    bps = (price / start_price - 1) * 10000
    """
    return (prices / start_price - 1.0) * 10000.0


def score_predictions(
    predicted_paths: np.ndarray,
    actual_prices: np.ndarray,
    start_price: float,
    interval_seconds: int = 300,
) -> dict[str, float]:
    """Score predictions using the Synth CRPS methodology.

    Args:
        predicted_paths: (n_paths, n_steps) price paths
        actual_prices: (n_steps,) actual prices
        start_price: starting price for bps conversion
        interval_seconds: 300 for 24h (5-min), 60 for 1h (1-min)

    Returns:
        dict with crps_sum and per-interval CRPS
    """
    # Convert to basis points
    pred_bps = prices_to_bps(predicted_paths, start_price)
    actual_bps = prices_to_bps(actual_prices, start_price)

    n_steps = len(actual_prices)

    # Define scoring intervals based on challenge type
    if interval_seconds == 300:  # 24h challenge
        intervals = {
            "5min": list(range(n_steps)),
            "30min": list(range(5, n_steps, 6)),
            "3hr": list(range(35, n_steps, 36)),
            "absolute": [n_steps - 1] if n_steps > 0 else [],
        }
    else:  # 1h challenge
        intervals = {
            "1min": list(range(n_steps)),
            "5min": list(range(4, n_steps, 5)),
            "15min": list(range(14, n_steps, 15)),
            "absolute": [n_steps - 1] if n_steps > 0 else [],
        }

    crps_by_interval = {}
    crps_values = []

    for interval_name, indices in intervals.items():
        interval_crps = []
        for idx in indices:
            if idx < n_steps and idx < pred_bps.shape[1]:
                c = crps_ensemble(pred_bps[:, idx], actual_bps[idx])
                interval_crps.append(c)
                crps_values.append(c)
        if interval_crps:
            crps_by_interval[interval_name] = float(np.mean(interval_crps))

    crps_sum = float(np.sum(crps_values)) if crps_values else float("inf")

    return {"crps_sum": crps_sum, "intervals": crps_by_interval}


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

ASSETS_24H = ["BTC", "ETH", "XAU", "SOL", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
ASSETS_1H = ["BTC", "ETH", "XAU", "SOL"]

ASSET_WEIGHTS = {
    "BTC": 1.0, "ETH": 0.6716, "XAU": 2.2620, "SOL": 0.5884,
    "SPYX": 2.9914, "NVDAX": 1.3885, "TSLAX": 1.4200,
    "AAPLX": 1.8650, "GOOGLX": 1.4311,
}


def generate_synthetic_data(
    n_train: int = 5000,
    n_val: int = 1000,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate synthetic return data for testing.

    In production, this would fetch real OHLCV from Binance/yfinance
    and compute causal features with .shift(1).

    Returns:
        (train_data, val_data) dicts mapping asset -> returns in bps
    """
    rng = np.random.RandomState(42)

    train_data = {}
    val_data = {}

    for asset in ASSETS_24H:
        # Simulate returns with asset-specific volatility
        vol = rng.uniform(5, 30)  # bps per step
        drift = rng.uniform(-0.5, 0.5)

        train_data[asset] = rng.randn(n_train) * vol + drift
        val_data[asset] = rng.randn(n_val) * vol + drift

    return train_data, val_data


# ──────────────────────────────────────────────
# Main: run train.py and evaluate
# ──────────────────────────────────────────────

def main() -> None:
    """Load train.py, run training, evaluate with CRPS, report results."""
    print("[prepare] Loading data...")
    train_data, val_data = generate_synthetic_data()

    print(f"[prepare] Assets: {list(train_data.keys())}")
    print(f"[prepare] Train size: {len(next(iter(train_data.values())))} per asset")
    print(f"[prepare] Val size: {len(next(iter(val_data.values())))} per asset")

    # Import train.py (evoloop mutates this file)
    print("[prepare] Importing train module...")
    train_module = importlib.import_module("train")

    # Run training and evaluation
    print("[prepare] Running train_and_evaluate()...")
    results = train_module.train_and_evaluate(train_data, val_data)

    # Report in evoloop-parseable format
    print()
    print(f"crps_weighted={results['crps_weighted']:.4f}")
    print(f"crps_worst_asset={results['crps_worst_asset']:.4f}")
    print(f"train_time={results['train_time']:.2f}")


if __name__ == "__main__":
    main()
