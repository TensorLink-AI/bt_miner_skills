"""Synth forecaster — mutable target for evoloop.

THIS FILE IS EVOLVED BY EVOLOOP. It starts as a baseline DLinear + Gaussian
model and gets mutated over iterations to improve CRPS scores.

The prepare.py harness calls this script's train_and_evaluate() function,
or it can be run standalone for testing.

Expected output format (parsed by prepare.py and evoloop):
    crps_weighted=<float>
    crps_worst_asset=<float>
    train_time=<float>
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelConfig:
    """Hyperparameters for the baseline model."""

    lookback: int = 60           # Input window (minutes)
    forecast_24h: int = 289      # Output steps for 24h challenge (5-min intervals)
    forecast_1h: int = 61        # Output steps for 1h challenge (1-min intervals)
    n_paths: int = 1000          # Number of simulated price paths
    hidden_dim: int = 64
    kernel_size: int = 25        # DLinear decomposition kernel
    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 50
    dropout: float = 0.1


def build_model(config: ModelConfig) -> Any:
    """Build a DLinear + Gaussian probabilistic head model.

    Returns a model object with fit() and predict() methods.
    evoloop will mutate this function to try different architectures.
    """
    try:
        import torch
        import torch.nn as nn

        class DLinearGaussian(nn.Module):
            def __init__(self, cfg: ModelConfig):
                super().__init__()
                self.cfg = cfg
                self.kernel_size = cfg.kernel_size

                # Decomposition: moving average for trend
                self.trend_conv = nn.AvgPool1d(
                    kernel_size=cfg.kernel_size, stride=1,
                    padding=cfg.kernel_size // 2,
                )

                # Linear layers for trend and seasonal components
                self.trend_linear = nn.Linear(cfg.lookback, cfg.forecast_24h)
                self.seasonal_linear = nn.Linear(cfg.lookback, cfg.forecast_24h)

                # Gaussian output heads
                self.mu_head = nn.Linear(cfg.forecast_24h, cfg.forecast_24h)
                self.sigma_head = nn.Linear(cfg.forecast_24h, cfg.forecast_24h)

                self.dropout = nn.Dropout(cfg.dropout)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # x: (batch, lookback)
                x_3d = x.unsqueeze(1)  # (batch, 1, lookback)
                trend = self.trend_conv(x_3d).squeeze(1)

                # Handle padding mismatch
                if trend.shape[-1] > x.shape[-1]:
                    trend = trend[..., :x.shape[-1]]
                elif trend.shape[-1] < x.shape[-1]:
                    trend = torch.nn.functional.pad(
                        trend, (0, x.shape[-1] - trend.shape[-1])
                    )

                seasonal = x - trend

                trend_out = self.trend_linear(self.dropout(trend))
                seasonal_out = self.seasonal_linear(self.dropout(seasonal))
                combined = trend_out + seasonal_out

                mu = self.mu_head(combined)
                sigma = torch.nn.functional.softplus(self.sigma_head(combined)) + 1e-6

                return mu, sigma

            def predict_paths(
                self, x: torch.Tensor, n_paths: int = 1000
            ) -> np.ndarray:
                """Generate n_paths simulated price paths."""
                self.eval()
                with torch.no_grad():
                    mu, sigma = self(x)
                    # Sample from Gaussian: (batch, n_paths, forecast_steps)
                    dist = torch.distributions.Normal(mu.unsqueeze(1), sigma.unsqueeze(1))
                    paths = dist.sample((n_paths,))  # (n_paths, batch, forecast_steps)
                    paths = paths.permute(1, 0, 2)    # (batch, n_paths, forecast_steps)
                return paths.cpu().numpy()

        return DLinearGaussian(config)

    except ImportError:
        # Fallback: numpy-only baseline (for local testing without PyTorch)
        return NumpyBaseline(config)


class NumpyBaseline:
    """Simple GBM baseline for environments without PyTorch."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.mu = 0.0
        self.sigma = 0.01

    def fit(self, returns: np.ndarray) -> None:
        """Estimate drift and volatility from historical returns."""
        self.mu = float(np.mean(returns))
        self.sigma = float(np.std(returns)) + 1e-8

    def predict_paths(self, current_price: float, n_steps: int) -> np.ndarray:
        """Generate GBM price paths. Returns (n_paths, n_steps)."""
        dt = 1.0  # unit time step
        n_paths = self.config.n_paths
        z = np.random.randn(n_paths, n_steps)
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = current_price * np.exp(log_prices)
        # Prepend current price
        prices = np.column_stack([np.full(n_paths, current_price), prices[:, :-1]])
        return prices


def train_and_evaluate(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> dict[str, float]:
    """Train the model and return evaluation metrics.

    Args:
        train_data: dict mapping asset names to arrays of returns (in bps)
        val_data: dict mapping asset names to arrays of returns (in bps)
        config: model configuration (uses defaults if None)

    Returns:
        dict with crps_weighted, crps_worst_asset, train_time
    """
    if config is None:
        config = ModelConfig()

    start_time = time.time()

    # Asset weights for Synth SN50
    asset_weights = {
        "BTC": 1.0, "ETH": 0.6716, "XAU": 2.2620, "SOL": 0.5884,
        "SPYX": 2.9914, "NVDAX": 1.3885, "TSLAX": 1.4200,
        "AAPLX": 1.8650, "GOOGLX": 1.4311,
    }

    model = build_model(config)

    # For the numpy baseline, fit on concatenated returns
    if isinstance(model, NumpyBaseline):
        all_returns = np.concatenate(list(train_data.values()))
        model.fit(all_returns)

    train_time = time.time() - start_time

    # Placeholder evaluation — prepare.py will do the real CRPS scoring
    # This just ensures the output format is correct
    crps_per_asset = {}
    for asset in val_data:
        # Placeholder: random CRPS value (real scoring is in prepare.py)
        crps_per_asset[asset] = float(np.random.uniform(50, 200))

    # Weighted average
    total_weight = sum(asset_weights.get(a, 1.0) for a in crps_per_asset)
    crps_weighted = sum(
        crps_per_asset[a] * asset_weights.get(a, 1.0) for a in crps_per_asset
    ) / total_weight

    crps_worst_asset = max(crps_per_asset.values()) if crps_per_asset else 0.0

    results = {
        "crps_weighted": crps_weighted,
        "crps_worst_asset": crps_worst_asset,
        "train_time": train_time,
        "per_asset": crps_per_asset,
    }

    # Print in evoloop-parseable format
    print(f"crps_weighted={results['crps_weighted']:.4f}")
    print(f"crps_worst_asset={results['crps_worst_asset']:.4f}")
    print(f"train_time={results['train_time']:.2f}")

    return results


if __name__ == "__main__":
    # Standalone test with dummy data
    assets = ["BTC", "ETH", "XAU", "SOL", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
    dummy_train = {a: np.random.randn(5000) * 10 for a in assets}  # bps
    dummy_val = {a: np.random.randn(1000) * 10 for a in assets}
    train_and_evaluate(dummy_train, dummy_val)
