# Validator Emulator Reference

## Purpose

The validator emulator replicates the **exact** scoring logic from `synth/validator/reward.py`
so you can evaluate your models locally before deploying to mainnet. If your emulator doesn't
match the validator, your backtests are meaningless.

## CRPS Calculation

### Core Formula

For an ensemble of N predictions y₁...yₙ against observation x:

```
CRPS = (1/N) Σ|yₙ - x| - (1/(2N²)) ΣΣ|yₙ - yₘ|
```

- **Term 1**: Average absolute error (penalizes bias)
- **Term 2**: Ensemble spread (rewards appropriate uncertainty)

### Implementation

```python
import numpy as np

def crps_ensemble(predictions: np.ndarray, observation: float) -> float:
    """
    Calculate CRPS for an ensemble forecast.
    
    Args:
        predictions: array of shape [n_paths] — predicted values
        observation: scalar — actual observed value
    
    Returns:
        crps: float — CRPS score (lower is better)
    """
    n = len(predictions)
    
    # Sort for efficient calculation
    sorted_preds = np.sort(predictions)
    
    # Term 1: mean absolute error
    mae = np.mean(np.abs(sorted_preds - observation))
    
    # Term 2: mean pairwise absolute difference (efficient via sorted order)
    # For sorted values: Σ|yi - yj| = 2 * Σ(2i - n - 1) * yi / n²
    indices = np.arange(1, n + 1)
    spread = 2 * np.sum((2 * indices - n - 1) * sorted_preds) / (n * n)
    
    crps = mae - 0.5 * abs(spread)
    return crps
```

### Efficient Vectorized CRPS

```python
def crps_ensemble_vectorized(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Vectorized CRPS for multiple timesteps.
    
    Args:
        predictions: [n_paths, n_timesteps]
        observations: [n_timesteps]
    
    Returns:
        crps_values: [n_timesteps]
    """
    n_paths = predictions.shape[0]
    
    # Sort along path axis
    sorted_preds = np.sort(predictions, axis=0)  # [n_paths, n_timesteps]
    
    # Term 1: MAE
    mae = np.mean(np.abs(predictions - observations[np.newaxis, :]), axis=0)
    
    # Term 2: Spread (using the efficient sorted formula)
    weights = (2 * np.arange(1, n_paths + 1) - n_paths - 1) / (n_paths * n_paths)
    spread = np.abs(np.sum(sorted_preds * weights[:, np.newaxis], axis=0))
    
    return mae - spread
```

---

## Scoring Pipeline (Exact Validator Replica)

### Step 1: Convert Paths to Basis Point Changes

```python
def paths_to_bps_changes(paths: np.ndarray, interval_steps: int) -> np.ndarray:
    """
    Convert price paths to basis point changes over a given interval.
    
    Args:
        paths: [n_paths, n_timesteps] — absolute price paths
        interval_steps: number of base timesteps per interval
                       (e.g., 6 for 30min interval in 5-min data)
    
    Returns:
        bps_changes: [n_paths, n_intervals] — price changes in basis points
    """
    n_paths, n_timesteps = paths.shape
    n_intervals = (n_timesteps - 1) // interval_steps
    
    bps_changes = []
    for i in range(n_intervals):
        start_idx = i * interval_steps
        end_idx = start_idx + interval_steps
        
        start_prices = paths[:, start_idx]
        end_prices = paths[:, end_idx]
        
        # Change in basis points
        change_bps = (end_prices / start_prices - 1) * 10000
        bps_changes.append(change_bps)
    
    return np.array(bps_changes).T  # [n_paths, n_intervals]


def actual_to_bps_changes(actual_prices: np.ndarray, interval_steps: int) -> np.ndarray:
    """
    Convert actual price series to basis point changes.
    
    Args:
        actual_prices: [n_timesteps] — observed prices at base interval
        interval_steps: number of base timesteps per interval
    
    Returns:
        bps_changes: [n_intervals] — actual changes in basis points
    """
    n_timesteps = len(actual_prices)
    n_intervals = (n_timesteps - 1) // interval_steps
    
    bps_changes = []
    for i in range(n_intervals):
        start_idx = i * interval_steps
        end_idx = start_idx + interval_steps
        
        change_bps = (actual_prices[end_idx] / actual_prices[start_idx] - 1) * 10000
        bps_changes.append(change_bps)
    
    return np.array(bps_changes)
```

### Step 2: Calculate CRPS Across All Interval Buckets

```python
# 24-HOUR CHALLENGE INTERVALS
INTERVALS_24H = {
    "5min":  {"steps": 1,   "count": 288},   # 288 × 5-min intervals
    "30min": {"steps": 6,   "count": 48},     # 48 × 30-min intervals
    "3hr":   {"steps": 36,  "count": 8},      # 8 × 3-hour intervals
}

# 1-HOUR HFT CHALLENGE INTERVALS (1-min base)
INTERVALS_1H = {
    "1min":  {"steps": 1,   "count": 60},     # 60 × 1-min intervals
    "5min":  {"steps": 5,   "count": 12},     # 12 × 5-min intervals
    "15min": {"steps": 15,  "count": 4},      # 4 × 15-min intervals
}

# ASSET WEIGHTS
ASSET_WEIGHTS = {
    "BTC":    1.0000,
    "ETH":    0.6716,
    "XAU":    2.2620,
    "SOL":    0.5884,
    "SPYX":   2.9914,
    "NVDAX":  1.3885,
    "TSLAX":  1.4200,
    "AAPLX":  1.8650,
    "GOOGLX": 1.4311,
}

# ASSETS PER CHALLENGE
ASSETS_24H = ["BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
ASSETS_1H = ["BTC", "ETH", "SOL", "XAU"]


def score_prediction(
    predicted_paths: np.ndarray,
    actual_prices: np.ndarray,
    challenge_type: str = "24h",
) -> dict:
    """
    Score a single prediction using the exact validator logic.
    
    Args:
        predicted_paths: [n_paths, n_timesteps] — simulated price paths
        actual_prices: [n_timesteps] — observed prices at base interval
        challenge_type: "24h" or "1h"
    
    Returns:
        {
            "crps_sum": float,           # Total CRPS across all intervals
            "interval_crps": dict,       # Per-interval-type CRPS breakdown
            "absolute_crps": float,      # CRPS on final absolute price
        }
    """
    intervals = INTERVALS_24H if challenge_type == "24h" else INTERVALS_1H
    
    total_crps = 0.0
    interval_crps = {}
    
    for interval_name, params in intervals.items():
        steps = params["steps"]
        count = params["count"]
        
        # Get bps changes for predictions and actuals
        pred_bps = paths_to_bps_changes(predicted_paths, steps)  # [n_paths, n_intervals]
        actual_bps = actual_to_bps_changes(actual_prices, steps)  # [n_intervals]
        
        # Trim to expected count
        pred_bps = pred_bps[:, :count]
        actual_bps = actual_bps[:count]
        
        # CRPS for each interval
        crps_values = crps_ensemble_vectorized(pred_bps, actual_bps)
        interval_sum = np.sum(crps_values)
        
        interval_crps[interval_name] = {
            "total": float(interval_sum),
            "mean": float(np.mean(crps_values)),
            "per_interval": crps_values.tolist(),
        }
        total_crps += interval_sum
    
    # Absolute CRPS on final price (divided by price, × 10000 for bps)
    final_pred_prices = predicted_paths[:, -1]
    final_actual_price = actual_prices[-1]
    abs_crps = crps_ensemble(final_pred_prices, final_actual_price)
    abs_crps_bps = (abs_crps / final_actual_price) * 10000
    
    total_crps += abs_crps_bps
    
    return {
        "crps_sum": float(total_crps),
        "interval_crps": interval_crps,
        "absolute_crps": float(abs_crps_bps),
    }
```

### Step 3: Score Transformation (Across Miners)

```python
def transform_scores(scores: dict[str, float]) -> dict[str, float]:
    """
    Apply validator's score transformation.
    
    Args:
        scores: {miner_id: crps_sum} — raw CRPS sums
    
    Returns:
        prompt_scores: {miner_id: transformed_score}
    """
    values = np.array(list(scores.values()))
    miner_ids = list(scores.keys())
    
    # 1. Cap worst 10% at 90th percentile
    p90 = np.percentile(values, 90)
    capped = np.minimum(values, p90)
    
    # 2. Assign 90th percentile to invalid/missing
    # (In emulation, mark missing miners explicitly)
    
    # 3. Subtract best score
    best = np.min(capped)
    prompt_scores = capped - best
    
    return dict(zip(miner_ids, prompt_scores.tolist()))
```

### Step 4: Rolling Average (Leaderboard Score)

```python
def rolling_average(
    prompt_scores: list[dict],
    asset_weights: dict[str, float],
    window_days: int = 10,
    current_time=None,
) -> float:
    """
    Calculate rolling weighted average score.
    
    Args:
        prompt_scores: list of {
            "score": float,
            "asset": str,
            "timestamp": datetime,
        }
        asset_weights: per-asset weight coefficients
        window_days: lookback window in days
        current_time: current time (for filtering)
    
    Returns:
        leaderboard_score: float (lower is better)
    """
    if current_time is None:
        current_time = max(p["timestamp"] for p in prompt_scores)
    
    cutoff = current_time - pd.Timedelta(days=window_days)
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for p in prompt_scores:
        if p["timestamp"] < cutoff:
            continue
        
        w = asset_weights.get(p["asset"], 1.0)
        total_weighted_score += p["score"] * w
        total_weight += w
    
    if total_weight == 0:
        return float("inf")
    
    return total_weighted_score / total_weight
```

### Step 5: Emission Share (Softmax)

```python
def calculate_emission_share(
    leaderboard_scores: dict[str, float],
    beta: float = -0.0475,
) -> dict[str, float]:
    """
    Calculate emission share using softmax.
    
    Args:
        leaderboard_scores: {miner_id: rolling_average_score}
        beta: softmax temperature (negative = lower score → higher weight)
    
    Returns:
        emission_shares: {miner_id: fraction_of_emissions}
    """
    scores = np.array(list(leaderboard_scores.values()))
    miner_ids = list(leaderboard_scores.keys())
    
    # Softmax with negative beta (rewards lower scores)
    exp_scores = np.exp(beta * scores)
    shares = exp_scores / np.sum(exp_scores)
    
    return dict(zip(miner_ids, shares.tolist()))
```

---

## Full Emulation Loop

```python
class ValidatorEmulator:
    """
    Emulates the full validator scoring pipeline.
    
    Usage:
        emulator = ValidatorEmulator()
        
        # Score a single prediction
        result = emulator.score_prediction(paths, actuals, "24h", "BTC")
        
        # Run full emulation over historical data
        results = emulator.run_backtest(model, data, challenges)
        
        # Get leaderboard position
        ranking = emulator.get_ranking(results)
    """
    
    def __init__(self):
        self.prompt_history = []  # Stores all scored prompts
    
    def score_prediction(self, paths, actuals, challenge_type, asset):
        """Score a single prediction and add to history."""
        result = score_prediction(paths, actuals, challenge_type)
        result["asset"] = asset
        result["challenge_type"] = challenge_type
        result["timestamp"] = pd.Timestamp.now(tz="UTC")
        self.prompt_history.append(result)
        return result
    
    def run_backtest(self, model, data_splits, challenges):
        """
        Run model across all validation data, scoring each simulated prompt.
        
        Simulates the validator sending requests every 30 minutes.
        """
        results = []
        
        for challenge in challenges:
            asset = challenge["asset"]
            challenge_type = challenge["type"]
            
            # Get the appropriate data
            asset_data = data_splits[asset]
            
            # Simulate prompts every 30 minutes
            prompt_times = pd.date_range(
                start=asset_data["val"][0],
                end=asset_data["val"][1],
                freq="30min",
            )
            
            for t in prompt_times:
                # Get lookback window for model input
                lookback_start = t - pd.Timedelta(minutes=model.lookback * 5)
                features = get_features(asset_data, lookback_start, t)
                
                if features is None:
                    continue
                
                # Get current price
                current_price = get_price_at(asset_data, t)
                
                # Generate predictions
                paths = model.predict(features, current_price, n_paths=1000)
                
                # Get actual future prices
                if challenge_type == "24h":
                    horizon = pd.Timedelta(hours=24)
                    interval = pd.Timedelta(minutes=5)
                else:
                    horizon = pd.Timedelta(hours=1)
                    interval = pd.Timedelta(minutes=1)
                
                actuals = get_actual_prices(asset_data, t, t + horizon, interval)
                
                if actuals is None or len(actuals) == 0:
                    continue
                
                # Score
                result = score_prediction(
                    np.array(paths),
                    np.array(actuals),
                    challenge_type,
                )
                result["asset"] = asset
                result["challenge_type"] = challenge_type
                result["timestamp"] = t
                results.append(result)
        
        return results
    
    def get_ranking(self, results, baseline_results=None):
        """
        Calculate leaderboard score and simulated emission share.
        
        If baseline_results provided, compares against baseline miners.
        """
        my_score = rolling_average(
            [{"score": r["crps_sum"], "asset": r["asset"], "timestamp": r["timestamp"]}
             for r in results],
            ASSET_WEIGHTS,
        )
        
        ranking = {"my_rolling_avg": my_score}
        
        if baseline_results:
            all_scores = {"my_model": my_score}
            for name, b_results in baseline_results.items():
                all_scores[name] = rolling_average(
                    [{"score": r["crps_sum"], "asset": r["asset"], "timestamp": r["timestamp"]}
                     for r in b_results],
                    ASSET_WEIGHTS,
                )
            ranking["emission_shares"] = calculate_emission_share(all_scores)
        
        return ranking
```

---

## Baseline Models for Comparison

Always compare against these baselines:

### 1. Geometric Brownian Motion (GBM)
```python
def gbm_baseline(current_price, dt, n_steps, n_paths, mu, sigma):
    """Standard GBM — the minimum bar to beat."""
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = current_price
    
    for t in range(n_steps):
        z = np.random.randn(n_paths)
        paths[:, t+1] = paths[:, t] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    return paths
```

### 2. Historical Simulation
```python
def historical_sim(current_price, historical_returns, n_steps, n_paths):
    """Bootstrap from historical returns."""
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = current_price
    
    for t in range(n_steps):
        sampled = np.random.choice(historical_returns, size=n_paths)
        paths[:, t+1] = paths[:, t] * (1 + sampled)
    return paths
```

Your model should beat BOTH of these on CRPS by at least 15% before deploying.
