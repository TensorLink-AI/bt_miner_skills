---
name: synth-miner-mlops
description: >
  Build, train, backtest, and deploy probabilistic forecasting models as a competitive miner on the
  Synth Subnet (Bittensor SN50). This skill covers the full MLOps pipeline: sourcing historical OHLCV
  data with strict train/val/test splits to prevent data leakage, training DLinear + Gaussian
  probabilistic head models (and swapping in custom architectures), running a local validator emulator
  that replicates exact CRPS scoring across all challenge types (1-hour HFT and 24-hour for BTC, ETH,
  SOL, XAU, SPYX, NVDAX, TSLAX, AAPLX, GOOGLX), tracking model competitiveness on a leaderboard,
  and deploying winning models to production. Think of it as AutoSearch for Synth — automated model
  development with infrastructure to track which models would actually win on mainnet. Trigger whenever
  the user mentions Synth subnet, SN50, CRPS scoring, probabilistic price forecasting, price path
  simulation, Synth miner, Synth MLOps, synth-subnet, Mode Network subnet, DLinear model, Gaussian
  probabilistic head, ensemble forecasting for Bittensor, or wants to build/improve a miner for the
  Synth subnet. Also trigger when someone says "set up synth miner", "train synth models", "backtest
  my synth miner", "emulate validator scoring", "which model wins on synth", or any reference to
  competitive probabilistic forecasting on Bittensor.
---

# Synth Subnet MLOps Miner Skill

## Quick Context: What Is Synth?

Synth (SN50 mainnet / SN247 testnet) is a Bittensor subnet where miners produce **probabilistic
price forecasts** — not point predictions. Miners generate 1000 simulated price paths per asset per
request. Validators score these using **CRPS (Continuous Ranked Probability Score)** which rewards
both accuracy and well-calibrated uncertainty. Lower CRPS = better. The subnet runs two competitions
with a 50/50 emissions split:

| Competition | Horizon | Assets | Interval | Paths | Frequency |
|------------|---------|--------|----------|-------|-----------|
| **24-Hour** | 24h | BTC, ETH, SOL, XAU, SPYX, NVDAX, TSLAX, AAPLX, GOOGLX | 5 min | 1000 | Every 30 min |
| **1-Hour HFT** | 1h | BTC, ETH, SOL, XAU | 1 min | 1000 | Every 30 min |

### Asset Weights (used in rolling average)
```
BTC:    1.0000    ETH:    0.6716    XAU:    2.2620    SOL:    0.5884
SPYX:   2.9914    NVDAX:  1.3885    TSLAX:  1.4200    AAPLX:  1.8650
GOOGLX: 1.4311
```

### CRPS Scoring Intervals (24-Hour Challenge)
- 288 × 5-minute intervals
- 48 × 30-minute intervals
- 8 × 3-hour intervals
- 1 × absolute final price comparison
- All computed in **basis points** (price change / start price × 10000)

### Score Transformation Pipeline
1. Calculate CRPS sum across all intervals for each miner
2. Cap worst 10% at 90th percentile
3. Assign 90th percentile to invalid/missing predictions
4. Subtract best (lowest) CRPS sum → best miner gets 0
5. Rolling average over 10-day window, weighted by asset weights
6. Softmax with β = -0.0475 → emission weights

---

## Step 1: Read the Reference Files

Before doing anything, read the appropriate reference files from `references/` relative to this skill:

| Task | Read These |
|------|-----------|
| **Setting up the full pipeline** | `architecture.md` → `data_pipeline.md` → `models.md` → `validator_emulator.md` |
| **Training a model** | `data_pipeline.md` → `models.md` |
| **Backtesting / Emulating scoring** | `validator_emulator.md` → `leaderboard.md` |
| **Deploying to mainnet** | `deployment.md` |
| **Checking live competition** | `synth_api.md` |
| **Understanding data leakage risks** | `data_pipeline.md` (Section: Anti-Leakage) |
| **Adding a new model architecture** | `models.md` (Section: Custom Models) |
| **Training on remote GPU (Basilica)** | Install **basilica-cli-helper** external skill or use pre-installed `basilica-sdk` skill |

---

## Step 2: Understand the Architecture

The pipeline has 5 stages:

```
[1. Data Sourcing] → [2. Feature Engineering] → [3. Model Training] → [4. Validator Emulation] → [5. Deployment]
     ↓                      ↓                        ↓                       ↓                       ↓
  Pyth/Binance         Anti-leakage            DLinear+Gaussian        CRPS scoring            PM2 miner
  OHLCV data           time splits             probabilistic head      rolling average          hot-swap
  all 9 assets         walk-forward            train on history        leaderboard rank         model registry
```

### Critical Anti-Leakage Rules
1. **Temporal splits only** — never random. Train < Val < Test < Live.
2. **Feature computation must be causal** — no lookahead in rolling stats.
3. **Walk-forward validation** — retrain on expanding window, test on next unseen block.
4. **Purge gap** — leave a gap between train end and val start equal to the forecast horizon (24h or 1h).
5. **Embargo period** — no overlapping forecast windows between train and test sets.

---

## Step 3: Build It

### Default Model: DLinear + Gaussian Probabilistic Head

The default model is intentionally simple — a DLinear backbone (decomposition-linear from "Are Transformers Effective for Time Series Forecasting?") with a Gaussian probabilistic output head. This serves as your baseline. The architecture:

```
Input (lookback window of returns in bps)
  → Trend/Seasonal Decomposition
  → Linear layers (one per component)
  → Concatenate
  → μ head (mean prediction)
  → σ head (std prediction, softplus activation)
  → Sample N=1000 paths from N(μ, σ²) at each timestep
  → Accumulate to get price levels
```

**Why this default?**
- DLinear is fast to train (seconds, not hours)
- Gaussian head gives calibrated uncertainty out of the box
- Easy to iterate: swap backbone, change distribution, add features
- Beats naive GBM baselines on CRPS because the head is designed for it

### Model Development Loop (AutoSearch-style)

```python
# Pseudocode for the automated model search
for model_config in search_space:
    model = build_model(model_config)
    for fold in walk_forward_folds:
        train(model, fold.train_data)
        predictions = model.predict(fold.val_data, n_paths=1000)
        crps_score = validator_emulator.score(predictions, fold.val_actuals)
        leaderboard.log(model_config, fold, crps_score)
    
    if leaderboard.is_competitive(model_config):
        model_registry.register(model, stage="candidate")
        live_test_score = validator_emulator.live_score(model, hours=48)
        if live_test_score < leaderboard.mainnet_threshold:
            model_registry.promote(model, stage="production")
            deployer.hot_swap(model)
```

---

## Step 4: Run the Scripts

Generate the implementation scripts using the reference files. Key scripts to create:

| Script | Purpose |
|--------|---------|
| `data_pipeline.py` | Fetch OHLCV data, compute features, enforce temporal splits |
| `models/dlinear_gaussian.py` | Default DLinear + Gaussian head model |
| `models/base_model.py` | Abstract base class for all models |
| `validator_emulator.py` | Replicate exact CRPS scoring logic |
| `leaderboard.py` | Track model scores across assets, challenges, time |
| `model_registry.py` | Version, store, promote/demote models |
| `train.py` | Training loop with walk-forward validation |
| `live_tester.py` | Run models against live price feeds, score in real-time |
| `deploy.py` | Hot-swap models into the production miner |
| `config.py` | All constants: assets, weights, intervals, horizons |

---

## Step 5: Track Competitiveness

The leaderboard tracks:

| Metric | Description |
|--------|-------------|
| **CRPS Sum** | Raw CRPS across all intervals for a single prompt |
| **Prompt Score** | CRPS Sum after percentile capping + best subtraction |
| **Rolling Avg** | 10-day weighted rolling average (the actual leaderboard metric) |
| **Emission Share** | Softmax(β × rolling_avg) — what % of emissions you'd get |
| **Per-Asset Breakdown** | Individual CRPS per asset (critical: bad on one asset tanks everything) |
| **Per-Interval Breakdown** | Which time horizons are you weak at? 5min? 3hr? 24hr? |

### Competitiveness Thresholds

Before deploying to mainnet, your model should:
1. Beat a naive GBM (Geometric Brownian Motion) baseline by >15% on CRPS
2. Show consistent performance across ALL assets (90th percentile penalty is brutal)
3. Demonstrate stability over at least 48 hours of live-testing
4. Not degrade significantly on any single CRPS interval bucket
5. Be below the network median CRPS on ≥70% of assets (check via Synth API)

---

## Key Implementation Details

### Response Format (What Your Miner Must Return)

```python
# For 24-hour challenge:
# [start_timestamp, time_interval, [path1], [path2], ..., [path1000]]
# Each path: 289 floats (24h / 5min + 1 = 289 points)
# Max 8 decimal digits per price

# For 1-hour challenge:
# Each path: 61 floats (1h / 1min + 1 = 61 points)

response = [
    int(start_time.timestamp()),  # Unix timestamp
    300,                           # 5 min = 300 seconds (or 60 for 1-hour)
    [104856.23, 104972.01, ...],  # Path 1: 289 prices
    [104856.23, 104724.54, ...],  # Path 2: 289 prices
    # ... 1000 paths total
]
```

### Pyth Oracle Price IDs
```python
PYTH_PRICE_IDS = {
    "BTC": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "XAU": "0x765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2",
}
```

### Walk-Forward Validation Setup
```
Total data: 90 days of 1-min OHLCV
├── Fold 1: Train [day 0-30] → Purge [day 30-31] → Val [day 31-40]
├── Fold 2: Train [day 0-40] → Purge [day 40-41] → Val [day 41-50]
├── Fold 3: Train [day 0-50] → Purge [day 50-51] → Val [day 51-60]
├── Fold 4: Train [day 0-60] → Purge [day 60-61] → Val [day 61-70]
└── Final:  Train [day 0-70] → Purge [day 70-71] → Test [day 71-90]
```

---

## Common Pitfalls

1. **Data leakage via features** — Computing rolling volatility using future data. Always use `.shift(1)` or causal windows.
2. **Ignoring asset weights** — XAU has 2.26× weight, SPYX has 2.99×. Bad XAU predictions hurt ~2× more than bad ETH.
3. **Overconfident paths** — Too-narrow distribution gets hammered by CRPS. Better to be slightly wide than too narrow.
4. **Ignoring the 1-hour challenge** — It's 50% of emissions. Many miners focus only on 24-hour.
5. **Not testing across all assets** — Missing one asset = 90th percentile penalty = destroyed emissions.
6. **Training on close prices only** — OHLCV gives you vol information. Use it.
7. **Not checking basis point conversion** — CRPS is on bps changes, not raw prices. A bug here means your emulator is wrong.

---

## Reference File Index

| File | Contents |
|------|----------|
| `references/architecture.md` | Full system architecture, directory structure, component diagram |
| `references/data_pipeline.md` | Data sourcing, feature engineering, anti-leakage, temporal splits |
| `references/models.md` | DLinear+Gaussian implementation, custom model interface, search space |
| `references/validator_emulator.md` | Exact CRPS calculation, scoring pipeline, interval definitions |
| `references/leaderboard.md` | Tracking metrics, competitiveness analysis, per-asset/interval breakdown |
| `references/deployment.md` | Model registry, hot-swap, PM2 config, production monitoring |
| `references/synth_api.md` | Live leaderboard, miner validation, competitive benchmarking via Synth API |
