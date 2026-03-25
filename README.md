# Synth Subnet Miner — Autonomous Agent Prompt

## Mission

Build a competitive miner for the Synth Subnet (Bittensor SN50) that generates probabilistic
price path forecasts scored by CRPS. Start from zero, end with a backtested model beating
baseline by >15% on CRPS, validated by a local emulator that replicates exact validator scoring,
and ready to deploy.

---

## Context

Read the skill file at `skill/SKILL.md` first. It contains:
- What the Synth subnet is and how it scores miners
- The two competitions (24-hour and 1-hour HFT) and their specs
- Asset weights, CRPS scoring intervals, and the score transformation pipeline
- Anti-leakage rules for data handling

Then read the reference files in order as you need them (don't load all at once):
- `skill/references/architecture.md` — system design and directory structure
- `skill/references/data_pipeline.md` — data sourcing, features, temporal splits
- `skill/references/models.md` — DLinear+Gaussian implementation, custom model interface
- `skill/references/validator_emulator.md` — exact CRPS scoring logic
- `skill/references/leaderboard.md` — tracking model competitiveness
- `skill/references/deployment.md` — model registry and production deployment
- `skill/references/basilica_gpu_training.md` — optional: Basilica GPU training

---

## External Skills

Some phases benefit from specialist skills maintained outside this package. When a phase
references an external skill, install it before starting that phase.

### How to Install an External Skill

**If you have shell access (Claude Code, Ralph, Cline, etc.):**
```bash
mkdir -p external_skills && cd external_skills
curl -L "<url>" -o <skill-name>.zip && unzip <skill-name>.zip -d <skill-name>/
# or: git clone <repo-url> <skill-name>/
```

**If you're in Claude.ai (no shell for arbitrary downloads):**
- Use `web_fetch` to read the skill's page directly from its URL
- Extract the key instructions into your working context
- You don't need to "install" anything — just read the content and follow it
- If the URL is blocked, check if `skill/references/` already covers the topic

### Available External Skills

| Skill | Use In Phase | Source | Purpose |
|-------|-------------|--------|---------|
| **basilica-cli-helper** | Phase 5, 6 (if `COMPUTE_BACKEND=basilica`) | [mcpmarket.com/tools/skills/basilica-cli-helper](https://mcpmarket.com/tools/skills/basilica-cli-helper) | Basilica CLI commands (`basilica up`, `exec`, `cp`), SSH config, troubleshooting. Complements `skill/references/basilica_gpu_training.md` which covers SDK patterns. |
| **bittensor-subnet-design** | Phase 8 (if customizing incentive mechanism) | Pre-installed at `/mnt/skills/user/bittensor-subnet-design/` in Claude | General Bittensor subnet design patterns, validator rules, CLI reference. |

### Adding More External Skills

To plug in another skill, add a row to the table above and reference it from the relevant
phase. The agent fetches it (or reads via web_fetch) and follows its SKILL.md.

---

## Phases

Work through these phases sequentially. Each phase has a **gate** — a concrete check you must
pass before moving on. If a gate fails, fix the issue before proceeding.

### Phase 1: Scaffold
**Goal**: Set up the project structure and install dependencies.

1. Create the directory structure from `references/architecture.md`
2. Set up a Python virtual environment with PyTorch, pandas, numpy, requests
3. Create `config.py` with all constants from the SKILL.md:
   - All 9 assets and their weights
   - Both challenge types (24h and 1h) with their interval definitions
   - CRPS scoring parameters (softmax beta, rolling window, etc.)
   - Pyth Oracle price IDs
   - Binance symbol mappings

**Gate**: `python -c "from config import ASSETS_24H, ASSET_WEIGHTS, INTERVALS_24H; print('OK')"` succeeds.

### Phase 2: Data Pipeline
**Goal**: Fetch historical data and create walk-forward splits with anti-leakage guarantees.

1. Read `references/data_pipeline.md` thoroughly
2. Implement `data_pipeline.py`:
   - Fetch 90 days of 1-minute OHLCV from Binance for BTC, ETH, SOL
   - For XAU, use PAXGUSDT as proxy
   - For equity assets (SPYX, NVDAX, TSLAX, AAPLX, GOOGLX), use yfinance for SPY, NVDA, TSLA, AAPL, GOOGL
   - Handle market hours gaps for equities
3. Implement causal feature computation (every feature uses `.shift(1)` — NO EXCEPTIONS):
   - Returns in basis points at multiple horizons
   - Realized volatility at multiple windows
   - VWAP deviation
   - High-low range
   - Volume ratio
   - Cyclical time encoding (hour, day of week)
4. Create walk-forward folds with purge gaps:
   - Minimum 5 folds
   - Purge gap = forecast horizon + 2h buffer
   - Final held-out test set = last 20 days
5. Save processed data as parquet files

**Gate**: Run these checks and all must pass:
- No NaN values in any feature columns
- All timestamps are UTC and monotonically increasing
- Verify anti-leakage: for each fold, `train.timestamp.max() + purge_gap <= val.timestamp.min()`
- At least 50,000 rows per asset in training data
- Print fold summary: train size, val size, date ranges

### Phase 3: Model — DLinear + Gaussian Head
**Goal**: Implement the default model and verify it produces valid outputs.

1. Read `references/models.md`
2. Implement `models/base_model.py` — the abstract interface
3. Implement `models/dlinear_gaussian.py` — the full DLinear + Gaussian model:
   - MovingAvgBlock with causal padding
   - Trend/seasonal decomposition
   - Linear projections to forecast horizon
   - μ head and σ head (σ uses sigmoid bounded between min_sigma and max_sigma)
   - `sample_paths()` method that generates N=1000 price paths
4. Implement both loss functions:
   - `CRPSLoss` — closed-form Gaussian CRPS (use this as primary training loss)
   - `GaussianNLLLoss` — negative log-likelihood (use as secondary/comparison)
5. Write a unit test that:
   - Creates a model with default params
   - Passes random input through it
   - Generates 1000 paths
   - Verifies output shapes

**Gate**: 
- Model forward pass produces `(mu, sigma)` with correct shapes `[batch, horizon]`
- `sigma` is always positive (min_sigma ≤ σ ≤ max_sigma)
- `sample_paths()` returns `[batch, 1000, horizon+1]` with all positive values
- First price in every path equals `current_price`
- No NaN or Inf in any output

### Phase 4: Validator Emulator
**Goal**: Implement exact CRPS scoring that matches the real validator.

1. Read `references/validator_emulator.md` carefully
2. Implement `evaluation/crps.py`:
   - `crps_ensemble()` for single observation
   - `crps_ensemble_vectorized()` for multiple timesteps
3. Implement `evaluation/validator_emulator.py`:
   - `paths_to_bps_changes()` — convert price paths to basis point changes
   - `actual_to_bps_changes()` — convert actual prices to basis point changes
   - `score_prediction()` — full scoring across all interval buckets
   - `transform_scores()` — percentile capping, best subtraction
   - `rolling_average()` — weighted 10-day rolling average
   - `calculate_emission_share()` — softmax with β = -0.0475
4. Implement baselines for comparison:
   - GBM (Geometric Brownian Motion) baseline
   - Historical simulation baseline
5. Write validation tests:
   - Construct a known scenario where CRPS can be hand-calculated
   - Verify basis point conversion is correct
   - Verify interval bucketing matches expected counts (288×5min, 48×30min, 8×3hr)

**Gate**:
- CRPS of a perfect prediction (all paths = actual) is approximately 0
- CRPS increases as predictions diverge from actual
- Narrower ensemble has lower CRPS when centered on truth, higher when off-center
- Interval counts match exactly: 288 + 48 + 8 + 1 = 345 CRPS values for 24h
- Basis point conversion test: price change from 100000 to 102000 = 200 bps ✓
- **Synth API cross-check**: Read `skill/references/synth_api.md`, then fetch
  `GET https://api.synthdata.co/validation/scores/latest?asset=BTC&time_length=86400`
  to see real CRPS values from the live network. Verify your emulator's output is in the
  same ballpark (same order of magnitude). If live scores are ~150 and yours are ~15000,
  your basis point conversion is wrong.

### Phase 5: Training Loop
**Goal**: Train the DLinear model across all assets using walk-forward validation.

**Compute option**: Check `config.py` for `COMPUTE_BACKEND`. If set to `"basilica"`:
- Read `skill/references/basilica_gpu_training.md` for SDK deployment patterns
- If you need lower-level CLI control (SSH, `basilica exec`, troubleshooting), install
  the **basilica-cli-helper** external skill (see External Skills section above)
- Deploy training to Basilica GPU instances instead of running locally
If `"local"` (default), run everything on the local machine.

1. Implement `training/train.py`:
   - DataLoader creation from walk-forward folds
   - Training loop with CRPS loss
   - AdamW optimizer with cosine annealing
   - Gradient clipping (max_norm=1.0)
   - Early stopping (patience=10)
   - Save best model per fold
2. If using Basilica:
   - Upload processed data to a Basilica Volume
   - Deploy a GPU training job using `basilica-sdk`
   - Monitor training via `deployment.logs()`
   - Download results from the Volume when complete
3. If local: train on each fold for each asset directly
4. After training, generate 1000-path predictions on each validation fold
5. Score using the validator emulator
6. Also score the GBM and historical simulation baselines on the same data

**Gate**:
- Training loss decreases over epochs (not diverging)
- Validation CRPS is finite and reasonable (not 0, not infinity)
- Model predictions for all 9 assets produce valid paths
- DLinear CRPS < GBM baseline CRPS (if not, debug before proceeding)
- Print per-asset CRPS comparison table: DLinear vs GBM vs HistSim

### Phase 6: Leaderboard & Search
**Goal**: Track results and run automated model search to find competitive configs.

1. Read `skill/references/leaderboard.md`
2. Implement `evaluation/leaderboard.py` with SQLite tracking
3. Log all training results to the leaderboard
4. Implement `training/search.py`:
   - Define search space (lookback, kernel_size, min/max_sigma, lr, batch_size)
   - Random search over 20-30 configurations
   - For each config, run full walk-forward training + emulator scoring
   - Log everything to leaderboard
   - If `COMPUTE_BACKEND == "basilica"`: deploy the search as a single GPU job that
     iterates through all configs. Use a Basilica Volume for checkpoints so partial
     results survive if the job times out. Set `ttl_seconds=14400` (4 hours).
5. After search completes, analyze results:
   - Which configs beat baseline by >15%?
   - Which assets are weakest across all configs?
   - Is 5-min, 30-min, or 3-hr interval the hardest?
   - What's the simulated emission share of the best model?

**Gate**:
- At least 20 model configs evaluated
- Best model beats GBM by >15% on rolling average CRPS
- Leaderboard comparison table generated showing all models ranked
- Per-asset heatmap shows no single asset contributing >30% of total CRPS
- If no model beats baseline by 15%, expand search space and retry
- **Synth API benchmark**: Fetch live scores from `GET /validation/scores/latest` for each
  asset. Compare your best model's CRPS against the network median. Print a side-by-side
  table: your CRPS vs live best / median / p25. You should be below the median on most
  assets to have a shot at positive emissions.

### Phase 7: Live Testing
**Goal**: Run the best model against live price data to validate real-world performance.

1. Implement `evaluation/live_tester.py`:
   - Connect to Pyth Oracle for live prices
   - Every 30 minutes, generate predictions for each asset
   - Wait for the forecast horizon to elapse
   - Score predictions using the emulator
   - Log to leaderboard with `is_live=True`
2. Run for minimum 4 hours (8 prompt cycles) — ideally 48 hours
3. Compare live CRPS to backtest CRPS
4. If live CRPS is >20% worse than backtest, investigate:
   - Data distribution shift?
   - Feature computation bug in live mode?
   - Pyth vs Binance price discrepancy?

**Gate**:
- Live predictions are valid format for all tested assets
- Live CRPS within 20% of backtest CRPS
- No errors or timeouts in any prediction cycle
- Response time < 30 seconds per prediction

### Phase 8: Deployment Preparation
**Goal**: Set up model registry and deployment infrastructure.

1. Read `skill/references/deployment.md`
2. Implement `deployment/model_registry.py`:
   - Register best model as candidate
   - Promote through stages: candidate → live_testing → production
   - Symlink-based current model pointer
3. Implement `miner/forward.py`:
   - Load production model from registry
   - Hot-swap detection (check symlink on each request)
   - Response formatting matching validator expectations
4. Implement `miner/response_formatter.py`:
   - Correct format: `[timestamp, interval, path1, path2, ..., path1000]`
   - Round to 8 decimal places
   - Validate path lengths (289 for 24h, 61 for 1h)
5. Create PM2 config for production miner

**Gate**:
- Model registry has at least one model in "production" stage
- Response formatter produces correctly formatted output
- Path lengths are exactly correct (289 or 61)
- All prices rounded to ≤8 decimal places
- First price in each path matches current_price
- **Post-deployment check** (if miner UID is known): Use the Synth API to verify:
  - `GET /validation/miner?uid=<YOUR_UID>` returns `"validated": true`
  - `GET /v2/leaderboard/latest` shows your UID with non-zero rewards
  - `GET /validation/scores/latest?miner_uid=<YOUR_UID>` shows CRPS scores being recorded
  - If validation fails, the `reason` field tells you exactly what's wrong with your format

---

## Critical Rules (Never Violate These)

1. **No data leakage** — Every feature computation must use `.shift(1)` or equivalent. Every
   split must be temporal with purge gaps. If you're unsure whether something leaks, it leaks.
   Fix it.

2. **All 9 assets matter** — Missing predictions for ANY asset = 90th percentile penalty.
   Don't skip equity assets just because they're harder to source.

3. **Basis points, not prices** — CRPS is calculated on price changes in basis points. If you
   compute CRPS on raw prices, your emulator is wrong and everything downstream is invalid.

4. **1000 paths, not 100** — The network upgraded to 1000 simulations. Using 100 will be scored
   as invalid.

5. **Both challenges** — 24-hour AND 1-hour HFT each get 50% of emissions. Build for both.

6. **Response format is exact** — Wrong timestamp format, wrong path length, >8 decimal places,
   or late response = score of 0 (assigned 90th percentile).

---

## When Stuck

- If data fetching fails: Check Binance API rate limits (1200 req/min). Add sleep(0.1) between requests.
- If CRPS is NaN: Check for zero or negative sigma values. Check for price = 0 in bps calculation.
- If model doesn't improve over baseline: The Gaussian head might be too thin-tailed. Try increasing max_sigma or switching to a Student-t distribution.
- If live CRPS ≫ backtest CRPS: Most likely a feature leakage bug that only manifests in live mode. Check that live feature computation is identical to training.
- If equity asset data has gaps: That's normal (market hours). Fill gaps with last known price or interpolate, but flag these periods in your scoring.

---

## Definition of Done

The task is complete when:
1. ✅ All 8 phase gates pass
2. ✅ A model is registered in the model registry at "production" stage
3. ✅ The model beats GBM baseline by >15% on CRPS across all assets
4. ✅ Live testing shows CRPS within 20% of backtest
5. ✅ Leaderboard shows per-asset and per-interval breakdown with no catastrophic weaknesses
6. ✅ PM2 config and miner forward function are ready for mainnet deployment
7. ✅ Synth API comparison shows model CRPS is below network median on ≥70% of assets
8. ✅ A summary report is generated showing: best model config, CRPS comparison vs baselines,
   CRPS comparison vs live network (from Synth API), per-asset breakdown, estimated
   emission share, and known weaknesses to improve next
