# Synth Subnet Miner — Autonomous Agent Prompt

## Mission

Build and deploy a competitive miner for the Synth Subnet (Bittensor SN50) that generates
probabilistic price path forecasts scored by CRPS. You start from zero. You're done when
your miner is earning non-zero emissions on mainnet.

---

## Context

Read `skill/SKILL.md` first — it defines the subnet, scoring mechanics, asset weights,
CRPS intervals, and the score transformation pipeline.

Reference files in `skill/references/` contain implementation-level detail. Pull them in
as needed — don't front-load everything:
- `architecture.md` — system design and suggested directory structure
- `data_pipeline.md` — data sourcing, features, temporal splits
- `models.md` — DLinear+Gaussian reference implementation, custom model interface
- `validator_emulator.md` — exact CRPS scoring logic
- `leaderboard.md` — tracking model competitiveness
- `deployment.md` — model registry and production deployment
- `synth_api.md` — live leaderboard and miner validation via Synth API

---

## External Skills

Some work benefits from specialist skills maintained outside this package.

### How to Install an External Skill

**If you have shell access (Claude Code, Ralph, Cline, etc.):**
```bash
mkdir -p external_skills && cd external_skills
curl -L "<url>" -o <skill-name>.zip && unzip <skill-name>.zip -d <skill-name>/
```

**If you're in Claude.ai (no shell for arbitrary downloads):**
- Use `web_fetch` to read the skill's page directly from its URL
- Extract the key instructions into your working context
- For Basilica: the `basilica-sdk` skill is pre-installed at `/mnt/skills/user/basilica-sdk/`

### Available External Skills

| Skill | Purpose | Source |
|-------|---------|--------|
| **basilica-cli-helper** | Basilica CLI commands, SSH config, troubleshooting | [mcpmarket.com/tools/skills/basilica-cli-helper](https://mcpmarket.com/tools/skills/basilica-cli-helper) |
| **basilica-sdk** | Basilica Python SDK for deploying GPU workloads | Pre-installed at `/mnt/skills/user/basilica-sdk/` |
| **bittensor-subnet-design** | Bittensor subnet design patterns, validator rules, CLI reference | Pre-installed at `/mnt/skills/user/bittensor-subnet-design/` |

If using Basilica for GPU compute, only use cheap GPUs (A4000, V100, L40). Do NOT use
A100, H100, A6000, or other expensive instances.

---

## Principles

These guide every decision you make. They're not suggestions.

### 1. Verify Against Reality

Never trust your own scoring. Before making any deployment decision, cross-check your
emulator output against live network scores from the Synth API. If your CRPS values are
a different order of magnitude from the network, your emulator is wrong — fix it before
training more models.

Concretely: fetch `GET /validation/scores/latest?asset=BTC&time_length=86400` and compare.
If live scores are ~150 and yours are ~15,000, your basis point conversion is broken.

### 2. Parameters Change

Before relying on hardcoded asset weights, interval definitions, or scoring parameters
(especially the softmax β), verify them against the current validator source at
https://github.com/mode-network/synth-subnet. The skill's reference values may be outdated.

### 3. Anti-Leakage Is Non-Negotiable

Every feature must be causal (`.shift(1)` or equivalent). Every split must be temporal
with purge gaps. Walk-forward validation only — no random splits, no shuffling. If you're
unsure whether something leaks, it leaks. Fix it.

### 4. The Baseline Is a Floor, Not a Ceiling

DLinear + Gaussian is a minimal viable model — fast to implement, good enough to validate
the pipeline end-to-end. Read `skill/references/models.md` for the reference implementation.
Get it working, verify the full pipeline with it, then move on to better approaches quickly.

**Once you have a working pipeline validated against the Synth API, spend most of your
effort on model improvement, not infrastructure polish.** The DLinear baseline exists so
you can validate your emulator and data pipeline quickly — it is not expected to be
competitive on mainnet.

The real competitive edge comes from:
- **Distribution choice**: Gaussian is provably suboptimal for CRPS at longer intervals
  because crypto returns are fat-tailed. Student-t distributions capture tail behavior
  with a single extra parameter (degrees of freedom ν). Mixture density networks can model
  multimodal outcomes. Either is a significant CRPS improvement over Gaussian at 3hr+ horizons.
- **Feature engineering**: Regime-aware features, cross-asset signals, volatility term
  structure.
- **Uncertainty scaling**: Uncertainty should grow with forecast horizon — a fixed σ across
  all intervals is leaving CRPS on the table.
- **Equity asset handling**: Getting clean data and handling market hours correctly is a
  real differentiator since many miners do this poorly.

---

## Critical Rules (Never Violate These)

0. **No local training** — NEVER run PyTorch model training, hyperparameter search, or any
   GPU-intensive workload on the local machine. The sandbox has limited CPU and no GPU — training
   locally WILL crash the session. Always use `COMPUTE_BACKEND = "basilica"` and offload training
   to Basilica GPU instances. Data pipeline, feature engineering, and lightweight operations
   (scoring, formatting, file I/O) are fine to run locally.

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

## What to Build

The work breaks into three layers: foundations that must be correct before anything else,
the iterative development loop you drive autonomously, and deployment readiness checks.

### Non-Negotiable Foundations

These must be built and validated first because everything downstream depends on their
correctness. Order between them is up to you, but both must be solid before you trust
any model evaluation result.

**Data Pipeline**
- Fetch historical OHLCV data for all 9 assets (see `skill/references/data_pipeline.md`)
- Compute causal features (all shifted, no exceptions)
- Create walk-forward splits with purge gaps (expanding window, temporal only)
- Save processed data in a reusable format

**Equity data — know this going in**: yfinance only provides 1-minute data for ~7-30 days.
For 90-day backtests, you'll need to use daily data and resample, find an alternative
source, or adjust your backtest window for equities. Don't discover this after building
your whole pipeline assuming 90 days of minute data exists.

**Market hours gaps**: Equity assets don't trade 24/7. You need a concrete strategy —
forward-fill during closed hours, mask scoring during gaps, or interpolate. This is a
real source of bugs that will make your emulator scores wrong for equities if unhandled.

**Validator Emulator**
- Implement exact CRPS scoring matching the real validator (see `skill/references/validator_emulator.md`)
- Basis point conversion, interval bucketing, score transformation, rolling averages
- Implement GBM and historical simulation baselines for comparison

**Verify before trusting**: Your emulator's CRPS output must be in the same order of
magnitude as live network scores. Check via Synth API (`/validation/scores/latest`). If
they diverge by 10x+, your emulator is wrong. Fix it before training any models — every
evaluation result from a broken emulator is garbage.

### The Development Loop

Once foundations are solid, you enter the core cycle: **train → evaluate → improve**.
There is no fixed number of iterations or prescribed search space. You drive this loop
autonomously.

**Compute**: All training MUST run on Basilica (`COMPUTE_BACKEND = "basilica"`). The local
sandbox has no GPU and limited CPU — training locally will crash the session. Use the
`basilica-sdk` skill to deploy GPU jobs (A4000, V100, or L40 only). Data pipeline work,
feature engineering, scoring, and evaluation are fine to run locally.

Each iteration:
1. **Train** a model on your walk-forward folds (on Basilica, not locally)
2. **Evaluate** using your emulator — score CRPS across all assets and intervals
3. **Compare** against baselines (GBM, historical sim) and against live network scores
   from the Synth API
4. **Identify weaknesses** — which assets, which intervals, which conditions are worst?
5. **Improve** — change the model, the features, the distribution, the hyperparameters,
   or the data processing based on what you learned

You decide when to:
- Try a new architecture vs. tune the current one
- Expand the hyperparameter search vs. fix a data issue
- Focus on a specific weak asset vs. improve across the board
- Move from Gaussian to a heavier-tailed distribution

**Track everything.** Use a leaderboard (see `skill/references/leaderboard.md`) to log
every experiment with its config and scores. You'll need this to understand what's working.

**Benchmark against the network regularly.** Fetch live scores from the Synth API and
compare your best model's CRPS against the network median. You should be below the median
on most assets to have a shot at positive emissions.

**The Pyth/Binance gap**: Training data comes from Binance, but live scoring uses Pyth
Oracle prices. There can be meaningful differences, especially during volatile periods.
If your live CRPS is significantly worse than backtest CRPS, this is one of the first
things to investigate.

### Deployment Readiness

Before going to mainnet, these checks must all pass. They're strict because a bad
deployment wastes emissions and compute.

**Model quality:**
- Your best model beats GBM baseline on CRPS across the asset-weighted average
- Live CRPS (tested against real price feeds) is in the same ballpark as backtest CRPS
- Synth API comparison shows your CRPS is below the network median on the majority of
  asset/interval pairs

**Output correctness:**
- Predictions produce exactly 1000 paths per asset per interval
- Path lengths are exactly correct (289 for 24h, 61 for 1h)
- All prices rounded to ≤8 decimal places
- First price in each path equals current_price
- Response format matches validator expectations exactly

**Infrastructure:**
- Model registry has a model promoted to production
- Miner forward function loads the production model and formats responses correctly
- Hot-swap support so you can update models without restarting

**Post-deployment verification:**
- `GET /validation/miner?uid=<YOUR_UID>` returns `"validated": true`
- `GET /v2/leaderboard/latest` shows your UID with non-zero rewards
- If validation fails, the `reason` field tells you exactly what's wrong

---

## When Stuck

- **Data fetching fails**: Check Binance API rate limits (1200 req/min). Add sleep(0.1)
  between requests.
- **CRPS is NaN**: Check for zero or negative sigma values. Check for price = 0 in bps
  calculation.
- **Model doesn't improve over baseline**: The Gaussian head is probably too thin-tailed.
  Crypto returns have excess kurtosis — a Student-t distribution with learnable degrees of
  freedom ν (constrained to ν > 2 for finite variance) directly addresses this. Alternatively,
  a mixture of 2-3 Gaussians can capture multimodal outcomes. Either approach should show
  immediate CRPS improvement at 3hr+ intervals where tail behavior matters most.
- **Live CRPS ≫ backtest CRPS**: Three likely causes in order of probability: (1) feature
  leakage bug that only manifests in live mode — check that live feature computation is
  identical to training; (2) Pyth vs Binance price discrepancy — compare the price feeds
  directly during a test window; (3) regime shift — retrain on more recent data.
- **Equity asset data has gaps**: That's normal (market hours). You must handle this
  explicitly. Forward-fill is the simplest approach, but flag these periods so you can
  analyze whether gap handling is hurting your scores. Consider masking equity CRPS during
  market-closed periods in your internal evaluation to get cleaner signal.
- **yfinance doesn't have enough minute data**: Expected. For equities, either use daily
  OHLCV and resample to your needed frequency, use an alternative data source (Alpha Vantage,
  Polygon.io), or shorten the backtest window for equity assets specifically.

---

## Definition of Done

You're done when your miner is deployed to mainnet, validated by the Synth API
(`/validation/miner?uid=<YOUR_UID>` returns `validated: true`), and earning non-zero
emissions visible on the leaderboard. Everything before that is iteration.

Intermediate signals that you're on track (check these, but they don't mean "done"):
- Your best model beats GBM baseline by a meaningful margin on CRPS
- Synth API comparison shows you below the network median on most asset/interval pairs
- Live testing shows no catastrophic divergence from backtest performance
- No single asset contributes a disproportionate share of your total CRPS error

Generate a summary report when deploying: best model config, CRPS vs baselines, CRPS vs
live network (from Synth API), per-asset breakdown, estimated emission share, and known
weaknesses to improve next.
