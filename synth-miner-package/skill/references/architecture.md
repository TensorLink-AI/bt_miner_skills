# Architecture Reference

## System Overview

The Synth MLOps miner is a pipeline that automates model development, evaluation, and deployment
for competitive mining on Bittensor's Synth Subnet (SN50).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYNTH MINER MLOPS PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐  │
│  │   DATA    │──▶│   FEATURE    │──▶│   MODEL    │──▶│  VALIDATOR   │  │
│  │ SOURCING  │   │ ENGINEERING  │   │  TRAINING  │   │  EMULATOR    │  │
│  └──────────┘   └──────────────┘   └────────────┘   └──────┬───────┘  │
│       │                                                      │          │
│       │              ┌──────────────┐                        │          │
│       │              │  LEADERBOARD │◀───────────────────────┘          │
│       │              │   TRACKER    │                                   │
│       │              └──────┬───────┘                                   │
│       │                     │                                           │
│       │              ┌──────▼───────┐   ┌──────────────┐               │
│       │              │    MODEL     │──▶│  DEPLOYMENT   │              │
│       │              │  REGISTRY    │   │  (HOT-SWAP)   │              │
│       │              └──────────────┘   └──────────────┘               │
│       │                                        │                        │
│       │              ┌─────────────────────────▼──────┐                │
│       └──────────────│    LIVE PRODUCTION MINER       │                │
│                      │  (PM2 + synth-subnet neuron)   │                │
│                      └────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
synth-miner-mlops/
├── config/
│   ├── assets.yaml           # Asset definitions, weights, Pyth IDs
│   ├── challenges.yaml       # Challenge types (1h, 24h), intervals
│   └── pipeline.yaml         # Training hyperparams, search space
│
├── data/
│   ├── raw/                  # Raw OHLCV from exchanges
│   │   ├── btc_1m.parquet
│   │   ├── eth_1m.parquet
│   │   └── ...
│   ├── processed/            # Feature-engineered data
│   │   ├── btc_features.parquet
│   │   └── ...
│   └── splits/               # Temporal train/val/test splits
│       ├── fold_0/
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   └── metadata.json  # Split timestamps, sizes
│       └── ...
│
├── models/
│   ├── base_model.py         # Abstract base class
│   ├── dlinear_gaussian.py   # Default: DLinear + Gaussian head
│   └── custom/               # User-added models
│
├── training/
│   ├── train.py              # Walk-forward training loop
│   ├── search.py             # Hyperparameter / architecture search
│   └── losses.py             # CRPS loss, NLL loss, etc.
│
├── evaluation/
│   ├── validator_emulator.py # Exact replica of validator scoring
│   ├── crps.py               # CRPS calculation functions
│   ├── leaderboard.py        # Model comparison and tracking
│   └── live_tester.py        # Real-time scoring against live prices
│
├── deployment/
│   ├── model_registry.py     # Version, stage, promote/demote
│   ├── deploy.py             # Hot-swap into production miner
│   └── monitor.py            # Production health checks
│
├── pipeline/
│   ├── data_pipeline.py      # End-to-end data sourcing + features
│   ├── run_search.py         # Full automated search loop
│   └── run_live_test.py      # Live testing before deployment
│
├── miner/
│   ├── forward.py            # The actual miner forward function
│   ├── model_loader.py       # Load current production model
│   └── response_formatter.py # Format predictions for validator
│
├── db/
│   ├── models.py             # SQLAlchemy models for tracking
│   └── init_db.py            # Database initialization
│
└── notebooks/
    ├── exploration.ipynb     # Data exploration
    └── analysis.ipynb        # Model comparison analysis
```

## Component Interactions

### Data Flow
1. `data_pipeline.py` fetches 1-minute OHLCV from Binance/Pyth for all 9 assets
2. Computes causal features (returns in bps, rolling vol, VWAP, etc.)
3. Creates walk-forward splits with purge gaps
4. Saves to `data/splits/` as parquet files

### Training Flow
1. `train.py` loads splits, instantiates model from config
2. Trains with CRPS-aware loss (or NLL for Gaussian)
3. Generates 1000-path predictions on validation set
4. Passes predictions to `validator_emulator.py` for scoring
5. Logs results to `leaderboard.py`

### Search Flow
1. `search.py` defines a search space (model configs)
2. For each config, runs full walk-forward training
3. Evaluates on emulator across all assets and challenge types
4. Ranks by simulated emission share
5. Promotes top candidates to `model_registry.py`

### Deployment Flow
1. `live_tester.py` runs candidate models against live Pyth prices for 48h
2. If model beats current production model on rolling CRPS, flag for promotion
3. `deploy.py` hot-swaps the model in the running miner process
4. `monitor.py` tracks production CRPS and alerts on degradation

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | PyTorch |
| Data | Pandas, Polars, Parquet |
| Database | SQLite (local tracking) or PostgreSQL |
| Process Manager | PM2 (production miner) |
| Price Feed | Pyth Oracle, Binance API |
| Bittensor | bittensor SDK, btcli |
| GPU Backend | Basilica (optional, `COMPUTE_BACKEND="basilica"`) — **cheap GPUs only: A4000, V100, L40** |
| Monitoring | Logging + optional Grafana/Prometheus |

## Network Configuration

| Network | NetUID | Use |
|---------|--------|-----|
| Mainnet (finney) | 50 | Production mining |
| Testnet | 247 | Testing and development |

Validators send requests every 30 minutes, alternating across assets. Your miner must respond
within 1 minute (before `start_time`). Failure to respond = 90th percentile penalty.
