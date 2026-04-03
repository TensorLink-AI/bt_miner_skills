# Synth Subnet Miner — MLOps Package

Everything needed to build a competitive miner for the Synth Subnet (Bittensor SN50).

## Package Layout

```
synth-miner-package/
├── README.md              ← You are here
├── AGENT_PROMPT.md        ← Task prompt for autonomous agents
├── subnet.json            ← Subnet metadata (netuid, name)
└── skill/
    ├── SKILL.md           ← Entry point: subnet overview, challenge specs, scoring rules
    └── references/
        ├── architecture.md       ← System design, directory structure, tech stack
        ├── data_pipeline.md      ← Data sourcing, causal features, anti-leakage, walk-forward splits
        ├── models.md             ← DLinear+Gaussian PyTorch implementation, custom model interface
        ├── validator_emulator.md ← Exact CRPS scoring logic replicating the real validator
        ├── leaderboard.md        ← Model tracking, competitiveness analysis, per-asset breakdown
        ├── deployment.md         ← Model registry, hot-swap, PM2 config, production monitoring
        └── synth_api.md          ← Live leaderboard, miner validation, competitive benchmarking
```

## How to Use

### With the Orchestrator + Evoloop (recommended)

The orchestrator provides a generic lifecycle engine that uses [evoloop](https://github.com/TensorLink-AI/evoloop)
for evolutionary model search. The Synth subnet package is configured in `subnets/synth/`.

```bash
# Run the full lifecycle: setup → search (evoloop) → deploy → monitor
python -m orchestrator.orchestrator --subnet synth

# Run a specific phase
python -m orchestrator.orchestrator --subnet synth --phase search
python -m orchestrator.orchestrator --subnet synth --phase monitor

# List available subnets
python -m orchestrator.orchestrator --list
```

The orchestrator:
1. **Setup** — validates data sources, Basilica credentials, evoloop task files
2. **Search** — runs evoloop to evolve `train.py` against CRPS objectives
3. **Deploy** — promotes the best model to production, starts the miner
4. **Monitor** — checks Synth API for live performance, re-evolves if degrading

See `subnets/synth/subnet.yaml` for configuration (convergence thresholds,
competitiveness gates, monitoring intervals).

### With Claude (interactive)

1. Install `skill/` as a Claude skill (drop into `/mnt/skills/user/synth-miner-mlops/`)
2. Ask Claude to help you build a Synth miner — the skill triggers automatically
3. Use the `AGENT_PROMPT.md` as a roadmap for what to work on in order

### Manual (just reading)

Start with `skill/SKILL.md` for the overview, then read the reference files as needed.
The reference files contain complete Python implementations you can copy and adapt.

## What Gets Built

By the end of the pipeline, you'll have:

- Historical data pipeline for all 9 assets with anti-leakage walk-forward splits
- Probabilistic forecasting model evolved by evoloop for optimal CRPS
- Local validator emulator that replicates exact CRPS scoring
- Model registry with hot-swap deployment to production miner
- PM2-managed miner process on Bittensor mainnet (SN50)
- Continuous monitoring with automatic re-evolution on performance degradation

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Mainnet NetUID | 50 |
| Testnet NetUID | 247 |
| Simulated Paths | 1000 |
| 24h Challenge | 9 assets, 5-min intervals, 289 price points per path |
| 1h HFT Challenge | 4 assets, 1-min intervals, 61 price points per path |
| Emissions Split | 50% / 50% (24h / 1h) |
| Scoring | CRPS on basis point changes |
| Softmax β | -0.0475 |
| Rolling Window | 10 days |

## External Skills

| Skill | What It Does | When Needed | Source |
|-------|-------------|-------------|--------|
| **basilica-cli-helper** | Basilica CLI for GPU instance management | Training on Basilica GPUs | [mcpmarket.com](https://mcpmarket.com/tools/skills/basilica-cli-helper) |
| **basilica-sdk** | Basilica Python SDK for deploying GPU workloads | Training on Basilica GPUs | Pre-installed in Claude at `/mnt/skills/user/basilica-sdk/` |
| **bittensor-subnet-design** | General Bittensor subnet patterns and CLI | Understanding SN50 architecture | Pre-installed in Claude at `/mnt/skills/user/bittensor-subnet-design/` |
