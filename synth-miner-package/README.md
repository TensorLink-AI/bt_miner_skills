# Synth Subnet Miner — MLOps Package

Everything needed to build a competitive miner for the Synth Subnet (Bittensor SN50).

## Package Layout

```
synth-miner-package/
├── README.md              ← You are here
├── AGENT_PROMPT.md        ← Give this to your loop agent (Ralph, Claude Code, etc.)
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

### With a loop agent (Ralph, Claude Code, Cline, etc.)

1. Point the agent at this directory as its workspace
2. Give it the contents of `AGENT_PROMPT.md` as the task prompt
3. Make sure the agent can read files from `skill/` during execution
4. Let it run through the 8 phases autonomously

The agent prompt has phase gates — concrete pass/fail checks at each stage — so the agent
can't skip ahead past broken steps.

### With Claude (interactive)

1. Install `skill/` as a Claude skill (drop into `/mnt/skills/user/synth-miner-mlops/`)
2. Ask Claude to help you build a Synth miner — the skill triggers automatically
3. Use the `AGENT_PROMPT.md` phases as a roadmap for what to work on in order

### Manual (just reading)

Start with `skill/SKILL.md` for the overview, then read the reference files as needed.
The reference files contain complete Python implementations you can copy and adapt.

## What Gets Built

By the end of the agent prompt's 8 phases, you'll have:

- Historical data pipeline for all 9 assets with anti-leakage walk-forward splits
- DLinear + Gaussian probabilistic head model (baseline, easily swappable)
- Local validator emulator that replicates exact CRPS scoring
- Automated model search (hyperparameter sweep + evaluation)
- Leaderboard tracking per-model, per-asset, per-interval CRPS
- Live testing against Pyth Oracle price feeds
- Model registry with hot-swap deployment to production miner
- PM2-managed miner process ready for Bittensor mainnet (SN50)

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

This package is self-contained, but some tasks benefit from specialist skills maintained
elsewhere. The agent prompt tells the agent when and how to fetch these.

| Skill | What It Does | When Needed | Source |
|-------|-------------|-------------|--------|
| **basilica-cli-helper** | Basilica CLI for GPU instance management | Training on Basilica GPUs | [mcpmarket.com](https://mcpmarket.com/tools/skills/basilica-cli-helper) |
| **basilica-sdk** | Basilica Python SDK for deploying GPU workloads | Training on Basilica GPUs | Pre-installed in Claude at `/mnt/skills/user/basilica-sdk/` |
| **bittensor-subnet-design** | General Bittensor subnet patterns and CLI | Understanding SN50 architecture | Pre-installed in Claude at `/mnt/skills/user/bittensor-subnet-design/` |

### Adding Your Own

To plug in another skill, add a row to the table above in this README, add it to the
"Available External Skills" table in `AGENT_PROMPT.md`, and reference it from the relevant phase.
