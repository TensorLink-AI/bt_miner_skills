# bt_miner_skills

Autonomous Bittensor subnet miner development — orchestrate model search, deployment, and monitoring across any subnet.

## Architecture

```
bt_miner_skills/
├── orchestrator/           # Generic lifecycle engine
│   ├── orchestrator.py     # Main: setup → search → deploy → monitor
│   ├── config.py           # SubnetConfig loader (from subnet.yaml)
│   ├── lifecycle.py        # Deploy + monitor (universal across subnets)
│   └── strategies/         # Pluggable search strategies
│       ├── base.py         # Strategy interface
│       ├── evoloop.py      # Evolutionary model search via evoloop
│       ├── config_search.py# Grid/random config search
│       ├── model_selection.py # Benchmark existing models
│       └── custom.py       # User-provided script (escape hatch)
│
├── subnets/                # One directory per subnet
│   ├── synth/              # SN50 — probabilistic price forecasting
│   │   ├── subnet.yaml     # Config: strategy, convergence, competitiveness, monitoring
│   │   ├── setup.py        # Validate prerequisites
│   │   ├── deploy.py       # Promote model to production
│   │   ├── monitor.py      # Check live performance via Synth API
│   │   └── evoloop_task/   # Task files for evoloop
│   │       ├── task.yaml   # Objectives, strategies, constraints
│   │       ├── train.py    # Mutable target (evoloop evolves this)
│   │       └── prepare.py  # Frozen eval harness (CRPS scoring)
│   │
│   └── template/           # Skeleton for adding new subnets
│
├── synth-miner-package/    # Skill docs: full Synth MLOps reference
│   ├── AGENT_PROMPT.md     # Autonomous agent task prompt
│   └── skill/              # SKILL.md + reference files
│
└── bittensor-rational-miner.md  # Mining economics decision framework
```

## How It Works

The **orchestrator** knows the lifecycle but not the technique:

1. **Setup** — run `subnets/<name>/setup.py` to validate prerequisites
2. **Search** — invoke a strategy (evoloop, config search, model selection, or custom)
3. **Deploy** — run `subnets/<name>/deploy.py` to put the result into production
4. **Monitor** — run `subnets/<name>/monitor.py` periodically, re-search if degrading

Each subnet provides its own setup/deploy/monitor scripts. The orchestrator sequences them.

## Quick Start

```bash
pip install pyyaml

# List available subnets
python -m orchestrator.orchestrator --list

# Run the full lifecycle for Synth
EVOLOOP_DIR=/path/to/evoloop python -m orchestrator.orchestrator --subnet synth

# Run a specific phase
python -m orchestrator.orchestrator --subnet synth --phase setup
python -m orchestrator.orchestrator --subnet synth --phase search
python -m orchestrator.orchestrator --subnet synth --phase deploy
python -m orchestrator.orchestrator --subnet synth --phase monitor

# Dry run (show what would execute)
python -m orchestrator.orchestrator --subnet synth --dry-run
```

## Search Strategies

| Strategy | Use When | Example |
|----------|----------|---------|
| **evoloop** | Train/evolve ML models | Synth price forecasting, pretraining |
| **config_search** | Find best params for existing tool | Grid/random search over config space |
| **model_selection** | Pick best existing model | LLM serving, image generation |
| **custom** | Nothing else fits | Infrastructure tasks, data pipelines |

Set `strategy.type` in `subnet.yaml` to choose. Each strategy has its own config options documented in `subnets/template/subnet.yaml`.

## Adding a New Subnet

```bash
cp -r subnets/template subnets/my-subnet
# Edit subnets/my-subnet/subnet.yaml
# Implement setup.py, deploy.py, monitor.py
# If using evoloop: create evoloop_task/ with task.yaml, train.py, prepare.py
python -m orchestrator.orchestrator --subnet my-subnet
```

## Dependencies

- Python 3.10+
- `pyyaml` (for config loading)
- [evoloop](https://github.com/TensorLink-AI/evoloop) (for evoloop strategy — set `EVOLOOP_DIR`)
- Subnet-specific deps installed by each subnet's setup.py
