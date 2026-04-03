# bt_miner_skills

Autonomous Bittensor subnet miner development — an LLM agent that manages model search, deployment, and monitoring across any subnet.

## Architecture

```
bt_miner_skills/
├── orchestrator/              # LLM agent that manages the lifecycle
│   ├── orchestrator.py        # Agent loop: gather status → ask LLM → execute tool
│   ├── snapshot.py            # Builds status snapshot + system prompt for the agent
│   ├── tools.py               # Agent tools: start_search, deploy, monitor, etc.
│   ├── state.py               # Persistent state (survives restarts)
│   ├── config.py              # SubnetConfig loader (from subnet.yaml)
│   ├── lifecycle.py           # Deploy + monitor subprocess helpers
│   └── strategies/            # Pluggable search strategies
│       ├── base.py            # Strategy interface
│       ├── evoloop.py         # Evolutionary model search via evoloop
│       ├── config_search.py   # Grid/random config search
│       ├── model_selection.py # Benchmark existing models
│       └── custom.py          # User-provided script (escape hatch)
│
├── subnets/                   # One directory per subnet
│   ├── synth/                 # SN50 — probabilistic price forecasting
│   │   ├── subnet.yaml        # Goals, strategy config, competitiveness gates
│   │   ├── setup.py           # Validate prerequisites
│   │   ├── deploy.py          # Promote model to production
│   │   ├── monitor.py         # Check live performance via Synth API
│   │   └── evoloop_task/      # Task files for evoloop
│   │       ├── task.yaml      # Objectives, strategies, constraints
│   │       ├── train.py       # Mutable target (evoloop evolves this)
│   │       └── prepare.py     # Frozen eval harness (CRPS scoring)
│   │
│   └── template/              # Skeleton for adding new subnets
│
├── synth-miner-package/       # Skill docs: full Synth MLOps reference
│   ├── AGENT_PROMPT.md        # Autonomous agent task prompt
│   └── skill/                 # SKILL.md + reference files
│
└── bittensor-rational-miner.md  # Mining economics decision framework
```

## How It Works

The orchestrator is an **LLM agent**, not a rigid pipeline. Every tick it:

1. **Gathers status** — search progress, live metrics, errors, decision history
2. **Asks the LLM** — "here's the situation, what should we do?"
3. **Executes the decision** — one tool call per tick
4. **Records the outcome** — persists state and decision log

The agent has 7 tools: `run_setup`, `start_search`, `stop_search`, `get_search_status`, `deploy`, `check_live_performance`, `wait`. It decides when to use each based on the actual situation, not hardcoded rules.

When no LLM is configured, a **heuristic fallback** follows the obvious path: setup → search → deploy → monitor → re-search if degrading.

## Quick Start

```bash
pip install pyyaml openai

# List available subnets
python -m orchestrator.orchestrator --list

# Run one tick (test mode — agent makes one decision and exits)
python -m orchestrator.orchestrator --subnet synth --once

# Run the agent loop (ticks every 5 minutes)
python -m orchestrator.orchestrator --subnet synth

# Custom tick interval (every 2 minutes)
python -m orchestrator.orchestrator --subnet synth --tick-interval 120

# Check current state
python -m orchestrator.orchestrator --status

# Run with a specific LLM
ORCHESTRATOR_API_KEY=sk-... ORCHESTRATOR_MODEL=gpt-4.1-mini \
  python -m orchestrator.orchestrator --subnet synth
```

## Agent Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `ORCHESTRATOR_API_KEY` | `$OPENAI_API_KEY` | API key for the agent's LLM |
| `ORCHESTRATOR_MODEL` | `gpt-4.1-mini` | Model for agent decisions (cheap is fine) |
| `ORCHESTRATOR_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `EVOLOOP_DIR` | — | Path to evoloop repo (if not pip-installed) |

## Search Strategies

| Strategy | Use When | Example |
|----------|----------|---------|
| **evoloop** | Train/evolve ML models | Synth price forecasting, pretraining |
| **config_search** | Find best params for existing tool | Grid/random search over config space |
| **model_selection** | Pick best existing model | LLM serving, image generation |
| **custom** | Nothing else fits | Infrastructure tasks, data pipelines |

## Adding a New Subnet

```bash
cp -r subnets/template subnets/my-subnet
# Edit subnets/my-subnet/subnet.yaml
# Implement setup.py, deploy.py, monitor.py
# If using evoloop: create evoloop_task/ with task.yaml, train.py, prepare.py
python -m orchestrator.orchestrator --subnet my-subnet --once
```

## Dependencies

- Python 3.10+
- `pyyaml` — config loading
- `openai` — agent LLM calls (optional; heuristic fallback without it)
- [evoloop](https://github.com/TensorLink-AI/evoloop) — for evoloop strategy
