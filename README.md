# bt_miner_skills

Agent-driven Bittensor miner builder. AI agents ingest subnet configurations, build competitive miners, deploy them, and iterate to improve.

## How It Works

1. **Define a subnet config** in `subnet_configs/` describing what the subnet expects
2. **Run the agent loop** to research, build, test, deploy, and monitor a miner
3. **Iterate** - the agent analyzes performance and improves the miner each cycle

## Quick Start

```bash
# Install
pip install -e .

# See available skills
python -m bt_miner_skills.cli list-skills

# Scaffold a miner from a subnet config
python -m bt_miner_skills.cli scaffold subnet_configs/example_sn1.yaml -o workspace/sn1/miner

# Check loop status
python -m bt_miner_skills.cli loop-status 1
```

## For Agents

See [CLAUDE.md](CLAUDE.md) for agent-specific instructions on how to use this repo to build miners.

## Structure

```
bt_miner_skills/
├── config/          # Subnet config schema and loaders
├── agent/           # Agent loop (research → build → test → deploy → monitor → improve)
├── chain/           # Bittensor chain data fetchers
├── skills/          # Reusable mining patterns (inference, caching, etc.)
├── scaffolder.py    # Code generator from templates
└── cli.py           # CLI entry point
subnet_configs/      # YAML configs describing each subnet
templates/           # Jinja2 templates for miner scaffolding
```
