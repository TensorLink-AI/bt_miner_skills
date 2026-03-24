# bt_miner_skills - Agent Instructions

You are an agent building competitive Bittensor miners. This repo provides the
framework for an iterative loop: **research → build → test → deploy → monitor → improve**.

## How This Works

1. **Subnet configs** in `subnet_configs/` describe what each subnet expects from miners
2. **The agent loop** in `bt_miner_skills/agent/` drives the iterative development cycle
3. **Skills** in `bt_miner_skills/skills/` are reusable mining patterns you compose
4. **Templates** in `templates/` are Jinja2 templates for scaffolding miner code
5. **Workspace** directories (e.g., `workspace/sn1/`) hold generated miners and state

## Your Workflow

### Starting a New Subnet Miner

1. Check `subnet_configs/` for an existing config for the target subnet
2. If none exists, research the subnet and create one:
   - Use Bittensor MCP tools to fetch chain params (`get_subnet_info`, `get_metagraph`)
   - Find the subnet's GitHub repo and study the validator/miner code
   - Fill in the `SubnetConfig` schema (see `bt_miner_skills/config/subnet_config.py`)
3. Run `python -m bt_miner_skills.cli scaffold subnet_configs/<file>.yaml -o workspace/sn<N>/miner`
4. Customize the generated miner with actual mining logic
5. Write tests in `workspace/sn<N>/tests/`

### The Agent Loop

Use `bt_miner_skills/agent/orchestrator.py` to manage loop state:

```python
from bt_miner_skills.agent.orchestrator import Orchestrator

orch = Orchestrator("subnet_configs/sn1.yaml", workspace="workspace")
print(orch.get_current_prompt())  # What to do next
orch.complete_phase({"result": "..."})  # Advance to next phase
```

Or via CLI:
```bash
python -m bt_miner_skills.cli loop-context subnet_configs/sn1.yaml
python -m bt_miner_skills.cli loop-status 1
```

### Phase Details

- **RESEARCH**: Study the subnet config, repo, scoring, and metagraph. Output findings as artifacts.
- **BUILD**: Scaffold + customize the miner. Use skills from the registry.
- **TEST**: Run unit tests, protocol compliance tests, benchmarks.
- **DEPLOY**: Set up environment, register on subnet, start miner.
- **MONITOR**: Check incentive, rank, errors. Collect metrics.
- **IMPROVE**: Analyze what top miners do differently, plan next iteration.

## Key Files

- `bt_miner_skills/config/subnet_config.py` - The SubnetConfig schema (read this first)
- `bt_miner_skills/agent/loop.py` - Phase definitions and state machine
- `bt_miner_skills/agent/orchestrator.py` - Main orchestrator
- `bt_miner_skills/skills/registry.py` - Available skills
- `bt_miner_skills/scaffolder.py` - Code generator
- `subnet_configs/` - Subnet configuration files
- `templates/` - Jinja2 templates for code generation

## Available Bittensor MCP Tools

You have access to Bittensor blockchain data via MCP:
- `get_subnet_info(netuid)` - Chain params (tempo, difficulty, emission, etc.)
- `get_metagraph(netuid)` - All neurons with stakes, incentives, ranks
- `get_neurons_lite(netuid)` - Lighter neuron data
- `get_subnet_overview()` - Overview of all subnets
- `get_subnet_emissions()` - Emission distribution

Use these to populate `chain_params` in subnet configs and to monitor miner performance.

## Creating a Subnet Config

When researching a new subnet, fill in ALL fields of `SubnetConfig`:

1. **Identity**: netuid, name, description, repo/docs URLs
2. **Task**: What miners actually do (task_type, task_description)
3. **Protocol**: The Synapse class, request/response fields, timeout
4. **Scoring**: How validators score miners (mechanism, criteria, weights)
5. **Hardware**: What's needed to run competitively
6. **Dependencies**: Python packages, models, env vars
7. **Strategy**: Hints, pitfalls, reference implementations

The more detail you provide, the better the scaffolded miner will be.

## Conventions

- Subnet workspaces live in `workspace/sn<netuid>/`
- Each workspace has `miner/`, `tests/`, and `logs/` subdirs
- Loop state is persisted in `workspace/sn<netuid>/loop_state.json`
- Config files are YAML in `subnet_configs/`
- New skills go in `bt_miner_skills/skills/` and get registered in `registry.py`
