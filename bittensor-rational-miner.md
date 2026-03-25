---
name: bittensor-rational-miner
version: 1.0.0
description: >
  Injects rational, competitive Bittensor mining decision-making into an agent.
  Covers subnet selection, registration cost monitoring, break-even analysis,
  deregistration risk, P&L tracking, and exit strategy. Subnet-agnostic in
  framework but aware that mining mechanics vary radically per subnet.
tags:
  - bittensor
  - crypto
  - mining
  - decentralized-ai
  - tao
  - dtao
  - subnet
---
# Bittensor Rational Miner Skill
## Purpose
This skill transforms an agent into a disciplined, economically rational Bittensor
subnet miner. It does **not** implement the actual mining logic for any specific
subnet — that varies radically between subnets (LLM inference on SN1, GPU compute
on SN27, protein folding on SN25, trading signals on SN8, etc.). Instead, it
provides the agent with a persistent decision framework covering:
- When to register (and when NOT to)
- How to monitor registration cost and break-even
- How to track P&L against hardware and opportunity cost
- How to assess deregistration risk and act before it happens
- When to exit a subnet and reallocate
The agent must always treat the actual mining work as subnet-specific — consult
the subnet's GitHub repo, SKILL.md, or documentation for the actual miner logic.
---
## Core Mental Model: The Three Clocks
A rational miner always tracks three concurrent timelines:
```
1. IMMUNITY CLOCK    — countdown to when you can be deregistered
2. EMISSION CLOCK    — when you start receiving meaningful emissions vs. peers
3. COST CLOCK        — rate at which your registration cost + hardware cost is bleeding
```
If Emission Clock > Cost Clock before Immunity Clock expires → you are on track.
If Cost Clock > Emission Clock before Immunity Clock expires → you have a problem.
---
## Phase 1: Subnet Selection (Pre-Registration)
Before spending any TAO on registration, the agent MUST assess these factors:
### 1.1 Emission Rate
- Fetch subnet emission via `btcli subnets list` or taostats.io/subnets
- Higher emission = more TAO distributed per epoch, but also more competition
- Under Dynamic TAO (dTAO), emission is driven by staked TAO in the subnet pool
- **Formula**: `Daily TAO per top miner ≈ (subnet_emission_per_tempo × 0.41 × miner_share) × (tempos_per_day)`
  - Miners receive ~41% of subnet emission; validators receive ~41%; subnet owner ~18%
  - `tempos_per_day = 86400 / (tempo × 12)` (blocks are ~12 seconds each)
### 1.2 Registration Cost
- Query current registration cost: `btcli subnets register --netuid <N> --wallet.name <W> --wallet.hotkey <H>`
- Or read from taostats: `https://taostats.io/subnets/netuid-<N>/#registration`
- Registration cost is **sunk** — it is burned on registration and cannot be recovered
- Registration cost fluctuates dynamically based on recent registration rate:
  - If many miners registered recently → cost rises
  - If few registrations → cost decays back down
- **Never register when cost is at a local peak** without explicit justification
### 1.3 Immunity Period
- Each new miner gets immunity for a defined number of blocks (subnet-specific, typically 4096 blocks ≈ ~14 hours)
- During immunity, you cannot be deregistered regardless of performance
- **Key rule**: Your emission must be competitive BEFORE immunity expires, or you risk immediate deregistration
### 1.4 Competitive Landscape
- Pull the metagraph for the target subnet and examine incentive distribution
- If the incentive distribution is tight (low std dev), the subnet is highly competitive — small drops in performance cause deregistration
- If distribution is spread, there is more headroom for performance variance
- Check: What is the emission of the **lowest active (non-immune) miner**? That is your floor.
### 1.5 Hardware Cost vs. Emission
- Estimate daily hardware cost (GPU server, electricity)
- Estimate daily emission at median miner rank
- **Break-even threshold**: `break_even_days = registration_cost_TAO / (daily_emission_TAO - daily_hardware_cost_TAO)`
- If break-even > 60 days → require strong conviction before registering
### 1.6 Subnet-Specific Mining Complexity
⚠️ **CRITICAL**: The agent must acknowledge that mining difficulty varies enormously:
| Mining Type | Example Subnets | What Miners Actually Do |
|---|---|---|
| LLM Inference | SN1, SN4, SN11 | Run GPU inference servers responding to validator queries |
| Distributed Training | SN3, SN9, SN37, SN81 | Contribute GPU to collaborative model training runs |
| Data Collection | SN13, SN22, SN24, SN42 | Scrape, clean, and submit structured datasets |
| Financial Prediction | SN8, SN28, SN50, SN55 | Submit price forecasts evaluated vs. ground truth |
| Compute Marketplace | SN12, SN27, SN39, SN51, SN64 | Provide raw GPU/CPU capacity on demand |
| Scientific Compute | SN18, SN25, SN57, SN68 | Run domain-specific scientific models |
| Event Prediction | SN6, SN30, SN41, SN44 | Submit predictions on sports/events |
| Code Generation | SN35, SN45, SN54, SN62 | Generate and execute code on demand |
Always review the subnet's `min_compute.yml` and README before committing. Do not
assume GPU type, VRAM, or bandwidth requirements — they vary from consumer GPU to
H100-class hardware.
---
## Phase 2: Registration Decision Gate
Run this checklist before every registration. If any FAIL, do not register:
```
[ ] Registration cost < 10% of estimated 30-day emission?          PASS / FAIL
[ ] Break-even < 45 days at median miner performance?              PASS / FAIL
[ ] Hardware is available and configured for subnet requirements?  PASS / FAIL
[ ] Miner code tested on testnet or locally?                       PASS / FAIL
[ ] Emission floor (lowest active miner) above hardware daily cost? PASS / FAIL
[ ] Registration cost is not at a recent local peak (declining)?   PASS / FAIL
[ ] You understand the scoring mechanism well enough to optimize?  PASS / FAIL
```
A FAIL on any item = wait, investigate, or target a different subnet.
---
## Phase 3: Active Mining — Ongoing Monitoring Loop
Once registered, the agent should run a monitoring cadence:
### Every Epoch (every `tempo` blocks ≈ 12-60 min depending on subnet):
- Check your miner's `incentive` rank vs. all peers in the metagraph
- Check your `trust` score (validators scoring you as reliable)
- Compare your emission this epoch vs. last epoch — is it trending up or down?
- Check your position relative to the **deregistration threshold** (lowest non-immune miner)
### Every Day:
- Calculate **actual daily TAO earned** vs. **hardware cost in TAO equivalent**
- Calculate **cumulative P&L** since registration:
  ```
  cumulative_PnL = total_TAO_earned - registration_cost - (hardware_cost_per_day × days_active)
  ```
- Note TAO price movement: if TAO price drops significantly, re-evaluate break-even
- Check registration cost for any subnets you may want to enter next
### Every Week:
- Full competitive review: where do you rank among all miners?
- Are new miners entering with better performance?
- Has the subnet's emission allocation changed (check taostats)?
- Review alpha token price if under dTAO — your emissions are in alpha, not TAO directly
---
## Phase 4: Deregistration Risk Management
### Deregistration Trigger
When immunity expires, if your emission is the **lowest among non-immune miners**,
the next new registrant will take your UID slot. You receive nothing and your
registration cost is permanently lost.
### Risk Signals (in order of urgency):
1. 🟡 **Yellow**: Your incentive rank drops to bottom 20% of non-immune miners
2. 🟠 **Orange**: Your incentive rank drops to bottom 10%, or performance gap widening
3. 🔴 **Red**: You are the lowest-ranked non-immune miner AND new registrations are occurring
### Response Actions by Signal:
- 🟡 Yellow → Debug performance, check validator connectivity, review scoring criteria
- 🟠 Orange → Implement emergency performance optimizations, consider temporarily pausing to fix
- 🔴 Red → Either fix within next 1-2 epochs or **voluntarily deregister** to avoid forced deregistration
  - Voluntary deregistration before forced = you preserve reputation and avoid paying for the next immunity period in vain
---
## Phase 5: Exit Criteria
Exit a subnet (voluntarily deregister) when ANY of these are true:
1. **Sustained loss**: 7 consecutive days of negative P&L with no improvement path visible
2. **Structural competition shift**: Multiple high-resource competitors entered and pushed floor emissions below break-even
3. **Subnet emission decline**: Subnet's TAO allocation decreased significantly (check taostats for trend)
4. **Alpha price collapse**: If subnet uses dTAO alpha tokens and alpha price dropped >60% vs. entry
5. **Opportunity cost**: Another subnet offers >2x the expected return for equivalent or lower hardware
6. **Mining mechanism change**: Subnet updated its incentive mechanism and your setup is no longer optimal — requires rebuild
---
## Phase 6: Multi-Subnet Strategy
Running multiple subnets is viable but requires careful management:
### Diversification Rules:
- Never put >50% of total mining capacity in one subnet
- Prefer subnets with uncorrelated scoring mechanisms (e.g., don't run two LLM inference subnets with identical hardware — validators may notice and score you lower)
- Maintain a **reserve TAO wallet** of at least 3× the highest current registration cost — for fast entry into opportunities
### Hotkey Management:
- Each UID on each subnet requires a unique hotkey within that subnet
- A single hotkey can hold UIDs across multiple different subnets
- Register subnet UIDs with separate hotkeys per subnet to isolate risk
### Resource Allocation:
- Allocate hardware to subnets in proportion to expected daily emission
- When a subnet's expected emission drops below hardware cost, reallocate to another subnet
---
## Live Data Queries
The agent should use these data sources for real-time decision-making:
### Chain Data (via btcli or Bittensor SDK):
```python
import bittensor as bt
# Registration cost
sub = bt.Subtensor(network="wss://gateway-dev.flamewire.io/public/rpc/bittensor")
cost = sub.burn_cost(netuid=<N>)
# Metagraph (for incentive rankings)
mg = bt.Metagraph(netuid=<N>)
mg.sync()
print(mg.incentive)   # incentive per UID
print(mg.emission)    # emission per UID
print(mg.trust)       # trust per UID
# Subnet info
info = sub.get_subnet_info(netuid=<N>)
```
### Taostats URLs for monitoring:
- Subnet overview: `https://taostats.io/subnets/netuid-<N>/`
- Registration cost: `https://taostats.io/subnets/netuid-<N>/#registration`
- Metagraph table: `https://taostats.io/subnets/netuid-<N>/#metagraph`
- Emission distribution: `https://taostats.io/subnets/netuid-<N>/#incentive`
---
## P&L Tracking Template
Maintain this ledger for each active miner:
```
SUBNET: SN<N> — <Name>
Registered:         <date>
Registration cost:  <X> TAO (burned)
Hardware setup:     <GPU type, monthly $, $/day in TAO equiv>
DAILY LOG:
Date | Epoch | Incentive Rank | Daily Emission | Daily HW Cost | Daily P&L | Cumulative P&L
-----|-------|----------------|----------------|---------------|-----------|---------------
...
BREAK-EVEN DATE (projected): <date>
ACTUAL BREAK-EVEN DATE:      <date or pending>
```
---
## Key Constants & Formulas
```
Block time:              ~12 seconds
Blocks per day:          ~7,200
Miner emission share:    41% of subnet emission
Validator share:         41%
Subnet owner share:      18%
Max miners per subnet:   192 (256 total UIDs, 64 reserved for validators)
Immunity period:         Subnet-specific, typically 1,000–4,096 blocks
Daily tempos = 86400 / (tempo_blocks × 12)
Daily miner emission (TAO equiv) = subnet_TAO_emission_per_tempo × 0.41 × (your_incentive / sum_all_incentives) × daily_tempos
Break-even days = registration_cost / (daily_emission - daily_hardware_cost)
```
---
## Behavioural Defaults
When operating under this skill, the agent defaults to:
- **Conservative entry**: Never register unless break-even < 45 days
- **Continuous monitoring**: Check metagraph every epoch, P&L every day
- **Early warning exits**: Trigger review at Yellow signal, not Red
- **Subnet-agnostic framing**: Always caveat that actual mining implementation is subnet-specific and must be sourced from the subnet's own documentation
- **Cost transparency**: Always express costs in both TAO and fiat equivalent
- **No overconfidence**: Registration cost projections use median miner rank, not top-rank optimism
