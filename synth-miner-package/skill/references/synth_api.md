# Synth API Reference

## Overview

The Synth API at `https://api.synthdata.co` provides live leaderboard data, miner validation
scores, and historical performance. Use this to:

1. **Benchmark your emulator** — compare your local CRPS calculations against real scores
2. **See what you're competing against** — know the top miners' scores before deploying
3. **Monitor your miner post-deployment** — check if your predictions are being scored correctly
4. **Validate your response format** — the validation endpoint tells you if submissions are valid

**Base URL**: `https://api.synthdata.co`
**Dashboard**: `https://miners.synthdata.co` (visual leaderboard)
**No auth required** for leaderboard and scores endpoints.

---

## Endpoints

### 1. Latest Leaderboard (What Am I Competing Against?)

**Call this first** to understand the current competitive landscape.

```
GET /v2/leaderboard/latest?prompt_name=low    # 24-hour challenge
GET /v2/leaderboard/latest?prompt_name=high   # 1-hour HFT challenge
```

**Response:**
```json
[
  {
    "coldkey": "5F1a...",
    "ip_address": "1.2.3.4",
    "neuron_uid": 42,
    "rewards": 0.0234,        // Emission share (higher = better)
    "updated_at": "2026-03-25T12:00:00Z"
  },
  ...
]
```

**What to look for:**
- `rewards` is the emission share — higher means better CRPS
- Sort by `rewards` descending to see top miners
- The top 10 miners are your real competition
- Note the gap between #1 and #10 — that's how tight the competition is

```python
import requests

def get_leaderboard(prompt_name="low"):
    """Fetch current Synth leaderboard.
    
    Args:
        prompt_name: "low" for 24-hour, "high" for 1-hour HFT
    """
    resp = requests.get(
        "https://api.synthdata.co/v2/leaderboard/latest",
        params={"prompt_name": prompt_name},
    )
    resp.raise_for_status()
    miners = resp.json()
    
    # Sort by rewards (emission share)
    miners.sort(key=lambda m: m["rewards"], reverse=True)
    return miners

def print_leaderboard(prompt_name="low"):
    """Print a readable leaderboard summary."""
    miners = get_leaderboard(prompt_name)
    horizon = "24-hour" if prompt_name == "low" else "1-hour HFT"
    
    print(f"\n{'='*60}")
    print(f" SYNTH LEADERBOARD — {horizon} Challenge")
    print(f"{'='*60}")
    print(f" {'Rank':<6} {'UID':<8} {'Rewards':<12} {'Coldkey':<20}")
    print(f" {'-'*6} {'-'*8} {'-'*12} {'-'*20}")
    
    for i, m in enumerate(miners[:20]):
        print(f" {i+1:<6} {m['neuron_uid']:<8} {m['rewards']:<12.6f} {m['coldkey'][:18]}...")
    
    total_top10 = sum(m["rewards"] for m in miners[:10])
    print(f"\n Top 10 combined: {total_top10:.4f} ({total_top10*100:.1f}% of emissions)")
    print(f" Total miners: {len(miners)}")
```

### 2. Meta-Leaderboard (Smoothed Rankings)

Aggregates performance over N days — more stable than the instant leaderboard.

```
GET /v2/meta-leaderboard/latest?days=14&prompt_name=low
GET /v2/meta-leaderboard/latest?days=14&prompt_name=high
```

Same response format as leaderboard. The `days` parameter controls the aggregation window
(default 14). This is closer to what actually determines emissions.

### 3. Latest Validation Scores (Per-Asset CRPS)

**This is the most useful endpoint for benchmarking.** Shows raw CRPS and prompt scores
for all miners, filterable by asset and challenge type.

```
GET /validation/scores/latest?asset=BTC&time_length=86400&time_increment=300   # BTC 24h
GET /validation/scores/latest?asset=ETH&time_length=3600&time_increment=60     # ETH 1h
```

**Parameters:**
- `asset`: BTC, ETH, SOL, XAU, SPYX, NVDAX, TSLAX, AAPLX, GOOGLX
- `time_length`: 86400 (24h) or 3600 (1h)
- `time_increment`: 300 (5-min for 24h) or 60 (1-min for 1h)

**Response:**
```json
[
  {
    "asset": "BTC",
    "crps": 145.23,           // Raw CRPS sum
    "miner_uid": 42,
    "prompt_score": 12.45,    // After transformation (0 = best)
    "scored_time": "2026-03-25T12:00:00Z",
    "time_length": 86400
  },
  ...
]
```

**What to look for:**
- `crps` is the raw CRPS sum — compare directly to your emulator output
- `prompt_score` is after transformation (best = 0)
- Check scores across ALL assets to see per-asset difficulty
- Compare your backtest CRPS to the median `crps` here

```python
def get_latest_scores(asset="BTC", challenge="24h"):
    """Fetch latest validation scores for an asset."""
    params = {"asset": asset}
    if challenge == "24h":
        params["time_length"] = 86400
        params["time_increment"] = 300
    else:
        params["time_length"] = 3600
        params["time_increment"] = 60
    
    resp = requests.get(
        "https://api.synthdata.co/validation/scores/latest",
        params=params,
    )
    resp.raise_for_status()
    return resp.json()

def get_competitive_thresholds():
    """Get the CRPS scores you need to beat to be competitive."""
    thresholds = {}
    
    for asset in ["BTC", "ETH", "SOL", "XAU"]:
        for challenge in ["24h", "1h"]:
            scores = get_latest_scores(asset, challenge)
            crps_values = [s["crps"] for s in scores if s["crps"] > 0]
            
            if crps_values:
                thresholds[f"{asset}_{challenge}"] = {
                    "best": min(crps_values),
                    "median": sorted(crps_values)[len(crps_values)//2],
                    "p25": sorted(crps_values)[len(crps_values)//4],
                    "n_miners": len(crps_values),
                }
    
    return thresholds

def print_competitive_analysis(my_crps: dict):
    """Compare your model's CRPS against live network.
    
    Args:
        my_crps: {"BTC_24h": 150.0, "ETH_24h": 160.0, ...}
    """
    thresholds = get_competitive_thresholds()
    
    print(f"\n{'='*70}")
    print(f" YOUR MODEL vs LIVE NETWORK")
    print(f"{'='*70}")
    print(f" {'Asset':<12} {'Yours':<10} {'Best':<10} {'Median':<10} {'Status':<15}")
    print(f" {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")
    
    for key, mine in my_crps.items():
        if key in thresholds:
            t = thresholds[key]
            if mine <= t["p25"]:
                status = "✅ TOP 25%"
            elif mine <= t["median"]:
                status = "⚠️  TOP 50%"
            else:
                status = "❌ BELOW MED"
            
            print(f" {key:<12} {mine:<10.1f} {t['best']:<10.1f} {t['median']:<10.1f} {status}")
```

### 4. Historical Scores (Track Performance Over Time)

```
GET /validation/scores/historical?from=2026-03-01&to=2026-03-25&miner_uid=42&asset=BTC
```

**Parameters:**
- `from`, `to`: ISO 8601 date range
- `miner_uid`: Optional — filter to your miner
- `asset`: Optional — filter to specific asset
- `time_increment`, `time_length`: Same as latest scores

Use this to:
- Track your miner's performance after deployment
- Detect degradation trends
- Compare your per-asset performance over time

### 5. Miner Validation (Is My Format Correct?)

```
GET /validation/miner?uid=42
```

**Response:**
```json
{
  "reason": "Valid",
  "response_time": "2026-03-25T12:00:00Z",
  "validated": true
}
```

If `validated` is `false`, the `reason` tells you what's wrong with your response format.
**Check this immediately after deploying** — format errors mean 90th percentile penalty.

---

## Integration with the Pipeline

### Pre-Deployment: Benchmark Phase

Before deploying to mainnet, compare your backtest CRPS against live scores:

```python
def should_deploy(my_backtest_crps: dict) -> bool:
    """Decide if model is ready for mainnet based on live competition."""
    thresholds = get_competitive_thresholds()
    
    competitive_count = 0
    total_count = 0
    
    for key, my_score in my_backtest_crps.items():
        if key in thresholds:
            total_count += 1
            if my_score <= thresholds[key]["median"]:
                competitive_count += 1
    
    ratio = competitive_count / total_count if total_count > 0 else 0
    
    print(f"Competitive on {competitive_count}/{total_count} asset-challenges ({ratio:.0%})")
    
    # Deploy if competitive on >70% of assets
    return ratio > 0.7
```

### Post-Deployment: Monitoring Phase

After your miner is live, continuously check:

```python
def monitor_my_miner(my_uid: int):
    """Post-deployment health check using Synth API."""
    
    # 1. Is my format valid?
    validation = requests.get(
        "https://api.synthdata.co/validation/miner",
        params={"uid": my_uid},
    ).json()
    
    if not validation["validated"]:
        print(f"🚨 VALIDATION FAILED: {validation['reason']}")
        return False
    
    # 2. Where am I on the leaderboard?
    for prompt in ["low", "high"]:
        lb = get_leaderboard(prompt)
        my_rank = next(
            (i+1 for i, m in enumerate(lb) if m["neuron_uid"] == my_uid),
            None
        )
        horizon = "24h" if prompt == "low" else "1h"
        if my_rank:
            rewards = next(m["rewards"] for m in lb if m["neuron_uid"] == my_uid)
            print(f"📊 {horizon}: Rank #{my_rank}/{len(lb)}, Rewards: {rewards:.6f}")
        else:
            print(f"⚠️  {horizon}: Not on leaderboard yet")
    
    # 3. Per-asset scores
    for asset in ["BTC", "ETH", "SOL", "XAU"]:
        scores = get_latest_scores(asset, "24h")
        my_score = next(
            (s for s in scores if s["miner_uid"] == my_uid),
            None
        )
        if my_score:
            print(f"  {asset}: CRPS={my_score['crps']:.1f}, Score={my_score['prompt_score']:.1f}")
    
    return True
```

---

## Key URLs

| Resource | URL |
|----------|-----|
| API Docs | `https://api.synthdata.co/docs` |
| Miner Dashboard | `https://miners.synthdata.co` |
| Main Website | `https://www.synthdata.co` |
| API Base | `https://api.synthdata.co` |
| Testnet API | `https://api-testnet.synthdata.co` |
