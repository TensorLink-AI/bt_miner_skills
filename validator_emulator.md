# Leaderboard & Competitiveness Tracking Reference

## Overview

The leaderboard tracks every model you train, test, and deploy — broken down by challenge type,
asset, time interval, and over time. This is how you know whether a model is ready for mainnet.

## What Gets Tracked

### Per-Prompt Metrics
```python
PromptResult = {
    "model_id": str,            # Unique model identifier
    "model_config": dict,       # Full config for reproducibility
    "challenge_type": str,      # "24h" or "1h"
    "asset": str,               # "BTC", "ETH", etc.
    "timestamp": datetime,      # When the prompt was scored
    "crps_sum": float,          # Raw CRPS sum
    "prompt_score": float,      # After percentile capping + best subtraction
    "interval_breakdown": {     # Per-interval CRPS
        "5min": {"total": float, "mean": float},
        "30min": {"total": float, "mean": float},
        "3hr": {"total": float, "mean": float},
        "absolute": float,
    },
}
```

### Aggregate Metrics (Per Model)
```python
ModelMetrics = {
    "model_id": str,
    "rolling_avg": float,           # 10-day weighted rolling average
    "emission_share": float,        # Simulated % of emissions
    "per_asset_rolling": {          # Rolling avg broken by asset
        "BTC": float, "ETH": float, "SOL": float, ...
    },
    "per_challenge_rolling": {      # Rolling avg by challenge type
        "24h": float, "1h": float,
    },
    "weakest_asset": str,           # Worst-performing asset
    "weakest_interval": str,        # Worst-performing interval bucket
    "consistency_score": float,     # Std of prompt scores (lower = more consistent)
    "n_prompts_scored": int,        # Total prompts in window
    "time_in_registry": float,      # Hours since first registered
}
```

## Leaderboard Implementation

```python
import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import numpy as np


class Leaderboard:
    """Track model competitiveness across all dimensions."""
    
    def __init__(self, db_path="data/leaderboard.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_config TEXT,
                challenge_type TEXT NOT NULL,
                asset TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                crps_sum REAL NOT NULL,
                prompt_score REAL,
                interval_breakdown TEXT,
                is_live BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                model_config TEXT,
                model_path TEXT,
                stage TEXT DEFAULT 'candidate',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                promoted_at TEXT,
                retired_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_model 
            ON prompt_results(model_id, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_asset 
            ON prompt_results(asset, challenge_type, timestamp)
        """)
        conn.commit()
        conn.close()
    
    def log_result(self, result: dict):
        """Log a single prompt result."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO prompt_results 
               (model_id, model_config, challenge_type, asset, timestamp, 
                crps_sum, prompt_score, interval_breakdown, is_live)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result["model_id"],
                json.dumps(result.get("model_config", {})),
                result["challenge_type"],
                result["asset"],
                result["timestamp"].isoformat(),
                result["crps_sum"],
                result.get("prompt_score"),
                json.dumps(result.get("interval_breakdown", {})),
                result.get("is_live", False),
            )
        )
        conn.commit()
        conn.close()
    
    def get_model_metrics(self, model_id: str, window_days: int = 10) -> dict:
        """Calculate aggregate metrics for a model."""
        conn = sqlite3.connect(self.db_path)
        cutoff = (datetime.utcnow() - timedelta(days=window_days)).isoformat()
        
        df = pd.read_sql_query(
            """SELECT * FROM prompt_results 
               WHERE model_id = ? AND timestamp > ?
               ORDER BY timestamp""",
            conn,
            params=(model_id, cutoff),
        )
        conn.close()
        
        if len(df) == 0:
            return {"model_id": model_id, "status": "no_data"}
        
        # Asset weights
        ASSET_WEIGHTS = {
            "BTC": 1.0, "ETH": 0.6716, "XAU": 2.2620, "SOL": 0.5884,
            "SPYX": 2.9914, "NVDAX": 1.3885, "TSLAX": 1.4200,
            "AAPLX": 1.8650, "GOOGLX": 1.4311,
        }
        
        # Rolling average
        df["weight"] = df["asset"].map(ASSET_WEIGHTS).fillna(1.0)
        rolling_avg = (df["crps_sum"] * df["weight"]).sum() / df["weight"].sum()
        
        # Per-asset breakdown
        per_asset = df.groupby("asset").agg(
            mean_crps=("crps_sum", "mean"),
            std_crps=("crps_sum", "std"),
            count=("crps_sum", "count"),
        ).to_dict("index")
        
        # Per-challenge breakdown
        per_challenge = df.groupby("challenge_type").agg(
            mean_crps=("crps_sum", "mean"),
            count=("crps_sum", "count"),
        ).to_dict("index")
        
        # Weakest points
        worst_asset = max(per_asset.items(), key=lambda x: x[1]["mean_crps"])[0]
        
        return {
            "model_id": model_id,
            "rolling_avg": float(rolling_avg),
            "per_asset": per_asset,
            "per_challenge": per_challenge,
            "weakest_asset": worst_asset,
            "consistency": float(df["crps_sum"].std()),
            "n_prompts": len(df),
        }
    
    def compare_models(self, model_ids: list[str], window_days: int = 10) -> pd.DataFrame:
        """Compare multiple models side by side."""
        rows = []
        for mid in model_ids:
            metrics = self.get_model_metrics(mid, window_days)
            if metrics.get("status") == "no_data":
                continue
            rows.append({
                "model_id": mid,
                "rolling_avg": metrics["rolling_avg"],
                "consistency": metrics["consistency"],
                "n_prompts": metrics["n_prompts"],
                "weakest_asset": metrics["weakest_asset"],
            })
        
        df = pd.DataFrame(rows).sort_values("rolling_avg")
        
        # Add simulated emission share
        if len(df) > 0:
            scores = dict(zip(df["model_id"], df["rolling_avg"]))
            exp_scores = np.exp(-0.0475 * np.array(list(scores.values())))
            shares = exp_scores / exp_scores.sum()
            df["emission_share"] = shares
        
        return df
    
    def is_competitive(self, model_id: str, threshold_percentile: float = 50) -> bool:
        """
        Check if model would be competitive on mainnet.
        
        A model is competitive if its rolling average is better than the
        threshold percentile of all tracked models.
        """
        conn = sqlite3.connect(self.db_path)
        all_models = pd.read_sql_query(
            "SELECT DISTINCT model_id FROM prompt_results", conn
        )["model_id"].tolist()
        conn.close()
        
        all_metrics = []
        for mid in all_models:
            m = self.get_model_metrics(mid)
            if m.get("status") != "no_data":
                all_metrics.append(m["rolling_avg"])
        
        if not all_metrics:
            return True  # No competition yet
        
        target = self.get_model_metrics(model_id)
        if target.get("status") == "no_data":
            return False
        
        threshold = np.percentile(all_metrics, threshold_percentile)
        return target["rolling_avg"] <= threshold
    
    def get_asset_heatmap(self, model_id: str) -> pd.DataFrame:
        """
        Get per-asset, per-interval CRPS breakdown as a heatmap.
        Shows where the model is strong/weak.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """SELECT asset, interval_breakdown FROM prompt_results
               WHERE model_id = ? ORDER BY timestamp DESC LIMIT 100""",
            conn,
            params=(model_id,),
        )
        conn.close()
        
        rows = []
        for _, row in df.iterrows():
            breakdown = json.loads(row["interval_breakdown"])
            for interval, values in breakdown.items():
                if isinstance(values, dict):
                    rows.append({
                        "asset": row["asset"],
                        "interval": interval,
                        "mean_crps": values.get("mean", values.get("total", 0)),
                    })
        
        if not rows:
            return pd.DataFrame()
        
        return pd.DataFrame(rows).pivot_table(
            index="asset", columns="interval", values="mean_crps", aggfunc="mean"
        )
```

## Competitiveness Dashboard

When reviewing model competitiveness, present this information:

### 1. Model Ranking Table
```
╔══════════════════════╦═══════════╦══════════════╦═══════════════╦═══════════════╗
║ Model                ║ Roll Avg  ║ Emission %   ║ Weakest Asset ║ Consistency   ║
╠══════════════════════╬═══════════╬══════════════╬═══════════════╬═══════════════╣
║ dlinear_v3_wide      ║ 142.3     ║ 28.4%        ║ GOOGLX        ║ 18.2         ║
║ dlinear_v2_base      ║ 158.7     ║ 22.1%        ║ XAU           ║ 22.4         ║
║ gbm_baseline         ║ 203.4     ║ 12.8%        ║ SOL           ║ 31.5         ║
║ historical_sim       ║ 245.1     ║ 8.2%         ║ TSLAX         ║ 45.3         ║
╚══════════════════════╩═══════════╩══════════════╩═══════════════╩═══════════════╝
```

### 2. Asset Heatmap
```
           5min    30min   3hr     absolute
BTC        12.3    18.4    42.1    8.2
ETH        14.1    21.2    48.3    9.5
SOL        18.7    28.9    55.2    12.1
XAU        15.3    22.1    38.4    7.8      ← Highest weight!
SPYX       22.1    34.5    61.2    14.3     ← Highest weight!
...
```

### 3. Key Questions to Ask
- Is any single asset contributing >30% of the total CRPS? → Fix that asset specifically
- Is the 3-hour interval much worse than 5-min? → Model captures noise but not drift
- Is consistency (std) high? → Model is unstable, might need longer training or regularization
- Is emission share < 5%? → Not competitive, don't deploy yet
