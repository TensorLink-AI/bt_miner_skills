# Deployment Reference

## Overview

Deployment covers: model registry (versioning/staging), hot-swapping models into the production
miner, and monitoring production performance.

## Model Registry

### Stages
```
candidate  →  live_testing  →  production  →  retired
    │              │                │
    └── rejected   └── rejected     └── replaced
```

### Implementation

```python
import json
import shutil
from pathlib import Path
from datetime import datetime


class ModelRegistry:
    """
    Version and manage models through their lifecycle.
    
    Directory structure:
    model_registry/
    ├── candidates/
    │   ├── dlinear_v3_20250301_143022/
    │   │   ├── model.pt
    │   │   ├── config.json
    │   │   └── metrics.json
    │   └── ...
    ├── live_testing/
    │   └── ...
    ├── production/
    │   ├── current -> dlinear_v3_20250301_143022  (symlink)
    │   └── dlinear_v3_20250301_143022/
    └── retired/
        └── ...
    """
    
    def __init__(self, base_dir="model_registry"):
        self.base_dir = Path(base_dir)
        for stage in ["candidates", "live_testing", "production", "retired"]:
            (self.base_dir / stage).mkdir(parents=True, exist_ok=True)
    
    def register(self, model, config: dict, metrics: dict, name_prefix: str = "model"):
        """Register a new model as candidate."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_id = f"{name_prefix}_{timestamp}"
        model_dir = self.base_dir / "candidates" / model_id
        model_dir.mkdir(parents=True)
        
        # Save model weights
        import torch
        torch.save(model.state_dict(), model_dir / "model.pt")
        
        # Save config and metrics
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return model_id
    
    def promote(self, model_id: str, to_stage: str):
        """Move model to next stage."""
        # Find current location
        current_path = self._find_model(model_id)
        if current_path is None:
            raise ValueError(f"Model {model_id} not found")
        
        target_dir = self.base_dir / to_stage / model_id
        shutil.move(str(current_path), str(target_dir))
        
        if to_stage == "production":
            # Update symlink
            current_link = self.base_dir / "production" / "current"
            if current_link.is_symlink():
                # Retire old production model
                old_target = current_link.resolve()
                old_id = old_target.name
                current_link.unlink()
                self.promote(old_id, "retired")
            
            current_link.symlink_to(target_dir)
        
        return target_dir
    
    def get_production_model_path(self) -> Path:
        """Get path to current production model."""
        current_link = self.base_dir / "production" / "current"
        if not current_link.exists():
            raise FileNotFoundError("No production model deployed")
        return current_link.resolve()
    
    def _find_model(self, model_id: str) -> Path:
        """Find model across all stages."""
        for stage in ["candidates", "live_testing", "production", "retired"]:
            path = self.base_dir / stage / model_id
            if path.exists():
                return path
        return None
    
    def list_models(self, stage: str = None) -> list[dict]:
        """List all models, optionally filtered by stage."""
        stages = [stage] if stage else ["candidates", "live_testing", "production", "retired"]
        models = []
        
        for s in stages:
            stage_dir = self.base_dir / s
            for model_dir in stage_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != "current":
                    config_path = model_dir / "config.json"
                    metrics_path = model_dir / "metrics.json"
                    
                    config = json.loads(config_path.read_text()) if config_path.exists() else {}
                    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
                    
                    models.append({
                        "model_id": model_dir.name,
                        "stage": s,
                        "config": config,
                        "metrics": metrics,
                    })
        
        return models
```

## Hot-Swap Deployment

### How the Production Miner Works

The production miner runs as a PM2 process using the synth-subnet codebase. Your custom model
integrates via the miner's `forward` function.

```python
# miner/forward.py — The function called by the synth-subnet miner neuron

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class MinerForward:
    """
    Production miner forward handler.
    
    Loads the current production model and generates predictions on demand.
    Supports hot-swapping models without restarting the miner process.
    """
    
    def __init__(self, registry_path="model_registry"):
        self.registry_path = Path(registry_path)
        self.model = None
        self.model_id = None
        self.last_loaded = None
        self._load_production_model()
    
    def _load_production_model(self):
        """Load or reload the current production model."""
        current_link = self.registry_path / "production" / "current"
        
        if not current_link.exists():
            raise FileNotFoundError("No production model. Deploy one first!")
        
        model_dir = current_link.resolve()
        new_model_id = model_dir.name
        
        # Only reload if model changed
        if new_model_id == self.model_id:
            return
        
        # Load config
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        
        # Instantiate model (import the right class based on config)
        model_class = self._get_model_class(config["model_type"])
        self.model = model_class(**config["model_params"])
        
        # Load weights
        state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.model_id = new_model_id
        self.last_loaded = datetime.utcnow()
        print(f"[MINER] Loaded model: {new_model_id}")
    
    def _get_model_class(self, model_type: str):
        """Dynamic model class loading."""
        if model_type == "dlinear_gaussian":
            from models.dlinear_gaussian import DLinearGaussian
            return DLinearGaussian
        # Add more model types here
        raise ValueError(f"Unknown model type: {model_type}")
    
    def generate_predictions(
        self,
        asset: str,
        start_time: datetime,
        time_increment: int,
        time_horizon: int,
        num_simulations: int,
        current_price: float,
    ) -> list:
        """
        Generate predictions in the format expected by the validator.
        
        Returns:
            [start_timestamp, time_interval, [path1], [path2], ..., [pathN]]
        """
        # Check for model updates (hot-swap)
        self._load_production_model()
        
        # Determine challenge type
        if time_horizon == 86400:  # 24 hours
            challenge_type = "24h"
            n_steps = 288  # 24h / 5min
        elif time_horizon == 3600:  # 1 hour
            challenge_type = "1h"
            n_steps = 60   # 1h / 1min
        else:
            raise ValueError(f"Unknown time_horizon: {time_horizon}")
        
        # Get features (lookback window of recent data)
        features = self._get_live_features(asset, challenge_type)
        
        # Generate paths
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            mu, sigma = self.model(features_tensor)
            paths = self.model.sample_paths(
                mu, sigma,
                n_paths=num_simulations,
                current_price=current_price,
            )
            paths = paths.squeeze(0).numpy()  # [n_paths, n_steps+1]
        
        # Round to 8 decimal places (validator requirement)
        paths = np.round(paths, 8)
        
        # Format response
        response = [
            int(start_time.timestamp()),
            time_increment,
        ]
        for i in range(num_simulations):
            response.append(paths[i].tolist())
        
        return response
    
    def _get_live_features(self, asset: str, challenge_type: str) -> np.ndarray:
        """
        Fetch recent price data and compute features for prediction.
        Uses Pyth Oracle for live prices.
        """
        # Implementation: fetch from Pyth, compute features causally
        # This connects to your data_pipeline.py logic
        pass
```

### PM2 Configuration

```javascript
// miner.config.js
module.exports = {
    apps: [
        {
            name: "synth-miner",
            interpreter: "python3",
            script: "./neurons/miner.py",
            args: [
                "--netuid", "50",
                "--wallet.name", "miner",
                "--wallet.hotkey", "default",
                "--subtensor.network", "finney",
                "--axon.port", "8091",
                "--logging.info",
            ].join(" "),
            env: {
                PYTHONPATH: ".",
                MODEL_REGISTRY_PATH: "./model_registry",
            },
        },
    ],
};
```

### Hot-Swap Procedure

```bash
# 1. Register new model
python -c "
from deployment.model_registry import ModelRegistry
from models.dlinear_gaussian import DLinearGaussian
import torch

registry = ModelRegistry()

# Load your trained model
model = DLinearGaussian(lookback=288, horizon=288)
model.load_state_dict(torch.load('best_model.pt'))

model_id = registry.register(
    model,
    config={'model_type': 'dlinear_gaussian', 'model_params': {...}},
    metrics={'val_crps': 142.3, 'live_test_crps': 148.7},
    name_prefix='dlinear_v4',
)
print(f'Registered: {model_id}')
"

# 2. Promote to production (automatic symlink update)
python -c "
from deployment.model_registry import ModelRegistry
registry = ModelRegistry()
registry.promote('dlinear_v4_20250301_143022', 'production')
print('Promoted to production — miner will hot-swap on next request')
"

# 3. The miner picks up the new model automatically (no restart needed)
# Check logs:
pm2 logs synth-miner --lines 20
```

## Production Monitoring

```python
class ProductionMonitor:
    """Monitor production miner health."""
    
    def __init__(self, leaderboard, registry):
        self.leaderboard = leaderboard
        self.registry = registry
    
    def health_check(self) -> dict:
        """Run production health checks."""
        model_path = self.registry.get_production_model_path()
        model_id = model_path.name
        
        metrics = self.leaderboard.get_model_metrics(model_id)
        
        alerts = []
        
        # Check if model is degrading
        if metrics.get("rolling_avg", 999) > 300:
            alerts.append("WARNING: Rolling average CRPS > 300 — model may be degrading")
        
        # Check per-asset health
        for asset, data in metrics.get("per_asset", {}).items():
            if data["mean_crps"] > 200:
                alerts.append(f"WARNING: {asset} mean CRPS = {data['mean_crps']:.1f} — poor")
        
        # Check consistency
        if metrics.get("consistency", 999) > 50:
            alerts.append("WARNING: High variance in scores — model is unstable")
        
        # Check if any assets are missing
        expected = set(["BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"])
        scored = set(metrics.get("per_asset", {}).keys())
        missing = expected - scored
        if missing:
            alerts.append(f"CRITICAL: Missing predictions for: {missing}")
        
        return {
            "model_id": model_id,
            "rolling_avg": metrics.get("rolling_avg"),
            "emission_share_estimate": metrics.get("emission_share"),
            "alerts": alerts,
            "status": "HEALTHY" if not alerts else "DEGRADED",
        }
```

## Pre-Deployment Checklist

Before promoting to production:

- [ ] Model trained on latest data (< 24h old)
- [ ] Walk-forward validation CRPS beats GBM baseline by >15%
- [ ] All 9 assets (24h) or 4 assets (1h) produce valid predictions
- [ ] Response format validated (correct timestamps, path lengths, decimal places)
- [ ] Live-tested for minimum 48 hours without errors
- [ ] Live CRPS within 10% of backtest CRPS (no train/live gap)
- [ ] Prediction latency < 30 seconds (must return before start_time)
- [ ] No NaN or Inf values in any paths
- [ ] All paths start at current_price (first value must match)
- [ ] Paths are realistic (no negative prices, no >10× jumps in 5 min)
