# Basilica GPU Training Reference

## Overview

Basilica (Bittensor Subnet 39) is a decentralized GPU compute marketplace. Instead of renting
from AWS/GCP/RunPod, you can spin up GPU instances on Basilica's network to train your Synth
models. This is optional — everything works on local hardware too — but Basilica gives you
access to A100/H100 GPUs on demand without vendor lock-in, and you're paying with TAO into
the Bittensor ecosystem you're already mining on.

## When to Use Basilica vs Local

| Scenario | Use Basilica | Use Local |
|----------|-------------|-----------|
| DLinear baseline (fast, CPU-fine) | ❌ | ✅ |
| Hyperparameter search (20+ configs) | ✅ | ❌ (too slow) |
| Transformer/large model training | ✅ | Maybe (if you have a GPU) |
| Live testing (needs uptime) | ✅ | ✅ |
| Quick iteration / debugging | ❌ | ✅ |

## Installation

```bash
pip install basilica-sdk
```

## Authentication

```bash
# Generate API token
basilica tokens create

# Set in environment
export BASILICA_API_TOKEN="basilica_..."
```

## Pattern 1: Deploy a Training Job (SDK — Recommended)

Use the Basilica SDK to deploy a training job as a containerized workload. Your training
script runs on remote GPU hardware, results sync back via persistent storage.

```python
import basilica

# Create persistent volume for data and model checkpoints
data_vol = basilica.Volume.from_name("synth-training-data", create_if_missing=True)
models_vol = basilica.Volume.from_name("synth-model-registry", create_if_missing=True)

@basilica.deployment(
    name="synth-train",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    gpu_count=1,
    min_gpu_memory_gb=16,
    memory="16Gi",
    cpu="4000m",
    pip_packages=[
        "pandas", "polars", "pyarrow", "numpy",
        "requests", "scikit-learn", "tqdm",
    ],
    volumes={
        "/data": data_vol,
        "/models": models_vol,
    },
    ttl_seconds=14400,  # Auto-cleanup after 4 hours
)
def train():
    """Training job that runs on Basilica GPU."""
    import subprocess
    import sys
    
    # Sync training code (uploaded to /data volume beforehand)
    sys.path.insert(0, "/data/synth-miner-mlops")
    
    from training.train import run_full_training
    from training.search import run_search
    
    # Run hyperparameter search across all assets
    results = run_search(
        data_dir="/data/processed",
        output_dir="/models/search_results",
        n_configs=30,
        device="cuda",
    )
    
    print(f"Search complete. Best CRPS: {results['best_crps']:.2f}")
    print(f"Best config saved to: /models/search_results/best_model.pt")

# Deploy and wait
deployment = train()
deployment.wait_until_ready()
print(f"Training running at: {deployment.url}")
print(f"Check logs: deployment.logs()")
```

## Pattern 2: Interactive GPU Shell

For development and debugging, deploy an interactive instance:

```python
from basilica import BasilicaClient

client = BasilicaClient()

data_vol = basilica.Volume.from_name("synth-training-data", create_if_missing=True)

# Deploy a Jupyter-like environment
deployment = client.deploy(
    name="synth-dev",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    port=8888,
    gpu_count=1,
    min_gpu_memory_gb=24,
    memory="32Gi",
    pip_packages=[
        "jupyter", "pandas", "polars", "pyarrow",
        "matplotlib", "seaborn", "tqdm",
    ],
    volumes={"/data": data_vol},
    env={"JUPYTER_TOKEN": "synth-dev-2025"},
    # Run Jupyter on startup
    source="""
import subprocess
subprocess.run([
    "jupyter", "notebook",
    "--ip=0.0.0.0", "--port=8888",
    "--no-browser", "--allow-root",
    f"--NotebookApp.token={os.environ['JUPYTER_TOKEN']}"
])
""",
)

deployment.wait_until_ready()
print(f"Jupyter at: {deployment.url}?token=synth-dev-2025")
```

## Pattern 3: Batch Training Pipeline

Run the full model search as a batch job — upload data first, run training, download results:

```python
import basilica
import json

# Step 1: Upload data to shared volume
data_vol = basilica.Volume.from_name("synth-data", create_if_missing=True)
models_vol = basilica.Volume.from_name("synth-models", create_if_missing=True)

# Step 2: Data preparation job (CPU only — cheap)
@basilica.deployment(
    name="synth-data-prep",
    image="python:3.11-slim",
    memory="8Gi",
    cpu="4000m",
    pip_packages=["pandas", "pyarrow", "requests", "numpy"],
    volumes={"/data": data_vol},
    ttl_seconds=3600,
)
def prep_data():
    """Fetch and prepare training data."""
    import sys
    sys.path.insert(0, "/data/code")
    
    from pipeline.data_pipeline import fetch_all_assets, compute_features, create_splits
    
    # Fetch 90 days of data for all assets
    raw = fetch_all_assets(days=90, save_dir="/data/raw")
    
    # Compute features with anti-leakage
    for asset, df in raw.items():
        featured = compute_features(df, asset)
        featured.to_parquet(f"/data/processed/{asset}_features.parquet")
    
    # Create walk-forward splits
    create_splits(data_dir="/data/processed", output_dir="/data/splits")
    
    print("Data preparation complete!")

# Step 3: GPU training job
@basilica.deployment(
    name="synth-gpu-train",
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    gpu_count=1,
    min_gpu_memory_gb=16,
    memory="16Gi",
    pip_packages=["pandas", "pyarrow", "numpy", "tqdm", "scikit-learn"],
    volumes={"/data": data_vol, "/models": models_vol},
    ttl_seconds=14400,  # 4 hours max
)
def gpu_train():
    """Run full hyperparameter search on GPU."""
    import sys, torch
    sys.path.insert(0, "/data/code")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    from training.search import run_search
    
    results = run_search(
        data_dir="/data/splits",
        output_dir="/models/search",
        n_configs=30,
        device="cuda",
    )
    
    # Save results summary
    with open("/models/search/summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Best model CRPS: {results['best_crps']:.2f}")

# Step 4: Run the pipeline
print("Starting data prep...")
prep_deployment = prep_data()
prep_deployment.wait_until_ready()
# Wait for data prep to complete (check logs)

print("Starting GPU training...")
train_deployment = gpu_train()
train_deployment.wait_until_ready()
print(f"Training logs: {train_deployment.logs(tail=50)}")
```

## Syncing Code to Basilica Volumes

Before running training, upload your code to the shared volume:

```python
import basilica
import tarfile
import io

data_vol = basilica.Volume.from_name("synth-data", create_if_missing=True)

# Deploy a quick uploader
@basilica.deployment(
    name="synth-upload",
    image="python:3.11-slim",
    volumes={"/data": data_vol},
    ttl_seconds=300,  # 5 min
)
def upload():
    """Upload codebase to volume."""
    import subprocess
    # Volume is mounted, any writes persist
    subprocess.run(["mkdir", "-p", "/data/code"])
    # Code will be available at /data/code for other deployments
    print("Volume ready for code upload")

deployment = upload()
deployment.wait_until_ready()
```

Alternatively, include your training script directly as the `source` parameter:

```python
# For simple scripts, inline the code
deployment = client.deploy(
    name="synth-quick-train",
    source="training/train.py",  # Local file path — Basilica uploads it
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    gpu_count=1,
    pip_packages=["pandas", "numpy", "pyarrow"],
)
```

## GPU Selection for Synth Models

| Model Type | Recommended GPU | VRAM | Basilica Config |
|-----------|----------------|------|-----------------|
| DLinear (baseline) | Any / CPU | <2 GB | `gpu_count=0` |
| DLinear + search (30 configs) | A4000 / A10 | 16 GB | `gpu_count=1, min_gpu_memory_gb=16` |
| Transformer models | A100 / H100 | 40+ GB | `gpu_count=1, min_gpu_memory_gb=40` |
| Multi-asset parallel training | A100 × 2 | 80+ GB | `gpu_count=2, gpu_models=["A100"]` |

## Cost Management

```python
# Always set TTL to prevent runaway costs
ttl_seconds=14400  # 4 hours max

# Check deployment status
deployment.status()

# Clean up when done
deployment.delete()

# List all running deployments
client = BasilicaClient()
for d in client.list():
    print(f"{d.name}: {d.state}")
    if d.state == "running" and is_stale(d):
        d.delete()
```

## Downloading Results

After training completes, results are on the persistent volume. Deploy a lightweight
instance to retrieve them:

```python
@basilica.deployment(
    name="synth-download",
    image="python:3.11-slim",
    port=8000,
    volumes={"/models": models_vol},
    ttl_seconds=600,
)
def serve_results():
    """Serve trained models for download."""
    import http.server
    import os
    os.chdir("/models")
    handler = http.server.SimpleHTTPRequestHandler
    http.server.HTTPServer(("", 8000), handler).serve_forever()

deployment = serve_results()
deployment.wait_until_ready()
print(f"Download models from: {deployment.url}")
# Use wget/curl to download best_model.pt, summary.json, etc.
```

## Error Handling for Training Jobs

```python
from basilica import DeploymentTimeout, DeploymentFailed

try:
    deployment = train()
    deployment.wait_until_ready(timeout=600)
except DeploymentTimeout:
    print("Training deployment timed out during startup")
    print(f"Last logs: {deployment.logs(tail=20)}")
    # Common cause: GPU not available, increase timeout or try different GPU
except DeploymentFailed as e:
    print(f"Deployment failed: {e.reason}")
    # Common causes: OOM, bad image, pip install failure

# Monitor long-running training
import time
while True:
    status = deployment.status()
    if status.get("is_failed"):
        print(f"Training failed! Logs:\n{deployment.logs(tail=50)}")
        break
    
    logs = deployment.logs(tail=5)
    print(logs)
    
    if "Training complete" in logs:
        print("Done!")
        break
    
    time.sleep(60)
```

## Integration with the MLOps Pipeline

The Basilica training option slots into Phase 5 (Training) and Phase 6 (Search) of the
agent prompt. The flow changes to:

```
LOCAL:   Data sourced → Features computed → Splits created
         ↓
BASILICA: Upload data to Volume → Deploy GPU training job → Run search
         ↓
LOCAL:   Download results → Score in emulator → Update leaderboard → Deploy
```

The key integration points:

1. **Data lives on Basilica Volumes** — upload once, reuse across training runs
2. **Model checkpoints on Volumes** — persist between deployments
3. **Training script is the same** — just runs on remote GPU instead of local
4. **Results come back to local** — for emulator scoring and leaderboard tracking
5. **Production miner runs locally** — Basilica is for training, not serving predictions

### Config Flag

Add to `config.py`:
```python
# Training compute backend
COMPUTE_BACKEND = "local"  # or "basilica"

# Basilica settings (only used if COMPUTE_BACKEND == "basilica")
BASILICA_GPU_COUNT = 1
BASILICA_MIN_GPU_MEMORY_GB = 16
BASILICA_GPU_MODELS = ["A100", "H100", "A4000"]
BASILICA_TTL_SECONDS = 14400  # 4 hours
BASILICA_DATA_VOLUME = "synth-training-data"
BASILICA_MODELS_VOLUME = "synth-model-registry"
```
