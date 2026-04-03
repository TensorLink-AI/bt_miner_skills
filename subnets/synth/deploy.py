"""Synth subnet deployment — promote a model artifact to production.

This script handles:
1. Copying the model artifact to the model registry
2. Updating the production symlink for hot-swap
3. Validating the miner format against Synth API
4. Starting/restarting the PM2 miner process

Usage:
    python deploy.py <artifact_path>

Outputs JSON on the last line with deployment metadata.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def deploy(artifact_path: str) -> dict:
    """Deploy a model artifact to production."""
    workspace = Path(os.environ.get("WORKSPACE_DIR", Path(__file__).parent / "workspace"))
    registry = workspace / "model_registry"

    # Create registry structure
    for stage in ["candidates", "live_testing", "production", "retired"]:
        (registry / stage).mkdir(parents=True, exist_ok=True)

    if not artifact_path:
        return {"success": False, "error": "No artifact path provided."}

    artifact = Path(artifact_path)
    if not artifact.exists():
        return {"success": False, "error": f"Artifact not found: {artifact}"}

    # Generate model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"synth_model_{timestamp}"

    # Copy artifact to production
    model_dir = registry / "production" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    if artifact.is_dir():
        shutil.copytree(artifact, model_dir, dirs_exist_ok=True)
    else:
        shutil.copy2(artifact, model_dir / artifact.name)

    # Update current symlink
    current_link = registry / "production" / "current"
    if current_link.is_symlink() or current_link.exists():
        current_link.unlink()
    current_link.symlink_to(model_dir)

    print(f"[deploy] Model {model_id} promoted to production.")
    print(f"[deploy] Registry: {registry / 'production'}")
    print(f"[deploy] Current -> {model_dir}")

    # TODO: Start/restart PM2 miner process
    # subprocess.run(["pm2", "restart", "synth-miner"], check=False)

    # TODO: Validate via Synth API
    # GET /validation/miner?uid=<UID> should return validated: true

    result = {
        "success": True,
        "model_id": model_id,
        "model_dir": str(model_dir),
        "registry": str(registry),
    }

    # Output JSON for orchestrator to parse
    print(json.dumps(result))
    return result


def main() -> None:
    artifact_path = sys.argv[1] if len(sys.argv) > 1 else ""
    result = deploy(artifact_path)
    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
