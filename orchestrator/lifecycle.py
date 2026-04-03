"""Lifecycle management — deploy and monitor are universal across subnets.

Each subnet provides its own deploy.sh and monitor.py, but the orchestrator
drives the lifecycle: deploy after search, monitor continuously, re-evolve
when performance degrades.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DeployResult:
    success: bool
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorResult:
    healthy: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    should_re_evolve: bool = False
    message: str = ""


def run_setup(subnet_dir: Path, workspace_dir: Path) -> bool:
    """Run the subnet's setup script to prepare prerequisites.

    setup.py should:
    - Validate data source access (APIs, credentials)
    - Build the eval harness (prepare.py for evoloop)
    - Verify the eval harness produces sane output
    - Return exit code 0 on success
    """
    setup_script = subnet_dir / "setup.py"
    if not setup_script.exists():
        print(f"[lifecycle] No setup.py found in {subnet_dir}, skipping setup phase.")
        return True

    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lifecycle] Running setup: {setup_script}")

    try:
        result = subprocess.run(
            ["python", str(setup_script)],
            cwd=str(subnet_dir),
            env={
                **__import__("os").environ,
                "WORKSPACE_DIR": str(workspace_dir),
                "SUBNET_DIR": str(subnet_dir),
            },
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[lifecycle] Setup failed with exit code {result.returncode}")
            return False
        print("[lifecycle] Setup complete.")
        return True
    except subprocess.TimeoutExpired:
        print("[lifecycle] Setup timed out after 600s.")
        return False
    except Exception as e:
        print(f"[lifecycle] Setup error: {e}")
        return False


def run_deploy(subnet_dir: Path, artifact_path: Path | None, workspace_dir: Path) -> DeployResult:
    """Run the subnet's deploy script to put a model/config into production.

    deploy.sh / deploy.py receives the artifact path as its first argument.
    It should handle whatever subnet-specific deployment is needed (PM2, Docker,
    bittensor registration, etc.).
    """
    # Try deploy.py first, then deploy.sh
    deploy_script = subnet_dir / "deploy.py"
    if not deploy_script.exists():
        deploy_script = subnet_dir / "deploy.sh"
    if not deploy_script.exists():
        return DeployResult(
            success=False,
            message=f"No deploy script found in {subnet_dir}",
        )

    artifact_arg = str(artifact_path) if artifact_path else ""
    cmd = (
        ["python", str(deploy_script), artifact_arg]
        if deploy_script.suffix == ".py"
        else ["bash", str(deploy_script), artifact_arg]
    )

    print(f"[lifecycle] Deploying: {deploy_script}")
    if artifact_path:
        print(f"[lifecycle] Artifact: {artifact_path}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(subnet_dir),
            env={
                **__import__("os").environ,
                "WORKSPACE_DIR": str(workspace_dir),
                "ARTIFACT_PATH": artifact_arg,
            },
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return DeployResult(
                success=False,
                message=f"Deploy failed (exit {result.returncode}): {result.stderr[:500]}",
            )

        # Try to parse metadata from stdout
        metadata = {}
        stdout_lines = result.stdout.strip().split("\n")
        if stdout_lines:
            try:
                metadata = json.loads(stdout_lines[-1])
            except json.JSONDecodeError:
                pass

        return DeployResult(
            success=True,
            message="Deployment complete.",
            metadata=metadata,
        )

    except subprocess.TimeoutExpired:
        return DeployResult(success=False, message="Deploy timed out after 300s.")
    except Exception as e:
        return DeployResult(success=False, message=f"Deploy error: {e}")


def run_monitor(subnet_dir: Path, workspace_dir: Path) -> MonitorResult:
    """Run the subnet's monitor script to check live performance.

    monitor.py should print a JSON object to stdout with at least:
    - "healthy": bool
    - "metrics": dict of current performance metrics
    - "should_re_evolve": bool (optional, triggers re-search)
    """
    monitor_script = subnet_dir / "monitor.py"
    if not monitor_script.exists():
        return MonitorResult(healthy=True, message="No monitor.py — skipping.")

    try:
        result = subprocess.run(
            ["python", str(monitor_script)],
            cwd=str(subnet_dir),
            env={
                **__import__("os").environ,
                "WORKSPACE_DIR": str(workspace_dir),
            },
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return MonitorResult(
                healthy=False,
                message=f"Monitor failed (exit {result.returncode}): {result.stderr[:500]}",
            )

        # Parse monitor output
        stdout_lines = result.stdout.strip().split("\n")
        if stdout_lines:
            try:
                data = json.loads(stdout_lines[-1])
                return MonitorResult(
                    healthy=data.get("healthy", True),
                    metrics=data.get("metrics", {}),
                    should_re_evolve=data.get("should_re_evolve", False),
                    message=data.get("message", ""),
                )
            except json.JSONDecodeError:
                return MonitorResult(
                    healthy=True,
                    message=f"Monitor output not JSON: {stdout_lines[-1][:200]}",
                )

        return MonitorResult(healthy=True, message="Monitor produced no output.")

    except subprocess.TimeoutExpired:
        return MonitorResult(healthy=False, message="Monitor timed out after 120s.")
    except Exception as e:
        return MonitorResult(healthy=False, message=f"Monitor error: {e}")
