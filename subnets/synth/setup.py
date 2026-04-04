"""Synth subnet setup — validate orchestrator prerequisites.

Only checks what the orchestrator needs to launch evoloop. Data pipelines,
dependencies, and training infra are evoloop's responsibility.

Exit 0 = ready to proceed to search phase.
Exit 1 = something is broken, fix before continuing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def check_evoloop_task_files() -> bool:
    """Verify the evoloop task directory has the required files."""
    subnet_dir = Path(os.environ.get("SUBNET_DIR", Path(__file__).parent))
    task_dir = subnet_dir / "evoloop_task"

    required = ["task.yaml"]

    ok = True
    for f in required:
        path = task_dir / f
        if not path.exists():
            print(f"[setup] MISSING required file: {path}")
            ok = False
        else:
            print(f"[setup] Found: {path}")

    return ok


def check_basilica() -> bool:
    """Check that Basilica credentials are available."""
    token = os.environ.get("BASILICA_API_TOKEN", "")
    if not token:
        print("[setup] WARNING: BASILICA_API_TOKEN not set.")
        print("[setup]   GPU training on Basilica will fail without this.")
        return True  # Warn but don't fail — might be using local backend

    print("[setup] Basilica token: SET")
    return True


def main() -> None:
    print("=" * 50)
    print("  Synth Subnet (SN50) — Setup Check")
    print("=" * 50)
    print()

    checks = [
        ("Evoloop task files", check_evoloop_task_files),
        ("Basilica compute", check_basilica),
    ]

    all_ok = True
    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        if not check_fn():
            all_ok = False

    print()
    if all_ok:
        print("[setup] All checks passed. Ready for search phase.")
        sys.exit(0)
    else:
        print("[setup] Some checks failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
