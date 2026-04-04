"""Synth subnet setup — validate prerequisites before evoloop runs.

This script:
1. Checks that numpy is available (used by prepare.py for synthetic data + CRPS)
2. Validates that evoloop task files exist (task.yaml, train.py, prepare.py)
3. Verifies the Synth API is reachable for post-search benchmarking

Exit 0 = ready to proceed to search phase.
Exit 1 = something is broken, fix before continuing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def check_dependencies() -> bool:
    """Verify that required Python packages are installed."""
    ok = True

    # numpy is required by prepare.py (CRPS scoring, synthetic data)
    try:
        import importlib
        importlib.import_module("numpy")
        print("[setup] numpy: OK")
    except ImportError:
        print("[setup] numpy: NOT INSTALLED (pip install numpy)")
        ok = False

    return ok


def check_evoloop_task_files() -> bool:
    """Verify the evoloop task directory has the required files."""
    subnet_dir = Path(os.environ.get("SUBNET_DIR", Path(__file__).parent))
    task_dir = subnet_dir / "evoloop_task"

    required = ["task.yaml"]
    recommended = ["train.py", "prepare.py"]

    ok = True
    for f in required:
        path = task_dir / f
        if not path.exists():
            print(f"[setup] MISSING required file: {path}")
            ok = False
        else:
            print(f"[setup] Found: {path}")

    for f in recommended:
        path = task_dir / f
        if not path.exists():
            print(f"[setup] WARNING: recommended file missing: {path}")
            print(f"[setup]   evoloop will need to create {f} from scratch.")
        else:
            print(f"[setup] Found: {path}")

    return ok


def check_synth_api() -> bool:
    """Check that the Synth API is reachable (for benchmarking)."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.synthdata.co/v2/leaderboard/latest",
            headers={"User-Agent": "synth-setup"},
        )
        urllib.request.urlopen(req, timeout=10)
        print("[setup] Synth API: OK")
        return True
    except Exception as e:
        print(f"[setup] Synth API: UNREACHABLE ({e})")
        print("[setup]   Live benchmarking won't work, but search can still run.")
        return True  # Not fatal


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
        ("Dependencies", check_dependencies),
        ("Evoloop task files", check_evoloop_task_files),
        ("Synth API", check_synth_api),
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
