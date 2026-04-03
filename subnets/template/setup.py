"""Template setup script — validate prerequisites before the search strategy runs.

Customize this for your subnet. Common checks:
- Data source accessibility (APIs, credentials)
- Required packages installed
- Eval harness produces sane output
- Compute backend available (Basilica token, local GPU, etc.)

Exit 0 = ready. Exit 1 = not ready.
"""

from __future__ import annotations

import sys


def main() -> None:
    print("[setup] Running setup checks...")

    # TODO: Add your subnet-specific checks here
    # Example:
    # - check_api_access()
    # - check_dependencies()
    # - validate_eval_harness()
    # - check_compute_backend()

    print("[setup] All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
