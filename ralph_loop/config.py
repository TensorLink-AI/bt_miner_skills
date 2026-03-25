"""Configuration for the Ralph loop."""

import os
from pathlib import Path

# Load .env file from repo root if present
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.is_file():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip("\"'")
            if _key and _key not in os.environ:  # don't override existing env vars
                os.environ[_key] = _val

# Chutes AI configuration (OpenAI-compatible API)
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")
CHUTES_BASE_URL = "https://api.chutes.ai/v1"
CHUTES_MODEL = os.environ.get("CHUTES_MODEL", "deepseek-ai/DeepSeek-V3-0324")

# Loop configuration
MAX_ITERATIONS = int(os.environ.get("RALPH_MAX_ITERATIONS", "0"))  # 0 = infinite
MAX_TOKENS_PER_TURN = int(os.environ.get("RALPH_MAX_TOKENS", "4096"))

# Skill discovery
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIRS_PATTERN = "*-package"  # e.g. synth-miner-package

# State persistence
STATE_DIR = os.path.join(REPO_ROOT, ".ralph_state")

# Workspace — where generated miner code is written and executed
WORKSPACE_ROOT = os.environ.get(
    "RALPH_WORKSPACE_ROOT",
    os.path.join(REPO_ROOT, "workspace"),
)
