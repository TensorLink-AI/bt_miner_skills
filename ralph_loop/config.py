"""Configuration for the Ralph loop."""

import os

from dotenv import load_dotenv, find_dotenv

# Load .env file — find_dotenv() walks up from cwd to locate it
load_dotenv(find_dotenv(usecwd=True), override=False)

# Chutes AI configuration (OpenAI-compatible API)
CHUTES_API_KEY = os.environ.get("CHUTES_API_KEY", "")
CHUTES_BASE_URL = os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
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

# the-commons knowledge sharing
COMMONS_API_TOKEN = os.environ.get("COMMONS_API_TOKEN", "")
COMMONS_URL = os.environ.get("COMMONS_URL", "")  # e.g. https://commons.tensorlink.ai
