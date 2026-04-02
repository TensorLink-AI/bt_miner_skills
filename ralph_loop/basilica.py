"""Basilica GPU compute tool definitions and execution.

Provides tool/function definitions for the LLM to call Basilica operations
instead of running GPU workloads locally. Also includes a guard that detects
and blocks local training attempts.
"""

import json
import logging
import os
import re
import subprocess
import time

logger = logging.getLogger(__name__)

# Allowed GPU types (cheap only)
ALLOWED_GPU_TYPES = {"A4000", "V100", "L40"}

# Patterns that indicate GPU/training workloads in shell commands
_LOCAL_TRAINING_PATTERNS = [
    re.compile(r"\bpython\b.*\btrain", re.IGNORECASE),
    re.compile(r"\bpython\b.*\bsearch\.py\b", re.IGNORECASE),
    re.compile(r"\btorch\.cuda\b", re.IGNORECASE),
    re.compile(r"\bimport\s+torch\b.*\btrain", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:model|net|network)\.train\(\)", re.IGNORECASE),
    re.compile(r"\bbackward\(\)", re.IGNORECASE),
    re.compile(r"\boptimizer\.step\(\)", re.IGNORECASE),
]

# Patterns that are safe even if they mention torch (e.g. torch.load for inference)
_SAFE_TORCH_PATTERNS = [
    re.compile(r"torch\.load\b", re.IGNORECASE),
    re.compile(r"torch\.no_grad\b", re.IGNORECASE),
    re.compile(r"model\.eval\(\)", re.IGNORECASE),
    re.compile(r"pip\s+install.*torch", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------

BASILICA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "basilica_submit_job",
            "description": (
                "Submit a GPU training job to Basilica. The job runs on a remote GPU "
                "instance. You must provide the Python script content, requirements, "
                "and GPU type. Returns a job ID for tracking. "
                "IMPORTANT: Use this instead of running training locally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script_content": {
                        "type": "string",
                        "description": (
                            "Full Python script content to execute on the GPU instance. "
                            "Must be self-contained — include all imports, data loading, "
                            "training logic, and result saving. Results should be saved "
                            "to files that can be retrieved."
                        ),
                    },
                    "requirements": {
                        "type": "string",
                        "description": (
                            "pip requirements (one per line) needed for the job. "
                            "Example: 'torch\\npandas\\nnumpy\\nscipy'"
                        ),
                    },
                    "gpu_type": {
                        "type": "string",
                        "enum": ["A4000", "V100", "L40"],
                        "description": "GPU type. Only cheap GPUs allowed: A4000, V100, L40.",
                    },
                    "job_name": {
                        "type": "string",
                        "description": "Human-readable name for this job (e.g. 'dlinear_v2_training').",
                    },
                    "timeout_minutes": {
                        "type": "integer",
                        "description": "Maximum runtime in minutes (default: 30).",
                        "default": 30,
                    },
                    "workspace_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of workspace file paths to upload with the job "
                            "(e.g. data files, config files the script needs)."
                        ),
                    },
                },
                "required": ["script_content", "gpu_type", "job_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "basilica_check_job",
            "description": (
                "Check the status of a Basilica GPU job. Returns status "
                "(queued/running/completed/failed), runtime, and output logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by basilica_submit_job.",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "basilica_fetch_results",
            "description": (
                "Download result files from a completed Basilica job back to the "
                "local workspace. Retrieves model checkpoints, score files, logs, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID of the completed job.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of file paths (relative to job workspace) to download. "
                            "Use ['*'] to download all output files."
                        ),
                    },
                    "destination": {
                        "type": "string",
                        "description": "Local directory to save files to (relative to workspace).",
                        "default": "basilica_results",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "basilica_list_jobs",
            "description": "List recent Basilica jobs and their statuses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of jobs to return (default: 10).",
                        "default": 10,
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": ["all", "running", "completed", "failed"],
                        "description": "Filter by job status.",
                        "default": "all",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

# In-memory job tracking (persisted via LoopState in production)
_jobs: dict[str, dict] = {}
_job_counter = 0


def execute_tool_call(
    tool_name: str,
    arguments: dict,
    workspace_dir: str,
) -> str:
    """Execute a Basilica tool call and return the result as a string.

    In a real deployment, these would call the Basilica SDK/API.
    Currently implements the scaffolding that:
    1. Writes the training script to workspace
    2. Attempts to call basilica CLI/SDK if available
    3. Falls back to recording the job for manual execution
    """
    if tool_name == "basilica_submit_job":
        return _submit_job(arguments, workspace_dir)
    elif tool_name == "basilica_check_job":
        return _check_job(arguments, workspace_dir)
    elif tool_name == "basilica_fetch_results":
        return _fetch_results(arguments, workspace_dir)
    elif tool_name == "basilica_list_jobs":
        return _list_jobs(arguments)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _submit_job(args: dict, workspace_dir: str) -> str:
    """Submit a training job to Basilica."""
    global _job_counter

    gpu_type = args.get("gpu_type", "A4000")
    if gpu_type not in ALLOWED_GPU_TYPES:
        return json.dumps({
            "error": f"GPU type '{gpu_type}' not allowed. Use one of: {', '.join(ALLOWED_GPU_TYPES)}",
        })

    job_name = args.get("job_name", "unnamed_job")
    script_content = args.get("script_content", "")
    requirements = args.get("requirements", "")
    timeout = args.get("timeout_minutes", 30)
    workspace_files = args.get("workspace_files", [])

    # Write the training script to a basilica jobs directory
    jobs_dir = os.path.join(workspace_dir, "basilica_jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    _job_counter += 1
    job_id = f"job_{_job_counter}_{job_name}_{int(time.time())}"
    job_dir = os.path.join(jobs_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Write script
    script_path = os.path.join(job_dir, "train.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    # Write requirements
    if requirements:
        req_path = os.path.join(job_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write(requirements)

    # Write job metadata
    job_meta = {
        "job_id": job_id,
        "job_name": job_name,
        "gpu_type": gpu_type,
        "timeout_minutes": timeout,
        "workspace_files": workspace_files,
        "submitted_at": time.time(),
        "status": "pending",
        "script_path": script_path,
    }

    meta_path = os.path.join(job_dir, "job_meta.json")
    with open(meta_path, "w") as f:
        json.dump(job_meta, f, indent=2)

    _jobs[job_id] = job_meta

    # Try to submit via basilica CLI if available
    result = _try_basilica_cli_submit(job_dir, gpu_type, timeout, workspace_dir)
    if result:
        job_meta["status"] = "submitted"
        job_meta["cli_output"] = result
        with open(meta_path, "w") as f:
            json.dump(job_meta, f, indent=2)

        return json.dumps({
            "job_id": job_id,
            "status": "submitted",
            "gpu_type": gpu_type,
            "message": f"Job submitted to Basilica ({gpu_type}). Use basilica_check_job to monitor.",
            "cli_output": result,
        })

    # Try the basilica Python SDK
    sdk_result = _try_basilica_sdk_submit(script_content, requirements, gpu_type, job_dir, workspace_dir)
    if sdk_result:
        job_meta["status"] = "submitted"
        job_meta["sdk_output"] = sdk_result
        with open(meta_path, "w") as f:
            json.dump(job_meta, f, indent=2)

        return json.dumps({
            "job_id": job_id,
            "status": "submitted",
            "gpu_type": gpu_type,
            "message": f"Job submitted via Basilica SDK ({gpu_type}).",
            "sdk_output": sdk_result,
        })

    # Fallback: job saved locally, needs manual submission or SDK setup
    job_meta["status"] = "saved_locally"
    with open(meta_path, "w") as f:
        json.dump(job_meta, f, indent=2)

    logger.warning(
        "Basilica CLI/SDK not available. Job saved to %s for manual submission.", job_dir
    )

    return json.dumps({
        "job_id": job_id,
        "status": "saved_locally",
        "gpu_type": gpu_type,
        "job_dir": job_dir,
        "message": (
            f"Job '{job_name}' saved to {job_dir}. "
            "Basilica CLI/SDK not found — install basilica-sdk or basilica-cli "
            "to enable automatic submission. Script is ready to run on any "
            f"{gpu_type} instance."
        ),
    })


def _try_basilica_cli_submit(
    job_dir: str, gpu_type: str, timeout: int, workspace_dir: str,
) -> str | None:
    """Try submitting via basilica CLI tool."""
    try:
        result = subprocess.run(
            ["which", "basilica"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        cmd = [
            "basilica", "run",
            "--gpu", gpu_type,
            "--timeout", str(timeout),
            "--script", os.path.join(job_dir, "train.py"),
        ]

        req_path = os.path.join(job_dir, "requirements.txt")
        if os.path.exists(req_path):
            cmd.extend(["--requirements", req_path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, cwd=workspace_dir,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning("basilica CLI failed: %s", result.stderr)
            return None

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _try_basilica_sdk_submit(
    script_content: str,
    requirements: str,
    gpu_type: str,
    job_dir: str,
    workspace_dir: str,
) -> str | None:
    """Try submitting via basilica Python SDK."""
    try:
        # Check common SDK locations
        sdk_paths = [
            "/mnt/skills/user/basilica-sdk",
            os.path.join(workspace_dir, "external_skills", "basilica-sdk"),
        ]

        sdk_path = None
        for p in sdk_paths:
            if os.path.isdir(p):
                sdk_path = p
                break

        if not sdk_path:
            # Try importing directly (might be pip installed)
            try:
                import importlib
                importlib.import_module("basilica")
            except ImportError:
                return None

        # Execute a submission script that uses the SDK
        submit_script = f"""
import sys
import json
sys.path.insert(0, "{sdk_path or ''}")

try:
    from basilica import BasilicaClient
    client = BasilicaClient()
    job = client.submit(
        script=open("{os.path.join(job_dir, 'train.py')}").read(),
        gpu_type="{gpu_type}",
    )
    print(json.dumps({{"job_id": str(job.id), "status": job.status}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
"""
        result = subprocess.run(
            ["python", "-c", submit_script],
            capture_output=True, text=True, timeout=30,
            cwd=workspace_dir,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    except Exception:
        return None


def _check_job(args: dict, workspace_dir: str) -> str:
    """Check status of a Basilica job."""
    job_id = args.get("job_id", "")

    # Check local metadata
    job_dir = os.path.join(workspace_dir, "basilica_jobs", job_id)
    meta_path = os.path.join(job_dir, "job_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Try checking via CLI
        try:
            result = subprocess.run(
                ["basilica", "status", job_id],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                meta["live_status"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return json.dumps(meta)

    return json.dumps({"error": f"Job '{job_id}' not found."})


def _fetch_results(args: dict, workspace_dir: str) -> str:
    """Fetch results from a completed Basilica job."""
    job_id = args.get("job_id", "")
    files = args.get("files", ["*"])
    destination = args.get("destination", "basilica_results")

    dest_dir = os.path.join(workspace_dir, destination, job_id)
    os.makedirs(dest_dir, exist_ok=True)

    # Try CLI fetch
    try:
        cmd = ["basilica", "fetch", job_id, "--output", dest_dir]
        if files != ["*"]:
            cmd.extend(["--files"] + files)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=workspace_dir,
        )
        if result.returncode == 0:
            return json.dumps({
                "status": "fetched",
                "destination": dest_dir,
                "output": result.stdout,
            })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check if results exist locally (from local job dir)
    job_dir = os.path.join(workspace_dir, "basilica_jobs", job_id)
    if os.path.isdir(job_dir):
        local_files = os.listdir(job_dir)
        return json.dumps({
            "status": "local_only",
            "job_dir": job_dir,
            "files_available": local_files,
            "message": "Results not yet fetched from remote. Job files available locally.",
        })

    return json.dumps({"error": f"Job '{job_id}' not found."})


def _list_jobs(args: dict) -> str:
    """List recent Basilica jobs."""
    limit = args.get("limit", 10)
    status_filter = args.get("status_filter", "all")

    jobs = list(_jobs.values())
    if status_filter != "all":
        jobs = [j for j in jobs if j.get("status") == status_filter]

    jobs = sorted(jobs, key=lambda j: j.get("submitted_at", 0), reverse=True)[:limit]

    return json.dumps({
        "total_jobs": len(_jobs),
        "showing": len(jobs),
        "jobs": [
            {
                "job_id": j["job_id"],
                "job_name": j["job_name"],
                "gpu_type": j["gpu_type"],
                "status": j["status"],
                "submitted_at": j.get("submitted_at"),
            }
            for j in jobs
        ],
    })


# ---------------------------------------------------------------------------
# Local training guard
# ---------------------------------------------------------------------------


def check_for_local_training(command: str) -> str | None:
    """Check if a shell command appears to run GPU training locally.

    Returns a warning message if local training is detected, None if safe.
    """
    # First check if it matches safe patterns (pip install, torch.load, etc.)
    for pattern in _SAFE_TORCH_PATTERNS:
        if pattern.search(command):
            return None

    # Then check for training patterns
    for pattern in _LOCAL_TRAINING_PATTERNS:
        if pattern.search(command):
            return (
                f"BLOCKED: This command appears to run GPU training locally. "
                f"Local training is not allowed — it will crash the sandbox. "
                f"Use the basilica_submit_job tool to run training on a remote GPU.\n"
                f"Command: {command[:200]}"
            )

    return None
