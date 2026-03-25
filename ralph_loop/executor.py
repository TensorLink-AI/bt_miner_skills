"""Code execution engine for the Ralph loop.

Parses fenced code blocks from LLM responses, executes them in order
(bash → python file writes → run commands), and captures output.
"""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 120  # seconds per command
MAX_OUTPUT_CHARS = 8000  # truncate captured output beyond this


@dataclass
class ExecResult:
    """Result of a single execution step."""

    block_type: str  # "bash", "python", "run"
    command: str  # the command or filename
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    error: str = ""  # internal error (timeout, path traversal, etc.)


@dataclass
class ExecutionReport:
    """Aggregated results from executing all blocks in one LLM response."""

    results: list[ExecResult] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)

    def format_for_llm(self) -> str:
        """Format execution results as context for the next LLM iteration."""
        if not self.results:
            return "No code blocks were found in your last response."

        parts = ["## Execution Output from Last Iteration\n"]
        for i, r in enumerate(self.results, 1):
            parts.append(f"### Step {i}: `{r.block_type}` — {r.command[:80]}")
            if r.error:
                parts.append(f"**ERROR:** {r.error}")
            else:
                parts.append(f"Exit code: {r.exit_code} | Duration: {r.duration:.1f}s")
                if r.stdout.strip():
                    out = _truncate(r.stdout.strip(), MAX_OUTPUT_CHARS // 2)
                    parts.append(f"stdout:\n```\n{out}\n```")
                if r.stderr.strip():
                    err = _truncate(r.stderr.strip(), MAX_OUTPUT_CHARS // 2)
                    parts.append(f"stderr:\n```\n{err}\n```")
                if not r.stdout.strip() and not r.stderr.strip():
                    parts.append("(no output)")
            parts.append("")

        if self.files_written:
            parts.append(f"**Files written:** {', '.join(self.files_written)}")
        if self.commands_run:
            parts.append(f"**Commands run:** {len(self.commands_run)}")

        return "\n".join(parts)


def parse_code_blocks(response: str) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Parse fenced code blocks from the LLM response.

    Returns:
        (bash_blocks, python_blocks, run_blocks)
        python_blocks are (filename, content) tuples.
    """
    bash_blocks: list[str] = []
    python_blocks: list[tuple[str, str]] = []
    run_blocks: list[str] = []

    pattern = re.compile(r"```(bash|python|run)\s*\n(.*?)```", re.DOTALL)

    for match in pattern.finditer(response):
        lang = match.group(1)
        content = match.group(2).rstrip("\n")

        if lang == "bash":
            bash_blocks.append(content)
        elif lang == "python":
            filename, body = _extract_filename(content)
            if filename:
                python_blocks.append((filename, body))
            else:
                logger.warning("Python block missing # FILENAME: directive, skipping")
        elif lang == "run":
            run_blocks.append(content)

    return bash_blocks, python_blocks, run_blocks


def execute_response(response: str, workspace_dir: str) -> ExecutionReport:
    """Parse and execute all code blocks from an LLM response.

    Execution order: bash blocks → python file writes → run blocks.
    """
    os.makedirs(workspace_dir, exist_ok=True)
    bash_blocks, python_blocks, run_blocks = parse_code_blocks(response)
    report = ExecutionReport()

    # 1. Execute bash blocks
    for block in bash_blocks:
        result = _run_shell(block, workspace_dir, "bash")
        report.results.append(result)
        report.commands_run.append(f"bash: {block[:60]}")

    # 2. Write python files
    for filename, content in python_blocks:
        result = _write_file(filename, content, workspace_dir)
        report.results.append(result)
        if not result.error:
            report.files_written.append(filename)

    # 3. Execute run blocks
    for block in run_blocks:
        result = _run_shell(block, workspace_dir, "run")
        report.results.append(result)
        report.commands_run.append(f"run: {block[:60]}")

    return report


def get_workspace_snapshot(workspace_dir: str, max_files: int = 100) -> str:
    """Get a listing of files in the workspace for LLM context."""
    if not os.path.isdir(workspace_dir):
        return "(workspace directory does not exist yet)"

    files = []
    for root, dirs, filenames in os.walk(workspace_dir):
        # Skip hidden dirs and common noise
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(root, fname), workspace_dir)
            try:
                size = os.path.getsize(os.path.join(root, fname))
                files.append(f"  {rel} ({_human_size(size)})")
            except OSError:
                files.append(f"  {rel}")

        if len(files) >= max_files:
            files.append(f"  ... and more (truncated at {max_files} files)")
            break

    if not files:
        return "(workspace is empty)"
    return "## Workspace Files\n```\n" + "\n".join(files) + "\n```"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_filename(content: str) -> tuple[str, str]:
    """Extract '# FILENAME: ...' from the first line of a python block."""
    lines = content.split("\n", 1)
    first_line = lines[0].strip()

    match = re.match(r"^#\s*FILENAME:\s*(.+)$", first_line)
    if match:
        filename = match.group(1).strip()
        body = lines[1] if len(lines) > 1 else ""
        return filename, body
    return "", content


def _validate_path(filename: str, workspace_dir: str) -> str | None:
    """Validate a file path is safe (no traversal, stays in workspace).

    Returns the resolved absolute path, or None if invalid.
    """
    # Block absolute paths
    if os.path.isabs(filename):
        return None

    # Normalize and resolve
    target = os.path.normpath(os.path.join(workspace_dir, filename))
    workspace_resolved = os.path.realpath(workspace_dir)

    # Must stay within workspace
    if not target.startswith(workspace_resolved + os.sep) and target != workspace_resolved:
        return None

    return target


def _write_file(filename: str, content: str, workspace_dir: str) -> ExecResult:
    """Write a python file to the workspace."""
    result = ExecResult(block_type="python", command=f"write {filename}")

    target = _validate_path(filename, workspace_dir)
    if target is None:
        result.error = f"Path traversal blocked: {filename}"
        result.exit_code = 1
        logger.warning("Path traversal attempt blocked: %s", filename)
        return result

    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write(content)
        result.stdout = f"Written: {filename} ({len(content)} bytes)"
        logger.info("Wrote file: %s", target)
    except OSError as e:
        result.error = str(e)
        result.exit_code = 1

    return result


def _run_shell(command: str, workspace_dir: str, block_type: str) -> ExecResult:
    """Execute a shell command in the workspace directory."""
    result = ExecResult(block_type=block_type, command=command[:120])
    start = time.monotonic()

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )
        result.exit_code = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
    except subprocess.TimeoutExpired:
        result.error = f"Timed out after {EXECUTION_TIMEOUT}s"
        result.exit_code = 124
    except Exception as e:
        result.error = str(e)
        result.exit_code = 1

    result.duration = time.monotonic() - start
    logger.info(
        "Executed [%s] (exit=%d, %.1fs): %s",
        block_type, result.exit_code, result.duration, command[:80],
    )
    return result


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text with an indicator if too long."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"


def _human_size(nbytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"
