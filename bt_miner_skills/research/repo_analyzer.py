"""Analyze a subnet's GitHub repo to extract protocol and reward function details.

Clones the repo (shallow) and searches for key patterns: Synapse definitions,
reward/scoring logic, and miner forward() implementations.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RepoAnalysis:
    """Results of analyzing a subnet repository."""

    repo_url: str
    synapse_classes: list[dict] = field(default_factory=list)
    reward_functions: list[dict] = field(default_factory=list)
    miner_forward_methods: list[dict] = field(default_factory=list)
    protocol_files: list[str] = field(default_factory=list)
    reward_files: list[str] = field(default_factory=list)
    miner_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    raw_excerpts: dict[str, str] = field(default_factory=dict)


def clone_repo(repo_url: str, dest: Path | None = None) -> Path:
    """Shallow clone a repo. Returns the path to the cloned directory."""
    if dest is None:
        dest = Path(tempfile.mkdtemp(prefix="bt_repo_"))
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return dest


def find_synapse_classes(repo_path: Path) -> list[dict]:
    """Find Synapse subclass definitions in the repo."""
    results = []
    for py_file in repo_path.rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")
        except Exception:
            continue
        # Match class definitions that inherit from Synapse
        for match in re.finditer(
            r"class\s+(\w+)\s*\(\s*(?:\w+\.)?Synapse\s*\).*?(?=\nclass |\Z)",
            content,
            re.DOTALL,
        ):
            class_name = match.group(1)
            class_body = match.group(0)[:2000]  # Cap size
            # Extract fields (Pydantic-style)
            fields = re.findall(
                r"(\w+)\s*:\s*([\w\[\], |]+)(?:\s*=\s*(.+))?",
                class_body,
            )
            results.append(
                {
                    "class_name": class_name,
                    "file": str(py_file.relative_to(repo_path)),
                    "fields": [
                        {"name": f[0], "type": f[1], "default": f[2] or None}
                        for f in fields
                        if f[0] not in ("class", "def", "self")
                    ],
                    "source": class_body,
                }
            )
    return results


def find_reward_functions(repo_path: Path) -> list[dict]:
    """Find reward/scoring functions in the repo."""
    results = []
    reward_patterns = [
        r"def\s+(reward|score|get_rewards?|calculate_rewards?)\s*\(",
        r"class\s+(\w*[Rr]eward\w*)\s*[:\(]",
    ]
    for py_file in repo_path.rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")
        except Exception:
            continue
        for pattern in reward_patterns:
            for match in re.finditer(pattern, content):
                # Grab surrounding context (up to 3000 chars after match start)
                start = max(0, match.start() - 100)
                excerpt = content[start : match.start() + 3000]
                results.append(
                    {
                        "name": match.group(1),
                        "file": str(py_file.relative_to(repo_path)),
                        "excerpt": excerpt,
                    }
                )
    return results


def find_miner_forward(repo_path: Path) -> list[dict]:
    """Find miner forward() method implementations."""
    results = []
    for py_file in repo_path.rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")
        except Exception:
            continue
        for match in re.finditer(
            r"(async\s+)?def\s+forward\s*\(self.*?\).*?(?=\n    def |\n    async def |\nclass |\Z)",
            content,
            re.DOTALL,
        ):
            results.append(
                {
                    "file": str(py_file.relative_to(repo_path)),
                    "is_async": bool(match.group(1)),
                    "source": match.group(0)[:3000],
                }
            )
    return results


def extract_dependencies(repo_path: Path) -> list[str]:
    """Extract Python dependencies from requirements files."""
    deps = []
    for req_file in ["requirements.txt", "requirements/main.txt", "requirements/base.txt"]:
        path = repo_path / req_file
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    deps.append(line)
    # Also check pyproject.toml
    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        # Simple extraction of dependencies list
        in_deps = False
        for line in content.splitlines():
            if "dependencies" in line and "[" in line:
                in_deps = True
                continue
            if in_deps:
                if "]" in line:
                    in_deps = False
                    continue
                dep = line.strip().strip('",')
                if dep:
                    deps.append(dep)
    return deps


def analyze_repo(repo_url: str) -> RepoAnalysis:
    """Full analysis of a subnet repository.

    Clones the repo, searches for protocol, reward, and miner patterns,
    then returns structured results the agent can use to build a config.
    """
    repo_path = clone_repo(repo_url)
    try:
        synapses = find_synapse_classes(repo_path)
        rewards = find_reward_functions(repo_path)
        forwards = find_miner_forward(repo_path)
        deps = extract_dependencies(repo_path)

        return RepoAnalysis(
            repo_url=repo_url,
            synapse_classes=synapses,
            reward_functions=rewards,
            miner_forward_methods=forwards,
            protocol_files=list({s["file"] for s in synapses}),
            reward_files=list({r["file"] for r in rewards}),
            miner_files=list({f["file"] for f in forwards}),
            dependencies=deps,
        )
    finally:
        # Clean up
        import shutil

        shutil.rmtree(repo_path, ignore_errors=True)
