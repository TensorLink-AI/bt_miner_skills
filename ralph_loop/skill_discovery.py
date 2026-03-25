"""Discover and load bittensor miner skill packages from the repo."""

import glob
import json
import os
from dataclasses import dataclass, field

from ralph_loop.config import REPO_ROOT, SKILLS_DIRS_PATTERN


@dataclass
class SkillPackage:
    """A discovered miner skill package."""

    name: str
    path: str
    agent_prompt: str
    skill_doc: str
    netuid: int | None = None
    subnet_name: str = ""
    references: dict[str, str] = field(default_factory=dict)


def discover_skills(
    filter_subnet: str | None = None,
) -> list[SkillPackage]:
    """Find all skill packages in the repo and load their contents.

    Args:
        filter_subnet: If set, only return skills matching this subnet.
            Accepts a netuid (e.g. "50"), subnet name (e.g. "synth"),
            or package directory name (e.g. "synth-miner-package").
    """
    pattern = os.path.join(REPO_ROOT, SKILLS_DIRS_PATTERN)
    packages = []

    for pkg_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(pkg_dir):
            continue

        agent_prompt_path = os.path.join(pkg_dir, "AGENT_PROMPT.md")
        skill_doc_path = os.path.join(pkg_dir, "skill", "SKILL.md")

        if not os.path.exists(agent_prompt_path) or not os.path.exists(skill_doc_path):
            continue

        with open(agent_prompt_path, "r") as f:
            agent_prompt = f.read()
        with open(skill_doc_path, "r") as f:
            skill_doc = f.read()

        # Load subnet metadata if present
        netuid = None
        subnet_name = ""
        meta_path = os.path.join(pkg_dir, "subnet.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            netuid = meta.get("netuid")
            subnet_name = meta.get("name", "")

        # Load reference files
        refs = {}
        refs_dir = os.path.join(pkg_dir, "skill", "references")
        if os.path.isdir(refs_dir):
            for ref_file in sorted(os.listdir(refs_dir)):
                if ref_file.endswith(".md"):
                    ref_path = os.path.join(refs_dir, ref_file)
                    with open(ref_path, "r") as f:
                        refs[ref_file] = f.read()

        pkg = SkillPackage(
            name=os.path.basename(pkg_dir),
            path=pkg_dir,
            agent_prompt=agent_prompt,
            skill_doc=skill_doc,
            netuid=netuid,
            subnet_name=subnet_name,
            references=refs,
        )
        packages.append(pkg)

    if filter_subnet is not None:
        packages = _filter_skills(packages, filter_subnet)

    return packages


def _filter_skills(skills: list[SkillPackage], query: str) -> list[SkillPackage]:
    """Filter skills by netuid, subnet name, or package name."""
    query_lower = query.lower().strip()

    # Try matching as netuid number
    try:
        target_uid = int(query_lower)
        matched = [s for s in skills if s.netuid == target_uid]
        if matched:
            return matched
    except ValueError:
        pass

    # Match by subnet name or package name (substring)
    return [
        s for s in skills
        if query_lower in s.subnet_name.lower()
        or query_lower in s.name.lower()
    ]
