"""Discover and load bittensor miner skill packages from the repo."""

import glob
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
    references: dict[str, str] = field(default_factory=dict)


def discover_skills() -> list[SkillPackage]:
    """Find all skill packages in the repo and load their contents."""
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
            references=refs,
        )
        packages.append(pkg)

    return packages
