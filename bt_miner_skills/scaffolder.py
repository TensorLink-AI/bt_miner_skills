"""Scaffold a miner from a subnet config and selected skills.

Generates the initial miner code from templates, wired up with
the right protocol and skill components.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from bt_miner_skills.config.subnet_config import SubnetConfig
from bt_miner_skills.skills.registry import Skill, get_skills_for_task


TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def scaffold_miner(
    config: SubnetConfig,
    output_dir: Path,
    skills: list[Skill] | None = None,
    iteration: int = 0,
) -> list[Path]:
    """Generate miner code from templates.

    Args:
        config: The subnet configuration
        output_dir: Where to write the generated files
        skills: Skills to include (auto-selected if None)
        iteration: Current agent loop iteration

    Returns:
        List of generated file paths
    """
    if skills is None:
        skills = get_skills_for_task(config.task_type.value)

    output_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    generated = []

    # Generate main miner file
    synapse_class = config.protocol.synapse_class or f"SN{config.netuid}Synapse"
    template = env.get_template("miner_base.py.j2")
    miner_code = template.render(
        config=config,
        synapse_class=synapse_class,
        skills=skills,
        iteration=iteration,
    )
    miner_path = output_dir / "miner.py"
    miner_path.write_text(miner_code)
    generated.append(miner_path)

    # Generate skill files
    for skill in skills:
        try:
            skill_template = env.get_template(
                skill.template_file.replace("templates/", "")
            )
            skill_code = skill_template.render(config=config)
            skill_path = output_dir / f"skill_{skill.name}.py"
            skill_path.write_text(skill_code)
            generated.append(skill_path)
        except Exception as e:
            print(f"Warning: Could not generate skill {skill.name}: {e}")

    # Generate requirements.txt
    all_deps = set(config.python_dependencies)
    for skill in skills:
        all_deps.update(skill.dependencies)
    all_deps.add("bittensor>=8.0.0")
    req_path = output_dir / "requirements.txt"
    req_path.write_text("\n".join(sorted(all_deps)) + "\n")
    generated.append(req_path)

    return generated
