"""Skills registry - reusable mining patterns that agents can compose.

A "skill" is a reusable piece of miner logic: a model loader, an inference
pipeline, a response formatter, etc. Agents discover and compose skills
to build miners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """A reusable mining skill/pattern."""

    name: str
    description: str
    category: str  # e.g., "inference", "data", "optimization", "protocol"
    template_file: str  # Relative path to the Jinja2 template
    dependencies: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


# Built-in skills that agents can use
BUILTIN_SKILLS: dict[str, Skill] = {
    "axon_server": Skill(
        name="axon_server",
        description="Base Bittensor axon server setup with blacklist and priority",
        category="protocol",
        template_file="templates/skills/axon_server.py.j2",
        dependencies=["bittensor"],
        tags=["base", "networking"],
    ),
    "hf_model_loader": Skill(
        name="hf_model_loader",
        description="Load a model from HuggingFace Hub with caching and device management",
        category="inference",
        template_file="templates/skills/hf_model_loader.py.j2",
        dependencies=["transformers", "torch"],
        tags=["model", "huggingface"],
    ),
    "vllm_inference": Skill(
        name="vllm_inference",
        description="High-performance LLM inference using vLLM",
        category="inference",
        template_file="templates/skills/vllm_inference.py.j2",
        dependencies=["vllm"],
        tags=["llm", "inference", "gpu"],
    ),
    "streaming_response": Skill(
        name="streaming_response",
        description="Stream responses back to validators token-by-token",
        category="protocol",
        template_file="templates/skills/streaming_response.py.j2",
        dependencies=["bittensor"],
        tags=["streaming", "protocol"],
    ),
    "request_validator": Skill(
        name="request_validator",
        description="Validate incoming synapse requests and reject malformed ones",
        category="protocol",
        template_file="templates/skills/request_validator.py.j2",
        dependencies=["bittensor"],
        tags=["validation", "security"],
    ),
    "gpu_manager": Skill(
        name="gpu_manager",
        description="GPU memory management, model offloading, and batch scheduling",
        category="optimization",
        template_file="templates/skills/gpu_manager.py.j2",
        dependencies=["torch"],
        tags=["gpu", "optimization"],
    ),
    "response_cache": Skill(
        name="response_cache",
        description="LRU cache for repeated/similar requests to reduce latency",
        category="optimization",
        template_file="templates/skills/response_cache.py.j2",
        dependencies=[],
        tags=["cache", "optimization"],
    ),
}


def get_skills_for_task(task_type: str) -> list[Skill]:
    """Return skills relevant to a given task type."""
    task_skill_map: dict[str, list[str]] = {
        "text_generation": [
            "axon_server",
            "hf_model_loader",
            "vllm_inference",
            "streaming_response",
            "request_validator",
            "gpu_manager",
            "response_cache",
        ],
        "image_generation": [
            "axon_server",
            "hf_model_loader",
            "request_validator",
            "gpu_manager",
        ],
        "embedding": [
            "axon_server",
            "hf_model_loader",
            "request_validator",
            "response_cache",
        ],
        "inference": [
            "axon_server",
            "hf_model_loader",
            "request_validator",
            "gpu_manager",
            "response_cache",
        ],
    }
    skill_names = task_skill_map.get(task_type, ["axon_server", "request_validator"])
    return [BUILTIN_SKILLS[name] for name in skill_names if name in BUILTIN_SKILLS]


def list_skills() -> list[Skill]:
    """List all available skills."""
    return list(BUILTIN_SKILLS.values())
