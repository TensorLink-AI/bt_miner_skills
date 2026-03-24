"""Subnet configuration schema.

This is the core data model that captures everything an agent needs to know
about a subnet in order to build a competitive miner.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """The type of work a subnet's miners are expected to perform."""

    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    EMBEDDING = "embedding"
    DATA_SCRAPING = "data_scraping"
    INFERENCE = "inference"
    TRAINING = "training"
    STORAGE = "storage"
    COMPUTE = "compute"
    CUSTOM = "custom"


class ScoringMechanism(str, Enum):
    """How validators score miner responses."""

    QUALITY = "quality"  # Output quality comparison
    SPEED = "speed"  # Latency-based
    ACCURACY = "accuracy"  # Ground-truth comparison
    COMPOSITE = "composite"  # Multi-factor
    CUSTOM = "custom"


class ProtocolSpec(BaseModel):
    """Describes the wire protocol between validators and miners."""

    synapse_class: str = Field(description="The Synapse subclass name used for requests")
    request_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Field name -> type mapping for the request",
    )
    response_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Field name -> type mapping for the expected response",
    )
    timeout_seconds: float = Field(default=12.0, description="Expected response timeout")
    notes: str = Field(default="", description="Free-form notes about the protocol")


class ScoringSpec(BaseModel):
    """Describes how miners are scored/ranked."""

    mechanism: ScoringMechanism = ScoringMechanism.COMPOSITE
    criteria: list[str] = Field(
        default_factory=list,
        description="Human-readable scoring criteria",
    )
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Criteria name -> weight mapping",
    )
    penalty_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that result in score penalties",
    )
    notes: str = Field(default="")


class HardwareRequirements(BaseModel):
    """Minimum hardware to run a competitive miner."""

    gpu: str = Field(default="none", description="e.g. 'A100 40GB', 'RTX 4090', 'none'")
    vram_gb: float = Field(default=0)
    ram_gb: float = Field(default=8)
    storage_gb: float = Field(default=50)
    cpu_cores: int = Field(default=4)
    bandwidth_mbps: float = Field(default=100)
    notes: str = Field(default="")


class ChainParams(BaseModel):
    """On-chain parameters pulled from subtensor."""

    netuid: int
    tempo: int = Field(default=360)
    immunity_period: int = Field(default=7200)
    max_allowed_validators: int = Field(default=128)
    min_allowed_weights: int = Field(default=1)
    max_weight_limit: int = Field(default=65535)
    difficulty: int = Field(default=10000000)
    burn_cost: str = Field(default="0")
    subnetwork_n: int = Field(default=256, description="Current number of neurons")
    max_n: int = Field(default=256, description="Max neurons allowed")
    kappa: int = Field(default=32767)
    rho: int = Field(default=10)
    emission_value: float = Field(default=0, description="Fraction of total emissions")


class SubnetConfig(BaseModel):
    """Complete configuration for a Bittensor subnet from the agent's perspective.

    This is the single source of truth that drives the agent loop:
    research -> build -> test -> deploy -> monitor -> improve.
    """

    # Identity
    netuid: int = Field(description="Subnet ID on the Bittensor network")
    name: str = Field(description="Human-readable subnet name")
    description: str = Field(default="", description="What this subnet does")
    repo_url: str = Field(default="", description="GitHub repo for the subnet code")
    docs_url: str = Field(default="", description="Documentation URL")

    # What the subnet does
    task_type: TaskType = TaskType.CUSTOM
    task_description: str = Field(
        default="",
        description="Detailed description of what miners must do",
    )

    # Protocol
    protocol: ProtocolSpec = Field(default_factory=ProtocolSpec)

    # Scoring
    scoring: ScoringSpec = Field(default_factory=ScoringSpec)

    # Hardware
    hardware: HardwareRequirements = Field(default_factory=HardwareRequirements)

    # Chain params (auto-populated from subtensor)
    chain_params: ChainParams | None = None

    # Dependencies & environment
    python_dependencies: list[str] = Field(
        default_factory=list,
        description="Additional pip packages needed",
    )
    model_dependencies: list[str] = Field(
        default_factory=list,
        description="HuggingFace model IDs or URLs needed",
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables the miner needs (keys only, no secrets)",
    )

    # Agent hints
    strategy_hints: list[str] = Field(
        default_factory=list,
        description="Tips for the agent on how to build a competitive miner",
    )
    known_pitfalls: list[str] = Field(
        default_factory=list,
        description="Common mistakes to avoid",
    )
    reference_implementations: list[str] = Field(
        default_factory=list,
        description="URLs to reference miner implementations",
    )

    # Extra
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Subnet-specific extra configuration",
    )
