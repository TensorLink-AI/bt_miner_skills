"""Configuration loading for subnet packages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvoloopConfig:
    """Settings passed to evoloop when using the evoloop strategy."""

    task_dir: str = "evoloop_task/"
    backend: str = "basilica"
    gpu: str = "A4000"
    gpu_count: int = 1
    time_budget: int = 600
    max_experiments: int = 0  # 0 = unlimited
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1"
    llm_model_strong: str = "o3"


@dataclass
class ConvergenceConfig:
    """When to stop the search strategy."""

    stale_threshold: int = 15  # stop after N iterations with no improvement
    min_experiments: int = 10  # always run at least this many
    max_experiments: int = 0   # hard ceiling (0 = use strategy default)


@dataclass
class CompetitivenessConfig:
    """Quality gates before deployment."""

    baseline_improvement: float = 0.15  # must beat baseline by this fraction
    min_asset_coverage: float = 0.70    # must beat median on this fraction of assets
    consistency_window_hours: int = 48  # stable over this many hours


@dataclass
class MonitorConfig:
    """Post-deployment monitoring settings."""

    check_interval_minutes: int = 30
    re_evolve_trigger: str = ""  # expression evaluated against live metrics


@dataclass
class StrategyConfig:
    """Strategy selection and its type-specific config."""

    type: str = "evoloop"  # evoloop | config_search | model_selection | custom
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubnetConfig:
    """Full configuration for a subnet miner."""

    name: str = ""
    netuid: int = 0
    network: str = "mainnet"

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    competitiveness: CompetitivenessConfig = field(default_factory=CompetitivenessConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    # Paths (set after loading)
    subnet_dir: Path = field(default_factory=Path)
    workspace_dir: Path = field(default_factory=Path)


def load_subnet_config(subnet_dir: str | Path) -> SubnetConfig:
    """Load a SubnetConfig from a subnet directory's subnet.yaml."""
    subnet_dir = Path(subnet_dir)
    config_path = subnet_dir / "subnet.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"No subnet.yaml found in {subnet_dir}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Build config from yaml
    subnet = raw.get("subnet", {})
    strategy_raw = raw.get("strategy", {})
    convergence_raw = raw.get("convergence", {})
    competitiveness_raw = raw.get("competitiveness", {})
    monitor_raw = raw.get("monitor", {})

    config = SubnetConfig(
        name=subnet.get("name", subnet_dir.name),
        netuid=subnet.get("netuid", 0),
        network=subnet.get("network", "mainnet"),
        strategy=StrategyConfig(
            type=strategy_raw.get("type", "evoloop"),
            config=strategy_raw.get("config", {}),
        ),
        convergence=ConvergenceConfig(
            stale_threshold=convergence_raw.get("stale_threshold", 15),
            min_experiments=convergence_raw.get("min_experiments", 10),
            max_experiments=convergence_raw.get("max_experiments", 0),
        ),
        competitiveness=CompetitivenessConfig(
            baseline_improvement=competitiveness_raw.get("baseline_improvement", 0.15),
            min_asset_coverage=competitiveness_raw.get("min_asset_coverage", 0.70),
            consistency_window_hours=competitiveness_raw.get(
                "consistency_window_hours", 48
            ),
        ),
        monitor=MonitorConfig(
            check_interval_minutes=monitor_raw.get("check_interval_minutes", 30),
            re_evolve_trigger=monitor_raw.get("re_evolve_trigger", ""),
        ),
        subnet_dir=subnet_dir.resolve(),
        workspace_dir=Path(
            os.environ.get(
                "ORCHESTRATOR_WORKSPACE",
                subnet_dir / "workspace",
            )
        ).resolve(),
    )

    return config
