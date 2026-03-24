"""Load and validate subnet configurations from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from .subnet_config import SubnetConfig


def load_subnet_config(path: str | Path) -> SubnetConfig:
    """Load a subnet config from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return SubnetConfig(**data)


def load_all_configs(config_dir: str | Path = "subnet_configs") -> dict[int, SubnetConfig]:
    """Load all subnet configs from a directory, keyed by netuid."""
    config_dir = Path(config_dir)
    configs = {}
    if not config_dir.exists():
        return configs
    for path in sorted(config_dir.glob("*.yaml")):
        config = load_subnet_config(path)
        configs[config.netuid] = config
    return configs
