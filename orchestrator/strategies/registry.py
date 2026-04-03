"""Strategy registry — maps strategy type names to implementations."""

from __future__ import annotations

from orchestrator.config import SubnetConfig
from orchestrator.strategies.base import Strategy

# Lazy imports to avoid loading unused dependencies
_STRATEGY_MAP = {
    "evoloop": "orchestrator.strategies.evoloop.EvoloopStrategy",
    "config_search": "orchestrator.strategies.config_search.ConfigSearchStrategy",
    "model_selection": "orchestrator.strategies.model_selection.ModelSelectionStrategy",
    "custom": "orchestrator.strategies.custom.CustomStrategy",
}


def get_strategy(config: SubnetConfig) -> Strategy:
    """Instantiate the correct strategy for a subnet config."""
    strategy_type = config.strategy.type

    if strategy_type not in _STRATEGY_MAP:
        available = ", ".join(_STRATEGY_MAP.keys())
        raise ValueError(
            f"Unknown strategy type: '{strategy_type}'. Available: {available}"
        )

    # Dynamic import
    module_path, class_name = _STRATEGY_MAP[strategy_type].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    strategy_class = getattr(module, class_name)

    return strategy_class(config)
