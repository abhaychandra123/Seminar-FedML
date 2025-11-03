"""Client selection strategies for Federated Learning."""

from fl_sim.strategies.base_strategy import BaseSelectionStrategy
from fl_sim.strategies.selection_strategies import (
    # Data sample-based
    NumSamplesStrategy,
    HighGradientNormStrategy,
    HighLossStrategy,
    SumLossStrategy,
    # Model-based
    ModelDivergenceStrategy,
    SameSignStrategy,
    UpdateDirectionStrategy,
    LocalChangeStrategy,
    PostGradientNormStrategy,
    GradientDotProductStrategy,
    # System-based
    HardDeadlineStrategy,
    SoftDeadlineStrategy,
    # Baseline
    RandomStrategy,
    # Registry
    STRATEGY_REGISTRY,
    get_strategy,
)

__all__ = [
    "BaseSelectionStrategy",
    "NumSamplesStrategy",
    "HighGradientNormStrategy",
    "HighLossStrategy",
    "SumLossStrategy",
    "ModelDivergenceStrategy",
    "SameSignStrategy",
    "UpdateDirectionStrategy",
    "LocalChangeStrategy",
    "PostGradientNormStrategy",
    "GradientDotProductStrategy",
    "HardDeadlineStrategy",
    "SoftDeadlineStrategy",
    "RandomStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]