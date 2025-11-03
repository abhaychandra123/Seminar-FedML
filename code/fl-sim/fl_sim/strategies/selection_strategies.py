"""Concrete implementations of 12 client selection strategies."""

import numpy as np
from typing import Dict
from fl_sim.strategies.base_strategy import BaseSelectionStrategy


# ============================================================================
# DATA SAMPLE-BASED STRATEGIES (Statistical Utility)
# ============================================================================

class NumSamplesStrategy(BaseSelectionStrategy):
    """APP-ONE (Eq. 2): Select clients based on number of data samples."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Higher number of examples = higher utility."""
        return metrics.get("num_examples", 0)


class HighGradientNormStrategy(BaseSelectionStrategy):
    """APP-TWO (Eq. 3): Select clients with high L2 norm of gradients (pre-training)."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Higher gradient norm = more divergent = higher utility."""
        # Uses pre_grad_norm (∇w_i)
        return metrics.get("pre_grad_norm", 0)


class HighLossStrategy(BaseSelectionStrategy):
    """APP-THREE (Note-based): Select clients with high AVERAGE loss."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Higher average loss = more important = higher utility.
        This is distinct from SumLossStrategy (APP-FOUR).
        We use eval_loss as it's a more stable metric collected from all clients.
        """
        return metrics.get("eval_loss", 0)


class SumLossStrategy(BaseSelectionStrategy):
    """APP-FOUR (Eq. 5): Select clients based on sum of losses (TOTAL loss)."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Total loss = higher utility."""
        loss = metrics.get("eval_loss", 0) 
        num_examples = metrics.get("num_examples", 1)
        # Return total loss across all samples
        return loss * num_examples


# ============================================================================
# MODEL-BASED STRATEGIES (Statistical Utility)
# ============================================================================

class ModelDivergenceStrategy(BaseSelectionStrategy):
    """APP-FIVE (Eq. 6): Select clients with high model divergence (L2 distance)."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Higher divergence = more different = higher utility."""
        # This is || w_local_final - w_global_initial ||
        divergence = metrics.get("model_divergence", 0)
        return divergence


class SameSignStrategy(BaseSelectionStrategy):
    """APP-SIX (Eq. 7): Select clients with lower same-sign weight percentage."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Lower same-sign percentage = more diverse = higher utility."""
        same_sign_pct = metrics.get("same_sign_percentage", 1.0)
        # Return inverse: lower percentage = higher utility
        return 1.0 - same_sign_pct


class UpdateDirectionStrategy(BaseSelectionStrategy):
    """APP-SEVEN (Eq. 8): Select clients based on gradient cosine similarity."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Higher cosine similarity (alignment) = higher utility."""
        # This is cos(θ) ≈ <∇w_i, ∇w_bar> / (||∇w_i|| * ||∇w_bar||)
        dot_product = metrics.get("grad_dot_product", 0)
        local_grad_norm = metrics.get("pre_grad_norm", 1.0)
        global_update_norm = metrics.get("global_update_norm", 1.0)
        
        denominator = (local_grad_norm * global_update_norm) + 1e-6
        return dot_product / denominator


class LocalChangeStrategy(BaseSelectionStrategy):
    """APP-EIGHT (Eq. 9): Select clients with large change in local model."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Larger local change = more learning = higher utility."""
        # This is || w_local_final - w_local_initial ||
        return metrics.get("update_magnitude", 0)


class PostGradientNormStrategy(BaseSelectionStrategy):
    """APP-NINE (Eq. 10): Select clients based on L2-norm of gradients (post-training)."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Higher post-training gradient norm = higher utility."""
        return metrics.get("post_grad_norm", 0)


class GradientDotProductStrategy(BaseSelectionStrategy):
    """APP-TEN (Eq. 11): Select clients with positive gradient dot product."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Utility = <∇w_i, ∇w_bar>. Prefers positive alignment, penalizes negative.
        This follows the paper's text: "remove clients that have negative inner products."
        """
        return metrics.get("grad_dot_product", -1.0)


# ============================================================================
# SYSTEM-BASED STRATEGIES (System Utility)
# ============================================================================

class HardDeadlineStrategy(BaseSelectionStrategy):
    """APP-ELEVEN (Eq. 12): Hard deadline - remove clients slower than threshold."""
    
    def __init__(self, fraction_train: float = 1.0, min_available_clients: int = 2, deadline: float = 5.0):
        """
        Args:
            fraction_train: Fraction of clients to select
            deadline: Time threshold in seconds
        """
        super().__init__(fraction_train, min_available_clients)
        self.deadline = deadline
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Binary utility: statistical utility if fast, else -infinity."""
        training_time = metrics.get("training_time", float('inf'))
        
        # Use a statistical metric (e.g., avg loss) as the base utility
        stat_utility = metrics.get("eval_loss", 0)
        
        if training_time > self.deadline:
            return -float('inf') # Hard cutoff
        else:
            return stat_utility # Prioritize by loss if within deadline


class SoftDeadlineStrategy(BaseSelectionStrategy):
    """APP-TWELVE (Eq. 13): Soft deadline - exponential punishment for stragglers."""
    
    def __init__(
        self, 
        fraction_train: float = 1.0, 
        min_available_clients: int = 2,
        deadline: float = 3.0,
        penalty_factor: float = 2.0
    ):
        """
        Args:
            fraction_train: Fraction of clients to select
            deadline: Time threshold for non-stragglers
            penalty_factor: Exponential penalty factor
        """
        super().__init__(fraction_train, min_available_clients)
        self.deadline = deadline
        self.penalty_factor = penalty_factor
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Utility = Statistical_Utility * System_Utility_Penalty"""
        
        # Statistical Utility (using total loss)
        stat_utility = metrics.get("eval_loss", 0) * metrics.get("num_examples", 1)
        
        # System Utility (Penalty)
        training_time = metrics.get("training_time", 0)
        
        if training_time <= self.deadline:
            penalty = 1.0
        else:
            # Exponential punishment
            excess_time = training_time - self.deadline
            penalty = np.exp(-self.penalty_factor * excess_time)
        
        # Paper Eq. 1: Util(i) = Util_stat(i) * Util_sys(i)
        return stat_utility * penalty


# ============================================================================
# RANDOM BASELINE
# ============================================================================

class RandomStrategy(BaseSelectionStrategy):
    """Baseline: Random client selection."""
    
    def compute_utility(
        self,
        client_id: str,
        metrics: Dict[str, float],
        round_num: int,
    ) -> float:
        """Random utility for fair random selection."""
        return np.random.random()


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "num_samples": NumSamplesStrategy,
    "high_grad_norm": HighGradientNormStrategy,
    "high_loss": HighLossStrategy,
    "sum_loss": SumLossStrategy,
    "model_divergence": ModelDivergenceStrategy,
    "same_sign": SameSignStrategy,
    "update_direction": UpdateDirectionStrategy,
    "local_change": LocalChangeStrategy,
    "post_grad_norm": PostGradientNormStrategy,
    "grad_dot_product": GradientDotProductStrategy,
    "hard_deadline": HardDeadlineStrategy,
    "soft_deadline": SoftDeadlineStrategy,
}


def get_strategy(strategy_name: str, **kwargs) -> BaseSelectionStrategy:
    """
    Factory function to get a strategy by name.
    
    Args:
        strategy_name: Name of the strategy
        **kwargs: Additional arguments for the strategy
        
    Returns:
        Instance of the requested strategy
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    
    # Pass all relevant kwargs to the constructor
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    # Filter kwargs to only those accepted by the constructor
    import inspect
    sig = inspect.signature(strategy_class.__init__)
    allowed_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    return strategy_class(**allowed_kwargs)
