"""Reward functions for reinforcement learning trading agents.

Provides various reward formulations including:
- Simple profit/loss
- Risk-adjusted returns (Sharpe, Sortino)
- Custom composite rewards
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardMetrics:
    """Container for reward calculation metrics."""

    returns: list[float]
    portfolio_values: list[float]
    positions: list[float]
    transaction_count: int
    drawdown: float
    max_drawdown: float


class BaseReward(ABC):
    """Base class for reward functions."""

    def __init__(self, scaling: float = 1.0):
        """Initialize reward function.

        Args:
            scaling: Reward scaling factor
        """
        self.scaling = scaling

    @abstractmethod
    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate reward from metrics.

        Args:
            metrics: Current trading metrics

        Returns:
            Calculated reward value
        """
        pass

    def __call__(self, metrics: RewardMetrics) -> float:
        """Calculate and scale reward."""
        return self.calculate(metrics) * self.scaling


class ProfitReward(BaseReward):
    """Simple profit-based reward."""

    def calculate(self, metrics: RewardMetrics) -> float:
        """Return the latest return as reward."""
        if len(metrics.returns) < 1:
            return 0.0
        return metrics.returns[-1]


class LogReturnReward(BaseReward):
    """Log return reward for better gradient properties."""

    def calculate(self, metrics: RewardMetrics) -> float:
        """Return log of 1 + return."""
        if len(metrics.returns) < 1:
            return 0.0

        ret = metrics.returns[-1]
        if ret <= -1.0:
            return -10.0  # Clip extreme losses
        return np.log1p(ret)


class SharpeReward(BaseReward):
    """Sharpe ratio based reward."""

    def __init__(
        self,
        window: int = 20,
        annualization: float = 252.0,
        scaling: float = 1.0,
    ):
        """Initialize Sharpe reward.

        Args:
            window: Rolling window for calculation
            annualization: Annualization factor (252 for daily)
            scaling: Reward scaling factor
        """
        super().__init__(scaling)
        self.window = window
        self.annualization = annualization

    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate rolling Sharpe ratio."""
        if len(metrics.returns) < self.window:
            # Fall back to simple return
            return metrics.returns[-1] if metrics.returns else 0.0

        window_returns = np.array(metrics.returns[-self.window:])

        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)

        if std_return < 1e-8:
            return mean_return * np.sqrt(self.annualization)

        sharpe = (mean_return / std_return) * np.sqrt(self.annualization)
        return sharpe


class SortinoReward(BaseReward):
    """Sortino ratio based reward (penalizes downside volatility only)."""

    def __init__(
        self,
        window: int = 20,
        annualization: float = 252.0,
        scaling: float = 1.0,
    ):
        """Initialize Sortino reward.

        Args:
            window: Rolling window for calculation
            annualization: Annualization factor
            scaling: Reward scaling factor
        """
        super().__init__(scaling)
        self.window = window
        self.annualization = annualization

    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate rolling Sortino ratio."""
        if len(metrics.returns) < self.window:
            return metrics.returns[-1] if metrics.returns else 0.0

        window_returns = np.array(metrics.returns[-self.window:])

        mean_return = np.mean(window_returns)
        downside_returns = window_returns[window_returns < 0]

        if len(downside_returns) == 0:
            # No downside volatility - very good
            return mean_return * np.sqrt(self.annualization) * 2

        downside_std = np.std(downside_returns)

        if downside_std < 1e-8:
            return mean_return * np.sqrt(self.annualization)

        sortino = (mean_return / downside_std) * np.sqrt(self.annualization)
        return sortino


class CalmarReward(BaseReward):
    """Calmar ratio based reward (return / max drawdown)."""

    def __init__(
        self,
        window: int = 252,
        min_drawdown: float = 0.01,
        scaling: float = 1.0,
    ):
        """Initialize Calmar reward.

        Args:
            window: Window for return calculation
            min_drawdown: Minimum drawdown to avoid division issues
            scaling: Reward scaling factor
        """
        super().__init__(scaling)
        self.window = window
        self.min_drawdown = min_drawdown

    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate Calmar ratio."""
        if len(metrics.portfolio_values) < 2:
            return 0.0

        # Calculate total return over window
        start_idx = max(0, len(metrics.portfolio_values) - self.window)
        start_value = metrics.portfolio_values[start_idx]
        end_value = metrics.portfolio_values[-1]

        if start_value == 0:
            return 0.0

        total_return = (end_value - start_value) / start_value

        # Use max drawdown with minimum threshold
        max_dd = max(metrics.max_drawdown, self.min_drawdown)

        return total_return / max_dd


class RiskAdjustedReward(BaseReward):
    """Composite risk-adjusted reward.

    Combines multiple factors:
    - Returns
    - Volatility penalty
    - Drawdown penalty
    - Transaction cost penalty
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        volatility_weight: float = 0.1,
        drawdown_weight: float = 0.2,
        transaction_weight: float = 0.001,
        window: int = 20,
        scaling: float = 1.0,
    ):
        """Initialize risk-adjusted reward.

        Args:
            return_weight: Weight for returns
            volatility_weight: Weight for volatility penalty
            drawdown_weight: Weight for drawdown penalty
            transaction_weight: Weight for transaction penalty
            window: Rolling window for calculations
            scaling: Reward scaling factor
        """
        super().__init__(scaling)
        self.return_weight = return_weight
        self.volatility_weight = volatility_weight
        self.drawdown_weight = drawdown_weight
        self.transaction_weight = transaction_weight
        self.window = window

    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate composite risk-adjusted reward."""
        reward = 0.0

        # Return component
        if metrics.returns:
            reward += self.return_weight * metrics.returns[-1]

        # Volatility penalty
        if len(metrics.returns) >= self.window:
            window_returns = np.array(metrics.returns[-self.window:])
            volatility = np.std(window_returns)
            reward -= self.volatility_weight * volatility

        # Drawdown penalty
        reward -= self.drawdown_weight * metrics.drawdown

        # Transaction penalty
        reward -= self.transaction_weight * metrics.transaction_count

        return reward


class DifferentialSharpeReward(BaseReward):
    """Differential Sharpe Ratio - online Sharpe estimation.

    More efficient than rolling Sharpe for RL as it updates incrementally.
    Based on Moody & Saffell (2001).
    """

    def __init__(
        self,
        eta: float = 0.001,
        scaling: float = 1.0,
    ):
        """Initialize differential Sharpe reward.

        Args:
            eta: Learning rate for running estimates
            scaling: Reward scaling factor
        """
        super().__init__(scaling)
        self.eta = eta
        self.A = 0.0  # Running mean of returns
        self.B = 0.0  # Running mean of squared returns

    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate differential Sharpe ratio."""
        if not metrics.returns:
            return 0.0

        r = metrics.returns[-1]

        # Update running estimates
        delta_A = r - self.A
        delta_B = r**2 - self.B

        # Calculate differential Sharpe
        denom = (self.B - self.A**2) ** 1.5
        if abs(denom) < 1e-8:
            dsr = delta_A
        else:
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / denom

        # Update estimates
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return dsr

    def reset(self) -> None:
        """Reset running estimates."""
        self.A = 0.0
        self.B = 0.0


class AsymmetricReward(BaseReward):
    """Asymmetric reward that penalizes losses more than it rewards gains.

    Useful for risk-averse training.
    """

    def __init__(
        self,
        loss_multiplier: float = 2.0,
        scaling: float = 1.0,
    ):
        """Initialize asymmetric reward.

        Args:
            loss_multiplier: How much more to penalize losses vs reward gains
            scaling: Reward scaling factor
        """
        super().__init__(scaling)
        self.loss_multiplier = loss_multiplier

    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate asymmetric reward."""
        if not metrics.returns:
            return 0.0

        ret = metrics.returns[-1]

        if ret < 0:
            return ret * self.loss_multiplier
        return ret


class ProfitFactorReward(BaseReward):
    """Profit factor based reward (gross profits / gross losses)."""

    def __init__(
        self,
        window: int = 50,
        scaling: float = 1.0,
    ):
        """Initialize profit factor reward.

        Args:
            window: Rolling window for calculation
            scaling: Reward scaling factor
        """
        super().__init__(scaling)
        self.window = window

    def calculate(self, metrics: RewardMetrics) -> float:
        """Calculate profit factor reward."""
        if len(metrics.returns) < self.window:
            return metrics.returns[-1] if metrics.returns else 0.0

        window_returns = np.array(metrics.returns[-self.window:])

        gross_profit = np.sum(window_returns[window_returns > 0])
        gross_loss = abs(np.sum(window_returns[window_returns < 0]))

        if gross_loss < 1e-8:
            return gross_profit  # No losses - excellent
        if gross_profit < 1e-8:
            return -gross_loss  # Only losses - terrible

        profit_factor = gross_profit / gross_loss

        # Transform to bounded reward
        # PF of 1 = break-even = 0 reward
        # PF > 1 = positive reward
        # PF < 1 = negative reward
        return np.log(profit_factor)


# Factory function
def create_reward(
    reward_type: str,
    **kwargs: Any,
) -> BaseReward:
    """Create a reward function by type.

    Args:
        reward_type: Type of reward function
        **kwargs: Additional arguments for the reward function

    Returns:
        Configured reward function instance
    """
    reward_map = {
        "profit": ProfitReward,
        "log_return": LogReturnReward,
        "sharpe": SharpeReward,
        "sortino": SortinoReward,
        "calmar": CalmarReward,
        "risk_adjusted": RiskAdjustedReward,
        "differential_sharpe": DifferentialSharpeReward,
        "asymmetric": AsymmetricReward,
        "profit_factor": ProfitFactorReward,
    }

    if reward_type not in reward_map:
        raise ValueError(f"Unknown reward type: {reward_type}. Available: {list(reward_map.keys())}")

    return reward_map[reward_type](**kwargs)
