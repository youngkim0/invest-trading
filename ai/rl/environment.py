"""Gymnasium trading environment for reinforcement learning.

Provides a customizable environment for training RL agents on financial data.
Supports both discrete and continuous action spaces.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger

from data.features.technical import TechnicalIndicators


class DiscreteAction(IntEnum):
    """Discrete action space."""

    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""

    # Data settings
    lookback_window: int = 60  # Number of candles for observation
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.001  # 0.1%

    # Action space
    action_type: str = "discrete"  # "discrete" or "continuous"
    max_position_size: float = 1.0  # Max position as fraction of portfolio

    # Reward settings
    reward_type: str = "sharpe"  # "sharpe", "sortino", "profit", "log_return"
    reward_window: int = 20  # Window for Sharpe/Sortino calculation
    reward_scaling: float = 1.0

    # Episode settings
    max_steps: int | None = None  # None = use all data
    random_start: bool = True  # Start at random point

    # Feature settings
    include_indicators: bool = True
    include_position: bool = True
    include_pnl: bool = True

    # Risk management
    max_drawdown: float = 0.2  # Max drawdown before termination
    stop_loss: float | None = 0.05  # Per-trade stop loss

    # Normalization
    normalize_obs: bool = True
    price_norm_window: int = 100


@dataclass
class PortfolioState:
    """Current portfolio state."""

    balance: float
    position: float = 0.0  # Positive = long, negative = short
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    peak_value: float = field(init=False)

    def __post_init__(self):
        self.peak_value = self.balance

    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        return self.balance + self.unrealized_pnl

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    def update_peak(self) -> None:
        """Update peak value if new high."""
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value


class TradingEnv(gym.Env):
    """Gymnasium environment for trading.

    Observation Space:
        - Price features: OHLCV normalized
        - Technical indicators: RSI, MACD, etc.
        - Position features: current position, unrealized PnL
        - Time features: optional time-of-day encoding

    Action Space:
        - Discrete: [Hold, Buy, Sell]
        - Continuous: Position size [-1, 1]

    Reward:
        - Sharpe ratio based (default)
        - Risk-adjusted returns
        - Transaction cost penalty
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: pd.DataFrame,
        config: TradingEnvConfig | None = None,
        render_mode: str | None = None,
    ):
        """Initialize trading environment.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            config: Environment configuration
            render_mode: Render mode for visualization
        """
        super().__init__()

        self.df = df.copy()
        self.config = config or TradingEnvConfig()
        self.render_mode = render_mode

        # Validate data
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Calculate technical indicators
        if self.config.include_indicators:
            self._add_indicators()

        # Drop NaN rows from indicator calculation
        self.df = self.df.dropna().reset_index(drop=True)

        # Calculate normalization parameters
        self._calculate_norm_params()

        # Define observation space
        self.n_features = self._calculate_feature_count()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.lookback_window, self.n_features),
            dtype=np.float32,
        )

        # Define action space
        if self.config.action_type == "discrete":
            self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        else:
            self.action_space = spaces.Box(
                low=-self.config.max_position_size,
                high=self.config.max_position_size,
                shape=(1,),
                dtype=np.float32,
            )

        # Episode state
        self.portfolio: PortfolioState | None = None
        self.current_step: int = 0
        self.start_step: int = 0
        self.returns_history: list[float] = []

    def _add_indicators(self) -> None:
        """Add technical indicators to dataframe."""
        ti = TechnicalIndicators(self.df)

        # Trend indicators
        self.df["sma_20"] = ti.sma(20)
        self.df["sma_50"] = ti.sma(50)
        self.df["ema_12"] = ti.ema(12)
        self.df["ema_26"] = ti.ema(26)

        macd = ti.macd()
        self.df["macd"] = macd["macd"]
        self.df["macd_signal"] = macd["signal"]
        self.df["macd_hist"] = macd["histogram"]

        # Momentum
        self.df["rsi"] = ti.rsi()

        stoch = ti.stochastic()
        self.df["stoch_k"] = stoch["stoch_k"]
        self.df["stoch_d"] = stoch["stoch_d"]

        adx = ti.adx()
        self.df["adx"] = adx["adx"]

        # Volatility
        self.df["atr"] = ti.atr()

        bb = ti.bollinger_bands()
        self.df["bb_upper"] = bb["upper"]
        self.df["bb_lower"] = bb["lower"]
        self.df["bb_pband"] = bb["pband"]

        # Volume
        self.df["obv"] = ti.obv()
        self.df["mfi"] = ti.mfi()

    def _calculate_norm_params(self) -> None:
        """Calculate normalization parameters."""
        self.price_mean = self.df["close"].mean()
        self.price_std = self.df["close"].std()
        self.volume_mean = self.df["volume"].mean()
        self.volume_std = self.df["volume"].std()

    def _calculate_feature_count(self) -> int:
        """Calculate number of features in observation."""
        count = 5  # OHLCV

        if self.config.include_indicators:
            # Add indicator columns
            indicator_cols = [
                "sma_20", "sma_50", "ema_12", "ema_26",
                "macd", "macd_signal", "macd_hist",
                "rsi", "stoch_k", "stoch_d", "adx",
                "atr", "bb_upper", "bb_lower", "bb_pband",
                "obv", "mfi",
            ]
            count += len([c for c in indicator_cols if c in self.df.columns])

        if self.config.include_position:
            count += 2  # position, entry_price_ratio

        if self.config.include_pnl:
            count += 2  # unrealized_pnl_ratio, drawdown

        return count

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        start_idx = max(0, self.current_step - self.config.lookback_window + 1)
        end_idx = self.current_step + 1

        # Get price data
        window_df = self.df.iloc[start_idx:end_idx].copy()

        # Normalize prices
        if self.config.normalize_obs:
            current_price = self.df.iloc[self.current_step]["close"]

            for col in ["open", "high", "low", "close"]:
                window_df[col] = (window_df[col] - current_price) / current_price

            window_df["volume"] = (window_df["volume"] - self.volume_mean) / (self.volume_std + 1e-8)

            # Normalize indicators that need it
            price_cols = ["sma_20", "sma_50", "ema_12", "ema_26", "bb_upper", "bb_lower"]
            for col in price_cols:
                if col in window_df.columns:
                    window_df[col] = (window_df[col] - current_price) / current_price

            # ATR as percentage
            if "atr" in window_df.columns:
                window_df["atr"] = window_df["atr"] / current_price

            # OBV normalize
            if "obv" in window_df.columns:
                obv_mean = window_df["obv"].mean()
                obv_std = window_df["obv"].std() + 1e-8
                window_df["obv"] = (window_df["obv"] - obv_mean) / obv_std

        # Build feature matrix
        features = []

        # OHLCV
        features.append(window_df[["open", "high", "low", "close", "volume"]].values)

        # Indicators
        if self.config.include_indicators:
            indicator_cols = [c for c in [
                "sma_20", "sma_50", "ema_12", "ema_26",
                "macd", "macd_signal", "macd_hist",
                "rsi", "stoch_k", "stoch_d", "adx",
                "atr", "bb_upper", "bb_lower", "bb_pband",
                "obv", "mfi",
            ] if c in window_df.columns]

            if indicator_cols:
                indicator_data = window_df[indicator_cols].values
                # Normalize RSI, Stochastic, MFI to 0-1
                for i, col in enumerate(indicator_cols):
                    if col in ["rsi", "stoch_k", "stoch_d", "mfi", "bb_pband"]:
                        indicator_data[:, i] = indicator_data[:, i] / 100
                    elif col == "adx":
                        indicator_data[:, i] = indicator_data[:, i] / 100

                features.append(indicator_data)

        # Portfolio features (broadcast to all timesteps)
        if self.config.include_position and self.portfolio:
            current_price = self.df.iloc[self.current_step]["close"]
            position_features = np.zeros((len(window_df), 2))
            position_features[:, 0] = self.portfolio.position / self.config.max_position_size
            if self.portfolio.entry_price > 0:
                position_features[:, 1] = (current_price - self.portfolio.entry_price) / self.portfolio.entry_price
            features.append(position_features)

        if self.config.include_pnl and self.portfolio:
            pnl_features = np.zeros((len(window_df), 2))
            pnl_features[:, 0] = self.portfolio.unrealized_pnl / self.portfolio.balance
            pnl_features[:, 1] = self.portfolio.drawdown
            features.append(pnl_features)

        # Concatenate and pad if needed
        obs = np.concatenate(features, axis=1)

        # Pad to lookback_window if needed
        if len(obs) < self.config.lookback_window:
            padding = np.zeros((self.config.lookback_window - len(obs), obs.shape[1]))
            obs = np.vstack([padding, obs])

        return obs.astype(np.float32)

    def _calculate_reward(self, action: int | np.ndarray) -> float:
        """Calculate reward based on configuration."""
        if not self.portfolio:
            return 0.0

        current_value = self.portfolio.total_value
        prev_value = self.returns_history[-1] if self.returns_history else self.config.initial_balance

        # Calculate return
        if prev_value > 0:
            ret = (current_value - prev_value) / prev_value
        else:
            ret = 0.0

        self.returns_history.append(current_value)

        # Apply reward type
        if self.config.reward_type == "profit":
            reward = ret

        elif self.config.reward_type == "log_return":
            reward = np.log1p(ret) if ret > -1 else -10.0

        elif self.config.reward_type == "sharpe":
            if len(self.returns_history) >= self.config.reward_window:
                returns = np.diff(self.returns_history[-self.config.reward_window:])
                returns = returns / self.returns_history[-self.config.reward_window:-1]
                if len(returns) > 1:
                    mean_ret = np.mean(returns)
                    std_ret = np.std(returns) + 1e-8
                    reward = mean_ret / std_ret
                else:
                    reward = ret
            else:
                reward = ret

        elif self.config.reward_type == "sortino":
            if len(self.returns_history) >= self.config.reward_window:
                returns = np.diff(self.returns_history[-self.config.reward_window:])
                returns = returns / self.returns_history[-self.config.reward_window:-1]
                if len(returns) > 1:
                    mean_ret = np.mean(returns)
                    downside = returns[returns < 0]
                    downside_std = np.std(downside) + 1e-8 if len(downside) > 0 else 1e-8
                    reward = mean_ret / downside_std
                else:
                    reward = ret
            else:
                reward = ret
        else:
            reward = ret

        # Transaction cost penalty
        if action != DiscreteAction.HOLD if isinstance(action, int) else abs(action) > 0.01:
            reward -= self.config.transaction_cost

        # Drawdown penalty
        if self.portfolio.drawdown > 0.1:
            reward -= self.portfolio.drawdown * 0.1

        return float(reward * self.config.reward_scaling)

    def _execute_action(self, action: int | np.ndarray) -> None:
        """Execute trading action."""
        if not self.portfolio:
            return

        current_price = float(self.df.iloc[self.current_step]["close"])

        if self.config.action_type == "discrete":
            if action == DiscreteAction.BUY and self.portfolio.position <= 0:
                # Close short if any, then go long
                if self.portfolio.position < 0:
                    self._close_position(current_price)
                self._open_position(1.0, current_price)

            elif action == DiscreteAction.SELL and self.portfolio.position >= 0:
                # Close long if any, then go short
                if self.portfolio.position > 0:
                    self._close_position(current_price)
                self._open_position(-1.0, current_price)

        else:
            # Continuous action: target position
            target_position = float(action[0])

            # Calculate position change
            position_delta = target_position - self.portfolio.position

            if abs(position_delta) > 0.01:  # Minimum change threshold
                if self.portfolio.position != 0 and np.sign(position_delta) != np.sign(self.portfolio.position):
                    # Closing or reversing
                    self._close_position(current_price)
                    if abs(target_position) > 0.01:
                        self._open_position(target_position, current_price)
                elif abs(target_position) > abs(self.portfolio.position):
                    # Adding to position
                    self._adjust_position(target_position, current_price)
                else:
                    # Reducing position
                    self._adjust_position(target_position, current_price)

        # Update unrealized PnL
        self._update_unrealized_pnl(current_price)

    def _open_position(self, size: float, price: float) -> None:
        """Open a new position."""
        if not self.portfolio:
            return

        position_value = self.portfolio.balance * abs(size) * self.config.max_position_size
        cost = position_value * self.config.transaction_cost

        self.portfolio.position = size
        self.portfolio.entry_price = price
        self.portfolio.balance -= cost

    def _close_position(self, price: float) -> None:
        """Close current position."""
        if not self.portfolio or self.portfolio.position == 0:
            return

        # Calculate PnL
        position_value = self.portfolio.balance * abs(self.portfolio.position) * self.config.max_position_size
        price_change = (price - self.portfolio.entry_price) / self.portfolio.entry_price
        pnl = position_value * price_change * np.sign(self.portfolio.position)

        # Apply transaction cost
        cost = position_value * self.config.transaction_cost

        self.portfolio.balance += pnl - cost
        self.portfolio.realized_pnl += pnl - cost
        self.portfolio.total_trades += 1
        if pnl > 0:
            self.portfolio.winning_trades += 1

        self.portfolio.position = 0.0
        self.portfolio.entry_price = 0.0
        self.portfolio.unrealized_pnl = 0.0

    def _adjust_position(self, target: float, price: float) -> None:
        """Adjust position size."""
        if not self.portfolio:
            return

        # Simplified: just set new position
        cost = self.portfolio.balance * abs(target - self.portfolio.position) * self.config.transaction_cost

        self.portfolio.position = target
        if self.portfolio.entry_price == 0:
            self.portfolio.entry_price = price
        self.portfolio.balance -= cost

    def _update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized PnL."""
        if not self.portfolio or self.portfolio.position == 0:
            return

        position_value = self.portfolio.balance * abs(self.portfolio.position) * self.config.max_position_size
        price_change = (current_price - self.portfolio.entry_price) / self.portfolio.entry_price
        self.portfolio.unrealized_pnl = position_value * price_change * np.sign(self.portfolio.position)
        self.portfolio.update_peak()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Determine starting point
        max_start = len(self.df) - self.config.lookback_window - 1
        if self.config.max_steps:
            max_start = min(max_start, len(self.df) - self.config.max_steps - self.config.lookback_window)

        if self.config.random_start and max_start > self.config.lookback_window:
            self.start_step = self.np_random.integers(
                self.config.lookback_window,
                max_start
            )
        else:
            self.start_step = self.config.lookback_window

        self.current_step = self.start_step

        # Reset portfolio
        self.portfolio = PortfolioState(balance=self.config.initial_balance)
        self.returns_history = [self.config.initial_balance]

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step in the environment."""
        # Execute action
        self._execute_action(action)

        # Move to next step
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check termination conditions
        terminated = False
        truncated = False

        # Max drawdown termination
        if self.portfolio and self.portfolio.drawdown > self.config.max_drawdown:
            terminated = True

        # End of data
        if self.current_step >= len(self.df) - 1:
            truncated = True

        # Max steps
        if self.config.max_steps and self.current_step - self.start_step >= self.config.max_steps:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        """Get episode info."""
        if not self.portfolio:
            return {}

        return {
            "step": self.current_step,
            "balance": self.portfolio.balance,
            "position": self.portfolio.position,
            "total_value": self.portfolio.total_value,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "realized_pnl": self.portfolio.realized_pnl,
            "drawdown": self.portfolio.drawdown,
            "total_trades": self.portfolio.total_trades,
            "winning_trades": self.portfolio.winning_trades,
            "win_rate": self.portfolio.winning_trades / max(1, self.portfolio.total_trades),
            "current_price": float(self.df.iloc[self.current_step]["close"]),
        }

    def render(self) -> str | None:
        """Render the environment."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            info = self._get_info()
            output = (
                f"Step: {info['step']} | "
                f"Price: {info['current_price']:.2f} | "
                f"Position: {info['position']:.2f} | "
                f"Value: {info['total_value']:.2f} | "
                f"PnL: {info['realized_pnl']:.2f} | "
                f"DD: {info['drawdown']:.2%}"
            )
            if self.render_mode == "human":
                print(output)
            return output
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass


def create_env(
    df: pd.DataFrame,
    config: TradingEnvConfig | None = None,
) -> TradingEnv:
    """Factory function to create trading environment.

    Args:
        df: OHLCV DataFrame
        config: Environment configuration

    Returns:
        Configured TradingEnv instance
    """
    return TradingEnv(df=df, config=config)
