"""Strategy configuration and parameters."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    """Available strategy types."""

    RL_PPO = "rl_ppo"
    RL_DQN = "rl_dqn"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"  # RL + LLM combined
    TECHNICAL = "technical"  # Pure technical analysis


class TimeFrame(str, Enum):
    """Supported timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class SignalStrength(str, Enum):
    """Signal strength levels."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TechnicalIndicatorConfig(BaseModel):
    """Configuration for technical indicators."""

    # Trend Indicators
    sma_periods: list[int] = Field(default_factory=lambda: [20, 50, 200])
    ema_periods: list[int] = Field(default_factory=lambda: [12, 26, 50])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Momentum Indicators
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    stochastic_k: int = 14
    stochastic_d: int = 3

    # Volatility Indicators
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14

    # Volume Indicators
    volume_sma_period: int = 20
    obv_enabled: bool = True
    vwap_enabled: bool = True


class RLStrategyConfig(BaseModel):
    """Configuration for RL-based strategies."""

    # Model Parameters
    model_path: str | None = None
    algorithm: str = "ppo"

    # Action Space
    action_type: str = "discrete"  # discrete or continuous
    discrete_actions: list[str] = Field(
        default_factory=lambda: ["hold", "buy", "sell"]
    )
    continuous_position_range: tuple[float, float] = (-1.0, 1.0)

    # Observation Space
    include_technical_indicators: bool = True
    include_price_features: bool = True
    include_volume_features: bool = True
    include_position_features: bool = True

    # Reward Configuration
    reward_type: str = "sharpe"  # sharpe, sortino, profit, risk_adjusted
    reward_lookback: int = 20
    transaction_cost_penalty: float = 0.001  # 0.1%


class LLMStrategyConfig(BaseModel):
    """Configuration for LLM-based strategies."""

    # Analysis Settings
    enable_market_analysis: bool = True
    enable_news_analysis: bool = True
    enable_sentiment_analysis: bool = True

    # Signal Integration
    signal_weight: float = 0.3  # Weight in hybrid strategy
    confidence_threshold: float = 0.6  # Min confidence to act

    # Analysis Frequency
    analysis_interval_minutes: int = 60  # How often to run analysis

    # Context Window
    news_lookback_hours: int = 24
    market_context_days: int = 7


class HybridStrategyConfig(BaseModel):
    """Configuration for hybrid RL + LLM strategies."""

    rl_config: RLStrategyConfig = Field(default_factory=RLStrategyConfig)
    llm_config: LLMStrategyConfig = Field(default_factory=LLMStrategyConfig)

    # Weight Distribution
    rl_weight: float = 0.7  # 70% RL, 30% LLM
    llm_weight: float = 0.3

    # Conflict Resolution
    conflict_resolution: str = "weighted_average"  # weighted_average, rl_priority, llm_veto
    llm_veto_threshold: float = 0.9  # LLM can veto if confidence > threshold

    # Meta Controller
    enable_regime_detection: bool = True
    regime_adaptation: bool = True  # Adapt weights based on market regime


class StrategyConfig(BaseModel):
    """Main strategy configuration."""

    # Strategy Selection
    strategy_type: StrategyType = StrategyType.HYBRID
    name: str = "default_strategy"
    description: str = ""

    # Timeframe
    primary_timeframe: TimeFrame = TimeFrame.H1
    secondary_timeframes: list[TimeFrame] = Field(
        default_factory=lambda: [TimeFrame.M15, TimeFrame.H4]
    )

    # Assets
    symbols: list[str] = Field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT"]
    )

    # Technical Indicators
    indicators: TechnicalIndicatorConfig = Field(default_factory=TechnicalIndicatorConfig)

    # Strategy-Specific Configs
    rl_config: RLStrategyConfig = Field(default_factory=RLStrategyConfig)
    llm_config: LLMStrategyConfig = Field(default_factory=LLMStrategyConfig)
    hybrid_config: HybridStrategyConfig = Field(default_factory=HybridStrategyConfig)

    # Risk Management
    max_position_per_symbol: float = 0.2  # Max 20% per symbol
    max_correlation_exposure: float = 0.5  # Max correlated positions
    max_sector_exposure: float = 0.4  # Max per sector (for stocks)

    # Entry/Exit Rules
    min_signal_strength: SignalStrength = SignalStrength.BUY
    entry_confirmation_candles: int = 1  # Candles to confirm entry
    exit_on_opposite_signal: bool = True

    def get_active_config(self) -> dict[str, Any]:
        """Get configuration for the active strategy type."""
        if self.strategy_type == StrategyType.HYBRID:
            return self.hybrid_config.model_dump()
        elif self.strategy_type in (StrategyType.RL_PPO, StrategyType.RL_DQN):
            return self.rl_config.model_dump()
        elif self.strategy_type == StrategyType.LLM_ONLY:
            return self.llm_config.model_dump()
        else:
            return self.indicators.model_dump()


# Predefined Strategy Templates
STRATEGY_TEMPLATES: dict[str, StrategyConfig] = {
    "aggressive_crypto": StrategyConfig(
        name="aggressive_crypto",
        description="Aggressive crypto trading with high risk tolerance",
        strategy_type=StrategyType.HYBRID,
        primary_timeframe=TimeFrame.M15,
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        rl_config=RLStrategyConfig(
            algorithm="ppo",
            reward_type="profit",
        ),
        hybrid_config=HybridStrategyConfig(
            rl_weight=0.8,
            llm_weight=0.2,
        ),
    ),
    "conservative_stocks": StrategyConfig(
        name="conservative_stocks",
        description="Conservative stock trading with lower risk",
        strategy_type=StrategyType.HYBRID,
        primary_timeframe=TimeFrame.D1,
        symbols=["SPY", "QQQ", "AAPL", "MSFT"],
        rl_config=RLStrategyConfig(
            algorithm="ppo",
            reward_type="sharpe",
        ),
        hybrid_config=HybridStrategyConfig(
            rl_weight=0.6,
            llm_weight=0.4,
            llm_veto_threshold=0.8,
        ),
    ),
    "balanced_multi_asset": StrategyConfig(
        name="balanced_multi_asset",
        description="Balanced strategy across crypto and stocks",
        strategy_type=StrategyType.HYBRID,
        primary_timeframe=TimeFrame.H4,
        symbols=["BTC/USDT", "ETH/USDT", "SPY", "QQQ"],
        hybrid_config=HybridStrategyConfig(
            rl_weight=0.7,
            llm_weight=0.3,
            enable_regime_detection=True,
        ),
    ),
}


def get_strategy_config(template_name: str | None = None) -> StrategyConfig:
    """Get strategy configuration, optionally from a template."""
    if template_name and template_name in STRATEGY_TEMPLATES:
        return STRATEGY_TEMPLATES[template_name]
    return StrategyConfig()
