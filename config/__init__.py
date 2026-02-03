"""Configuration module for AI Trading System."""

from config.exchanges import (
    AlpacaConfig,
    BinanceConfig,
    ExchangeSettings,
    ExchangeType,
    get_exchange_settings,
)
from config.settings import (
    DatabaseSettings,
    LLMSettings,
    RLSettings,
    Settings,
    TradingSettings,
    get_settings,
)
from config.strategies import (
    HybridStrategyConfig,
    LLMStrategyConfig,
    RLStrategyConfig,
    SignalStrength,
    StrategyConfig,
    StrategyType,
    TechnicalIndicatorConfig,
    TimeFrame,
    get_strategy_config,
)

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    "DatabaseSettings",
    "TradingSettings",
    "RLSettings",
    "LLMSettings",
    # Exchanges
    "ExchangeSettings",
    "get_exchange_settings",
    "BinanceConfig",
    "AlpacaConfig",
    "ExchangeType",
    # Strategies
    "StrategyConfig",
    "get_strategy_config",
    "StrategyType",
    "TimeFrame",
    "SignalStrength",
    "TechnicalIndicatorConfig",
    "RLStrategyConfig",
    "LLMStrategyConfig",
    "HybridStrategyConfig",
]
