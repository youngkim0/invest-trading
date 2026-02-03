"""Exchange configuration for Binance and Alpaca."""

from enum import Enum
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeType(str, Enum):
    """Supported exchange types."""

    BINANCE = "binance"
    BINANCE_TESTNET = "binance_testnet"
    ALPACA = "alpaca"
    ALPACA_PAPER = "alpaca_paper"


class BinanceConfig(BaseSettings):
    """Binance exchange configuration."""

    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")

    # Testnet
    testnet_api_key: SecretStr = SecretStr("")
    testnet_api_secret: SecretStr = SecretStr("")

    # Connection
    use_testnet: bool = True  # Default to testnet for safety
    rate_limit: int = 1200  # Requests per minute

    # Trading Pairs
    default_quote_currency: str = "USDT"
    enabled_pairs: list[str] = Field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
        ]
    )

    # Endpoints
    spot_endpoint: str = "https://api.binance.com"
    futures_endpoint: str = "https://fapi.binance.com"
    testnet_spot_endpoint: str = "https://testnet.binance.vision"
    testnet_futures_endpoint: str = "https://testnet.binancefuture.com"

    @property
    def active_api_key(self) -> str:
        """Get active API key based on testnet setting."""
        if self.use_testnet:
            return self.testnet_api_key.get_secret_value()
        return self.api_key.get_secret_value()

    @property
    def active_api_secret(self) -> str:
        """Get active API secret based on testnet setting."""
        if self.use_testnet:
            return self.testnet_api_secret.get_secret_value()
        return self.api_secret.get_secret_value()


class AlpacaConfig(BaseSettings):
    """Alpaca exchange configuration for stocks."""

    model_config = SettingsConfigDict(env_prefix="ALPACA_")

    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")

    # Paper Trading
    paper_api_key: SecretStr = SecretStr("")
    paper_api_secret: SecretStr = SecretStr("")

    # Connection
    use_paper: bool = True  # Default to paper trading for safety
    rate_limit: int = 200  # Requests per minute

    # Trading Symbols
    enabled_symbols: list[str] = Field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "SPY",
            "QQQ",
        ]
    )

    # Endpoints
    live_endpoint: str = "https://api.alpaca.markets"
    paper_endpoint: str = "https://paper-api.alpaca.markets"
    data_endpoint: str = "https://data.alpaca.markets"

    @property
    def active_api_key(self) -> str:
        """Get active API key based on paper setting."""
        if self.use_paper:
            return self.paper_api_key.get_secret_value()
        return self.api_key.get_secret_value()

    @property
    def active_api_secret(self) -> str:
        """Get active API secret based on paper setting."""
        if self.use_paper:
            return self.paper_api_secret.get_secret_value()
        return self.api_secret.get_secret_value()

    @property
    def base_url(self) -> str:
        """Get base URL based on paper setting."""
        return self.paper_endpoint if self.use_paper else self.live_endpoint


class ExchangeSettings(BaseSettings):
    """Combined exchange settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)

    # Default Exchange
    default_crypto_exchange: Literal["binance"] = "binance"
    default_stock_exchange: Literal["alpaca"] = "alpaca"


def get_exchange_settings() -> ExchangeSettings:
    """Get exchange settings instance."""
    return ExchangeSettings()
