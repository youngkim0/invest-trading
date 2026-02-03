"""Global settings for the AI Trading System."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupabaseSettings(BaseSettings):
    """Supabase connection settings."""

    model_config = SettingsConfigDict(env_prefix="SUPABASE_")

    url: str = ""
    anon_key: SecretStr = SecretStr("")
    service_role_key: SecretStr = SecretStr("")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is configured."""
        return bool(self.url and self.anon_key.get_secret_value())


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "invest"
    user: str = "postgres"
    password: SecretStr = SecretStr("")

    @property
    def url(self) -> str:
        """Get database URL."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        """Get async database URL."""
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"


class BinanceSettings(BaseSettings):
    """Binance exchange settings."""

    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")
    testnet: bool = True  # Use testnet by default for safety

    @property
    def is_configured(self) -> bool:
        """Check if Binance is configured."""
        return bool(self.api_key.get_secret_value() and self.api_secret.get_secret_value())


class AlpacaSettings(BaseSettings):
    """Alpaca exchange settings."""

    model_config = SettingsConfigDict(env_prefix="ALPACA_")

    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")
    paper: bool = True  # Use paper trading by default

    @property
    def is_configured(self) -> bool:
        """Check if Alpaca is configured."""
        return bool(self.api_key.get_secret_value() and self.api_secret.get_secret_value())


class TradingSettings(BaseSettings):
    """Trading-related settings."""

    model_config = SettingsConfigDict(env_prefix="TRADING_")

    # Risk Management
    max_position_size: float = Field(default=0.1, description="Max position size as fraction of portfolio")
    max_drawdown: float = Field(default=0.15, description="Max drawdown before stopping (15%)")
    daily_trade_limit: int = Field(default=50, description="Max trades per day")

    # Position Sizing
    position_sizing_method: Literal["fixed", "kelly", "volatility"] = "volatility"
    base_position_size: float = 0.02  # 2% of portfolio per trade

    # Stop Loss / Take Profit
    use_atr_stops: bool = True
    atr_stop_multiplier: float = 2.0
    default_stop_loss: float = 0.02  # 2%
    default_take_profit: float = 0.04  # 4%

    # Trading Mode
    mode: Literal["backtest", "paper", "live"] = "paper"


class RLSettings(BaseSettings):
    """Reinforcement Learning settings."""

    model_config = SettingsConfigDict(env_prefix="RL_")

    # Training
    algorithm: Literal["ppo", "dqn", "a2c"] = "ppo"
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda

    # Environment
    lookback_window: int = 60  # Number of candles for observation
    reward_scaling: float = 1.0

    # Model
    policy_network: str = "MlpPolicy"
    hidden_layers: list[int] = Field(default_factory=lambda: [256, 256])

    # Training Schedule
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    save_freq: int = 50_000


class LLMSettings(BaseSettings):
    """LLM Agent settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    # API Keys
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")

    # Model Selection
    primary_model: str = "gpt-4-turbo-preview"
    fallback_model: str = "gpt-3.5-turbo"

    # Agent Configuration
    market_analysis_enabled: bool = True
    news_analysis_enabled: bool = True
    sentiment_weight: float = 0.3  # Weight of LLM signals in final decision

    # Rate Limiting
    max_requests_per_minute: int = 60
    request_timeout: int = 30


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "AI Trading System"
    debug: bool = False
    log_level: str = "INFO"

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data_files")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # Nested Settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    rl: RLSettings = Field(default_factory=RLSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
