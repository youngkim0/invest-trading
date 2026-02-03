"""Database models for the AI Trading System.

Uses SQLAlchemy 2.0 with TimescaleDB for time-series data optimization.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


# =============================================================================
# Enums
# =============================================================================


class AssetType(str, PyEnum):
    """Asset type enumeration."""

    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    FUTURES = "futures"


class OrderSide(str, PyEnum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, PyEnum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, PyEnum):
    """Order status enumeration."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(str, PyEnum):
    """Position status enumeration."""

    OPEN = "open"
    CLOSED = "closed"


class SignalType(str, PyEnum):
    """Trading signal type."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SignalSource(str, PyEnum):
    """Signal source enumeration."""

    RL_AGENT = "rl_agent"
    LLM_AGENT = "llm_agent"
    TECHNICAL = "technical"
    HYBRID = "hybrid"
    MANUAL = "manual"


# =============================================================================
# Market Data Models (TimescaleDB Hypertables)
# =============================================================================


class OHLCV(Base):
    """OHLCV (candlestick) data model.

    This table should be converted to a TimescaleDB hypertable:
    SELECT create_hypertable('ohlcv', 'timestamp');
    """

    __tablename__ = "ohlcv"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)  # 1m, 5m, 1h, etc.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)

    # Optional fields
    quote_volume: Mapped[Decimal | None] = mapped_column(Numeric(30, 8))
    trades_count: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "timeframe", "timestamp", name="uq_ohlcv"),
        Index("ix_ohlcv_symbol_timeframe", "symbol", "timeframe"),
        Index("ix_ohlcv_timestamp", "timestamp"),
    )


class TickData(Base):
    """Tick-level trade data.

    This table should be converted to a TimescaleDB hypertable:
    SELECT create_hypertable('tick_data', 'timestamp');
    """

    __tablename__ = "tick_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    side: Mapped[str] = mapped_column(String(10))  # buy/sell
    trade_id: Mapped[str | None] = mapped_column(String(100))

    __table_args__ = (
        Index("ix_tick_symbol", "symbol"),
        Index("ix_tick_timestamp", "timestamp"),
    )


# =============================================================================
# Trading Models
# =============================================================================


class Order(Base):
    """Order model."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    exchange_order_id: Mapped[str | None] = mapped_column(String(100))

    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    asset_type: Mapped[AssetType] = mapped_column(Enum(AssetType), nullable=False)

    side: Mapped[OrderSide] = mapped_column(Enum(OrderSide), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(Enum(OrderType), nullable=False)
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus), default=OrderStatus.PENDING)

    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))  # Limit price
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    average_fill_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Fees
    commission: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    commission_asset: Mapped[str | None] = mapped_column(String(20))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Strategy reference
    strategy_name: Mapped[str | None] = mapped_column(String(100))
    signal_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("signals.id"))

    # Metadata
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    # Relationships
    signal: Mapped["Signal | None"] = relationship("Signal", back_populates="orders")
    position: Mapped["Position | None"] = relationship("Position", back_populates="orders", foreign_keys="Position.entry_order_id")

    __table_args__ = (
        Index("ix_orders_symbol", "symbol"),
        Index("ix_orders_status", "status"),
        Index("ix_orders_created_at", "created_at"),
    )


class Position(Base):
    """Position model."""

    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    asset_type: Mapped[AssetType] = mapped_column(Enum(AssetType), nullable=False)

    side: Mapped[OrderSide] = mapped_column(Enum(OrderSide), nullable=False)
    status: Mapped[PositionStatus] = mapped_column(Enum(PositionStatus), default=PositionStatus.OPEN)

    # Position details
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Stop loss / Take profit
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # PnL
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    total_commission: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Order references
    entry_order_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("orders.id"))
    exit_order_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))

    # Strategy reference
    strategy_name: Mapped[str | None] = mapped_column(String(100))

    # Metadata
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    # Relationships
    orders: Mapped["Order | None"] = relationship("Order", back_populates="position", foreign_keys=[entry_order_id])

    __table_args__ = (
        Index("ix_positions_symbol", "symbol"),
        Index("ix_positions_status", "status"),
        Index("ix_positions_opened_at", "opened_at"),
    )


# =============================================================================
# Signal Models
# =============================================================================


class Signal(Base):
    """Trading signal model."""

    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)

    signal_type: Mapped[SignalType] = mapped_column(Enum(SignalType), nullable=False)
    source: Mapped[SignalSource] = mapped_column(Enum(SignalSource), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)  # 0.0000 - 1.0000

    # Price levels
    entry_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Signal details
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text)

    # Execution
    is_executed: Mapped[bool] = mapped_column(Boolean, default=False)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Timestamps
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Metadata
    indicators: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # Technical indicators at signal time
    llm_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # LLM analysis data
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    # Relationships
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="signal")

    __table_args__ = (
        Index("ix_signals_symbol", "symbol"),
        Index("ix_signals_generated_at", "generated_at"),
        Index("ix_signals_source", "source"),
    )


# =============================================================================
# Performance Models
# =============================================================================


class PerformanceSnapshot(Base):
    """Performance snapshot model for tracking portfolio performance over time.

    This table should be converted to a TimescaleDB hypertable:
    SELECT create_hypertable('performance_snapshots', 'timestamp');
    """

    __tablename__ = "performance_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Portfolio value
    total_equity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    positions_value: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)

    # PnL
    daily_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    total_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))

    # Returns
    daily_return: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))
    cumulative_return: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))

    # Risk metrics
    drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    sortino_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    # Trading stats
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))

    # Strategy reference
    strategy_name: Mapped[str | None] = mapped_column(String(100))

    __table_args__ = (
        Index("ix_performance_timestamp", "timestamp"),
        Index("ix_performance_strategy", "strategy_name"),
    )


class TradeLog(Base):
    """Detailed trade log for investment journal."""

    __tablename__ = "trade_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    # Trade identification
    position_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("positions.id"), nullable=False)
    entry_order_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("orders.id"), nullable=False)
    exit_order_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("orders.id"))

    # Trade details
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    side: Mapped[OrderSide] = mapped_column(Enum(OrderSide), nullable=False)

    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)

    # PnL
    gross_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    total_commission: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=Decimal("0"))
    return_pct: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))

    # Duration
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)

    # Signal info
    signal_source: Mapped[SignalSource | None] = mapped_column(Enum(SignalSource))
    signal_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))

    # Strategy info
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Analysis
    entry_reasoning: Mapped[str | None] = mapped_column(Text)
    exit_reasoning: Mapped[str | None] = mapped_column(Text)
    post_trade_analysis: Mapped[str | None] = mapped_column(Text)

    # Market context at trade time
    market_context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    indicators_at_entry: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    indicators_at_exit: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Tags for categorization
    tags: Mapped[list[str] | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("ix_trade_logs_symbol", "symbol"),
        Index("ix_trade_logs_entry_time", "entry_time"),
        Index("ix_trade_logs_strategy", "strategy_name"),
    )


# =============================================================================
# AI/ML Models
# =============================================================================


class ModelCheckpoint(Base):
    """RL model checkpoint tracking."""

    __tablename__ = "model_checkpoints"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    # Model identification
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)  # ppo, dqn, etc.
    version: Mapped[str] = mapped_column(String(50), nullable=False)

    # File location
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)

    # Training info
    total_timesteps: Mapped[int] = mapped_column(Integer, nullable=False)
    training_episodes: Mapped[int] = mapped_column(Integer, default=0)

    # Performance metrics
    mean_reward: Mapped[Decimal | None] = mapped_column(Numeric(15, 6))
    std_reward: Mapped[Decimal | None] = mapped_column(Numeric(15, 6))
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))

    # Training parameters
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)  # Currently deployed model
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Notes
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_model_checkpoints_name", "model_name"),
        Index("ix_model_checkpoints_active", "is_active"),
    )


class LLMAnalysisLog(Base):
    """Log of LLM agent analyses for review and improvement."""

    __tablename__ = "llm_analysis_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    # Analysis context
    symbol: Mapped[str | None] = mapped_column(String(50))
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)  # market, news, sentiment

    # Input/Output
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)

    # Model info
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    # Extracted signals
    signal_type: Mapped[SignalType | None] = mapped_column(Enum(SignalType))
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))

    # Feedback
    was_accurate: Mapped[bool | None] = mapped_column(Boolean)
    feedback_notes: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_logs_symbol", "symbol"),
        Index("ix_llm_logs_type", "analysis_type"),
        Index("ix_llm_logs_created_at", "created_at"),
    )
