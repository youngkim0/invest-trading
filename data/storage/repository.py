"""Data repository for database operations."""

from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, TypeVar

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from config import get_settings
from data.storage.models import (
    OHLCV,
    Base,
    LLMAnalysisLog,
    ModelCheckpoint,
    Order,
    OrderStatus,
    PerformanceSnapshot,
    Position,
    PositionStatus,
    Signal,
    SignalSource,
    SignalType,
    TradeLog,
)

T = TypeVar("T", bound=Base)

settings = get_settings()


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str | None = None):
        """Initialize database manager.

        Args:
            database_url: Database URL. If None, uses settings.
        """
        url = database_url or settings.database.async_url
        self.engine = create_async_engine(
            url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def create_tables(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self):
        """Get a database session context manager."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


class OHLCVRepository:
    """Repository for OHLCV data operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def insert(self, data: OHLCV) -> OHLCV:
        """Insert a single OHLCV record."""
        self.session.add(data)
        await self.session.flush()
        return data

    async def bulk_insert(self, data: list[OHLCV]) -> int:
        """Bulk insert OHLCV records."""
        self.session.add_all(data)
        await self.session.flush()
        return len(data)

    async def get_candles(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> Sequence[OHLCV]:
        """Get OHLCV candles for a symbol."""
        query = (
            select(OHLCV)
            .where(
                and_(
                    OHLCV.symbol == symbol,
                    OHLCV.exchange == exchange,
                    OHLCV.timeframe == timeframe,
                    OHLCV.timestamp >= start_time,
                )
            )
            .order_by(OHLCV.timestamp.asc())
            .limit(limit)
        )

        if end_time:
            query = query.where(OHLCV.timestamp <= end_time)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_latest(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> OHLCV | None:
        """Get the latest candle for a symbol."""
        query = (
            select(OHLCV)
            .where(
                and_(
                    OHLCV.symbol == symbol,
                    OHLCV.exchange == exchange,
                    OHLCV.timeframe == timeframe,
                )
            )
            .order_by(OHLCV.timestamp.desc())
            .limit(1)
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def delete_old_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        before: datetime,
    ) -> int:
        """Delete old candle data."""
        query = delete(OHLCV).where(
            and_(
                OHLCV.symbol == symbol,
                OHLCV.exchange == exchange,
                OHLCV.timeframe == timeframe,
                OHLCV.timestamp < before,
            )
        )
        result = await self.session.execute(query)
        return result.rowcount


class OrderRepository:
    """Repository for order operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, order: Order) -> Order:
        """Create a new order."""
        self.session.add(order)
        await self.session.flush()
        return order

    async def get_by_id(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return await self.session.get(Order, order_id)

    async def get_by_exchange_id(
        self,
        exchange: str,
        exchange_order_id: str,
    ) -> Order | None:
        """Get order by exchange order ID."""
        query = select(Order).where(
            and_(
                Order.exchange == exchange,
                Order.exchange_order_id == exchange_order_id,
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Decimal | None = None,
        average_fill_price: Decimal | None = None,
    ) -> Order | None:
        """Update order status."""
        order = await self.get_by_id(order_id)
        if order:
            order.status = status
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
            if average_fill_price is not None:
                order.average_fill_price = average_fill_price
            if status == OrderStatus.FILLED:
                order.filled_at = datetime.utcnow()
            await self.session.flush()
        return order

    async def get_open_orders(
        self,
        symbol: str | None = None,
        exchange: str | None = None,
    ) -> Sequence[Order]:
        """Get all open orders."""
        conditions = [Order.status.in_([OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED])]

        if symbol:
            conditions.append(Order.symbol == symbol)
        if exchange:
            conditions.append(Order.exchange == exchange)

        query = select(Order).where(and_(*conditions)).order_by(Order.created_at.desc())
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_orders_by_signal(self, signal_id: str) -> Sequence[Order]:
        """Get all orders associated with a signal."""
        query = select(Order).where(Order.signal_id == signal_id).order_by(Order.created_at.asc())
        result = await self.session.execute(query)
        return result.scalars().all()


class PositionRepository:
    """Repository for position operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, position: Position) -> Position:
        """Create a new position."""
        self.session.add(position)
        await self.session.flush()
        return position

    async def get_by_id(self, position_id: str) -> Position | None:
        """Get position by ID."""
        return await self.session.get(Position, position_id)

    async def get_open_positions(
        self,
        symbol: str | None = None,
        exchange: str | None = None,
    ) -> Sequence[Position]:
        """Get all open positions."""
        conditions = [Position.status == PositionStatus.OPEN]

        if symbol:
            conditions.append(Position.symbol == symbol)
        if exchange:
            conditions.append(Position.exchange == exchange)

        query = select(Position).where(and_(*conditions)).order_by(Position.opened_at.desc())
        result = await self.session.execute(query)
        return result.scalars().all()

    async def close_position(
        self,
        position_id: str,
        exit_price: Decimal,
        exit_order_id: str,
        realized_pnl: Decimal,
    ) -> Position | None:
        """Close a position."""
        position = await self.get_by_id(position_id)
        if position:
            position.status = PositionStatus.CLOSED
            position.exit_price = exit_price
            position.exit_order_id = exit_order_id
            position.realized_pnl = realized_pnl
            position.unrealized_pnl = Decimal("0")
            position.closed_at = datetime.utcnow()
            await self.session.flush()
        return position

    async def update_unrealized_pnl(
        self,
        position_id: str,
        current_price: Decimal,
        unrealized_pnl: Decimal,
    ) -> Position | None:
        """Update position's unrealized PnL."""
        position = await self.get_by_id(position_id)
        if position:
            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl
            await self.session.flush()
        return position


class SignalRepository:
    """Repository for signal operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, signal: Signal) -> Signal:
        """Create a new signal."""
        self.session.add(signal)
        await self.session.flush()
        return signal

    async def get_by_id(self, signal_id: str) -> Signal | None:
        """Get signal by ID with orders."""
        query = (
            select(Signal)
            .options(selectinload(Signal.orders))
            .where(Signal.id == signal_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_recent_signals(
        self,
        symbol: str | None = None,
        source: SignalSource | None = None,
        signal_type: SignalType | None = None,
        limit: int = 100,
    ) -> Sequence[Signal]:
        """Get recent signals with optional filters."""
        conditions = []

        if symbol:
            conditions.append(Signal.symbol == symbol)
        if source:
            conditions.append(Signal.source == source)
        if signal_type:
            conditions.append(Signal.signal_type == signal_type)

        query = select(Signal).order_by(Signal.generated_at.desc()).limit(limit)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        return result.scalars().all()

    async def mark_executed(self, signal_id: str) -> Signal | None:
        """Mark a signal as executed."""
        signal = await self.get_by_id(signal_id)
        if signal:
            signal.is_executed = True
            signal.executed_at = datetime.utcnow()
            await self.session.flush()
        return signal


class PerformanceRepository:
    """Repository for performance data operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_snapshot(self, snapshot: PerformanceSnapshot) -> PerformanceSnapshot:
        """Create a performance snapshot."""
        self.session.add(snapshot)
        await self.session.flush()
        return snapshot

    async def get_snapshots(
        self,
        start_time: datetime,
        end_time: datetime | None = None,
        strategy_name: str | None = None,
    ) -> Sequence[PerformanceSnapshot]:
        """Get performance snapshots for a time range."""
        conditions = [PerformanceSnapshot.timestamp >= start_time]

        if end_time:
            conditions.append(PerformanceSnapshot.timestamp <= end_time)
        if strategy_name:
            conditions.append(PerformanceSnapshot.strategy_name == strategy_name)

        query = (
            select(PerformanceSnapshot)
            .where(and_(*conditions))
            .order_by(PerformanceSnapshot.timestamp.asc())
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_latest_snapshot(
        self,
        strategy_name: str | None = None,
    ) -> PerformanceSnapshot | None:
        """Get the latest performance snapshot."""
        conditions = []
        if strategy_name:
            conditions.append(PerformanceSnapshot.strategy_name == strategy_name)

        query = select(PerformanceSnapshot).order_by(PerformanceSnapshot.timestamp.desc()).limit(1)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_performance_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        strategy_name: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated performance summary."""
        conditions = [
            PerformanceSnapshot.timestamp >= start_time,
            PerformanceSnapshot.timestamp <= end_time,
        ]
        if strategy_name:
            conditions.append(PerformanceSnapshot.strategy_name == strategy_name)

        query = select(
            func.sum(PerformanceSnapshot.daily_pnl).label("total_pnl"),
            func.max(PerformanceSnapshot.max_drawdown).label("max_drawdown"),
            func.sum(PerformanceSnapshot.total_trades).label("total_trades"),
            func.sum(PerformanceSnapshot.winning_trades).label("winning_trades"),
            func.sum(PerformanceSnapshot.losing_trades).label("losing_trades"),
        ).where(and_(*conditions))

        result = await self.session.execute(query)
        row = result.one()

        total_trades = row.total_trades or 0
        winning_trades = row.winning_trades or 0

        return {
            "total_pnl": row.total_pnl or Decimal("0"),
            "max_drawdown": row.max_drawdown or Decimal("0"),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": row.losing_trades or 0,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
        }


class TradeLogRepository:
    """Repository for trade log operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, trade_log: TradeLog) -> TradeLog:
        """Create a trade log entry."""
        self.session.add(trade_log)
        await self.session.flush()
        return trade_log

    async def get_trades(
        self,
        symbol: str | None = None,
        strategy_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[TradeLog]:
        """Get trade logs with optional filters."""
        conditions = []

        if symbol:
            conditions.append(TradeLog.symbol == symbol)
        if strategy_name:
            conditions.append(TradeLog.strategy_name == strategy_name)
        if start_time:
            conditions.append(TradeLog.entry_time >= start_time)
        if end_time:
            conditions.append(TradeLog.entry_time <= end_time)

        query = select(TradeLog).order_by(TradeLog.entry_time.desc()).limit(limit)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_trade_statistics(
        self,
        strategy_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get aggregated trade statistics."""
        conditions = []

        if strategy_name:
            conditions.append(TradeLog.strategy_name == strategy_name)
        if start_time:
            conditions.append(TradeLog.entry_time >= start_time)
        if end_time:
            conditions.append(TradeLog.entry_time <= end_time)

        query = select(
            func.count(TradeLog.id).label("total_trades"),
            func.sum(TradeLog.net_pnl).label("total_pnl"),
            func.avg(TradeLog.net_pnl).label("avg_pnl"),
            func.avg(TradeLog.return_pct).label("avg_return"),
            func.avg(TradeLog.duration_seconds).label("avg_duration"),
        )

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        row = result.one()

        # Count winning/losing trades
        win_query = select(func.count(TradeLog.id)).where(
            and_(TradeLog.net_pnl > 0, *conditions) if conditions else TradeLog.net_pnl > 0
        )
        win_result = await self.session.execute(win_query)
        winning_trades = win_result.scalar() or 0

        total_trades = row.total_trades or 0

        return {
            "total_trades": total_trades,
            "total_pnl": row.total_pnl or Decimal("0"),
            "avg_pnl": row.avg_pnl or Decimal("0"),
            "avg_return": row.avg_return or Decimal("0"),
            "avg_duration_seconds": row.avg_duration or 0,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
        }


class ModelCheckpointRepository:
    """Repository for model checkpoint operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, checkpoint: ModelCheckpoint) -> ModelCheckpoint:
        """Create a new model checkpoint."""
        self.session.add(checkpoint)
        await self.session.flush()
        return checkpoint

    async def get_by_id(self, checkpoint_id: str) -> ModelCheckpoint | None:
        """Get checkpoint by ID."""
        return await self.session.get(ModelCheckpoint, checkpoint_id)

    async def get_active_model(self, model_name: str) -> ModelCheckpoint | None:
        """Get the active model checkpoint."""
        query = select(ModelCheckpoint).where(
            and_(
                ModelCheckpoint.model_name == model_name,
                ModelCheckpoint.is_active == True,  # noqa: E712
            )
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def set_active(self, checkpoint_id: str) -> ModelCheckpoint | None:
        """Set a checkpoint as active (deactivates others with same name)."""
        checkpoint = await self.get_by_id(checkpoint_id)
        if checkpoint:
            # Deactivate all other checkpoints with same model name
            await self.session.execute(
                update(ModelCheckpoint)
                .where(
                    and_(
                        ModelCheckpoint.model_name == checkpoint.model_name,
                        ModelCheckpoint.id != checkpoint_id,
                    )
                )
                .values(is_active=False)
            )
            checkpoint.is_active = True
            await self.session.flush()
        return checkpoint

    async def get_model_history(
        self,
        model_name: str,
        limit: int = 10,
    ) -> Sequence[ModelCheckpoint]:
        """Get checkpoint history for a model."""
        query = (
            select(ModelCheckpoint)
            .where(ModelCheckpoint.model_name == model_name)
            .order_by(ModelCheckpoint.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(query)
        return result.scalars().all()


class LLMAnalysisRepository:
    """Repository for LLM analysis log operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, log: LLMAnalysisLog) -> LLMAnalysisLog:
        """Create an LLM analysis log entry."""
        self.session.add(log)
        await self.session.flush()
        return log

    async def get_recent_analyses(
        self,
        symbol: str | None = None,
        analysis_type: str | None = None,
        limit: int = 50,
    ) -> Sequence[LLMAnalysisLog]:
        """Get recent LLM analyses."""
        conditions = []

        if symbol:
            conditions.append(LLMAnalysisLog.symbol == symbol)
        if analysis_type:
            conditions.append(LLMAnalysisLog.analysis_type == analysis_type)

        query = select(LLMAnalysisLog).order_by(LLMAnalysisLog.created_at.desc()).limit(limit)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(query)
        return result.scalars().all()

    async def update_feedback(
        self,
        log_id: str,
        was_accurate: bool,
        feedback_notes: str | None = None,
    ) -> LLMAnalysisLog | None:
        """Update feedback for an LLM analysis."""
        log = await self.session.get(LLMAnalysisLog, log_id)
        if log:
            log.was_accurate = was_accurate
            log.feedback_notes = feedback_notes
            await self.session.flush()
        return log

    async def get_accuracy_stats(
        self,
        analysis_type: str | None = None,
    ) -> dict[str, Any]:
        """Get accuracy statistics for LLM analyses."""
        conditions = [LLMAnalysisLog.was_accurate.isnot(None)]

        if analysis_type:
            conditions.append(LLMAnalysisLog.analysis_type == analysis_type)

        total_query = select(func.count(LLMAnalysisLog.id)).where(and_(*conditions))
        accurate_query = select(func.count(LLMAnalysisLog.id)).where(
            and_(*conditions, LLMAnalysisLog.was_accurate == True)  # noqa: E712
        )

        total_result = await self.session.execute(total_query)
        accurate_result = await self.session.execute(accurate_query)

        total = total_result.scalar() or 0
        accurate = accurate_result.scalar() or 0

        return {
            "total_evaluated": total,
            "accurate": accurate,
            "inaccurate": total - accurate,
            "accuracy_rate": accurate / total if total > 0 else 0,
        }


# Convenience function to get all repositories
def get_repositories(session: AsyncSession) -> dict[str, Any]:
    """Get all repository instances for a session."""
    return {
        "ohlcv": OHLCVRepository(session),
        "orders": OrderRepository(session),
        "positions": PositionRepository(session),
        "signals": SignalRepository(session),
        "performance": PerformanceRepository(session),
        "trade_logs": TradeLogRepository(session),
        "model_checkpoints": ModelCheckpointRepository(session),
        "llm_analysis": LLMAnalysisRepository(session),
    }
