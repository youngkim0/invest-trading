"""Trade logging system for the investment journal."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from loguru import logger

from data.storage.models import (
    OrderSide,
    PerformanceSnapshot,
    SignalSource,
    TradeLog,
)
from data.storage.repository import (
    DatabaseManager,
    PerformanceRepository,
    TradeLogRepository,
)


@dataclass
class TradeEntry:
    """Trade entry data for logging."""

    symbol: str
    exchange: str
    side: OrderSide
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime

    # Optional fields
    exit_price: Decimal | None = None
    exit_time: datetime | None = None
    gross_pnl: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")

    # Signal info
    signal_source: SignalSource | None = None
    signal_confidence: Decimal | None = None

    # Strategy info
    strategy_name: str = ""
    position_id: str | None = None
    entry_order_id: str | None = None
    exit_order_id: str | None = None

    # Analysis
    entry_reasoning: str | None = None
    exit_reasoning: str | None = None

    # Context
    market_context: dict[str, Any] | None = None
    indicators_at_entry: dict[str, Any] | None = None
    indicators_at_exit: dict[str, Any] | None = None

    # Tags
    tags: list[str] | None = None

    def calculate_pnl(self) -> None:
        """Calculate PnL from entry/exit prices."""
        if self.exit_price is None:
            return

        if self.side == OrderSide.BUY:
            self.gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            self.gross_pnl = (self.entry_price - self.exit_price) * self.quantity

        self.net_pnl = self.gross_pnl - self.commission

    @property
    def return_pct(self) -> Decimal:
        """Calculate return percentage."""
        entry_value = self.entry_price * self.quantity
        if entry_value == 0:
            return Decimal("0")
        return (self.net_pnl / entry_value) * 100

    @property
    def duration_seconds(self) -> int | None:
        """Calculate trade duration in seconds."""
        if self.exit_time is None:
            return None
        return int((self.exit_time - self.entry_time).total_seconds())

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0


class TradeLogger:
    """Logs trades and maintains the investment journal."""

    def __init__(self, db_manager: DatabaseManager | None = None):
        """Initialize trade logger.

        Args:
            db_manager: Database manager for persistence
        """
        self.db_manager = db_manager or DatabaseManager()

        # In-memory trade storage (for quick access)
        self._open_trades: dict[str, TradeEntry] = {}  # position_id -> trade
        self._closed_trades: list[TradeEntry] = []

        # Statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = Decimal("0")

    async def log_entry(
        self,
        trade: TradeEntry,
    ) -> str:
        """Log a trade entry.

        Args:
            trade: Trade entry data

        Returns:
            Trade/position ID
        """
        # Generate position ID if not provided
        if not trade.position_id:
            trade.position_id = f"pos_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        # Store in memory
        self._open_trades[trade.position_id] = trade

        logger.info(
            f"Trade entry logged: {trade.symbol} {trade.side.value} "
            f"@ {trade.entry_price} x {trade.quantity}"
        )

        return trade.position_id

    async def log_exit(
        self,
        position_id: str,
        exit_price: Decimal,
        exit_time: datetime | None = None,
        commission: Decimal = Decimal("0"),
        exit_reasoning: str | None = None,
        indicators_at_exit: dict[str, Any] | None = None,
    ) -> TradeEntry | None:
        """Log a trade exit.

        Args:
            position_id: Position ID from entry
            exit_price: Exit price
            exit_time: Exit timestamp
            commission: Additional commission
            exit_reasoning: Reason for exit
            indicators_at_exit: Indicators at exit time

        Returns:
            Completed trade entry or None
        """
        if position_id not in self._open_trades:
            logger.warning(f"Position {position_id} not found in open trades")
            return None

        trade = self._open_trades.pop(position_id)
        trade.exit_price = exit_price
        trade.exit_time = exit_time or datetime.utcnow()
        trade.commission += commission
        trade.exit_reasoning = exit_reasoning
        trade.indicators_at_exit = indicators_at_exit

        # Calculate PnL
        trade.calculate_pnl()

        # Update statistics
        self._total_trades += 1
        if trade.is_winner:
            self._winning_trades += 1
        self._total_pnl += trade.net_pnl

        # Store completed trade
        self._closed_trades.append(trade)

        # Persist to database
        await self._persist_trade(trade)

        logger.info(
            f"Trade exit logged: {trade.symbol} {trade.side.value} "
            f"@ {exit_price}, PnL: {trade.net_pnl}"
        )

        return trade

    async def _persist_trade(self, trade: TradeEntry) -> None:
        """Persist trade to database."""
        async with self.db_manager.session() as session:
            repo = TradeLogRepository(session)

            trade_log = TradeLog(
                position_id=trade.position_id,
                entry_order_id=trade.entry_order_id,
                exit_order_id=trade.exit_order_id,
                symbol=trade.symbol,
                exchange=trade.exchange,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                gross_pnl=trade.gross_pnl,
                net_pnl=trade.net_pnl,
                total_commission=trade.commission,
                return_pct=trade.return_pct,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                duration_seconds=trade.duration_seconds,
                signal_source=trade.signal_source,
                signal_confidence=trade.signal_confidence,
                strategy_name=trade.strategy_name,
                entry_reasoning=trade.entry_reasoning,
                exit_reasoning=trade.exit_reasoning,
                market_context=trade.market_context,
                indicators_at_entry=trade.indicators_at_entry,
                indicators_at_exit=trade.indicators_at_exit,
                tags=trade.tags,
            )

            await repo.create(trade_log)

    async def log_performance_snapshot(
        self,
        total_equity: Decimal,
        cash_balance: Decimal,
        positions_value: Decimal,
        strategy_name: str | None = None,
    ) -> None:
        """Log a performance snapshot.

        Args:
            total_equity: Total portfolio equity
            cash_balance: Cash balance
            positions_value: Value of open positions
            strategy_name: Strategy name
        """
        # Calculate metrics
        daily_pnl = sum(
            t.net_pnl for t in self._closed_trades
            if t.exit_time and t.exit_time.date() == datetime.utcnow().date()
        )

        total_pnl = self._total_pnl
        winning = self._winning_trades
        total = self._total_trades
        win_rate = Decimal(str(winning / total)) if total > 0 else Decimal("0")

        async with self.db_manager.session() as session:
            repo = PerformanceRepository(session)

            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                total_equity=total_equity,
                cash_balance=cash_balance,
                positions_value=positions_value,
                daily_pnl=daily_pnl,
                total_pnl=total_pnl,
                total_trades=total,
                winning_trades=winning,
                losing_trades=total - winning,
                win_rate=win_rate,
                strategy_name=strategy_name,
            )

            await repo.create_snapshot(snapshot)

        logger.debug(f"Performance snapshot logged: equity={total_equity}")

    def get_open_trades(self) -> list[TradeEntry]:
        """Get all open trades."""
        return list(self._open_trades.values())

    def get_closed_trades(
        self,
        limit: int = 100,
        symbol: str | None = None,
    ) -> list[TradeEntry]:
        """Get closed trades.

        Args:
            limit: Maximum number of trades
            symbol: Filter by symbol

        Returns:
            List of closed trades
        """
        trades = self._closed_trades

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        return trades[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get trading statistics."""
        if self._total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "profit_factor": 0.0,
            }

        winning_pnl = sum(
            float(t.net_pnl) for t in self._closed_trades if t.is_winner
        )
        losing_pnl = abs(sum(
            float(t.net_pnl) for t in self._closed_trades if not t.is_winner
        ))

        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": self._winning_trades / self._total_trades,
            "total_pnl": float(self._total_pnl),
            "avg_pnl": float(self._total_pnl) / self._total_trades,
            "profit_factor": profit_factor,
            "open_positions": len(self._open_trades),
        }

    def clear_history(self) -> None:
        """Clear in-memory trade history."""
        self._closed_trades.clear()
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = Decimal("0")
        logger.info("Trade history cleared")
