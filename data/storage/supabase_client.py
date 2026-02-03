"""Supabase client for the AI Trading System."""

from functools import lru_cache
from typing import Any

from loguru import logger
from supabase import create_client, Client

from config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """Get Supabase client singleton.

    Returns:
        Supabase client instance
    """
    settings = get_settings()

    supabase_url = settings.supabase.url
    supabase_key = settings.supabase.anon_key.get_secret_value()

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and key must be configured")

    client = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized")

    return client


class SupabaseRepository:
    """Base repository for Supabase operations."""

    def __init__(self, table_name: str):
        """Initialize repository.

        Args:
            table_name: Name of the Supabase table
        """
        self.client = get_supabase_client()
        self.table_name = table_name

    @property
    def table(self):
        """Get table reference."""
        return self.client.table(self.table_name)

    async def insert(self, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a record.

        Args:
            data: Record data

        Returns:
            Inserted record
        """
        result = self.table.insert(data).execute()
        return result.data[0] if result.data else {}

    async def insert_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert multiple records.

        Args:
            records: List of record data

        Returns:
            Inserted records
        """
        result = self.table.insert(records).execute()
        return result.data or []

    async def select(
        self,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """Select records.

        Args:
            columns: Columns to select
            filters: Filter conditions (column: value)
            order_by: Order by column (prefix with - for desc)
            limit: Maximum records to return
            offset: Number of records to skip

        Returns:
            List of records
        """
        query = self.table.select(columns)

        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)

        if order_by:
            if order_by.startswith("-"):
                query = query.order(order_by[1:], desc=True)
            else:
                query = query.order(order_by)

        if limit:
            query = query.limit(limit)

        if offset:
            query = query.offset(offset)

        result = query.execute()
        return result.data or []

    async def select_one(
        self,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Select a single record.

        Args:
            columns: Columns to select
            filters: Filter conditions

        Returns:
            Record or None
        """
        records = await self.select(columns, filters, limit=1)
        return records[0] if records else None

    async def update(
        self,
        data: dict[str, Any],
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Update records.

        Args:
            data: Update data
            filters: Filter conditions

        Returns:
            Updated records
        """
        query = self.table.update(data)

        for column, value in filters.items():
            query = query.eq(column, value)

        result = query.execute()
        return result.data or []

    async def upsert(self, data: dict[str, Any]) -> dict[str, Any]:
        """Upsert a record.

        Args:
            data: Record data

        Returns:
            Upserted record
        """
        result = self.table.upsert(data).execute()
        return result.data[0] if result.data else {}

    async def delete(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Delete records.

        Args:
            filters: Filter conditions

        Returns:
            Deleted records
        """
        query = self.table.delete()

        for column, value in filters.items():
            query = query.eq(column, value)

        result = query.execute()
        return result.data or []


class OHLCVRepository(SupabaseRepository):
    """Repository for OHLCV data."""

    def __init__(self):
        super().__init__("ohlcv")

    async def get_candles(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get OHLCV candles.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Candle timeframe
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            limit: Maximum records

        Returns:
            List of candles
        """
        query = self.table.select("*").eq("symbol", symbol).eq("exchange", exchange).eq("timeframe", timeframe)

        if start_time:
            query = query.gte("timestamp", start_time)
        if end_time:
            query = query.lte("timestamp", end_time)

        query = query.order("timestamp", desc=True).limit(limit)

        result = query.execute()
        return result.data or []

    async def save_candles(self, candles: list[dict[str, Any]]) -> int:
        """Save OHLCV candles.

        Args:
            candles: List of candle data

        Returns:
            Number of saved candles
        """
        if not candles:
            return 0

        result = self.table.upsert(candles).execute()
        return len(result.data) if result.data else 0


class TradeLogRepository(SupabaseRepository):
    """Repository for trade logs."""

    def __init__(self):
        super().__init__("trade_logs")

    async def get_trades(
        self,
        strategy_name: str | None = None,
        symbol: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get trade logs.

        Args:
            strategy_name: Filter by strategy
            symbol: Filter by symbol
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            limit: Maximum records

        Returns:
            List of trades
        """
        query = self.table.select("*")

        if strategy_name:
            query = query.eq("strategy_name", strategy_name)
        if symbol:
            query = query.eq("symbol", symbol)
        if start_time:
            query = query.gte("entry_time", start_time)
        if end_time:
            query = query.lte("entry_time", end_time)

        query = query.order("entry_time", desc=True).limit(limit)

        result = query.execute()
        return result.data or []

    async def log_trade(self, trade: dict[str, Any]) -> dict[str, Any]:
        """Log a trade.

        Args:
            trade: Trade data

        Returns:
            Saved trade
        """
        result = self.table.insert(trade).execute()
        return result.data[0] if result.data else {}


class PerformanceRepository(SupabaseRepository):
    """Repository for performance snapshots."""

    def __init__(self):
        super().__init__("performance_snapshots")

    async def get_snapshots(
        self,
        strategy_name: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get performance snapshots.

        Args:
            strategy_name: Filter by strategy
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            limit: Maximum records

        Returns:
            List of snapshots
        """
        query = self.table.select("*")

        if strategy_name:
            query = query.eq("strategy_name", strategy_name)
        if start_time:
            query = query.gte("timestamp", start_time)
        if end_time:
            query = query.lte("timestamp", end_time)

        query = query.order("timestamp", desc=True).limit(limit)

        result = query.execute()
        return result.data or []

    async def save_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Save performance snapshot.

        Args:
            snapshot: Snapshot data

        Returns:
            Saved snapshot
        """
        result = self.table.insert(snapshot).execute()
        return result.data[0] if result.data else {}


class SignalRepository(SupabaseRepository):
    """Repository for trading signals."""

    def __init__(self):
        super().__init__("signals")

    async def get_signals(
        self,
        symbol: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get trading signals.

        Args:
            symbol: Filter by symbol
            source: Filter by signal source
            limit: Maximum records

        Returns:
            List of signals
        """
        query = self.table.select("*")

        if symbol:
            query = query.eq("symbol", symbol)
        if source:
            query = query.eq("source", source)

        query = query.order("timestamp", desc=True).limit(limit)

        result = query.execute()
        return result.data or []

    async def save_signal(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Save a signal.

        Args:
            signal: Signal data

        Returns:
            Saved signal
        """
        result = self.table.insert(signal).execute()
        return result.data[0] if result.data else {}
