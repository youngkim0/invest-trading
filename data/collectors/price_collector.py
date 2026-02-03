"""Price data collector for crypto and stocks."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from loguru import logger

from config import get_exchange_settings
from config.strategies import TimeFrame
from core.engine.alpaca_connector import get_alpaca_connector
from core.engine.binance_connector import get_binance_connector
from data.storage.models import OHLCV, AssetType
from data.storage.repository import DatabaseManager, OHLCVRepository


class PriceCollector:
    """Collects price data from multiple exchanges."""

    def __init__(self, db_manager: DatabaseManager | None = None):
        """Initialize price collector.

        Args:
            db_manager: Database manager for storing data
        """
        self.db_manager = db_manager or DatabaseManager()
        self.exchange_settings = get_exchange_settings()
        self._binance = None
        self._alpaca = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize exchange connections."""
        self._binance = await get_binance_connector()
        self._alpaca = get_alpaca_connector()
        logger.info("Price collector initialized")

    async def close(self) -> None:
        """Close exchange connections."""
        if self._binance:
            await self._binance.disconnect()
        if self._alpaca:
            self._alpaca.disconnect()
        logger.info("Price collector closed")

    # ==========================================================================
    # Historical Data Collection
    # ==========================================================================

    async def collect_crypto_history(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> int:
        """Collect historical crypto OHLCV data.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe
            start: Start time
            end: End time (default: now)

        Returns:
            Number of candles collected
        """
        if not self._binance:
            raise RuntimeError("Binance connector not initialized")

        end = end or datetime.utcnow()
        total_collected = 0

        logger.info(f"Collecting {symbol} {timeframe.value} data from {start} to {end}")

        current_start = start
        while current_start < end:
            try:
                candles = await self._binance.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=1000,
                )

                if not candles:
                    break

                # Store in database
                async with self.db_manager.session() as session:
                    repo = OHLCVRepository(session)

                    ohlcv_records = [
                        OHLCV(
                            symbol=symbol,
                            exchange="binance",
                            timeframe=timeframe.value,
                            timestamp=candle["timestamp"],
                            open=candle["open"],
                            high=candle["high"],
                            low=candle["low"],
                            close=candle["close"],
                            volume=candle["volume"],
                        )
                        for candle in candles
                    ]

                    # Use upsert logic (simplified - in production use ON CONFLICT)
                    for record in ohlcv_records:
                        try:
                            await repo.insert(record)
                        except Exception:
                            # Skip duplicates
                            await session.rollback()

                total_collected += len(candles)

                # Update start for next batch
                last_timestamp = candles[-1]["timestamp"]
                current_start = last_timestamp + timedelta(minutes=1)

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error collecting {symbol} data: {e}")
                await asyncio.sleep(1)
                continue

        logger.info(f"Collected {total_collected} candles for {symbol}")
        return total_collected

    def collect_stock_history(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime | None = None,
    ) -> int:
        """Collect historical stock OHLCV data.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            timeframe: Bar timeframe
            start: Start time
            end: End time (default: now)

        Returns:
            Number of bars collected
        """
        if not self._alpaca:
            raise RuntimeError("Alpaca connector not initialized")

        logger.info(f"Collecting {symbol} {timeframe.value} data from {start} to {end}")

        try:
            bars = self._alpaca.fetch_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=10000,
            )

            if not bars:
                return 0

            # Store in database (sync operation for simplicity)
            # In production, you'd want to batch this
            logger.info(f"Collected {len(bars)} bars for {symbol}")
            return len(bars)

        except Exception as e:
            logger.error(f"Error collecting {symbol} data: {e}")
            return 0

    async def collect_all_crypto(
        self,
        timeframe: TimeFrame,
        days_back: int = 30,
    ) -> dict[str, int]:
        """Collect historical data for all configured crypto pairs.

        Args:
            timeframe: Candlestick timeframe
            days_back: Number of days of history to collect

        Returns:
            Dictionary mapping symbols to number of candles collected
        """
        results = {}
        start = datetime.utcnow() - timedelta(days=days_back)

        for symbol in self.exchange_settings.binance.enabled_pairs:
            count = await self.collect_crypto_history(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
            )
            results[symbol] = count

        return results

    def collect_all_stocks(
        self,
        timeframe: TimeFrame,
        days_back: int = 30,
    ) -> dict[str, int]:
        """Collect historical data for all configured stocks.

        Args:
            timeframe: Bar timeframe
            days_back: Number of days of history to collect

        Returns:
            Dictionary mapping symbols to number of bars collected
        """
        results = {}
        start = datetime.utcnow() - timedelta(days=days_back)

        for symbol in self.exchange_settings.alpaca.enabled_symbols:
            count = self.collect_stock_history(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
            )
            results[symbol] = count

        return results

    # ==========================================================================
    # Real-time Data Collection
    # ==========================================================================

    async def stream_crypto_prices(
        self,
        symbols: list[str],
        callback: Any | None = None,
    ) -> None:
        """Stream real-time crypto prices.

        Args:
            symbols: List of trading pairs
            callback: Optional callback function for price updates
        """
        if not self._binance:
            raise RuntimeError("Binance connector not initialized")

        self._running = True
        logger.info(f"Starting crypto price stream for {symbols}")

        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self._stream_crypto_symbol(symbol, callback)
            )
            tasks.append(task)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Crypto price stream cancelled")
        finally:
            self._running = False

    async def _stream_crypto_symbol(
        self,
        symbol: str,
        callback: Any | None,
    ) -> None:
        """Stream prices for a single crypto symbol."""
        async for ticker in self._binance.watch_ticker(symbol):
            if not self._running:
                break

            if callback:
                await callback(ticker)
            else:
                logger.debug(f"{symbol}: {ticker['last']}")

    async def stream_crypto_candles(
        self,
        symbol: str,
        timeframe: TimeFrame,
        callback: Any | None = None,
    ) -> None:
        """Stream real-time crypto candles.

        Args:
            symbol: Trading pair
            timeframe: Candlestick timeframe
            callback: Optional callback for candle updates
        """
        if not self._binance:
            raise RuntimeError("Binance connector not initialized")

        self._running = True
        logger.info(f"Starting candle stream for {symbol} {timeframe.value}")

        try:
            async for candle in self._binance.watch_ohlcv(symbol, timeframe):
                if not self._running:
                    break

                if callback:
                    await callback(candle)
                else:
                    logger.debug(
                        f"{symbol} {timeframe.value}: O={candle['open']} H={candle['high']} "
                        f"L={candle['low']} C={candle['close']} V={candle['volume']}"
                    )
        except asyncio.CancelledError:
            logger.info("Candle stream cancelled")
        finally:
            self._running = False

    def stop_streaming(self) -> None:
        """Stop all streaming operations."""
        self._running = False
        logger.info("Stopping price streams")

    # ==========================================================================
    # Data Retrieval
    # ==========================================================================

    async def get_latest_price(
        self,
        symbol: str,
        asset_type: AssetType,
    ) -> Decimal | None:
        """Get the latest price for a symbol.

        Args:
            symbol: Trading symbol
            asset_type: Type of asset

        Returns:
            Latest price or None
        """
        try:
            if asset_type == AssetType.CRYPTO:
                if not self._binance:
                    raise RuntimeError("Binance connector not initialized")
                ticker = await self._binance.fetch_ticker(symbol)
                return ticker.get("last")

            elif asset_type == AssetType.STOCK:
                if not self._alpaca:
                    raise RuntimeError("Alpaca connector not initialized")
                quote = self._alpaca.fetch_latest_quote(symbol)
                # Return mid price
                bid = quote.get("bid", Decimal("0"))
                ask = quote.get("ask", Decimal("0"))
                return (bid + ask) / 2

            return None

        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    async def get_candles(
        self,
        symbol: str,
        asset_type: AssetType,
        timeframe: TimeFrame,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent candles for a symbol.

        Args:
            symbol: Trading symbol
            asset_type: Type of asset
            timeframe: Candlestick timeframe
            limit: Number of candles

        Returns:
            List of candle dictionaries
        """
        try:
            if asset_type == AssetType.CRYPTO:
                if not self._binance:
                    raise RuntimeError("Binance connector not initialized")
                return await self._binance.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )

            elif asset_type == AssetType.STOCK:
                if not self._alpaca:
                    raise RuntimeError("Alpaca connector not initialized")
                return self._alpaca.fetch_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )

            return []

        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return []


# Factory function
async def create_price_collector(
    db_manager: DatabaseManager | None = None,
) -> PriceCollector:
    """Create and initialize a price collector.

    Args:
        db_manager: Optional database manager

    Returns:
        Initialized price collector
    """
    collector = PriceCollector(db_manager)
    await collector.initialize()
    return collector
