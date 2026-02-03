"""Real-time market data collector using public APIs."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import httpx
import pandas as pd
from loguru import logger


class MarketDataCollector:
    """Collect market data from public APIs (no authentication required)."""

    BINANCE_BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_binance_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get candlestick data from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            url = f"{self.BINANCE_BASE_URL}/klines"
            params = {
                "symbol": symbol.replace("/", ""),
                "interval": interval,
                "limit": limit,
            }

            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Keep only OHLCV columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch Binance klines: {e}")
            return pd.DataFrame()

    async def get_binance_ticker(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Get current ticker price from Binance.

        Args:
            symbol: Trading pair

        Returns:
            Ticker data
        """
        try:
            url = f"{self.BINANCE_BASE_URL}/ticker/24hr"
            params = {"symbol": symbol.replace("/", "")}

            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": symbol,
                "price": float(data["lastPrice"]),
                "change_24h": float(data["priceChangePercent"]),
                "high_24h": float(data["highPrice"]),
                "low_24h": float(data["lowPrice"]),
                "volume_24h": float(data["volume"]),
                "quote_volume_24h": float(data["quoteVolume"]),
            }

        except Exception as e:
            logger.error(f"Failed to fetch Binance ticker: {e}")
            return {}

    async def get_multiple_tickers(
        self,
        symbols: list[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    ) -> list[dict[str, Any]]:
        """Get tickers for multiple symbols.

        Args:
            symbols: List of trading pairs

        Returns:
            List of ticker data
        """
        tasks = [self.get_binance_ticker(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    async def get_orderbook(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get order book from Binance.

        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Order book data
        """
        try:
            url = f"{self.BINANCE_BASE_URL}/depth"
            params = {
                "symbol": symbol.replace("/", ""),
                "limit": limit,
            }

            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                "bids": [[float(p), float(q)] for p, q in data["bids"]],
                "asks": [[float(p), float(q)] for p, q in data["asks"]],
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return {}

    async def get_recent_trades(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent trades from Binance.

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)

        Returns:
            List of recent trades
        """
        try:
            url = f"{self.BINANCE_BASE_URL}/trades"
            params = {
                "symbol": symbol.replace("/", ""),
                "limit": limit,
            }

            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return [{
                "id": t["id"],
                "price": float(t["price"]),
                "quantity": float(t["qty"]),
                "time": datetime.fromtimestamp(t["time"] / 1000).isoformat(),
                "is_buyer_maker": t["isBuyerMaker"],
            } for t in data]

        except Exception as e:
            logger.error(f"Failed to fetch recent trades: {e}")
            return []


# Synchronous wrapper for use in Streamlit
def get_market_data_sync(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 100,
) -> pd.DataFrame:
    """Synchronous wrapper to get market data."""
    async def _fetch():
        collector = MarketDataCollector()
        try:
            return await collector.get_binance_klines(symbol, interval, limit)
        finally:
            await collector.close()

    return asyncio.run(_fetch())


def get_ticker_sync(symbol: str = "BTCUSDT") -> dict[str, Any]:
    """Synchronous wrapper to get ticker."""
    async def _fetch():
        collector = MarketDataCollector()
        try:
            return await collector.get_binance_ticker(symbol)
        finally:
            await collector.close()

    return asyncio.run(_fetch())


def get_multiple_tickers_sync(
    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
) -> list[dict[str, Any]]:
    """Synchronous wrapper to get multiple tickers."""
    async def _fetch():
        collector = MarketDataCollector()
        try:
            return await collector.get_multiple_tickers(symbols)
        finally:
            await collector.close()

    return asyncio.run(_fetch())
