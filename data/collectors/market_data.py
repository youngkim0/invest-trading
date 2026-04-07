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
    BINANCE_FAPI_URL = "https://fapi.binance.com"

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

    # === Derivatives (Futures) Data ===

    async def get_funding_rate(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Get funding rate history from Binance Futures."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/fapi/v1/fundingRate"
            params = {"symbol": symbol, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{
                "symbol": r["symbol"],
                "funding_rate": float(r["fundingRate"]),
                "funding_time": datetime.fromtimestamp(r["fundingTime"] / 1000).isoformat(),
            } for r in data]
        except Exception as e:
            logger.error(f"Failed to fetch funding rate: {e}")
            return []

    async def get_open_interest(
        self,
        symbol: str = "BTCUSDT",
    ) -> dict[str, Any]:
        """Get current open interest snapshot."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/fapi/v1/openInterest"
            params = {"symbol": symbol}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return {
                "symbol": data["symbol"],
                "open_interest": float(data["openInterest"]),
                "time": datetime.fromtimestamp(data["time"] / 1000).isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to fetch open interest: {e}")
            return {}

    async def get_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 48,
    ) -> list[dict[str, Any]]:
        """Get open interest history (4h of 5m data by default)."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/futures/data/openInterestHist"
            params = {"symbol": symbol, "period": period, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{
                "symbol": r["symbol"],
                "sum_open_interest": float(r["sumOpenInterest"]),
                "sum_open_interest_value": float(r["sumOpenInterestValue"]),
                "timestamp": datetime.fromtimestamp(r["timestamp"] / 1000).isoformat(),
            } for r in data]
        except Exception as e:
            logger.error(f"Failed to fetch OI history: {e}")
            return []

    async def get_premium_index(
        self,
        symbol: str = "BTCUSDT",
    ) -> dict[str, Any]:
        """Get premium index (mark price, index price, predicted funding)."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/fapi/v1/premiumIndex"
            params = {"symbol": symbol}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return {
                "symbol": data["symbol"],
                "mark_price": float(data["markPrice"]),
                "index_price": float(data["indexPrice"]),
                "last_funding_rate": float(data["lastFundingRate"]),
                "next_funding_time": datetime.fromtimestamp(data["nextFundingTime"] / 1000).isoformat(),
                "interest_rate": float(data.get("interestRate", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to fetch premium index: {e}")
            return {}

    async def get_taker_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Get taker buy/sell volume ratio."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/futures/data/takerlongshortRatio"
            params = {"symbol": symbol, "period": period, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{
                "buy_sell_ratio": float(r["buySellRatio"]),
                "buy_vol": float(r["buyVol"]),
                "sell_vol": float(r["sellVol"]),
                "timestamp": datetime.fromtimestamp(r["timestamp"] / 1000).isoformat(),
            } for r in data]
        except Exception as e:
            logger.error(f"Failed to fetch taker L/S ratio: {e}")
            return []

    async def get_top_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Get top trader long/short account ratio."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/futures/data/topLongShortAccountRatio"
            params = {"symbol": symbol, "period": period, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{
                "long_short_ratio": float(r["longShortRatio"]),
                "long_account": float(r["longAccount"]),
                "short_account": float(r["shortAccount"]),
                "timestamp": datetime.fromtimestamp(r["timestamp"] / 1000).isoformat(),
            } for r in data]
        except Exception as e:
            logger.error(f"Failed to fetch top L/S ratio: {e}")
            return []

    async def get_top_long_short_position_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Get top trader long/short POSITION ratio (capital-weighted, not account count)."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/futures/data/topLongShortPositionRatio"
            params = {"symbol": symbol, "period": period, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{
                "long_short_ratio": float(r["longShortRatio"]),
                "long_account": float(r["longAccount"]),
                "short_account": float(r["shortAccount"]),
                "timestamp": datetime.fromtimestamp(r["timestamp"] / 1000).isoformat(),
            } for r in data]
        except Exception as e:
            logger.error(f"Failed to fetch top position ratio: {e}")
            return []

    async def get_global_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Get global (all traders) long/short account ratio."""
        try:
            url = f"{self.BINANCE_FAPI_URL}/futures/data/globalLongShortAccountRatio"
            params = {"symbol": symbol, "period": period, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [{
                "long_short_ratio": float(r["longShortRatio"]),
                "long_account": float(r["longAccount"]),
                "short_account": float(r["shortAccount"]),
                "timestamp": datetime.fromtimestamp(r["timestamp"] / 1000).isoformat(),
            } for r in data]
        except Exception as e:
            logger.error(f"Failed to fetch global L/S ratio: {e}")
            return []

    async def get_fear_greed_index(self) -> dict[str, Any]:
        """Get crypto Fear & Greed Index (0-100, updates daily)."""
        try:
            url = "https://api.alternative.me/fng/"
            params = {"limit": 1, "format": "json"}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            entry = data.get("data", [{}])[0]
            return {
                "value": int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "Neutral"),
            }
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")
            return {"value": 50, "classification": "Neutral"}

    # =================================================================
    # v7.0.3: Free external data sources for AI intelligence
    # =================================================================

    async def get_crypto_news(self, limit: int = 10) -> list[dict]:
        """Fetch latest crypto news from CoinGecko (free, no key needed).

        Returns list of {title, description, url, published_at}.
        """
        try:
            url = "https://api.coingecko.com/api/v3/news"
            response = await self.client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            articles = data.get("data", [])[:limit]
            return [
                {
                    "title": a.get("title", ""),
                    "description": (a.get("description") or "")[:200],
                    "url": a.get("url", ""),
                    "published_at": a.get("updated_at") or a.get("created_at", ""),
                }
                for a in articles
            ]
        except Exception as e:
            logger.debug(f"Crypto news fetch failed: {e}")
            return []

    async def get_funding_rate_history(self, symbol: str = "BTCUSDT", limit: int = 30) -> list[dict]:
        """Fetch historical funding rates from Binance (free).

        Returns last N funding rate entries (each is 8h apart).
        30 entries = 10 days of history.
        """
        try:
            url = f"{self.BINANCE_FAPI_URL}/fapi/v1/fundingRate"
            params = {"symbol": symbol, "limit": limit}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [
                {
                    "symbol": symbol,
                    "rate": float(d.get("fundingRate", 0)),
                    "time": datetime.fromtimestamp(d.get("fundingTime", 0) / 1000).isoformat(),
                }
                for d in data
            ]
        except Exception as e:
            logger.debug(f"Funding rate history fetch failed for {symbol}: {e}")
            return []

    async def get_macro_dxy(self) -> dict:
        """Fetch DXY (US Dollar Index) approximation from free API.

        Uses EUR/USD as proxy (DXY is ~57% EUR/USD weighted).
        Free, no key needed.
        """
        try:
            # Use Binance forex proxy — EUR/USDT as DXY inverse proxy
            url = f"{self.BINANCE_BASE_URL}/ticker/price"
            params = {"symbol": "EURUSDT"}
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            eur_usd = float(data.get("price", 1.0))
            # Approximate DXY: when EUR/USD goes up, DXY goes down
            # DXY ~= 100 * (1 / EUR/USD) normalized
            dxy_approx = round(100.0 / eur_usd * 1.08, 2)  # Rough normalization
            return {
                "dxy_approx": dxy_approx,
                "eur_usd": eur_usd,
                "source": "binance_eurusdt_proxy",
            }
        except Exception as e:
            logger.debug(f"DXY proxy fetch failed: {e}")
            return {"dxy_approx": 100.0, "eur_usd": 1.0, "source": "fallback"}

    async def get_btc_dominance(self) -> dict:
        """Fetch BTC dominance from CoinGecko (free).

        BTC dominance rising = alts underperforming, risk-off in crypto.
        """
        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = await self.client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json().get("data", {})
            market_cap_pct = data.get("market_cap_percentage", {})
            return {
                "btc_dominance": round(market_cap_pct.get("btc", 0), 2),
                "eth_dominance": round(market_cap_pct.get("eth", 0), 2),
                "total_market_cap_usd": data.get("total_market_cap", {}).get("usd", 0),
                "total_volume_usd": data.get("total_volume", {}).get("usd", 0),
                "market_cap_change_24h": round(data.get("market_cap_change_percentage_24h_usd", 0), 2),
            }
        except Exception as e:
            logger.debug(f"BTC dominance fetch failed: {e}")
            return {"btc_dominance": 0, "eth_dominance": 0}

    async def get_derivatives_data(
        self,
        symbol: str = "BTCUSDT",
    ) -> dict[str, Any]:
        """Fetch all derivatives data for a symbol in parallel.

        Returns dict with keys: funding_rate, open_interest, oi_history,
        premium_index, taker_ratio_5m, taker_ratio_15m, taker_ratio_1h,
        top_long_short.
        """
        try:
            results = await asyncio.gather(
                self.get_funding_rate(symbol, limit=3),
                self.get_open_interest(symbol),
                self.get_open_interest_history(symbol, "5m", 48),
                self.get_premium_index(symbol),
                self.get_taker_long_short_ratio(symbol, "5m", 1),
                self.get_taker_long_short_ratio(symbol, "15m", 1),
                self.get_taker_long_short_ratio(symbol, "1h", 1),
                self.get_top_long_short_ratio(symbol, "1h", 1),
                self.get_top_long_short_position_ratio(symbol, "1h", 1),
                self.get_global_long_short_ratio(symbol, "1h", 1),
                return_exceptions=True,
            )

            return {
                "funding_rate": results[0] if not isinstance(results[0], Exception) else [],
                "open_interest": results[1] if not isinstance(results[1], Exception) else {},
                "oi_history": results[2] if not isinstance(results[2], Exception) else [],
                "premium_index": results[3] if not isinstance(results[3], Exception) else {},
                "taker_ratio_5m": results[4] if not isinstance(results[4], Exception) else [],
                "taker_ratio_15m": results[5] if not isinstance(results[5], Exception) else [],
                "taker_ratio_1h": results[6] if not isinstance(results[6], Exception) else [],
                "top_long_short": results[7] if not isinstance(results[7], Exception) else [],
                "top_position_ratio": results[8] if not isinstance(results[8], Exception) else [],
                "global_long_short": results[9] if not isinstance(results[9], Exception) else [],
            }
        except Exception as e:
            logger.error(f"Failed to fetch derivatives data: {e}")
            return {}


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
