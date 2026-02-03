"""Binance exchange connector using CCXT."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt
from loguru import logger

from config import get_exchange_settings
from config.strategies import TimeFrame


class BinanceConnector:
    """Async connector for Binance exchange using CCXT."""

    TIMEFRAME_MAP = {
        TimeFrame.M1: "1m",
        TimeFrame.M5: "5m",
        TimeFrame.M15: "15m",
        TimeFrame.M30: "30m",
        TimeFrame.H1: "1h",
        TimeFrame.H4: "4h",
        TimeFrame.D1: "1d",
        TimeFrame.W1: "1w",
    }

    def __init__(self):
        """Initialize Binance connector."""
        self.config = get_exchange_settings().binance
        self._exchange: ccxt.binance | None = None
        self._ws_connections: dict[str, Any] = {}

    async def connect(self) -> None:
        """Connect to Binance exchange."""
        options = {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        }

        if self.config.use_testnet:
            options["urls"] = {
                "api": {
                    "public": self.config.testnet_spot_endpoint,
                    "private": self.config.testnet_spot_endpoint,
                }
            }

        self._exchange = ccxt.binance(
            {
                "apiKey": self.config.active_api_key,
                "secret": self.config.active_api_secret,
                "enableRateLimit": True,
                "rateLimit": 60000 // self.config.rate_limit,  # ms per request
                "options": options,
            }
        )

        # Load markets
        await self._exchange.load_markets()
        logger.info(
            f"Connected to Binance {'testnet' if self.config.use_testnet else 'mainnet'}"
        )

    async def disconnect(self) -> None:
        """Disconnect from Binance exchange."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
            logger.info("Disconnected from Binance")

    @property
    def exchange(self) -> ccxt.binance:
        """Get exchange instance."""
        if not self._exchange:
            raise RuntimeError("Not connected to Binance. Call connect() first.")
        return self._exchange

    # ==========================================================================
    # Market Data
    # ==========================================================================

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame | str,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe
            since: Start time for candles
            limit: Maximum number of candles to fetch

        Returns:
            List of OHLCV dictionaries
        """
        tf = self.TIMEFRAME_MAP.get(timeframe, timeframe) if isinstance(timeframe, TimeFrame) else timeframe
        since_ts = int(since.timestamp() * 1000) if since else None

        raw_ohlcv = await self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=tf,
            since=since_ts,
            limit=limit,
        )

        return [
            {
                "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                "open": Decimal(str(candle[1])),
                "high": Decimal(str(candle[2])),
                "low": Decimal(str(candle[3])),
                "close": Decimal(str(candle[4])),
                "volume": Decimal(str(candle[5])),
            }
            for candle in raw_ohlcv
        ]

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """
        ticker = await self.exchange.fetch_ticker(symbol)
        return {
            "symbol": ticker["symbol"],
            "timestamp": datetime.fromtimestamp(ticker["timestamp"] / 1000) if ticker["timestamp"] else None,
            "bid": Decimal(str(ticker["bid"])) if ticker["bid"] else None,
            "ask": Decimal(str(ticker["ask"])) if ticker["ask"] else None,
            "last": Decimal(str(ticker["last"])) if ticker["last"] else None,
            "high": Decimal(str(ticker["high"])) if ticker["high"] else None,
            "low": Decimal(str(ticker["low"])) if ticker["low"] else None,
            "volume": Decimal(str(ticker["baseVolume"])) if ticker["baseVolume"] else None,
            "quote_volume": Decimal(str(ticker["quoteVolume"])) if ticker["quoteVolume"] else None,
            "change": Decimal(str(ticker["change"])) if ticker["change"] else None,
            "change_pct": Decimal(str(ticker["percentage"])) if ticker["percentage"] else None,
        }

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Fetch order book.

        Args:
            symbol: Trading pair symbol
            limit: Depth limit

        Returns:
            Order book dictionary
        """
        order_book = await self.exchange.fetch_order_book(symbol, limit)
        return {
            "symbol": symbol,
            "timestamp": datetime.fromtimestamp(order_book["timestamp"] / 1000) if order_book["timestamp"] else None,
            "bids": [
                {"price": Decimal(str(bid[0])), "quantity": Decimal(str(bid[1]))}
                for bid in order_book["bids"]
            ],
            "asks": [
                {"price": Decimal(str(ask[0])), "quantity": Decimal(str(ask[1]))}
                for ask in order_book["asks"]
            ],
        }

    async def fetch_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch recent trades.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch

        Returns:
            List of trade dictionaries
        """
        trades = await self.exchange.fetch_trades(symbol, limit=limit)
        return [
            {
                "id": trade["id"],
                "timestamp": datetime.fromtimestamp(trade["timestamp"] / 1000),
                "symbol": trade["symbol"],
                "side": trade["side"],
                "price": Decimal(str(trade["price"])),
                "quantity": Decimal(str(trade["amount"])),
                "cost": Decimal(str(trade["cost"])) if trade["cost"] else None,
            }
            for trade in trades
        ]

    # ==========================================================================
    # Account Data
    # ==========================================================================

    async def fetch_balance(self) -> dict[str, Any]:
        """Fetch account balance.

        Returns:
            Balance dictionary with free, used, and total for each asset
        """
        balance = await self.exchange.fetch_balance()

        result = {
            "timestamp": datetime.utcnow(),
            "total": {},
            "free": {},
            "used": {},
        }

        for currency, amounts in balance.items():
            if isinstance(amounts, dict) and "free" in amounts:
                if amounts.get("total", 0) > 0:
                    result["total"][currency] = Decimal(str(amounts["total"]))
                    result["free"][currency] = Decimal(str(amounts["free"]))
                    result["used"][currency] = Decimal(str(amounts["used"]))

        return result

    async def fetch_my_trades(
        self,
        symbol: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch user's trade history.

        Args:
            symbol: Trading pair symbol
            since: Start time
            limit: Maximum number of trades

        Returns:
            List of trade dictionaries
        """
        since_ts = int(since.timestamp() * 1000) if since else None
        trades = await self.exchange.fetch_my_trades(symbol, since=since_ts, limit=limit)

        return [
            {
                "id": trade["id"],
                "order_id": trade["order"],
                "timestamp": datetime.fromtimestamp(trade["timestamp"] / 1000),
                "symbol": trade["symbol"],
                "side": trade["side"],
                "price": Decimal(str(trade["price"])),
                "quantity": Decimal(str(trade["amount"])),
                "cost": Decimal(str(trade["cost"])),
                "fee": Decimal(str(trade["fee"]["cost"])) if trade.get("fee") else Decimal("0"),
                "fee_currency": trade["fee"]["currency"] if trade.get("fee") else None,
            }
            for trade in trades
        ]

    # ==========================================================================
    # Order Management
    # ==========================================================================

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
    ) -> dict[str, Any]:
        """Create a market order.

        Args:
            symbol: Trading pair symbol
            side: "buy" or "sell"
            quantity: Order quantity

        Returns:
            Order response dictionary
        """
        order = await self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=float(quantity),
        )
        return self._parse_order(order)

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> dict[str, Any]:
        """Create a limit order.

        Args:
            symbol: Trading pair symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Limit price

        Returns:
            Order response dictionary
        """
        order = await self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=float(quantity),
            price=float(price),
        )
        return self._parse_order(order)

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
        limit_price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Create a stop-loss order.

        Args:
            symbol: Trading pair symbol
            side: "buy" or "sell"
            quantity: Order quantity
            stop_price: Stop trigger price
            limit_price: Limit price (if None, creates stop-market)

        Returns:
            Order response dictionary
        """
        order_type = "STOP_LOSS_LIMIT" if limit_price else "STOP_LOSS"
        params = {"stopPrice": float(stop_price)}

        order = await self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=float(quantity),
            price=float(limit_price) if limit_price else None,
            params=params,
        )
        return self._parse_order(order)

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol

        Returns:
            Cancelled order dictionary
        """
        order = await self.exchange.cancel_order(order_id, symbol)
        return self._parse_order(order)

    async def fetch_order(
        self,
        order_id: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Fetch order details.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol

        Returns:
            Order dictionary
        """
        order = await self.exchange.fetch_order(order_id, symbol)
        return self._parse_order(order)

    async def fetch_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all open orders.

        Args:
            symbol: Trading pair symbol (optional)

        Returns:
            List of open order dictionaries
        """
        orders = await self.exchange.fetch_open_orders(symbol)
        return [self._parse_order(order) for order in orders]

    def _parse_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Parse CCXT order response to standard format."""
        return {
            "id": order["id"],
            "client_order_id": order.get("clientOrderId"),
            "timestamp": datetime.fromtimestamp(order["timestamp"] / 1000) if order["timestamp"] else None,
            "symbol": order["symbol"],
            "type": order["type"],
            "side": order["side"],
            "price": Decimal(str(order["price"])) if order["price"] else None,
            "quantity": Decimal(str(order["amount"])),
            "filled": Decimal(str(order["filled"])) if order["filled"] else Decimal("0"),
            "remaining": Decimal(str(order["remaining"])) if order["remaining"] else None,
            "average": Decimal(str(order["average"])) if order["average"] else None,
            "status": order["status"],
            "fee": Decimal(str(order["fee"]["cost"])) if order.get("fee") else Decimal("0"),
            "fee_currency": order["fee"]["currency"] if order.get("fee") else None,
        }

    # ==========================================================================
    # WebSocket Streaming
    # ==========================================================================

    async def watch_ticker(self, symbol: str) -> AsyncIterator[dict[str, Any]]:
        """Watch real-time ticker updates.

        Args:
            symbol: Trading pair symbol

        Yields:
            Ticker update dictionaries
        """
        while True:
            try:
                ticker = await self.exchange.watch_ticker(symbol)
                yield {
                    "symbol": ticker["symbol"],
                    "timestamp": datetime.fromtimestamp(ticker["timestamp"] / 1000) if ticker["timestamp"] else None,
                    "bid": Decimal(str(ticker["bid"])) if ticker["bid"] else None,
                    "ask": Decimal(str(ticker["ask"])) if ticker["ask"] else None,
                    "last": Decimal(str(ticker["last"])) if ticker["last"] else None,
                }
            except Exception as e:
                logger.error(f"Ticker stream error for {symbol}: {e}")
                await asyncio.sleep(1)

    async def watch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame | str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Watch real-time OHLCV updates.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe

        Yields:
            OHLCV update dictionaries
        """
        tf = self.TIMEFRAME_MAP.get(timeframe, timeframe) if isinstance(timeframe, TimeFrame) else timeframe

        while True:
            try:
                ohlcvs = await self.exchange.watch_ohlcv(symbol, tf)
                for candle in ohlcvs:
                    yield {
                        "symbol": symbol,
                        "timeframe": tf,
                        "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                        "open": Decimal(str(candle[1])),
                        "high": Decimal(str(candle[2])),
                        "low": Decimal(str(candle[3])),
                        "close": Decimal(str(candle[4])),
                        "volume": Decimal(str(candle[5])),
                    }
            except Exception as e:
                logger.error(f"OHLCV stream error for {symbol}: {e}")
                await asyncio.sleep(1)

    async def watch_trades(self, symbol: str) -> AsyncIterator[dict[str, Any]]:
        """Watch real-time trade updates.

        Args:
            symbol: Trading pair symbol

        Yields:
            Trade dictionaries
        """
        while True:
            try:
                trades = await self.exchange.watch_trades(symbol)
                for trade in trades:
                    yield {
                        "id": trade["id"],
                        "timestamp": datetime.fromtimestamp(trade["timestamp"] / 1000),
                        "symbol": trade["symbol"],
                        "side": trade["side"],
                        "price": Decimal(str(trade["price"])),
                        "quantity": Decimal(str(trade["amount"])),
                    }
            except Exception as e:
                logger.error(f"Trade stream error for {symbol}: {e}")
                await asyncio.sleep(1)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Get symbol trading information.

        Args:
            symbol: Trading pair symbol

        Returns:
            Symbol info dictionary or None
        """
        if not self._exchange or symbol not in self._exchange.markets:
            return None

        market = self._exchange.markets[symbol]
        return {
            "symbol": symbol,
            "base": market["base"],
            "quote": market["quote"],
            "active": market["active"],
            "precision": {
                "amount": market["precision"]["amount"],
                "price": market["precision"]["price"],
            },
            "limits": {
                "amount": market["limits"]["amount"],
                "price": market["limits"]["price"],
                "cost": market["limits"]["cost"],
            },
            "maker_fee": Decimal(str(market["maker"])) if market.get("maker") else None,
            "taker_fee": Decimal(str(market["taker"])) if market.get("taker") else None,
        }

    def format_quantity(self, symbol: str, quantity: Decimal) -> Decimal:
        """Format quantity according to exchange precision.

        Args:
            symbol: Trading pair symbol
            quantity: Raw quantity

        Returns:
            Formatted quantity
        """
        info = self.get_symbol_info(symbol)
        if info:
            precision = info["precision"]["amount"]
            return Decimal(str(round(float(quantity), precision)))
        return quantity

    def format_price(self, symbol: str, price: Decimal) -> Decimal:
        """Format price according to exchange precision.

        Args:
            symbol: Trading pair symbol
            price: Raw price

        Returns:
            Formatted price
        """
        info = self.get_symbol_info(symbol)
        if info:
            precision = info["precision"]["price"]
            return Decimal(str(round(float(price), precision)))
        return price


# Singleton instance
_binance_connector: BinanceConnector | None = None


async def get_binance_connector() -> BinanceConnector:
    """Get or create Binance connector singleton."""
    global _binance_connector
    if _binance_connector is None:
        _binance_connector = BinanceConnector()
        await _binance_connector.connect()
    return _binance_connector
