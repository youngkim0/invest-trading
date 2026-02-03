"""Alpaca exchange connector for stock trading."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)
from loguru import logger

from config import get_exchange_settings
from config.strategies import TimeFrame


class AlpacaConnector:
    """Connector for Alpaca stock trading API."""

    TIMEFRAME_MAP = {
        TimeFrame.M1: AlpacaTimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame.M5: AlpacaTimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame.M15: AlpacaTimeFrame(15, TimeFrameUnit.Minute),
        TimeFrame.M30: AlpacaTimeFrame(30, TimeFrameUnit.Minute),
        TimeFrame.H1: AlpacaTimeFrame(1, TimeFrameUnit.Hour),
        TimeFrame.H4: AlpacaTimeFrame(4, TimeFrameUnit.Hour),
        TimeFrame.D1: AlpacaTimeFrame(1, TimeFrameUnit.Day),
        TimeFrame.W1: AlpacaTimeFrame(1, TimeFrameUnit.Week),
    }

    def __init__(self):
        """Initialize Alpaca connector."""
        self.config = get_exchange_settings().alpaca
        self._trading_client: TradingClient | None = None
        self._data_client: StockHistoricalDataClient | None = None
        self._stream_client: StockDataStream | None = None
        self._stream_handlers: dict[str, Any] = {}

    def connect(self) -> None:
        """Connect to Alpaca API."""
        api_key = self.config.active_api_key
        api_secret = self.config.active_api_secret

        # Trading client
        self._trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=self.config.use_paper,
        )

        # Historical data client
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        # Streaming client
        self._stream_client = StockDataStream(
            api_key=api_key,
            secret_key=api_secret,
        )

        logger.info(
            f"Connected to Alpaca {'paper' if self.config.use_paper else 'live'} trading"
        )

    def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if self._stream_client:
            asyncio.create_task(self._stream_client.close())
        self._trading_client = None
        self._data_client = None
        self._stream_client = None
        logger.info("Disconnected from Alpaca")

    @property
    def trading_client(self) -> TradingClient:
        """Get trading client."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca. Call connect() first.")
        return self._trading_client

    @property
    def data_client(self) -> StockHistoricalDataClient:
        """Get data client."""
        if not self._data_client:
            raise RuntimeError("Not connected to Alpaca. Call connect() first.")
        return self._data_client

    @property
    def stream_client(self) -> StockDataStream:
        """Get stream client."""
        if not self._stream_client:
            raise RuntimeError("Not connected to Alpaca. Call connect() first.")
        return self._stream_client

    # ==========================================================================
    # Market Data
    # ==========================================================================

    def fetch_bars(
        self,
        symbol: str,
        timeframe: TimeFrame | str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch historical bar data.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            timeframe: Bar timeframe
            start: Start time
            end: End time
            limit: Maximum number of bars

        Returns:
            List of bar dictionaries
        """
        tf = self.TIMEFRAME_MAP.get(timeframe) if isinstance(timeframe, TimeFrame) else None
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Default to last 30 days if no start specified
        if start is None:
            start = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=30)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )

        bars = self.data_client.get_stock_bars(request)

        if symbol not in bars:
            return []

        return [
            {
                "timestamp": bar.timestamp,
                "open": Decimal(str(bar.open)),
                "high": Decimal(str(bar.high)),
                "low": Decimal(str(bar.low)),
                "close": Decimal(str(bar.close)),
                "volume": Decimal(str(bar.volume)),
                "vwap": Decimal(str(bar.vwap)) if bar.vwap else None,
                "trade_count": bar.trade_count,
            }
            for bar in bars[symbol]
        ]

    def fetch_latest_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote dictionary
        """
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.data_client.get_stock_latest_quote(request)

        if symbol not in quotes:
            raise ValueError(f"No quote found for {symbol}")

        quote = quotes[symbol]
        return {
            "symbol": symbol,
            "timestamp": quote.timestamp,
            "bid": Decimal(str(quote.bid_price)),
            "bid_size": quote.bid_size,
            "ask": Decimal(str(quote.ask_price)),
            "ask_size": quote.ask_size,
        }

    def fetch_latest_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch latest quotes for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to quotes
        """
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self.data_client.get_stock_latest_quote(request)

        return {
            symbol: {
                "symbol": symbol,
                "timestamp": quote.timestamp,
                "bid": Decimal(str(quote.bid_price)),
                "bid_size": quote.bid_size,
                "ask": Decimal(str(quote.ask_price)),
                "ask_size": quote.ask_size,
            }
            for symbol, quote in quotes.items()
        }

    def fetch_trades(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch recent trades.

        Args:
            symbol: Stock symbol
            start: Start time
            end: End time
            limit: Maximum number of trades

        Returns:
            List of trade dictionaries
        """
        if start is None:
            start = datetime.now(ZoneInfo("America/New_York")) - timedelta(hours=1)

        request = StockTradesRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            limit=limit,
        )

        trades = self.data_client.get_stock_trades(request)

        if symbol not in trades:
            return []

        return [
            {
                "id": str(trade.id),
                "timestamp": trade.timestamp,
                "price": Decimal(str(trade.price)),
                "size": trade.size,
                "exchange": trade.exchange,
            }
            for trade in trades[symbol]
        ]

    # ==========================================================================
    # Account Data
    # ==========================================================================

    def fetch_account(self) -> dict[str, Any]:
        """Fetch account information.

        Returns:
            Account dictionary
        """
        account = self.trading_client.get_account()

        return {
            "id": str(account.id),
            "status": account.status.value,
            "currency": account.currency,
            "cash": Decimal(str(account.cash)),
            "portfolio_value": Decimal(str(account.portfolio_value)),
            "equity": Decimal(str(account.equity)),
            "buying_power": Decimal(str(account.buying_power)),
            "long_market_value": Decimal(str(account.long_market_value)),
            "short_market_value": Decimal(str(account.short_market_value)),
            "initial_margin": Decimal(str(account.initial_margin)),
            "maintenance_margin": Decimal(str(account.maintenance_margin)),
            "daytrade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
        }

    def fetch_positions(self) -> list[dict[str, Any]]:
        """Fetch all open positions.

        Returns:
            List of position dictionaries
        """
        positions = self.trading_client.get_all_positions()

        return [
            {
                "symbol": pos.symbol,
                "quantity": Decimal(str(pos.qty)),
                "side": "long" if float(pos.qty) > 0 else "short",
                "market_value": Decimal(str(pos.market_value)),
                "cost_basis": Decimal(str(pos.cost_basis)),
                "avg_entry_price": Decimal(str(pos.avg_entry_price)),
                "current_price": Decimal(str(pos.current_price)),
                "unrealized_pl": Decimal(str(pos.unrealized_pl)),
                "unrealized_plpc": Decimal(str(pos.unrealized_plpc)),
                "change_today": Decimal(str(pos.change_today)),
            }
            for pos in positions
        ]

    def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        """Fetch position for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position dictionary or None
        """
        try:
            pos = self.trading_client.get_open_position(symbol)
            return {
                "symbol": pos.symbol,
                "quantity": Decimal(str(pos.qty)),
                "side": "long" if float(pos.qty) > 0 else "short",
                "market_value": Decimal(str(pos.market_value)),
                "cost_basis": Decimal(str(pos.cost_basis)),
                "avg_entry_price": Decimal(str(pos.avg_entry_price)),
                "current_price": Decimal(str(pos.current_price)),
                "unrealized_pl": Decimal(str(pos.unrealized_pl)),
            }
        except Exception:
            return None

    # ==========================================================================
    # Order Management
    # ==========================================================================

    def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Create a market order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            time_in_force: Order duration (day, gtc, ioc, fok)

        Returns:
            Order response dictionary
        """
        request = MarketOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce(time_in_force.upper()),
        )

        order = self.trading_client.submit_order(request)
        return self._parse_order(order)

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        limit_price: Decimal,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Create a limit order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            limit_price: Limit price
            time_in_force: Order duration

        Returns:
            Order response dictionary
        """
        request = LimitOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce(time_in_force.upper()),
            limit_price=float(limit_price),
        )

        order = self.trading_client.submit_order(request)
        return self._parse_order(order)

    def create_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Create a stop order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            stop_price: Stop trigger price
            time_in_force: Order duration

        Returns:
            Order response dictionary
        """
        request = StopOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce(time_in_force.upper()),
            stop_price=float(stop_price),
        )

        order = self.trading_client.submit_order(request)
        return self._parse_order(order)

    def create_stop_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
        limit_price: Decimal,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Create a stop-limit order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            stop_price: Stop trigger price
            limit_price: Limit price after trigger
            time_in_force: Order duration

        Returns:
            Order response dictionary
        """
        request = StopLimitOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce(time_in_force.upper()),
            stop_price=float(stop_price),
            limit_price=float(limit_price),
        )

        order = self.trading_client.submit_order(request)
        return self._parse_order(order)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            Cancelled order dictionary
        """
        self.trading_client.cancel_order_by_id(order_id)
        order = self.trading_client.get_order_by_id(order_id)
        return self._parse_order(order)

    def cancel_all_orders(self) -> int:
        """Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        result = self.trading_client.cancel_orders()
        return len(result)

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        """Fetch order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order dictionary
        """
        order = self.trading_client.get_order_by_id(order_id)
        return self._parse_order(order)

    def fetch_orders(
        self,
        status: str = "open",
        limit: int = 100,
        after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch orders with filters.

        Args:
            status: Order status filter (open, closed, all)
            limit: Maximum number of orders
            after: Only fetch orders after this time

        Returns:
            List of order dictionaries
        """
        request = GetOrdersRequest(
            status=status,
            limit=limit,
            after=after,
        )

        orders = self.trading_client.get_orders(request)
        return [self._parse_order(order) for order in orders]

    def _parse_order(self, order: Any) -> dict[str, Any]:
        """Parse Alpaca order to standard format."""
        return {
            "id": str(order.id),
            "client_order_id": order.client_order_id,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
            "symbol": order.symbol,
            "type": order.order_type.value if order.order_type else None,
            "side": order.side.value if order.side else None,
            "quantity": Decimal(str(order.qty)) if order.qty else None,
            "filled_quantity": Decimal(str(order.filled_qty)) if order.filled_qty else Decimal("0"),
            "limit_price": Decimal(str(order.limit_price)) if order.limit_price else None,
            "stop_price": Decimal(str(order.stop_price)) if order.stop_price else None,
            "filled_avg_price": Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
            "status": order.status.value if order.status else None,
            "time_in_force": order.time_in_force.value if order.time_in_force else None,
        }

    # ==========================================================================
    # Streaming
    # ==========================================================================

    async def subscribe_trades(
        self,
        symbols: list[str],
        handler: Any,
    ) -> None:
        """Subscribe to real-time trade updates.

        Args:
            symbols: List of stock symbols
            handler: Async callback function for trade updates
        """
        self.stream_client.subscribe_trades(handler, *symbols)

    async def subscribe_quotes(
        self,
        symbols: list[str],
        handler: Any,
    ) -> None:
        """Subscribe to real-time quote updates.

        Args:
            symbols: List of stock symbols
            handler: Async callback function for quote updates
        """
        self.stream_client.subscribe_quotes(handler, *symbols)

    async def subscribe_bars(
        self,
        symbols: list[str],
        handler: Any,
    ) -> None:
        """Subscribe to real-time bar updates.

        Args:
            symbols: List of stock symbols
            handler: Async callback function for bar updates
        """
        self.stream_client.subscribe_bars(handler, *symbols)

    async def start_streaming(self) -> None:
        """Start the streaming connection."""
        await self.stream_client.run()

    async def stop_streaming(self) -> None:
        """Stop the streaming connection."""
        await self.stream_client.stop()

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        Returns:
            True if market is open
        """
        clock = self.trading_client.get_clock()
        return clock.is_open

    def get_market_hours(self) -> dict[str, Any]:
        """Get market hours for today.

        Returns:
            Market hours dictionary
        """
        clock = self.trading_client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": clock.next_open,
            "next_close": clock.next_close,
        }


# Singleton instance
_alpaca_connector: AlpacaConnector | None = None


def get_alpaca_connector() -> AlpacaConnector:
    """Get or create Alpaca connector singleton."""
    global _alpaca_connector
    if _alpaca_connector is None:
        _alpaca_connector = AlpacaConnector()
        _alpaca_connector.connect()
    return _alpaca_connector
