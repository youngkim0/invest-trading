"""Order management for Binance USDT-M Futures via CCXT.

Handles: connection, leverage/margin setup, market orders, stop-loss placement,
order cancellation, position queries, and symbol format conversion.
"""

import asyncio
from typing import Any

import ccxt.async_support as ccxt
from loguru import logger


class OrderManager:
    """Manages order execution on Binance USDT-M Futures."""

    # Binance raw format -> CCXT format
    SYMBOL_MAP = {
        "BTCUSDT": "BTC/USDT",
        "ETHUSDT": "ETH/USDT",
        "XRPUSDT": "XRP/USDT",
    }

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._exchange: ccxt.binance | None = None
        self._leverage_set: set[str] = set()
        self._margin_set: set[str] = set()
        self._default_leverage = 10

    def ccxt_symbol(self, symbol: str) -> str:
        """Convert BTCUSDT -> BTC/USDT for CCXT."""
        return self.SYMBOL_MAP.get(symbol, symbol)

    async def connect(self, leverage: int = 10) -> None:
        """Connect to Binance Futures."""
        self._default_leverage = leverage

        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }

        if self.testnet:
            options["urls"] = {
                "api": {
                    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                }
            }

        self._exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": options,
        })

        await self._exchange.load_markets()
        mode = "testnet" if self.testnet else "MAINNET"
        logger.info(f"Connected to Binance Futures ({mode})")

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    @property
    def exchange(self) -> ccxt.binance:
        if not self._exchange:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._exchange

    # =========================================================================
    # Setup
    # =========================================================================

    async def ensure_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for symbol if not already set."""
        ccxt_sym = self.ccxt_symbol(symbol)
        key = f"{ccxt_sym}:{leverage}"
        if key not in self._leverage_set:
            try:
                await self.exchange.set_leverage(leverage, ccxt_sym)
                self._leverage_set.add(key)
                logger.info(f"Set leverage {leverage}x for {symbol}")
            except Exception as e:
                if "No need to change" not in str(e):
                    logger.warning(f"Could not set leverage for {symbol}: {e}")
                self._leverage_set.add(key)  # Don't retry

    async def ensure_margin_mode(self, symbol: str, mode: str = "isolated") -> None:
        """Set margin mode for symbol if not already set."""
        ccxt_sym = self.ccxt_symbol(symbol)
        if ccxt_sym not in self._margin_set:
            try:
                await self.exchange.set_margin_mode(mode, ccxt_sym)
                self._margin_set.add(ccxt_sym)
                logger.info(f"Set {mode} margin for {symbol}")
            except Exception as e:
                if "No need to change" not in str(e):
                    logger.warning(f"Could not set margin mode for {symbol}: {e}")
                self._margin_set.add(ccxt_sym)

    # =========================================================================
    # Account Queries
    # =========================================================================

    async def get_balance(self) -> dict[str, float]:
        """Get USDT futures balance."""
        balance = await self.exchange.fetch_balance({"type": "future"})
        usdt = balance.get("USDT", {})
        return {
            "total": float(usdt.get("total", 0)),
            "free": float(usdt.get("free", 0)),
            "used": float(usdt.get("used", 0)),
        }

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions from exchange."""
        positions = await self.exchange.fetch_positions()
        return [
            {
                "symbol": p["symbol"],
                "side": "long" if p["side"] == "long" else "short",
                "quantity": abs(float(p["contracts"] or 0)),
                "entry_price": float(p["entryPrice"] or 0),
                "unrealized_pnl": float(p["unrealizedPnl"] or 0),
                "leverage": int(p["leverage"] or 0),
                "margin_type": p.get("marginType", ""),
            }
            for p in positions
            if abs(float(p["contracts"] or 0)) > 0
        ]

    # =========================================================================
    # Order Placement
    # =========================================================================

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int = 10,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place a market order on futures.

        Args:
            symbol: Binance format (BTCUSDT)
            side: "buy" or "sell"
            quantity: Order quantity in base asset
            leverage: Leverage to use
            reduce_only: If True, only reduces existing position
        """
        ccxt_sym = self.ccxt_symbol(symbol)

        await self.ensure_margin_mode(symbol)
        await self.ensure_leverage(symbol, leverage)

        params = {}
        if reduce_only:
            params["reduceOnly"] = True

        order = await self.exchange.create_order(
            symbol=ccxt_sym,
            type="market",
            side=side,
            amount=quantity,
            params=params,
        )

        result = self._parse_order(order)
        logger.info(
            f"Market {side.upper()} {symbol}: qty={quantity:.6f} "
            f"filled={result['filled']:.6f} @ avg={result['average']}"
        )
        return result

    async def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> dict[str, Any]:
        """Place a STOP_MARKET order (exchange-side SL).

        Args:
            symbol: Binance format (BTCUSDT)
            side: "sell" for long SL, "buy" for short SL
            quantity: Position quantity
            stop_price: Trigger price
        """
        ccxt_sym = self.ccxt_symbol(symbol)

        order = await self.exchange.create_order(
            symbol=ccxt_sym,
            type="STOP_MARKET",
            side=side,
            amount=quantity,
            params={
                "stopPrice": stop_price,
                "reduceOnly": True,
            },
        )

        result = self._parse_order(order)
        logger.info(f"SL placed: {side} {symbol} @ ${stop_price:,.2f} (ID: {result['id']})")
        return result

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> dict[str, Any]:
        """Place a TAKE_PROFIT_MARKET order on exchange.

        Args:
            symbol: Binance format (BTCUSDT)
            side: "sell" for long TP, "buy" for short TP
            quantity: Position quantity
            price: Trigger price
        """
        ccxt_sym = self.ccxt_symbol(symbol)

        order = await self.exchange.create_order(
            symbol=ccxt_sym,
            type="TAKE_PROFIT_MARKET",
            side=side,
            amount=quantity,
            params={
                "stopPrice": price,
                "reduceOnly": True,
            },
        )

        result = self._parse_order(order)
        logger.info(f"TP placed: {side} {symbol} @ ${price:,.2f} (ID: {result['id']})")
        return result

    # =========================================================================
    # Order Management
    # =========================================================================

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order. Returns True if cancelled successfully."""
        try:
            ccxt_sym = self.ccxt_symbol(symbol)
            await self.exchange.cancel_order(order_id, ccxt_sym)
            logger.info(f"Cancelled order {order_id} on {symbol}")
            return True
        except ccxt.OrderNotFound:
            logger.debug(f"Order {order_id} already filled/cancelled")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for symbol. Returns count cancelled."""
        try:
            ccxt_sym = self.ccxt_symbol(symbol)
            orders = await self.exchange.fetch_open_orders(ccxt_sym)
            cancelled = 0
            for order in orders:
                if await self.cancel_order(symbol, order["id"]):
                    cancelled += 1
            return cancelled
        except Exception as e:
            logger.error(f"Failed to cancel orders for {symbol}: {e}")
            return 0

    async def fetch_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        """Fetch order status."""
        ccxt_sym = self.ccxt_symbol(symbol)
        order = await self.exchange.fetch_order(order_id, ccxt_sym)
        return self._parse_order(order)

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch all open orders."""
        ccxt_sym = self.ccxt_symbol(symbol) if symbol else None
        orders = await self.exchange.fetch_open_orders(ccxt_sym)
        return [self._parse_order(o) for o in orders]

    # =========================================================================
    # Formatting
    # =========================================================================

    def format_quantity(self, symbol: str, quantity: float) -> float:
        """Format quantity to exchange precision."""
        ccxt_sym = self.ccxt_symbol(symbol)
        if ccxt_sym in self.exchange.markets:
            market = self.exchange.markets[ccxt_sym]
            precision = market.get("precision", {}).get("amount", 8)
            return float(self.exchange.amount_to_precision(ccxt_sym, quantity))
        return quantity

    def format_price(self, symbol: str, price: float) -> float:
        """Format price to exchange precision."""
        ccxt_sym = self.ccxt_symbol(symbol)
        if ccxt_sym in self.exchange.markets:
            return float(self.exchange.price_to_precision(ccxt_sym, price))
        return price

    # =========================================================================
    # Internal
    # =========================================================================

    def _parse_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Parse CCXT order to standard format."""
        return {
            "id": order["id"],
            "client_order_id": order.get("clientOrderId"),
            "symbol": order["symbol"],
            "type": order["type"],
            "side": order["side"],
            "price": float(order["price"]) if order["price"] else None,
            "quantity": float(order["amount"]) if order["amount"] else 0,
            "filled": float(order["filled"]) if order["filled"] else 0,
            "remaining": float(order["remaining"]) if order["remaining"] else None,
            "average": float(order["average"]) if order["average"] else None,
            "status": order["status"],
            "fee": (
                float(order["fee"]["cost"])
                if order.get("fee") and order["fee"].get("cost")
                else 0
            ),
        }
