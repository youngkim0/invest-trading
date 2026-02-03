"""Paper trading script for live simulation."""

import argparse
import asyncio
import signal
from datetime import datetime
from decimal import Decimal
from typing import Any

from loguru import logger

from config import get_settings
from config.strategies import StrategyConfig
from core.engine.binance_connector import BinanceConnector
from core.engine.alpaca_connector import AlpacaConnector
from core.strategies.hybrid_strategy import HybridStrategy, create_hybrid_strategy
from data.storage.repository import DatabaseManager
from journal.trade_logger import TradeEntry, TradeLogger
from data.storage.models import OrderSide, SignalSource


class PaperTrader:
    """Paper trading engine for live simulation."""

    def __init__(
        self,
        strategy: HybridStrategy,
        symbols: list[str],
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.1,
    ):
        self.strategy = strategy
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct

        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.trade_logger = TradeLogger(self.db_manager)

        # Paper trading state
        self.capital = initial_capital
        self.positions: dict[str, dict] = {}
        self.running = False

        # Connectors
        self.binance: BinanceConnector | None = None
        self.alpaca: AlpacaConnector | None = None

        # Data buffers
        self.candle_buffers: dict[str, list] = {s: [] for s in symbols}

    async def initialize(self):
        """Initialize connectors."""
        # Initialize Binance for crypto
        crypto_symbols = [s for s in self.symbols if "/" in s]
        if crypto_symbols:
            self.binance = BinanceConnector(self.settings.binance)
            await self.binance.connect()
            logger.info("Binance connector initialized")

        # Initialize Alpaca for stocks
        stock_symbols = [s for s in self.symbols if "/" not in s]
        if stock_symbols:
            self.alpaca = AlpacaConnector(self.settings.alpaca)
            await self.alpaca.connect()
            logger.info("Alpaca connector initialized")

    async def start(self):
        """Start paper trading."""
        self.running = True
        logger.info("Starting paper trading...")

        await self.initialize()

        # Start data streams
        tasks = []

        for symbol in self.symbols:
            if "/" in symbol:
                # Crypto - use Binance
                task = asyncio.create_task(
                    self._stream_crypto(symbol)
                )
            else:
                # Stock - use Alpaca
                task = asyncio.create_task(
                    self._stream_stock(symbol)
                )
            tasks.append(task)

        # Run until stopped
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Paper trading stopped")

    async def stop(self):
        """Stop paper trading."""
        self.running = False

        # Close all positions
        for symbol, position in list(self.positions.items()):
            await self._close_position(symbol)

        # Disconnect
        if self.binance:
            await self.binance.close()
        if self.alpaca:
            await self.alpaca.disconnect()

        # Print final stats
        self._print_summary()

    async def _stream_crypto(self, symbol: str):
        """Stream crypto data and trade."""
        logger.info(f"Starting crypto stream for {symbol}")

        # Get initial historical data
        import pandas as pd
        from datetime import timedelta

        end = datetime.utcnow()
        start = end - timedelta(days=7)

        ohlcv = await self.binance.fetch_ohlcv(
            symbol=symbol,
            timeframe="1h",
            since=int(start.timestamp() * 1000),
            limit=168,  # 7 days of hourly data
        )

        if ohlcv:
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            self.candle_buffers[symbol] = df.to_dict("records")

        # Stream real-time data
        while self.running:
            try:
                ticker = await self.binance.fetch_ticker(symbol)
                if ticker:
                    await self._on_price_update(symbol, ticker)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in crypto stream: {e}")
                await asyncio.sleep(5)

    async def _stream_stock(self, symbol: str):
        """Stream stock data and trade."""
        logger.info(f"Starting stock stream for {symbol}")

        # Get initial historical data
        from datetime import timedelta

        end = datetime.utcnow()
        start = end - timedelta(days=7)

        bars = await self.alpaca.get_historical_bars(
            symbol=symbol,
            timeframe="1Hour",
            start=start,
            end=end,
        )

        if bars is not None:
            self.candle_buffers[symbol] = bars.to_dict("records")

        # Stream real-time quotes
        while self.running:
            try:
                quote = await self.alpaca.get_latest_quote(symbol)
                if quote:
                    await self._on_price_update(symbol, {
                        "last": (quote.get("ask_price", 0) + quote.get("bid_price", 0)) / 2,
                        "bid": quote.get("bid_price"),
                        "ask": quote.get("ask_price"),
                    })

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in stock stream: {e}")
                await asyncio.sleep(5)

    async def _on_price_update(self, symbol: str, ticker: dict):
        """Handle price update."""
        import pandas as pd

        current_price = ticker.get("last") or ticker.get("close", 0)

        if current_price <= 0:
            return

        # Update candle buffer with new "candle"
        new_candle = {
            "timestamp": datetime.utcnow(),
            "open": current_price,
            "high": current_price,
            "low": current_price,
            "close": current_price,
            "volume": ticker.get("volume", 0),
        }

        self.candle_buffers[symbol].append(new_candle)

        # Keep last 200 candles
        if len(self.candle_buffers[symbol]) > 200:
            self.candle_buffers[symbol] = self.candle_buffers[symbol][-200:]

        # Convert to DataFrame for strategy
        df = pd.DataFrame(self.candle_buffers[symbol])
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)

        if len(df) < 60:
            return  # Need enough data

        # Generate signals
        signals = self.strategy.generate_signals(symbol, df)

        if not signals:
            return

        signal = signals[0]

        # Execute paper trades
        if symbol in self.positions:
            # Check for exit
            position = self.positions[symbol]
            if position["side"] == "long" and signal.signal_type.value in ["sell", "strong_sell"]:
                await self._close_position(symbol)
            elif position["side"] == "short" and signal.signal_type.value in ["buy", "strong_buy"]:
                await self._close_position(symbol)
        else:
            # Check for entry
            if signal.confidence >= 0.6:
                if signal.signal_type.value in ["buy", "strong_buy"]:
                    await self._open_position(symbol, "long", current_price)
                elif signal.signal_type.value in ["sell", "strong_sell"]:
                    await self._open_position(symbol, "short", current_price)

    async def _open_position(self, symbol: str, side: str, price: float):
        """Open a paper position."""
        # Calculate position size
        position_value = self.capital * self.max_position_pct
        quantity = position_value / price

        self.positions[symbol] = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "entry_time": datetime.utcnow(),
        }

        logger.info(
            f"📈 PAPER {side.upper()} {symbol} @ {price:.2f} "
            f"qty={quantity:.4f} value=${position_value:.2f}"
        )

        # Log to journal
        trade = TradeEntry(
            symbol=symbol,
            exchange="paper",
            side=OrderSide.BUY if side == "long" else OrderSide.SELL,
            entry_price=Decimal(str(price)),
            quantity=Decimal(str(quantity)),
            entry_time=datetime.utcnow(),
            signal_source=SignalSource.HYBRID,
            strategy_name=self.strategy.name,
        )

        position_id = await self.trade_logger.log_entry(trade)
        self.positions[symbol]["position_id"] = position_id

    async def _close_position(self, symbol: str):
        """Close a paper position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Get current price
        if "/" in symbol and self.binance:
            ticker = await self.binance.fetch_ticker(symbol)
            current_price = ticker.get("last", 0) if ticker else 0
        elif self.alpaca:
            quote = await self.alpaca.get_latest_quote(symbol)
            current_price = (quote.get("ask_price", 0) + quote.get("bid_price", 0)) / 2 if quote else 0
        else:
            current_price = position["entry_price"]

        # Calculate PnL
        if position["side"] == "long":
            pnl = (current_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - current_price) * position["quantity"]

        self.capital += pnl

        logger.info(
            f"📉 PAPER CLOSE {symbol} @ {current_price:.2f} "
            f"PnL=${pnl:.2f} Capital=${self.capital:.2f}"
        )

        # Log to journal
        if "position_id" in position:
            await self.trade_logger.log_exit(
                position_id=position["position_id"],
                exit_price=Decimal(str(current_price)),
                exit_time=datetime.utcnow(),
            )

        del self.positions[symbol]

    def _print_summary(self):
        """Print trading summary."""
        stats = self.trade_logger.get_statistics()

        print("\n" + "=" * 50)
        print("PAPER TRADING SUMMARY")
        print("=" * 50)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: {((self.capital / self.initial_capital) - 1):.2%}")
        print("-" * 50)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        print(f"Total PnL: ${stats['total_pnl']:,.2f}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print("=" * 50)


async def main():
    """Main paper trading function."""
    parser = argparse.ArgumentParser(description="Run paper trading")

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC/USDT"],
        help="Symbols to trade",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.1,
        help="Max position size as fraction of capital",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Path to RL model",
    )

    args = parser.parse_args()

    # Setup logging
    logger.add(
        f"logs/paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB",
    )

    logger.info("Starting paper trading session...")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Capital: ${args.capital:,.2f}")

    # Create strategy
    strategy_config = StrategyConfig()
    strategy = create_hybrid_strategy(
        config=strategy_config,
        rl_model_path=args.rl_model,
    )

    # Create paper trader
    trader = PaperTrader(
        strategy=strategy,
        symbols=args.symbols,
        initial_capital=args.capital,
        max_position_pct=args.max_position,
    )

    # Handle shutdown
    def handle_shutdown(sig, frame):
        logger.info("Shutting down...")
        asyncio.create_task(trader.stop())

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run
    await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
