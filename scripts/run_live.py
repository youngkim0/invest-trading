"""Live trading script - USE WITH CAUTION."""

import argparse
import asyncio
import signal
from datetime import datetime, timedelta
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


class LiveTrader:
    """Live trading engine - executes real trades."""

    def __init__(
        self,
        strategy: HybridStrategy,
        symbols: list[str],
        max_position_pct: float = 0.05,
        max_daily_loss_pct: float = 0.02,
        dry_run: bool = True,
    ):
        self.strategy = strategy
        self.symbols = symbols
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.dry_run = dry_run

        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.trade_logger = TradeLogger(self.db_manager)

        # State
        self.running = False
        self.daily_pnl = 0.0
        self.starting_equity = 0.0

        # Connectors
        self.binance: BinanceConnector | None = None
        self.alpaca: AlpacaConnector | None = None

        # Data buffers
        self.candle_buffers: dict[str, list] = {s: [] for s in symbols}

        # Risk management
        self.positions: dict[str, dict] = {}
        self.last_trade_time: dict[str, datetime] = {}
        self.min_trade_interval = timedelta(minutes=5)

    async def initialize(self):
        """Initialize connectors and get account info."""
        crypto_symbols = [s for s in self.symbols if "/" in s]
        stock_symbols = [s for s in self.symbols if "/" not in s]

        if crypto_symbols:
            self.binance = BinanceConnector(self.settings.binance)
            await self.binance.connect()

            # Get account balance
            balance = await self.binance.get_balance()
            usdt_balance = balance.get("USDT", {}).get("free", 0)
            logger.info(f"Binance USDT balance: ${usdt_balance:,.2f}")
            self.starting_equity += usdt_balance

        if stock_symbols:
            self.alpaca = AlpacaConnector(self.settings.alpaca)
            await self.alpaca.connect()

            # Get account info
            account = await self.alpaca.get_account()
            if account:
                equity = float(account.get("equity", 0))
                logger.info(f"Alpaca equity: ${equity:,.2f}")
                self.starting_equity += equity

        logger.info(f"Total starting equity: ${self.starting_equity:,.2f}")

        if self.dry_run:
            logger.warning("⚠️  DRY RUN MODE - No real trades will be executed")

    async def start(self):
        """Start live trading."""
        self.running = True

        logger.info("=" * 50)
        logger.info("🚀 STARTING LIVE TRADING")
        logger.info("=" * 50)

        if not self.dry_run:
            logger.warning("⚠️  LIVE MODE - Real trades will be executed!")
            logger.warning("Press Ctrl+C within 10 seconds to cancel...")
            await asyncio.sleep(10)

        await self.initialize()

        # Load historical data
        await self._load_historical_data()

        # Start trading loop
        tasks = []

        for symbol in self.symbols:
            task = asyncio.create_task(self._trading_loop(symbol))
            tasks.append(task)

        # Risk monitoring task
        tasks.append(asyncio.create_task(self._risk_monitor()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Live trading stopped")

    async def stop(self):
        """Stop live trading."""
        self.running = False
        logger.info("Stopping live trading...")

        # Close all positions
        if not self.dry_run:
            for symbol in list(self.positions.keys()):
                await self._close_position(symbol, "shutdown")

        # Disconnect
        if self.binance:
            await self.binance.close()
        if self.alpaca:
            await self.alpaca.disconnect()

        self._print_summary()

    async def _load_historical_data(self):
        """Load historical data for all symbols."""
        import pandas as pd

        for symbol in self.symbols:
            logger.info(f"Loading historical data for {symbol}...")

            end = datetime.utcnow()
            start = end - timedelta(days=7)

            if "/" in symbol and self.binance:
                ohlcv = await self.binance.fetch_ohlcv(
                    symbol=symbol,
                    timeframe="1h",
                    since=int(start.timestamp() * 1000),
                    limit=168,
                )

                if ohlcv:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    self.candle_buffers[symbol] = df.to_dict("records")

            elif self.alpaca:
                bars = await self.alpaca.get_historical_bars(
                    symbol=symbol,
                    timeframe="1Hour",
                    start=start,
                    end=end,
                )

                if bars is not None:
                    self.candle_buffers[symbol] = bars.to_dict("records")

            logger.info(f"Loaded {len(self.candle_buffers[symbol])} candles for {symbol}")

    async def _trading_loop(self, symbol: str):
        """Main trading loop for a symbol."""
        logger.info(f"Starting trading loop for {symbol}")

        while self.running:
            try:
                # Check risk limits
                if not self._check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing trading")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue

                # Get current price
                current_price = await self._get_current_price(symbol)

                if current_price <= 0:
                    await asyncio.sleep(60)
                    continue

                # Update candle buffer
                await self._update_candles(symbol, current_price)

                # Generate signals
                import pandas as pd
                df = pd.DataFrame(self.candle_buffers[symbol])
                if "timestamp" in df.columns:
                    df.set_index("timestamp", inplace=True)

                if len(df) < 60:
                    await asyncio.sleep(60)
                    continue

                signals = self.strategy.generate_signals(symbol, df)

                if signals:
                    await self._process_signal(symbol, signals[0], current_price)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in trading loop for {symbol}: {e}")
                await asyncio.sleep(30)

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        if "/" in symbol and self.binance:
            ticker = await self.binance.fetch_ticker(symbol)
            return ticker.get("last", 0) if ticker else 0
        elif self.alpaca:
            quote = await self.alpaca.get_latest_quote(symbol)
            if quote:
                return (quote.get("ask_price", 0) + quote.get("bid_price", 0)) / 2
        return 0

    async def _update_candles(self, symbol: str, price: float):
        """Update candle buffer with current price."""
        new_candle = {
            "timestamp": datetime.utcnow(),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0,
        }

        self.candle_buffers[symbol].append(new_candle)

        if len(self.candle_buffers[symbol]) > 200:
            self.candle_buffers[symbol] = self.candle_buffers[symbol][-200:]

    async def _process_signal(self, symbol: str, signal, current_price: float):
        """Process trading signal."""
        # Check minimum trade interval
        if symbol in self.last_trade_time:
            if datetime.utcnow() - self.last_trade_time[symbol] < self.min_trade_interval:
                return

        # Check confidence threshold
        if signal.confidence < 0.7:
            return

        if symbol in self.positions:
            # Check for exit
            position = self.positions[symbol]
            should_exit = False

            if position["side"] == "long" and signal.signal_type.value in ["sell", "strong_sell"]:
                should_exit = True
            elif position["side"] == "short" and signal.signal_type.value in ["buy", "strong_buy"]:
                should_exit = True

            # Check stop loss
            if position["side"] == "long":
                if current_price <= position["stop_loss"]:
                    should_exit = True
                    logger.warning(f"Stop loss triggered for {symbol}")
            else:
                if current_price >= position["stop_loss"]:
                    should_exit = True
                    logger.warning(f"Stop loss triggered for {symbol}")

            if should_exit:
                await self._close_position(symbol, "signal")

        else:
            # Check for entry
            if signal.signal_type.value in ["buy", "strong_buy"]:
                await self._open_position(symbol, "long", current_price, signal)
            elif signal.signal_type.value in ["sell", "strong_sell"]:
                await self._open_position(symbol, "short", current_price, signal)

    def _check_risk_limits(self) -> bool:
        """Check if within risk limits."""
        # Check daily loss limit
        if self.starting_equity > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.starting_equity
            if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
                logger.error(f"Daily loss limit reached: {daily_loss_pct:.2%}")
                return False

        return True

    async def _open_position(self, symbol: str, side: str, price: float, signal):
        """Open a position."""
        if symbol in self.positions:
            return

        # Calculate position size
        if "/" in symbol and self.binance:
            balance = await self.binance.get_balance()
            available = balance.get("USDT", {}).get("free", 0)
        elif self.alpaca:
            account = await self.alpaca.get_account()
            available = float(account.get("buying_power", 0)) if account else 0
        else:
            available = 0

        position_value = available * self.max_position_pct
        quantity = position_value / price

        # Calculate stop loss
        stop_loss_pct = 0.02  # 2% stop loss
        if side == "long":
            stop_loss = price * (1 - stop_loss_pct)
        else:
            stop_loss = price * (1 + stop_loss_pct)

        logger.info(
            f"{'🟢' if not self.dry_run else '🔵'} "
            f"{'LIVE' if not self.dry_run else 'DRY'} {side.upper()} {symbol} "
            f"@ {price:.2f} qty={quantity:.4f} SL={stop_loss:.2f}"
        )

        if not self.dry_run:
            # Execute real order
            try:
                if "/" in symbol and self.binance:
                    order = await self.binance.create_market_order(
                        symbol=symbol,
                        side="buy" if side == "long" else "sell",
                        amount=quantity,
                    )
                elif self.alpaca:
                    order = await self.alpaca.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side="buy" if side == "long" else "sell",
                        order_type="market",
                        time_in_force="day",
                    )

                logger.info(f"Order executed: {order}")

            except Exception as e:
                logger.error(f"Failed to execute order: {e}")
                return

        self.positions[symbol] = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "entry_time": datetime.utcnow(),
        }

        self.last_trade_time[symbol] = datetime.utcnow()

        # Log to journal
        trade = TradeEntry(
            symbol=symbol,
            exchange="binance" if "/" in symbol else "alpaca",
            side=OrderSide.BUY if side == "long" else OrderSide.SELL,
            entry_price=Decimal(str(price)),
            quantity=Decimal(str(quantity)),
            entry_time=datetime.utcnow(),
            signal_source=SignalSource.HYBRID,
            signal_confidence=Decimal(str(signal.confidence)),
            strategy_name=self.strategy.name,
        )

        position_id = await self.trade_logger.log_entry(trade)
        self.positions[symbol]["position_id"] = position_id

    async def _close_position(self, symbol: str, reason: str):
        """Close a position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        current_price = await self._get_current_price(symbol)

        # Calculate PnL
        if position["side"] == "long":
            pnl = (current_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - current_price) * position["quantity"]

        self.daily_pnl += pnl

        logger.info(
            f"{'🔴' if not self.dry_run else '🔵'} "
            f"{'LIVE' if not self.dry_run else 'DRY'} CLOSE {symbol} "
            f"@ {current_price:.2f} PnL=${pnl:.2f} ({reason})"
        )

        if not self.dry_run:
            # Execute real order
            try:
                if "/" in symbol and self.binance:
                    side = "sell" if position["side"] == "long" else "buy"
                    order = await self.binance.create_market_order(
                        symbol=symbol,
                        side=side,
                        amount=position["quantity"],
                    )
                elif self.alpaca:
                    await self.alpaca.close_position(symbol)

            except Exception as e:
                logger.error(f"Failed to close position: {e}")

        # Log to journal
        if "position_id" in position:
            await self.trade_logger.log_exit(
                position_id=position["position_id"],
                exit_price=Decimal(str(current_price)),
                exit_time=datetime.utcnow(),
                exit_reasoning=reason,
            )

        del self.positions[symbol]
        self.last_trade_time[symbol] = datetime.utcnow()

    async def _risk_monitor(self):
        """Monitor risk in background."""
        while self.running:
            try:
                # Check positions for stop loss
                for symbol in list(self.positions.keys()):
                    position = self.positions[symbol]
                    current_price = await self._get_current_price(symbol)

                    if position["side"] == "long":
                        if current_price <= position["stop_loss"]:
                            await self._close_position(symbol, "stop_loss")
                    else:
                        if current_price >= position["stop_loss"]:
                            await self._close_position(symbol, "stop_loss")

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
                await asyncio.sleep(5)

    def _print_summary(self):
        """Print trading summary."""
        stats = self.trade_logger.get_statistics()

        print("\n" + "=" * 50)
        print("LIVE TRADING SUMMARY")
        print("=" * 50)
        print(f"Mode: {'LIVE' if not self.dry_run else 'DRY RUN'}")
        print(f"Daily PnL: ${self.daily_pnl:,.2f}")
        print("-" * 50)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        print(f"Total PnL: ${stats['total_pnl']:,.2f}")
        print("=" * 50)


async def main():
    """Main live trading function."""
    parser = argparse.ArgumentParser(description="Run live trading")

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC/USDT"],
        help="Symbols to trade",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.05,
        help="Max position size as fraction of capital",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=0.02,
        help="Max daily loss as fraction of capital",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Path to RL model",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute real trades (default is dry run)",
    )

    args = parser.parse_args()

    # Setup logging
    logger.add(
        f"logs/live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB",
    )

    if args.live:
        logger.warning("=" * 50)
        logger.warning("⚠️  LIVE TRADING MODE ENABLED")
        logger.warning("⚠️  REAL MONEY WILL BE USED")
        logger.warning("=" * 50)

    # Create strategy
    strategy_config = StrategyConfig()
    strategy = create_hybrid_strategy(
        config=strategy_config,
        rl_model_path=args.rl_model,
    )

    # Create live trader
    trader = LiveTrader(
        strategy=strategy,
        symbols=args.symbols,
        max_position_pct=args.max_position,
        max_daily_loss_pct=args.max_daily_loss,
        dry_run=not args.live,
    )

    # Handle shutdown
    def handle_shutdown(sig, frame):
        logger.info("Shutdown signal received...")
        asyncio.create_task(trader.stop())

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run
    await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
