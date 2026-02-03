#!/usr/bin/env python
"""Simple paper trading script using Binance real-time data and Supabase."""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import uuid

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import pandas as pd
import numpy as np
from loguru import logger

from data.collectors.market_data import MarketDataCollector
from data.storage.supabase_client import (
    TradeLogRepository,
    SignalRepository,
    PerformanceRepository,
)


class TechnicalSignalGenerator:
    """Generate trading signals from technical indicators."""

    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def calculate_macd(self, prices: pd.Series) -> dict:
        """Calculate MACD."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return {
            "macd": float(macd.iloc[-1]),
            "signal": float(signal.iloc[-1]),
            "histogram": float(histogram.iloc[-1]),
        }

    def calculate_sma_crossover(self, prices: pd.Series) -> dict:
        """Calculate SMA crossover."""
        sma20 = prices.rolling(20).mean()
        sma50 = prices.rolling(50).mean()
        return {
            "sma20": float(sma20.iloc[-1]),
            "sma50": float(sma50.iloc[-1]),
            "bullish_cross": bool(sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-2] <= sma50.iloc[-2]),
            "bearish_cross": bool(sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-2] >= sma50.iloc[-2]),
        }

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """Generate trading signal based on technical indicators."""
        if len(df) < 60:
            return {"signal": "hold", "confidence": 0.5, "reasoning": "Insufficient data"}

        prices = df["close"]
        current_price = float(prices.iloc[-1])

        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        sma = self.calculate_sma_crossover(prices)

        # Scoring system
        buy_score = 0
        sell_score = 0
        reasons = []

        # RSI signals
        if rsi < self.rsi_oversold:
            buy_score += 2
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            sell_score += 2
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 45:
            buy_score += 1
        elif rsi > 55:
            sell_score += 1

        # MACD signals
        if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
            buy_score += 1
            reasons.append("MACD bullish")
        elif macd["histogram"] < 0 and macd["macd"] < macd["signal"]:
            sell_score += 1
            reasons.append("MACD bearish")

        # SMA crossover
        if sma["bullish_cross"]:
            buy_score += 2
            reasons.append("SMA bullish crossover")
        elif sma["bearish_cross"]:
            sell_score += 2
            reasons.append("SMA bearish crossover")
        elif sma["sma20"] > sma["sma50"]:
            buy_score += 0.5
        else:
            sell_score += 0.5

        # Determine signal
        total_score = buy_score - sell_score
        max_score = 5.0

        if total_score >= 2:
            signal = "strong_buy" if total_score >= 3.5 else "buy"
            confidence = min(0.9, 0.6 + (total_score / max_score) * 0.3)
        elif total_score <= -2:
            signal = "strong_sell" if total_score <= -3.5 else "sell"
            confidence = min(0.9, 0.6 + (abs(total_score) / max_score) * 0.3)
        else:
            signal = "hold"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": ", ".join(reasons) if reasons else "No strong signals",
            "indicators": {
                "rsi": rsi,
                "macd": macd,
                "sma": sma,
                "price": current_price,
            },
        }


class SimplePaperTrader:
    """Simple paper trading engine."""

    def __init__(
        self,
        symbols: list[str] = ["BTCUSDT"],
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.2,
        check_interval: int = 60,  # seconds
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.check_interval = check_interval

        self.positions: dict[str, dict] = {}
        self.signal_generator = TechnicalSignalGenerator()

        # Supabase repositories
        self.trade_repo = TradeLogRepository()
        self.signal_repo = SignalRepository()
        self.perf_repo = PerformanceRepository()

        self.running = False
        self.trade_count = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    async def start(self):
        """Start paper trading."""
        self.running = True
        logger.info("=" * 60)
        logger.info("🚀 Starting Paper Trading")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"   Max Position: {self.max_position_pct:.0%}")
        logger.info(f"   Check Interval: {self.check_interval}s")
        logger.info("=" * 60)

        collector = MarketDataCollector()

        try:
            while self.running:
                for symbol in self.symbols:
                    await self._process_symbol(symbol, collector)

                # Save performance snapshot
                await self._save_performance_snapshot()

                # Status update
                self._print_status()

                # Wait for next check
                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Paper trading cancelled")
        finally:
            await collector.close()
            await self._close_all_positions(collector)
            self._print_summary()

    async def stop(self):
        """Stop paper trading."""
        self.running = False

    async def _process_symbol(self, symbol: str, collector: MarketDataCollector):
        """Process a single symbol."""
        try:
            # Fetch market data
            df = await collector.get_binance_klines(symbol, "1h", 100)
            if df.empty:
                return

            # Get current price
            ticker = await collector.get_binance_ticker(symbol)
            current_price = ticker.get("price", 0) if ticker else 0

            if current_price <= 0:
                return

            # Generate signal
            signal_result = self.signal_generator.generate_signal(df)

            # Log signal to Supabase
            await self._save_signal(symbol, signal_result, current_price)

            # Trading logic
            if symbol in self.positions:
                # Check for exit
                await self._check_exit(symbol, signal_result, current_price)
            else:
                # Check for entry
                await self._check_entry(symbol, signal_result, current_price)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    async def _check_entry(self, symbol: str, signal: dict, price: float):
        """Check if should enter a position."""
        if signal["confidence"] < 0.65:
            return

        if signal["signal"] in ["buy", "strong_buy"]:
            await self._open_position(symbol, "long", price, signal)
        elif signal["signal"] in ["sell", "strong_sell"]:
            await self._open_position(symbol, "short", price, signal)

    async def _check_exit(self, symbol: str, signal: dict, price: float):
        """Check if should exit a position."""
        position = self.positions.get(symbol)
        if not position:
            return

        # Exit conditions
        should_exit = False
        exit_reason = ""

        # Opposite signal
        if position["side"] == "long" and signal["signal"] in ["sell", "strong_sell"]:
            should_exit = True
            exit_reason = "Sell signal"
        elif position["side"] == "short" and signal["signal"] in ["buy", "strong_buy"]:
            should_exit = True
            exit_reason = "Buy signal"

        # Stop loss (2%)
        pnl_pct = self._calculate_pnl_pct(position, price)
        if pnl_pct < -0.02:
            should_exit = True
            exit_reason = "Stop loss triggered"

        # Take profit (4%)
        if pnl_pct > 0.04:
            should_exit = True
            exit_reason = "Take profit triggered"

        if should_exit:
            await self._close_position(symbol, price, exit_reason)

    def _calculate_pnl_pct(self, position: dict, current_price: float) -> float:
        """Calculate position PnL percentage."""
        entry_price = position["entry_price"]
        if position["side"] == "long":
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price

    async def _open_position(self, symbol: str, side: str, price: float, signal: dict):
        """Open a position."""
        position_value = self.capital * self.max_position_pct
        quantity = position_value / price

        self.positions[symbol] = {
            "position_id": str(uuid.uuid4()),
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "entry_time": datetime.utcnow(),
            "signal": signal,
        }

        logger.info(
            f"📈 OPEN {side.upper()} {symbol} @ ${price:,.2f} | "
            f"Qty: {quantity:.6f} | Value: ${position_value:,.2f} | "
            f"Confidence: {signal['confidence']:.0%}"
        )

        # Log to Supabase
        try:
            await self.trade_repo.log_trade({
                "position_id": self.positions[symbol]["position_id"],
                "symbol": symbol,
                "exchange": "binance",
                "side": "buy" if side == "long" else "sell",
                "entry_price": price,
                "entry_time": datetime.utcnow().isoformat(),
                "quantity": quantity,
                "strategy_name": "paper_technical",
                "signal_source": "technical",
                "signal_confidence": signal["confidence"],
                "entry_reasoning": signal["reasoning"],
                "indicators_at_entry": signal.get("indicators"),
            })
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    async def _close_position(self, symbol: str, price: float, reason: str):
        """Close a position."""
        position = self.positions.pop(symbol, None)
        if not position:
            return

        # Calculate PnL
        if position["side"] == "long":
            pnl = (price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - price) * position["quantity"]

        pnl_pct = self._calculate_pnl_pct(position, price)

        self.capital += pnl
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.winning_trades += 1

        duration = (datetime.utcnow() - position["entry_time"]).total_seconds()

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} CLOSE {symbol} @ ${price:,.2f} | "
            f"PnL: ${pnl:,.2f} ({pnl_pct:+.2%}) | "
            f"Reason: {reason} | Duration: {duration/60:.1f}min"
        )

        # Update trade in Supabase
        try:
            await self.trade_repo.table.update({
                "exit_price": price,
                "exit_time": datetime.utcnow().isoformat(),
                "gross_pnl": pnl,
                "net_pnl": pnl * 0.999,  # 0.1% fee estimate
                "return_pct": pnl_pct * 100,
                "duration_seconds": int(duration),
                "exit_reasoning": reason,
            }).eq("position_id", position["position_id"]).execute()
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")

    async def _close_all_positions(self, collector: MarketDataCollector):
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            try:
                ticker = await collector.get_binance_ticker(symbol)
                price = ticker.get("price", self.positions[symbol]["entry_price"]) if ticker else self.positions[symbol]["entry_price"]
                await self._close_position(symbol, price, "Session end")
            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")

    async def _save_signal(self, symbol: str, signal: dict, price: float):
        """Save signal to Supabase."""
        try:
            # Convert indicators to JSON-safe format
            indicators = signal.get("indicators", {})
            if indicators:
                # Convert any numpy/bool types to Python native types
                indicators = {
                    k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else
                        bool(v) if isinstance(v, (bool, np.bool_)) else
                        {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                         for kk, vv in v.items()} if isinstance(v, dict) else v)
                    for k, v in indicators.items()
                }

            await self.signal_repo.save_signal({
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "exchange": "binance",
                "timeframe": "1h",
                "signal_type": signal["signal"],
                "source": "technical",
                "confidence": signal["confidence"],
                "entry_price": price,
                "reasoning": signal["reasoning"],
                "indicators": indicators,
                "status": "pending",
            })
        except Exception as e:
            logger.debug(f"Failed to save signal: {e}")

    async def _save_performance_snapshot(self):
        """Save performance snapshot to Supabase."""
        try:
            positions_value = sum(
                p["quantity"] * p["entry_price"]
                for p in self.positions.values()
            )

            win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0

            await self.perf_repo.save_snapshot({
                "timestamp": datetime.utcnow().isoformat(),
                "strategy_name": "paper_technical",
                "total_equity": self.capital + positions_value,
                "cash_balance": self.capital,
                "positions_value": positions_value,
                "total_pnl": self.total_pnl,
                "total_trades": self.trade_count,
                "winning_trades": self.winning_trades,
                "losing_trades": self.trade_count - self.winning_trades,
                "win_rate": win_rate,
                "open_positions": {
                    s: {"side": p["side"], "entry": p["entry_price"]}
                    for s, p in self.positions.items()
                },
            })
        except Exception as e:
            logger.debug(f"Failed to save performance: {e}")

    def _print_status(self):
        """Print current status."""
        positions_str = ", ".join(
            f"{s}: {p['side']} @ ${p['entry_price']:,.0f}"
            for s, p in self.positions.items()
        ) or "None"

        win_rate = self.winning_trades / self.trade_count * 100 if self.trade_count > 0 else 0

        logger.info(
            f"💰 Capital: ${self.capital:,.2f} | "
            f"Trades: {self.trade_count} | "
            f"Win Rate: {win_rate:.0f}% | "
            f"Positions: {positions_str}"
        )

    def _print_summary(self):
        """Print trading summary."""
        total_return = (self.capital / self.initial_capital - 1) * 100
        win_rate = self.winning_trades / self.trade_count * 100 if self.trade_count > 0 else 0

        print("\n" + "=" * 60)
        print("📊 PAPER TRADING SUMMARY")
        print("=" * 60)
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Capital:    ${self.capital:,.2f}")
        print(f"Total Return:     {total_return:+.2f}%")
        print("-" * 60)
        print(f"Total Trades:     {self.trade_count}")
        print(f"Winning Trades:   {self.winning_trades}")
        print(f"Losing Trades:    {self.trade_count - self.winning_trades}")
        print(f"Win Rate:         {win_rate:.1f}%")
        print(f"Total PnL:        ${self.total_pnl:,.2f}")
        print("=" * 60)


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Paper Trading")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=10000.0,
        help="Initial capital"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Check interval in seconds"
    )
    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        f"logs/paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="10 MB",
    )

    # Create trader
    trader = SimplePaperTrader(
        symbols=args.symbols,
        initial_capital=args.capital,
        check_interval=args.interval,
    )

    # Handle shutdown
    loop = asyncio.get_event_loop()

    def handle_shutdown():
        logger.info("Shutting down...")
        loop.create_task(trader.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown)

    # Run
    await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
