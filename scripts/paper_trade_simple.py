#!/usr/bin/env python
"""Simple paper trading script using Binance real-time data and Supabase."""

import asyncio
import signal
import sys
from datetime import datetime, timedelta, timezone
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
from data.features.smc.detector import SMCDetector
from data.features.smc.confluence import ConfluenceEngine
from data.features.smc.zones import ZoneDirection


class TechnicalSignalGenerator:
    """Generate trading signals from technical indicators + SMC."""

    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Initialize SMC components
        self.smc_detector = SMCDetector(
            lookback=100,
            atr_period=14,
            swing_lookback=5,
            min_impulse_atr=1.5,  # Slightly lower for more OB detection
        )
        self.confluence_engine = ConfluenceEngine(
            min_confluence_score=0.55,  # Lower threshold for paper trading
            min_rr_ratio=0.5,  # Allow low zone R:R, we enforce R:R in exit logic
            atr_stop_multiplier=1.5,
        )

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> dict:
        """Calculate RSI with momentum detection."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        prev_rsi = float(rsi.iloc[-2]) if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else current_rsi
        return {
            "value": current_rsi,
            "prev": prev_rsi,
            "rising": current_rsi > prev_rsi,
            "falling": current_rsi < prev_rsi,
        }

    def calculate_macd(self, prices: pd.Series) -> dict:
        """Calculate MACD with histogram momentum."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0
        return {
            "macd": float(macd.iloc[-1]),
            "signal": float(signal.iloc[-1]),
            "histogram": float(histogram.iloc[-1]),
            "hist_rising": float(histogram.iloc[-1]) > prev_hist,
            "hist_falling": float(histogram.iloc[-1]) < prev_hist,
        }

    def calculate_sma_crossover(self, prices: pd.Series) -> dict:
        """Calculate SMA crossover and price position."""
        sma20 = prices.rolling(20).mean()
        sma50 = prices.rolling(50).mean()
        current_price = float(prices.iloc[-1])
        sma20_val = float(sma20.iloc[-1])
        sma50_val = float(sma50.iloc[-1])

        return {
            "sma20": sma20_val,
            "sma50": sma50_val,
            "bullish_cross": bool(sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-2] <= sma50.iloc[-2]),
            "bearish_cross": bool(sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-2] >= sma50.iloc[-2]),
            # Price position relative to SMAs (more reliable than crossovers)
            "price_above_sma20": current_price > sma20_val,
            "price_above_sma50": current_price > sma50_val,
            "sma20_above_sma50": sma20_val > sma50_val,
        }

    def calculate_price_momentum(self, prices: pd.Series) -> dict:
        """Calculate short-term price momentum."""
        current = float(prices.iloc[-1])
        price_5_ago = float(prices.iloc[-5]) if len(prices) >= 5 else current
        price_10_ago = float(prices.iloc[-10]) if len(prices) >= 10 else current

        momentum_5 = (current - price_5_ago) / price_5_ago * 100
        momentum_10 = (current - price_10_ago) / price_10_ago * 100

        return {
            "momentum_5": momentum_5,
            "momentum_10": momentum_10,
            "strong_up": momentum_5 > 1.5,  # >1.5% in 5 candles
            "strong_down": momentum_5 < -1.5,
            "weakening_up": momentum_5 > 0 and momentum_5 < momentum_10 / 2,  # Slowing uptrend
            "weakening_down": momentum_5 < 0 and momentum_5 > momentum_10 / 2,  # Slowing downtrend
        }

    def determine_htf_trend(self, df_1h: pd.DataFrame) -> str:
        """Determine higher-timeframe trend from 1h candles.

        Returns 'bullish', 'bearish', or 'neutral'.
        """
        if len(df_1h) < 50:
            return "neutral"

        prices = df_1h["close"]
        sma20 = prices.rolling(20).mean()
        sma50 = prices.rolling(50).mean()
        current_price = float(prices.iloc[-1])
        sma20_val = float(sma20.iloc[-1])
        sma50_val = float(sma50.iloc[-1])

        if sma20_val > sma50_val and current_price > sma20_val:
            return "bullish"
        elif sma20_val < sma50_val and current_price < sma20_val:
            return "bearish"
        return "neutral"

    def generate_signal(self, df: pd.DataFrame, htf_trend: str = "neutral") -> dict:
        """Generate trading signal requiring SMC + Technical AGREEMENT.

        Key rules:
        1. SMC and technical indicators MUST agree on direction
        2. If they conflict, return HOLD (no trade)
        3. Higher confidence when both strongly agree
        4. HTF trend acts as hard filter - no buying in downtrends, no selling in uptrends
        """
        if len(df) < 60:
            return {"signal": "hold", "confidence": 0.5, "reasoning": "Insufficient data"}

        prices = df["close"]
        current_price = float(prices.iloc[-1])

        # Calculate technical indicators
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        sma = self.calculate_sma_crossover(prices)
        momentum = self.calculate_price_momentum(prices)

        # === SMC Analysis ===
        smc_direction = None
        smc_score = 0
        smc_reasons = []
        smc_result = None
        market_trend = None

        try:
            smc_analysis = self.smc_detector.analyze(df)
            if smc_analysis:
                # Get market structure trend (most important)
                market_structure = smc_analysis.get("market_structure")
                if market_structure:
                    market_trend = market_structure.trend
                    if market_trend == ZoneDirection.BULLISH:
                        smc_reasons.append("Bullish structure")
                    elif market_trend == ZoneDirection.BEARISH:
                        smc_reasons.append("Bearish structure")

                # Get confluence result
                smc_result = self.confluence_engine.analyze(
                    current_price=current_price,
                    order_blocks=smc_analysis.get("order_blocks", []),
                    fair_value_gaps=smc_analysis.get("fair_value_gaps", []),
                    liquidity_sweeps=smc_analysis.get("liquidity_sweeps", []),
                    channels=smc_analysis.get("channels", []),
                    market_structure=market_structure,
                    atr=smc_analysis.get("atr", 0.0),
                )

                if smc_result and smc_result.score >= 0.55:
                    smc_score = smc_result.score
                    smc_direction = smc_result.direction
                    smc_reasons.append(f"SMC {smc_score:.0%}")

                    if smc_result.factors.get("zone_confluence", 0) > 0.15:
                        smc_reasons.append("OB+FVG")
        except Exception as e:
            logger.debug(f"SMC analysis failed: {e}")

        # === Technical Scoring ===
        tech_buy_score = 0
        tech_sell_score = 0
        reasons = []

        # RSI signals
        if rsi["value"] < self.rsi_oversold:
            tech_buy_score += 2
            reasons.append(f"RSI oversold ({rsi['value']:.0f})")
        elif rsi["value"] > self.rsi_overbought:
            tech_sell_score += 2
            reasons.append(f"RSI overbought ({rsi['value']:.0f})")
        elif rsi["value"] < 40:
            tech_buy_score += 1
        elif rsi["value"] > 60:
            tech_sell_score += 1

        # RSI momentum
        if rsi["value"] > 65 and rsi["falling"]:
            tech_sell_score += 1
        elif rsi["value"] < 35 and rsi["rising"]:
            tech_buy_score += 1

        # MACD signals
        if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
            tech_buy_score += 1.5 if macd["hist_rising"] else 0.5
            if macd["hist_rising"]:
                reasons.append("MACD bullish+")
        elif macd["histogram"] < 0 and macd["macd"] < macd["signal"]:
            tech_sell_score += 1.5 if macd["hist_falling"] else 0.5
            if macd["hist_falling"]:
                reasons.append("MACD bearish+")

        # SMA analysis - price position is more reliable than crossovers
        # Price above both SMAs = bullish trend
        if sma["price_above_sma20"] and sma["price_above_sma50"]:
            tech_buy_score += 1.5
            reasons.append("Price above SMAs")
        elif not sma["price_above_sma20"] and not sma["price_above_sma50"]:
            tech_sell_score += 1.5
            reasons.append("Price below SMAs")

        # Crossovers only as confirmation (reduced weight)
        if sma["bullish_cross"]:
            tech_buy_score += 1.0
            reasons.append("SMA bullish cross")
        elif sma["bearish_cross"]:
            tech_sell_score += 1.0
            reasons.append("SMA bearish cross")

        # Strong momentum only
        if momentum["strong_up"] and momentum["momentum_5"] > 2.0:
            tech_buy_score += 1.5
            reasons.append(f"Strong +{momentum['momentum_5']:.1f}%")
        elif momentum["strong_down"] and momentum["momentum_5"] < -2.0:
            tech_sell_score += 1.5
            reasons.append(f"Strong {momentum['momentum_5']:.1f}%")

        # === DETERMINE TECHNICAL DIRECTION ===
        tech_score = tech_buy_score - tech_sell_score
        if tech_score >= 1.5:  # Lowered from 2.0 - SMC provides additional confirmation
            tech_direction = "bullish"
        elif tech_score <= -1.5:  # Lowered from -2.0
            tech_direction = "bearish"
        else:
            tech_direction = "neutral"

        # === RSI EXTREME VETO - MOST IMPORTANT ===
        # RSI oversold/overbought are contrarian signals - they VETO opposite trades
        rsi_oversold = rsi["value"] < self.rsi_oversold  # < 30
        rsi_overbought = rsi["value"] > self.rsi_overbought  # > 70
        rsi_extreme_oversold = rsi["value"] < 25  # Very oversold - strong buy zone
        rsi_extreme_overbought = rsi["value"] > 75  # Very overbought - strong sell zone

        # === AGREEMENT CHECK - CRITICAL ===
        # Both SMC and technicals must agree, otherwise HOLD
        signal = "hold"
        confidence = 0.5

        smc_bullish = smc_direction == ZoneDirection.BULLISH
        smc_bearish = smc_direction == ZoneDirection.BEARISH

        # === RSI VETO LOGIC ===
        # NEVER sell when oversold - price likely to bounce UP
        if rsi_oversold and tech_direction == "bearish":
            signal = "hold"
            confidence = 0.5
            reasons = [f"RSI oversold ({rsi['value']:.0f}) - NO SELL, wait for bounce"]

        # NEVER buy when overbought - price likely to pull back
        elif rsi_overbought and tech_direction == "bullish":
            signal = "hold"
            confidence = 0.5
            reasons = [f"RSI overbought ({rsi['value']:.0f}) - NO BUY, wait for pullback"]

        # RSI EXTREME OVERSOLD + Rising = Strong BUY opportunity
        elif rsi_extreme_oversold and rsi["rising"]:
            signal = "buy"
            confidence = 0.70
            reasons = [f"RSI extreme oversold ({rsi['value']:.0f}) + rising - reversal BUY"]

        # RSI EXTREME OVERBOUGHT + Falling = Strong SELL opportunity
        elif rsi_extreme_overbought and rsi["falling"]:
            signal = "sell"
            confidence = 0.70
            reasons = [f"RSI extreme overbought ({rsi['value']:.0f}) + falling - reversal SELL"]

        # BULLISH: Both SMC and technicals agree bullish (and not overbought)
        elif tech_direction == "bullish" and smc_bullish:
            combined_score = abs(tech_score) + smc_score * 3
            if combined_score >= 6.0:
                signal = "strong_buy"
                confidence = min(0.85, 0.60 + combined_score * 0.03)
            elif combined_score >= 4.5:
                signal = "buy"
                confidence = min(0.75, 0.55 + combined_score * 0.03)
            reasons.extend(smc_reasons)
            reasons.append("SMC+Tech aligned")

        # BEARISH: Both SMC and technicals agree bearish (and not oversold)
        elif tech_direction == "bearish" and smc_bearish:
            combined_score = abs(tech_score) + smc_score * 3
            if combined_score >= 6.0:
                signal = "strong_sell"
                confidence = min(0.85, 0.60 + combined_score * 0.03)
            elif combined_score >= 4.5:
                signal = "sell"
                confidence = min(0.75, 0.55 + combined_score * 0.03)
            reasons.extend(smc_reasons)
            reasons.append("SMC+Tech aligned")

        # CONFLICT: SMC and technicals disagree - NO TRADE
        elif (tech_direction == "bullish" and smc_bearish) or (tech_direction == "bearish" and smc_bullish):
            signal = "hold"
            confidence = 0.5
            reasons = ["SMC↔Tech conflict - no trade"]

        # WEAK: Not enough conviction from either
        else:
            signal = "hold"
            confidence = 0.5
            if not reasons:
                reasons = ["Waiting for confluence"]

        # === HTF TREND HARD FILTER ===
        # Never buy in 1h downtrend, never sell in 1h uptrend
        if htf_trend == "bearish" and signal in ("buy", "strong_buy"):
            reasons = [f"HTF bearish - blocked {signal}"] + reasons
            signal = "hold"
            confidence = 0.5
        elif htf_trend == "bullish" and signal in ("sell", "strong_sell"):
            reasons = [f"HTF bullish - blocked {signal}"] + reasons
            signal = "hold"
            confidence = 0.5
        elif htf_trend == "neutral" and signal != "hold":
            confidence = max(0.5, confidence - 0.10)
            reasons.append("HTF neutral (-10% conf)")

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": ", ".join(reasons) if reasons else "No strong signals",
            "indicators": {
                "rsi": rsi["value"],
                "rsi_momentum": "rising" if rsi["rising"] else "falling",
                "macd": macd,
                "sma": sma,
                "momentum": momentum,
                "price": current_price,
                "smc_score": smc_score if smc_result else 0,
                "smc_direction": smc_direction.value if smc_direction else None,
                "htf_trend": htf_trend,
            },
        }


class SimplePaperTrader:
    """Simple paper trading engine."""

    def __init__(
        self,
        symbols: list[str] = ["BTCUSDT"],
        initial_capital: float = 1000.0,  # Realistic starting capital
        max_position_pct: float = 0.20,  # 20% per position (more conservative)
        leverage: int = 10,  # 10x leverage (realistic for crypto futures)
        check_interval: int = 60,  # seconds
        # Exit configuration - IMPROVED R:R RATIO (1:2 minimum)
        stop_loss_pct: float = 0.012,  # 1.2% stop loss (tighter = smaller losses)
        take_profit_pct: float = 0.025,  # 2.5% take profit (bigger wins)
        # R:R ratio = 2.5/1.2 = 2.08:1 (need 33% win rate to break even)
        trailing_stop_activation_pct: float = 0.015,  # Activate trailing at 1.5%
        trailing_stop_distance_pct: float = 0.008,  # Trail 0.8% behind peak
        signal_exit_min_confidence: float = 0.70,  # Higher confidence required
        signal_exit_min_profit_pct: float = 0.01,  # 1% profit threshold
        # Time-based exit
        max_position_hours: float = 4.0,  # Shorter hold = less exposure
        stale_position_min_profit_pct: float = 0.005,  # Exit stale if < 0.5%
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.leverage = leverage
        self.check_interval = check_interval

        # Exit parameters (adjusted for leverage)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        self.signal_exit_min_confidence = signal_exit_min_confidence
        self.signal_exit_min_profit_pct = signal_exit_min_profit_pct
        # Time-based exit
        self.max_position_hours = max_position_hours
        self.stale_position_min_profit_pct = stale_position_min_profit_pct

        # Liquidation tracking
        self.liquidation_pct = 1.0 / leverage * 0.9  # ~9% for 10x (with 10% buffer)

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

    async def _load_existing_positions(self):
        """Load existing open positions from database on startup."""
        try:
            # Query for open positions (no exit_time = still open)
            result = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("*")
                .is_("exit_time", "null")
                .execute()
            )

            if result.data:
                for trade in result.data:
                    symbol = trade.get("symbol")
                    if not symbol:
                        continue

                    # Reconstruct position from trade log
                    entry_price = float(trade.get("entry_price", 0))
                    quantity = float(trade.get("quantity", 0))
                    side = "long" if trade.get("side") == "buy" else "short"
                    entry_time_str = trade.get("entry_time")

                    if entry_price <= 0 or quantity <= 0:
                        continue

                    # Parse entry time
                    if entry_time_str:
                        try:
                            entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                        except:
                            entry_time = datetime.now(timezone.utc)
                    else:
                        entry_time = datetime.now(timezone.utc)

                    # Calculate stop/take profit prices
                    if side == "long":
                        stop_price = entry_price * (1 - self.stop_loss_pct)
                        take_profit_price = entry_price * (1 + self.take_profit_pct)
                    else:
                        stop_price = entry_price * (1 + self.stop_loss_pct)
                        take_profit_price = entry_price * (1 - self.take_profit_pct)

                    self.positions[symbol] = {
                        "position_id": trade.get("position_id", str(uuid.uuid4())),
                        "side": side,
                        "entry_price": entry_price,
                        "quantity": quantity,
                        "entry_time": entry_time,
                        "signal": {"confidence": trade.get("signal_confidence", 0.5)},
                        "trailing_stop_active": False,
                        "peak_pnl_pct": 0.0,
                        "stop_loss_price": stop_price,
                        "take_profit_price": take_profit_price,
                    }

                    hours_open = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                    logger.info(
                        f"📂 Loaded existing position: {symbol} {side.upper()} @ ${entry_price:,.2f} | "
                        f"Qty: {quantity:.6f} | Open for {hours_open:.1f}h"
                    )

                if self.positions:
                    logger.info(f"✅ Loaded {len(self.positions)} existing position(s) from database")

        except Exception as e:
            logger.warning(f"Could not load existing positions: {e}")

    async def start(self):
        """Start paper trading."""
        self.running = True

        # Load existing positions from database
        await self._load_existing_positions()

        rr_ratio = self.take_profit_pct / self.stop_loss_pct
        logger.info("=" * 60)
        logger.info("🚀 Starting Leveraged Paper Trading (SMC+Tech ALIGNED)")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Capital: ${self.initial_capital:,.2f} | Leverage: {self.leverage}x")
        logger.info(f"   Max Position: {self.max_position_pct:.0%} margin → ${self.initial_capital * self.max_position_pct * self.leverage:,.2f} size")
        logger.info("-" * 60)
        logger.info("📊 Risk Management (R:R = {:.1f}:1):".format(rr_ratio))
        logger.info(f"   Stop Loss: {self.stop_loss_pct:.1%} (={self.stop_loss_pct * self.leverage:.0%} ROE)")
        logger.info(f"   Take Profit: {self.take_profit_pct:.1%} (={self.take_profit_pct * self.leverage:.0%} ROE)")
        logger.info(f"   Break-even win rate: {1/(1+rr_ratio)*100:.0f}%")
        logger.info(f"   Trailing: activates {self.trailing_stop_activation_pct:.1%}, trails {self.trailing_stop_distance_pct:.1%}")
        logger.info(f"   Max hold: {self.max_position_hours}h | Signal requires SMC+Tech alignment")
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
            # Fetch market data - 1h for trend, 1m for entry signals
            df_1h = await collector.get_binance_klines(symbol, "1h", 100)
            df_1m = await collector.get_binance_klines(symbol, "1m", 100)
            if df_1m.empty:
                return

            # Get current price
            ticker = await collector.get_binance_ticker(symbol)
            current_price = ticker.get("price", 0) if ticker else 0

            if current_price <= 0:
                return

            # Determine higher-timeframe trend from 1h candles
            htf_trend = "neutral"
            if not df_1h.empty:
                htf_trend = self.signal_generator.determine_htf_trend(df_1h)

            # Generate signal from 1m candles, filtered by 1h trend
            signal_result = self.signal_generator.generate_signal(df_1m, htf_trend=htf_trend)

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
        # Double-check no position exists in memory
        if symbol in self.positions:
            return

        # Also check database for any open position (prevents duplicates across restarts)
        try:
            existing = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("id")
                .eq("symbol", symbol)
                .is_("exit_time", "null")
                .limit(1)
                .execute()
            )
            if existing.data:
                logger.debug(f"Skipping {symbol} entry - position exists in DB")
                return
        except Exception as e:
            logger.warning(f"Could not check DB for existing position: {e}")

        if signal["confidence"] < 0.65:
            return

        if signal["signal"] in ["buy", "strong_buy"]:
            await self._open_position(symbol, "long", price, signal)
        elif signal["signal"] in ["sell", "strong_sell"]:
            await self._open_position(symbol, "short", price, signal)

    async def _check_exit(self, symbol: str, signal: dict, price: float):
        """Check if should exit a leveraged position.

        Priority order:
        0. LIQUIDATION - forced exit (highest priority)
        1. Take profit - lock in gains
        2. Trailing stop - protect accumulated profits
        3. Stop loss - limit losses
        4. Time-based exit - stale positions
        5. RSI-based profit taking
        6. Signal-based exit
        """
        position = self.positions.get(symbol)
        if not position:
            return

        pnl_pct = self._calculate_pnl_pct(position, price)

        # Update peak profit and trailing stop
        self._update_trailing_stop(symbol, pnl_pct)

        should_exit = False
        exit_reason = ""

        # 0. LIQUIDATION CHECK - forced exit at max loss
        if pnl_pct <= -self.liquidation_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"💀 LIQUIDATED ({pnl_pct:.2%} = {roe:.0f}% ROE)"
            logger.warning(f"⚠️ Position liquidated! Lost margin on {symbol}")

        # 1. TAKE PROFIT - lock in gains
        elif pnl_pct >= self.take_profit_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"Take profit ({pnl_pct:.2%} = +{roe:.0f}% ROE)"

        # 2. TRAILING STOP - protect accumulated profits
        elif position.get("trailing_stop_active", False):
            peak_pnl = position.get("peak_pnl_pct", 0)
            trailing_stop_level = peak_pnl - self.trailing_stop_distance_pct
            if pnl_pct <= trailing_stop_level:
                should_exit = True
                exit_reason = f"Trailing stop (peak: {peak_pnl:.2%}, current: {pnl_pct:.2%})"

        # 3. STOP LOSS - limit losses
        elif pnl_pct <= -self.stop_loss_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"Stop loss ({pnl_pct:.2%} = {roe:.0f}% ROE)"

        # 4. TIME-BASED EXIT - exit stale positions with minimal profit
        elif self._is_stale_position(position, pnl_pct):
            should_exit = True
            hours_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 3600
            exit_reason = f"Stale position ({hours_open:.1f}h, only {pnl_pct:.2%} profit)"

        # 5. RSI-BASED PROFIT TAKING - exit profitable longs when overbought
        elif self._should_exit_on_rsi(position, signal, pnl_pct):
            indicators = signal.get("indicators", {})
            rsi = indicators.get("rsi", 50)
            should_exit = True
            exit_reason = f"RSI profit taking (RSI={rsi:.0f}, profit={pnl_pct:.2%})"

        # 6. SIGNAL-BASED EXIT - with conditions met
        elif self._should_exit_on_signal(position, signal, pnl_pct):
            should_exit = True
            confidence = signal.get("confidence", 0)
            exit_reason = f"{signal['signal'].replace('_', ' ').title()} signal (confidence: {confidence:.0%})"

        if should_exit:
            await self._close_position(symbol, price, exit_reason)

    def _is_stale_position(self, position: dict, pnl_pct: float) -> bool:
        """Check if position is stale (old with minimal profit)."""
        hours_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 3600

        # Exit if position is older than max_position_hours AND profit is below threshold
        if hours_open >= self.max_position_hours:
            if pnl_pct < self.stale_position_min_profit_pct:
                return True
        return False

    def _should_exit_on_rsi(self, position: dict, signal: dict, pnl_pct: float) -> bool:
        """Exit profitable positions when RSI indicates overbought/oversold reversal."""
        if pnl_pct < 0.005:  # Need at least 0.5% profit
            return False

        indicators = signal.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        rsi_momentum = indicators.get("rsi_momentum", "")
        side = position.get("side")

        # Long position + RSI overbought and falling = exit
        if side == "long" and rsi > 68 and rsi_momentum == "falling":
            return True

        # Short position + RSI oversold and rising = exit
        if side == "short" and rsi < 32 and rsi_momentum == "rising":
            return True

        return False

    def _update_trailing_stop(self, symbol: str, current_pnl_pct: float):
        """Update trailing stop based on current P&L."""
        position = self.positions.get(symbol)
        if not position:
            return

        # Activate trailing stop when profit threshold is reached
        if current_pnl_pct >= self.trailing_stop_activation_pct:
            if not position.get("trailing_stop_active", False):
                position["trailing_stop_active"] = True
                position["peak_pnl_pct"] = current_pnl_pct
                logger.info(
                    f"🔒 Trailing stop activated for {symbol} at {current_pnl_pct:.2%} profit"
                )
            # Update peak if current profit is higher
            elif current_pnl_pct > position.get("peak_pnl_pct", 0):
                position["peak_pnl_pct"] = current_pnl_pct
                logger.debug(f"📈 {symbol} new peak profit: {current_pnl_pct:.2%}")

    def _should_exit_on_signal(self, position: dict, signal: dict, pnl_pct: float) -> bool:
        """Determine if should exit based on opposite signal with strict conditions.

        Conditions for signal-based exit:
        1. Must be opposite signal (buy for short, sell for long)
        2. If profitable (> signal_exit_min_profit_pct), require STRONG signal
        3. Signal confidence must meet minimum threshold
        """
        signal_type = signal.get("signal", "hold")
        confidence = signal.get("confidence", 0)
        side = position.get("side")

        # Check for opposite signal
        is_opposite_signal = False
        is_strong_signal = False

        if side == "long" and signal_type in ["sell", "strong_sell"]:
            is_opposite_signal = True
            is_strong_signal = signal_type == "strong_sell"
        elif side == "short" and signal_type in ["buy", "strong_buy"]:
            is_opposite_signal = True
            is_strong_signal = signal_type == "strong_buy"

        if not is_opposite_signal:
            return False

        # If position is significantly profitable, require strong signal
        if pnl_pct >= self.signal_exit_min_profit_pct:
            if not is_strong_signal:
                logger.debug(
                    f"Ignoring weak {signal_type} signal - position is {pnl_pct:.2%} profitable"
                )
                return False

        # Check minimum confidence threshold
        if confidence < self.signal_exit_min_confidence:
            logger.debug(
                f"Ignoring {signal_type} signal - confidence {confidence:.0%} < {self.signal_exit_min_confidence:.0%}"
            )
            return False

        return True

    def _calculate_pnl_pct(self, position: dict, current_price: float) -> float:
        """Calculate position PnL percentage."""
        entry_price = position["entry_price"]
        if position["side"] == "long":
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price

    async def _open_position(self, symbol: str, side: str, price: float, signal: dict):
        """Open a leveraged position."""
        # Margin = capital allocated to this position
        margin = self.capital * self.max_position_pct
        # Position value = margin * leverage
        position_value = margin * self.leverage
        quantity = position_value / price

        # Calculate stop loss, take profit, and liquidation prices
        if side == "long":
            stop_price = price * (1 - self.stop_loss_pct)
            take_profit_price = price * (1 + self.take_profit_pct)
            liquidation_price = price * (1 - self.liquidation_pct)
        else:
            stop_price = price * (1 + self.stop_loss_pct)
            take_profit_price = price * (1 - self.take_profit_pct)
            liquidation_price = price * (1 + self.liquidation_pct)

        self.positions[symbol] = {
            "position_id": str(uuid.uuid4()),
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": margin,  # Actual capital at risk
            "leverage": self.leverage,
            "entry_time": datetime.now(timezone.utc),
            "signal": signal,
            # Trailing stop fields
            "trailing_stop_active": False,
            "peak_pnl_pct": 0.0,
            # Reference prices
            "stop_loss_price": stop_price,
            "take_profit_price": take_profit_price,
            "liquidation_price": liquidation_price,
        }

        logger.info(
            f"📈 OPEN {side.upper()} {symbol} @ ${price:,.2f} | "
            f"Margin: ${margin:,.2f} | Size: ${position_value:,.2f} ({self.leverage}x) | "
            f"Conf: {signal['confidence']:.0%}"
        )
        logger.info(
            f"   └─ SL: ${stop_price:,.2f} ({self.stop_loss_pct:.1%}) | "
            f"TP: ${take_profit_price:,.2f} ({self.take_profit_pct:.1%}) | "
            f"Liq: ${liquidation_price:,.2f}"
        )

        # Log to Supabase
        try:
            await self.trade_repo.log_trade({
                "position_id": self.positions[symbol]["position_id"],
                "symbol": symbol,
                "exchange": "binance",
                "side": "buy" if side == "long" else "sell",
                "entry_price": price,
                "entry_time": datetime.now(timezone.utc).isoformat(),
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
        """Close a leveraged position."""
        position = self.positions.pop(symbol, None)
        if not position:
            return

        # Calculate PnL on the leveraged position
        if position["side"] == "long":
            pnl = (price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - price) * position["quantity"]

        pnl_pct = self._calculate_pnl_pct(position, price)
        margin = position.get("margin", position["quantity"] * position["entry_price"] / self.leverage)
        leverage = position.get("leverage", self.leverage)

        # ROE (Return on Equity) = actual return on margin
        roe = (pnl / margin) * 100 if margin > 0 else 0

        self.capital += pnl
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.winning_trades += 1

        duration = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds()

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} CLOSE {symbol} @ ${price:,.2f} | "
            f"PnL: ${pnl:,.2f} (ROE: {roe:+.1f}%) | "
            f"Reason: {reason} | {duration/60:.1f}min"
        )

        # Update trade in Supabase
        try:
            await self.trade_repo.table.update({
                "exit_price": price,
                "exit_time": datetime.now(timezone.utc).isoformat(),
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "exchange": "binance",
                "timeframe": "1m",
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
        """Print current status with position details."""
        win_rate = self.winning_trades / self.trade_count * 100 if self.trade_count > 0 else 0

        logger.info(
            f"💰 Capital: ${self.capital:,.2f} | "
            f"Trades: {self.trade_count} | "
            f"Win Rate: {win_rate:.0f}% | "
            f"Open: {len(self.positions)}"
        )

        # Print detailed position info
        for symbol, pos in self.positions.items():
            trailing = "🔒" if pos.get("trailing_stop_active") else "⏳"
            peak = pos.get("peak_pnl_pct", 0)
            logger.info(
                f"   {trailing} {symbol}: {pos['side']} @ ${pos['entry_price']:,.2f} | "
                f"Peak: {peak:.2%} | "
                f"SL: ${pos.get('stop_loss_price', 0):,.2f} | "
                f"TP: ${pos.get('take_profit_price', 0):,.2f}"
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

    parser = argparse.ArgumentParser(description="Leveraged Paper Trading")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=1000.0,
        help="Initial capital (default: $1000)"
    )
    parser.add_argument(
        "--leverage", type=int, default=10,
        help="Leverage multiplier (default: 10x)"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Check interval in seconds"
    )
    # Exit configuration - IMPROVED R:R RATIO (2:1 minimum)
    parser.add_argument(
        "--stop-loss", type=float, default=0.012,
        help="Stop loss percentage (default: 0.012 = 1.2%% for tighter losses)"
    )
    parser.add_argument(
        "--take-profit", type=float, default=0.025,
        help="Take profit percentage (default: 0.025 = 2.5%% for bigger wins)"
    )
    parser.add_argument(
        "--trailing-activation", type=float, default=0.015,
        help="Trailing stop activation percentage (default: 0.015 = 1.5%%)"
    )
    parser.add_argument(
        "--trailing-distance", type=float, default=0.008,
        help="Trailing stop distance percentage (default: 0.008 = 0.8%%)"
    )
    parser.add_argument(
        "--signal-confidence", type=float, default=0.70,
        help="Minimum confidence for signal-based exits (default: 0.70 = 70%%)"
    )
    parser.add_argument(
        "--signal-profit-threshold", type=float, default=0.01,
        help="Profit threshold requiring strong signal to exit (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--max-position-hours", type=float, default=4.0,
        help="Maximum hours to hold a stale position (default: 4.0)"
    )
    parser.add_argument(
        "--stale-profit-threshold", type=float, default=0.005,
        help="Min profit to keep stale position (default: 0.005 = 0.5%%)"
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

    # Create trader with leverage configuration
    trader = SimplePaperTrader(
        symbols=args.symbols,
        initial_capital=args.capital,
        leverage=args.leverage,
        check_interval=args.interval,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        trailing_stop_activation_pct=args.trailing_activation,
        trailing_stop_distance_pct=args.trailing_distance,
        signal_exit_min_confidence=args.signal_confidence,
        signal_exit_min_profit_pct=args.signal_profit_threshold,
        max_position_hours=args.max_position_hours,
        stale_position_min_profit_pct=args.stale_profit_threshold,
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
