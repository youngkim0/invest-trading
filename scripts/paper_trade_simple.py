#!/usr/bin/env python
"""Multi-strategy paper trading script using Binance real-time data and Supabase.

Supports 3 strategies:
- agreement_classic: Original MACD+SMC+SMA agreement on 1m+1h
- agreement_mtf: Multi-timeframe agreement (1m+5m+30m+1h)
- momentum: Momentum/Breakout strategy with volume confirmation
"""

import asyncio
import signal
import sys
from dataclasses import dataclass
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

    def generate_signal(self, df: pd.DataFrame, htf_trend: str = "neutral", **kwargs) -> dict:
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
        # rsi_extreme thresholds removed in v3.6 — RSI reversal entries disabled

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

        # NOTE: RSI reversal entries removed in v3.6. Data showed 25% WR, -$65 over 8 trades.
        # RSI extremes on 1m candles indicate strong trend, not reversal.
        # RSI VETO (above) is kept — only the entry trigger is removed.

        # BULLISH: Both SMC and technicals agree bullish (and not overbought)
        elif tech_direction == "bullish" and smc_bullish:
            # Require 1m SMA20 > SMA50 (short-term trend must be up)
            if not sma["sma20_above_sma50"]:
                signal = "hold"
                confidence = 0.5
                reasons.append(f"SMA20 crossed below SMA50 - short-term trend bearish, no buy")
            # Require MACD histogram confirms direction (not near-zero)
            elif macd["histogram"] <= 0 or (abs(macd["histogram"]) / current_price * 100 if current_price > 0 else 0) < 0.001:
                signal = "hold"
                confidence = 0.5
                reasons.append(f"MACD too weak ({macd['histogram']:.4f}) - no momentum")
            else:
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
            # Require MACD histogram confirms direction (not near-zero)
            macd_pct = abs(macd["histogram"]) / current_price * 100 if current_price > 0 else 0
            if macd["histogram"] >= 0 or macd_pct < 0.001:
                signal = "hold"
                confidence = 0.5
                reasons.append(f"MACD too weak ({macd['histogram']:.4f}) - no momentum")
            else:
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
        # Only buy in bullish/neutral 1h trend, only sell in bearish 1h trend
        if htf_trend == "bearish" and signal in ("buy", "strong_buy"):
            reasons = [f"HTF bearish - blocked {signal}"] + reasons
            signal = "hold"
            confidence = 0.5
        elif htf_trend != "bearish" and signal in ("sell", "strong_sell"):
            reasons = [f"HTF {htf_trend} - blocked {signal} (need bearish)"] + reasons
            signal = "hold"
            confidence = 0.5
        elif htf_trend == "neutral" and signal in ("buy", "strong_buy"):
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


class MTFSignalGenerator(TechnicalSignalGenerator):
    """Multi-timeframe agreement signal generator (1m+5m+30m+1h).

    Uses the same agreement logic as TechnicalSignalGenerator on 1m candles,
    then adds confluence scoring from 5m and 30m timeframes.
    If MTF strongly disagrees, downgrades to hold.
    If MTF strongly confirms, upgrades signal strength.
    """

    def generate_signal(self, df: pd.DataFrame, htf_trend: str = "neutral",
                        df_5m: pd.DataFrame = None, df_30m: pd.DataFrame = None, **kwargs) -> dict:
        """Generate signal with multi-timeframe confluence."""
        # Get base signal from 1m (parent class)
        base_result = super().generate_signal(df, htf_trend)

        # If base is hold, no point adding MTF confluence
        if base_result["signal"] == "hold":
            return base_result

        is_buy = base_result["signal"] in ("buy", "strong_buy")
        mtf_bonus = 0.0
        mtf_reasons = []

        # === 5m confluence (+1.0 confirm / -0.5 disagree) ===
        if df_5m is not None and len(df_5m) >= 50:
            prices_5m = df_5m["close"]
            macd_5m = self.calculate_macd(prices_5m)
            sma_5m = self.calculate_sma_crossover(prices_5m)

            if is_buy:
                if macd_5m["histogram"] > 0 and sma_5m["sma20_above_sma50"]:
                    mtf_bonus += 1.0
                    mtf_reasons.append("5m confirms↑")
                elif macd_5m["histogram"] < 0 and not sma_5m["sma20_above_sma50"]:
                    mtf_bonus -= 0.5
                    mtf_reasons.append("5m disagrees")
            else:  # sell
                if macd_5m["histogram"] < 0 and not sma_5m["sma20_above_sma50"]:
                    mtf_bonus += 1.0
                    mtf_reasons.append("5m confirms↓")
                elif macd_5m["histogram"] > 0 and sma_5m["sma20_above_sma50"]:
                    mtf_bonus -= 0.5
                    mtf_reasons.append("5m disagrees")

        # === 30m confluence (+1.5 confirm / -1.0 disagree) ===
        if df_30m is not None and len(df_30m) >= 50:
            prices_30m = df_30m["close"]
            macd_30m = self.calculate_macd(prices_30m)
            sma_30m = self.calculate_sma_crossover(prices_30m)

            if is_buy:
                if macd_30m["histogram"] > 0 and sma_30m["sma20_above_sma50"]:
                    mtf_bonus += 1.5
                    mtf_reasons.append("30m confirms↑")
                elif macd_30m["histogram"] < 0 and not sma_30m["sma20_above_sma50"]:
                    mtf_bonus -= 1.0
                    mtf_reasons.append("30m disagrees")
            else:  # sell
                if macd_30m["histogram"] < 0 and not sma_30m["sma20_above_sma50"]:
                    mtf_bonus += 1.5
                    mtf_reasons.append("30m confirms↓")
                elif macd_30m["histogram"] > 0 and sma_30m["sma20_above_sma50"]:
                    mtf_bonus -= 1.0
                    mtf_reasons.append("30m disagrees")

        # If MTF strongly disagrees, downgrade to hold
        if mtf_bonus <= -1.5:
            return {
                **base_result,
                "signal": "hold",
                "confidence": 0.5,
                "reasoning": f"MTF disagreement ({', '.join(mtf_reasons)}), " + base_result["reasoning"],
            }

        # Adjust confidence based on MTF
        conf_adj = mtf_bonus * 0.03
        new_confidence = min(0.95, max(0.5, base_result["confidence"] + conf_adj))

        # If MTF strongly confirms, upgrade signal strength
        new_signal = base_result["signal"]
        if mtf_bonus >= 2.0:
            if new_signal == "buy":
                new_signal = "strong_buy"
            elif new_signal == "sell":
                new_signal = "strong_sell"

        reasoning = base_result["reasoning"]
        if mtf_reasons:
            reasoning += ", " + ", ".join(mtf_reasons)

        return {
            **base_result,
            "signal": new_signal,
            "confidence": new_confidence,
            "reasoning": reasoning,
        }


class MomentumBreakoutGenerator(TechnicalSignalGenerator):
    """Momentum/Breakout signal generator for aggressive entries.

    Entry rules:
    - Breakout: Price breaks 20-bar high (buy) or 20-bar low (sell) → 2.0 pts
    - Volume spike: Current volume >= 2x 20-bar average → 2.0 pts
    - RSI confirmation: 40-70 for buy, 30-60 for sell → 1.0 pt
    - MACD direction: Histogram positive+rising / negative+falling → 1.5 pts
    - SMA trend: SMA20 > SMA50 (buy) or SMA20 < SMA50 (sell) → 1.5 pts
    - Buy threshold >= 5.0, strong >= 7.0

    Reuses RSI/MACD/SMA from parent class. HTF filter same as agreement.
    """

    def generate_signal(self, df: pd.DataFrame, htf_trend: str = "neutral", **kwargs) -> dict:
        """Generate momentum/breakout signal."""
        if len(df) < 60:
            return {"signal": "hold", "confidence": 0.5, "reasoning": "Insufficient data"}

        prices = df["close"]
        volumes = df["volume"]
        current_price = float(prices.iloc[-1])

        # Calculate base indicators (reuse parent methods)
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        sma = self.calculate_sma_crossover(prices)

        # Breakout detection (20-bar high/low, excluding current bar)
        lookback = min(20, len(prices) - 1)
        high_20 = float(prices.iloc[-lookback - 1:-1].max())
        low_20 = float(prices.iloc[-lookback - 1:-1].min())
        breakout_up = current_price > high_20
        breakout_down = current_price < low_20

        # Volume spike detection
        avg_volume_20 = float(volumes.iloc[-lookback - 1:-1].mean()) if len(volumes) > lookback else 0
        current_volume = float(volumes.iloc[-1])
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        volume_spike = volume_ratio >= 2.0

        # Scoring
        buy_score = 0.0
        sell_score = 0.0
        reasons = []

        # Breakout (2.0 pts)
        if breakout_up:
            buy_score += 2.0
            reasons.append(f"Breakout above ${high_20:,.2f}")
        if breakout_down:
            sell_score += 2.0
            reasons.append(f"Breakdown below ${low_20:,.2f}")

        # Volume spike (2.0 pts) - only counts if breakout confirms direction
        if volume_spike:
            if breakout_up:
                buy_score += 2.0
                reasons.append(f"Vol spike {volume_ratio:.1f}x")
            elif breakout_down:
                sell_score += 2.0
                reasons.append(f"Vol spike {volume_ratio:.1f}x")

        # RSI confirmation (1.0 pt)
        # Buy: 40-70 (not overbought), Sell: 30-60 (not oversold)
        if 40 <= rsi["value"] <= 70:
            buy_score += 1.0
        if 30 <= rsi["value"] <= 60:
            sell_score += 1.0

        # MACD direction (1.5 pts)
        if macd["histogram"] > 0 and macd["hist_rising"]:
            buy_score += 1.5
            reasons.append("MACD rising+")
        elif macd["histogram"] < 0 and macd["hist_falling"]:
            sell_score += 1.5
            reasons.append("MACD falling-")

        # SMA trend (1.5 pts)
        if sma["sma20_above_sma50"]:
            buy_score += 1.5
        else:
            sell_score += 1.5

        # Determine signal
        signal_type = "hold"
        confidence = 0.5

        if buy_score >= 7.0:
            signal_type = "strong_buy"
            confidence = min(0.85, 0.60 + buy_score * 0.03)
        elif buy_score >= 5.0:
            signal_type = "buy"
            confidence = min(0.75, 0.55 + buy_score * 0.03)
        elif sell_score >= 7.0:
            signal_type = "strong_sell"
            confidence = min(0.85, 0.60 + sell_score * 0.03)
        elif sell_score >= 5.0:
            signal_type = "sell"
            confidence = min(0.75, 0.55 + sell_score * 0.03)

        # RSI Veto (same as agreement)
        if rsi["value"] < self.rsi_oversold and signal_type in ("sell", "strong_sell"):
            signal_type = "hold"
            confidence = 0.5
            reasons = [f"RSI oversold ({rsi['value']:.0f}) - veto sell"]
        elif rsi["value"] > self.rsi_overbought and signal_type in ("buy", "strong_buy"):
            signal_type = "hold"
            confidence = 0.5
            reasons = [f"RSI overbought ({rsi['value']:.0f}) - veto buy"]

        # HTF filter (same as agreement)
        if htf_trend == "bearish" and signal_type in ("buy", "strong_buy"):
            reasons = [f"HTF bearish - blocked {signal_type}"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend != "bearish" and signal_type in ("sell", "strong_sell"):
            reasons = [f"HTF {htf_trend} - blocked sell"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend == "neutral" and signal_type in ("buy", "strong_buy"):
            confidence = max(0.5, confidence - 0.10)
            reasons.append("HTF neutral (-10% conf)")

        if not reasons:
            reasons = ["No breakout signals"]

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "rsi": rsi["value"],
                "rsi_momentum": "rising" if rsi["rising"] else "falling",
                "macd": macd,
                "sma": sma,
                "price": current_price,
                "breakout_up": breakout_up,
                "breakout_down": breakout_down,
                "volume_spike": volume_spike,
                "volume_ratio": volume_ratio,
                "htf_trend": htf_trend,
            },
        }


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str                        # "agreement_classic", "agreement_mtf", "momentum"
    strategy_type: str               # "agreement", "agreement_mtf", "momentum"
    generator: object                # Signal generator instance
    stop_loss_pct: float             # e.g., 0.012
    take_profit_pct: float           # e.g., 0.025
    trailing_activation_pct: float   # e.g., 0.015
    trailing_distance_pct: float     # e.g., 0.008
    max_position_hours: float        # e.g., 4.0
    capital: float                   # Allocated capital

    @property
    def rr_ratio(self) -> float:
        return self.take_profit_pct / self.stop_loss_pct if self.stop_loss_pct > 0 else 0

    @property
    def source_label(self) -> str:
        """Signal source label for DB."""
        return self.strategy_type  # "agreement", "agreement_mtf", "momentum"


# Strategy name mapping for backward compat with old "paper_technical" trades
LEGACY_STRATEGY_MAP = {
    "paper_technical": "agreement_classic",
}


class SimplePaperTrader:
    """Multi-strategy paper trading engine."""

    def __init__(
        self,
        symbols: list[str] = ["BTCUSDT"],
        strategies: list[StrategyConfig] = None,
        initial_capital: float = 1000.0,
        max_position_pct: float = 0.20,
        leverage: int = 10,
        check_interval: int = 60,
        signal_exit_min_confidence: float = 0.70,
        signal_exit_min_profit_pct: float = 0.01,
        stale_position_min_profit_pct: float = 0.005,
    ):
        self.symbols = symbols
        self.max_position_pct = max_position_pct
        self.leverage = leverage
        self.check_interval = check_interval
        self.signal_exit_min_confidence = signal_exit_min_confidence
        self.signal_exit_min_profit_pct = signal_exit_min_profit_pct
        self.stale_position_min_profit_pct = stale_position_min_profit_pct

        # Liquidation tracking
        self.liquidation_pct = 1.0 / leverage * 0.9  # ~9% for 10x (with 10% buffer)

        # Default strategy if none provided (backward compat)
        if strategies is None:
            strategies = [
                StrategyConfig(
                    name="agreement_classic",
                    strategy_type="agreement",
                    generator=TechnicalSignalGenerator(),
                    stop_loss_pct=0.012,
                    take_profit_pct=0.025,
                    trailing_activation_pct=0.015,
                    trailing_distance_pct=0.008,
                    max_position_hours=4.0,
                    capital=initial_capital,
                ),
            ]

        self.strategies = strategies
        self.strategy_map = {s.name: s for s in strategies}

        # Per-strategy tracking
        self.strategy_stats = {}
        self.initial_capital = 0.0
        for s in strategies:
            self.strategy_stats[s.name] = {
                "capital": s.capital,
                "initial_capital": s.capital,
                "trade_count": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
            }
            self.initial_capital += s.capital

        # Positions keyed by "strategy_name:symbol"
        self.positions: dict[str, dict] = {}

        # Supabase repositories
        self.trade_repo = TradeLogRepository()
        self.signal_repo = SignalRepository()
        self.perf_repo = PerformanceRepository()

        self.running = False

        # Cooldown after stop loss: {strategy:symbol: datetime}
        self.stop_loss_cooldowns: dict[str, datetime] = {}
        self.stop_loss_cooldown_minutes = 60
        # Max consecutive stop losses per strategy:symbol per day
        self.daily_stop_losses: dict[str, int] = {}
        self.max_daily_stop_losses = 2
        self.last_reset_date: str = ""

    @property
    def total_capital(self):
        return sum(s["capital"] for s in self.strategy_stats.values())

    @property
    def total_pnl(self):
        return sum(s["total_pnl"] for s in self.strategy_stats.values())

    @property
    def total_trades(self):
        return sum(s["trade_count"] for s in self.strategy_stats.values())

    @property
    def total_winning(self):
        return sum(s["winning_trades"] for s in self.strategy_stats.values())

    def _pos_key(self, strategy_name: str, symbol: str) -> str:
        return f"{strategy_name}:{symbol}"

    def _parse_pos_key(self, pos_key: str) -> tuple[str, str]:
        parts = pos_key.split(":", 1)
        return parts[0], parts[1]

    async def _load_existing_positions(self):
        """Load existing open positions from database on startup."""
        try:
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

                    # Map strategy name (handle legacy "paper_technical")
                    strategy_name = trade.get("strategy_name", "paper_technical")
                    strategy_name = LEGACY_STRATEGY_MAP.get(strategy_name, strategy_name)

                    # Only load if we have this strategy active
                    strategy = self.strategy_map.get(strategy_name)
                    if not strategy:
                        logger.debug(f"Skipping position for inactive strategy: {strategy_name}")
                        continue

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

                    # Calculate stop/take profit using strategy-specific params
                    if side == "long":
                        stop_price = entry_price * (1 - strategy.stop_loss_pct)
                        take_profit_price = entry_price * (1 + strategy.take_profit_pct)
                    else:
                        stop_price = entry_price * (1 + strategy.stop_loss_pct)
                        take_profit_price = entry_price * (1 - strategy.take_profit_pct)

                    pos_key = self._pos_key(strategy_name, symbol)
                    self.positions[pos_key] = {
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
                        "strategy_name": strategy_name,
                    }

                    hours_open = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                    logger.info(
                        f"📂 Loaded: [{strategy_name}] {symbol} {side.upper()} @ ${entry_price:,.2f} | "
                        f"Qty: {quantity:.6f} | Open for {hours_open:.1f}h"
                    )

                loaded = len(self.positions)
                if loaded:
                    logger.info(f"✅ Loaded {loaded} existing position(s) from database")

        except Exception as e:
            logger.warning(f"Could not load existing positions: {e}")

    async def start(self):
        """Start paper trading."""
        self.running = True

        # Load existing positions from database
        await self._load_existing_positions()

        logger.info("=" * 70)
        logger.info("🚀 Starting Multi-Strategy Paper Trading")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Total Capital: ${self.initial_capital:,.2f} | Leverage: {self.leverage}x")
        logger.info("-" * 70)
        for s in self.strategies:
            logger.info(f"   📋 {s.name}: ${s.capital:,.2f} | SL={s.stop_loss_pct:.1%} TP={s.take_profit_pct:.1%} R:R={s.rr_ratio:.1f}:1 | Max hold: {s.max_position_hours}h")
        logger.info("-" * 70)
        logger.info(f"   Position size: {self.max_position_pct:.0%} margin | SL cooldown: {self.stop_loss_cooldown_minutes}min")
        logger.info("=" * 70)

        collector = MarketDataCollector()

        try:
            while self.running:
                for symbol in self.symbols:
                    # Fetch candles once per symbol (shared across strategies)
                    candles = await self._fetch_candles(symbol, collector)
                    if candles is None:
                        continue

                    # Run each strategy
                    for strategy in self.strategies:
                        await self._process_strategy_symbol(strategy, symbol, candles)

                # Save performance snapshots (one per strategy)
                await self._save_performance_snapshots()

                # Status update
                self._print_status()

                # Wait for next check
                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Paper trading cancelled")
        finally:
            await self._close_all_positions(collector)
            await collector.close()
            self._print_summary()

    async def stop(self):
        """Stop paper trading."""
        self.running = False

    async def _fetch_candles(self, symbol: str, collector: MarketDataCollector) -> dict | None:
        """Fetch all needed candles for a symbol (shared across strategies)."""
        try:
            df_1m = await collector.get_binance_klines(symbol, "1m", 100)
            if df_1m.empty:
                return None

            df_1h = await collector.get_binance_klines(symbol, "1h", 100)

            ticker = await collector.get_binance_ticker(symbol)
            current_price = ticker.get("price", 0) if ticker else 0
            if current_price <= 0:
                return None

            candles = {
                "1m": df_1m,
                "1h": df_1h,
                "current_price": current_price,
            }

            # Only fetch 5m/30m if any MTF strategy is active
            if any(s.strategy_type == "agreement_mtf" for s in self.strategies):
                candles["5m"] = await collector.get_binance_klines(symbol, "5m", 100)
                candles["30m"] = await collector.get_binance_klines(symbol, "30m", 100)

            return candles
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return None

    async def _process_strategy_symbol(self, strategy: StrategyConfig, symbol: str, candles: dict):
        """Process a single strategy+symbol combination."""
        try:
            df_1m = candles["1m"]
            df_1h = candles["1h"]
            current_price = candles["current_price"]

            # Determine HTF trend (shared 1h analysis)
            htf_trend = "neutral"
            if not df_1h.empty:
                htf_trend = strategy.generator.determine_htf_trend(df_1h)

            # Generate signal based on strategy type
            if strategy.strategy_type == "agreement_mtf":
                signal_result = strategy.generator.generate_signal(
                    df_1m, htf_trend,
                    df_5m=candles.get("5m"),
                    df_30m=candles.get("30m"),
                )
            else:
                signal_result = strategy.generator.generate_signal(df_1m, htf_trend)

            # Save signal to DB
            await self._save_signal(symbol, signal_result, current_price, strategy)

            # Trading logic
            pos_key = self._pos_key(strategy.name, symbol)
            if pos_key in self.positions:
                await self._check_exit(pos_key, signal_result, current_price, strategy)
            else:
                await self._check_entry(pos_key, symbol, signal_result, current_price, strategy)

        except Exception as e:
            logger.error(f"Error processing [{strategy.name}] {symbol}: {e}")

    async def _check_entry(self, pos_key: str, symbol: str, signal: dict, price: float, strategy: StrategyConfig):
        """Check if should enter a position for a given strategy."""
        if pos_key in self.positions:
            return

        # Check DB for existing open position for this strategy+symbol
        try:
            existing = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("id")
                .eq("symbol", symbol)
                .eq("strategy_name", strategy.name)
                .is_("exit_time", "null")
                .limit(1)
                .execute()
            )
            if existing.data:
                logger.debug(f"Skipping [{strategy.name}] {symbol} entry - position exists in DB")
                return
        except Exception as e:
            logger.warning(f"Could not check DB for existing position: {e}")

        if signal["confidence"] < 0.65:
            return

        # Reset daily stop loss counter at midnight UTC
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.daily_stop_losses = {}
            self.last_reset_date = today

        # Check stop loss cooldown
        cooldown_key = pos_key
        if cooldown_key in self.stop_loss_cooldowns:
            cooldown_until = self.stop_loss_cooldowns[cooldown_key]
            remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
            if remaining > 0:
                logger.info(f"   [{strategy.name}] {symbol}: SL cooldown ({remaining:.0f}min left)")
                return
            else:
                del self.stop_loss_cooldowns[cooldown_key]

        # Check max daily stop losses
        if self.daily_stop_losses.get(cooldown_key, 0) >= self.max_daily_stop_losses:
            logger.info(f"   [{strategy.name}] {symbol}: Max daily SL reached")
            return

        if signal["signal"] in ["buy", "strong_buy"]:
            await self._open_position(pos_key, symbol, "long", price, signal, strategy)
        elif signal["signal"] in ["sell", "strong_sell"]:
            await self._open_position(pos_key, symbol, "short", price, signal, strategy)

    async def _check_exit(self, pos_key: str, signal: dict, price: float, strategy: StrategyConfig):
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
        position = self.positions.get(pos_key)
        if not position:
            return

        pnl_pct = self._calculate_pnl_pct(position, price)

        # Update peak profit and trailing stop
        self._update_trailing_stop(pos_key, pnl_pct, strategy)

        should_exit = False
        exit_reason = ""

        # 0. LIQUIDATION CHECK
        if pnl_pct <= -self.liquidation_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"💀 LIQUIDATED ({pnl_pct:.2%} = {roe:.0f}% ROE)"
            logger.warning(f"⚠️ Position liquidated! [{strategy.name}] {pos_key}")

        # 1. TAKE PROFIT
        elif pnl_pct >= strategy.take_profit_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"Take profit ({pnl_pct:.2%} = +{roe:.0f}% ROE)"

        # 2. TRAILING STOP
        elif position.get("trailing_stop_active", False):
            peak_pnl = position.get("peak_pnl_pct", 0)
            trailing_stop_level = peak_pnl - strategy.trailing_distance_pct
            if pnl_pct <= trailing_stop_level:
                should_exit = True
                exit_reason = f"Trailing stop (peak: {peak_pnl:.2%}, current: {pnl_pct:.2%})"

        # 3. STOP LOSS
        elif pnl_pct <= -strategy.stop_loss_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"Stop loss ({pnl_pct:.2%} = {roe:.0f}% ROE)"
            # Activate cooldown
            self.stop_loss_cooldowns[pos_key] = datetime.now(timezone.utc) + timedelta(minutes=self.stop_loss_cooldown_minutes)
            self.daily_stop_losses[pos_key] = self.daily_stop_losses.get(pos_key, 0) + 1
            logger.info(f"   [{strategy.name}] SL cooldown activated. Daily: {self.daily_stop_losses[pos_key]}/{self.max_daily_stop_losses}")

        # 4. TIME-BASED EXIT
        elif self._is_stale_position(position, pnl_pct, strategy):
            should_exit = True
            hours_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 3600
            exit_reason = f"Stale position ({hours_open:.1f}h, only {pnl_pct:.2%} profit)"

        # 5. RSI-BASED PROFIT TAKING
        elif self._should_exit_on_rsi(position, signal, pnl_pct, strategy):
            indicators = signal.get("indicators", {})
            rsi = indicators.get("rsi", 50)
            should_exit = True
            exit_reason = f"RSI profit taking (RSI={rsi:.0f}, profit={pnl_pct:.2%})"

        # 6. SIGNAL-BASED EXIT
        elif self._should_exit_on_signal(position, signal, pnl_pct):
            should_exit = True
            confidence = signal.get("confidence", 0)
            exit_reason = f"{signal['signal'].replace('_', ' ').title()} signal (confidence: {confidence:.0%})"

        if should_exit:
            await self._close_position(pos_key, price, exit_reason, strategy)

    def _is_stale_position(self, position: dict, pnl_pct: float, strategy: StrategyConfig) -> bool:
        """Check if position is stale (old with minimal profit)."""
        hours_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 3600
        if hours_open >= strategy.max_position_hours:
            if pnl_pct < self.stale_position_min_profit_pct:
                return True
        return False

    def _should_exit_on_rsi(self, position: dict, signal: dict, pnl_pct: float, strategy: StrategyConfig) -> bool:
        """Exit profitable positions when RSI indicates overbought/oversold reversal."""
        if pnl_pct < strategy.trailing_activation_pct:
            return False

        indicators = signal.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        rsi_momentum = indicators.get("rsi_momentum", "")
        side = position.get("side")

        if side == "long" and rsi > 80 and rsi_momentum == "falling":
            return True
        if side == "short" and rsi < 20 and rsi_momentum == "rising":
            return True
        return False

    def _update_trailing_stop(self, pos_key: str, current_pnl_pct: float, strategy: StrategyConfig):
        """Update trailing stop based on current P&L."""
        position = self.positions.get(pos_key)
        if not position:
            return

        if current_pnl_pct >= strategy.trailing_activation_pct:
            if not position.get("trailing_stop_active", False):
                position["trailing_stop_active"] = True
                position["peak_pnl_pct"] = current_pnl_pct
                _, symbol = self._parse_pos_key(pos_key)
                logger.info(
                    f"🔒 Trailing stop activated [{strategy.name}] {symbol} at {current_pnl_pct:.2%}"
                )
            elif current_pnl_pct > position.get("peak_pnl_pct", 0):
                position["peak_pnl_pct"] = current_pnl_pct
                _, symbol = self._parse_pos_key(pos_key)
                logger.debug(f"📈 [{strategy.name}] {symbol} new peak: {current_pnl_pct:.2%}")

    def _should_exit_on_signal(self, position: dict, signal: dict, pnl_pct: float) -> bool:
        """Determine if should exit based on opposite signal."""
        signal_type = signal.get("signal", "hold")
        confidence = signal.get("confidence", 0)
        side = position.get("side")

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

        if pnl_pct >= self.signal_exit_min_profit_pct:
            if not is_strong_signal:
                return False

        if confidence < self.signal_exit_min_confidence:
            return False

        return True

    def _calculate_pnl_pct(self, position: dict, current_price: float) -> float:
        """Calculate position PnL percentage."""
        entry_price = position["entry_price"]
        if position["side"] == "long":
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price

    async def _open_position(self, pos_key: str, symbol: str, side: str, price: float,
                              signal: dict, strategy: StrategyConfig):
        """Open a leveraged position."""
        stats = self.strategy_stats[strategy.name]

        # Margin = strategy capital * position percentage
        margin = stats["capital"] * self.max_position_pct
        position_value = margin * self.leverage
        quantity = position_value / price

        # Calculate stop loss, take profit, and liquidation prices
        if side == "long":
            stop_price = price * (1 - strategy.stop_loss_pct)
            take_profit_price = price * (1 + strategy.take_profit_pct)
            liquidation_price = price * (1 - self.liquidation_pct)
        else:
            stop_price = price * (1 + strategy.stop_loss_pct)
            take_profit_price = price * (1 - strategy.take_profit_pct)
            liquidation_price = price * (1 + self.liquidation_pct)

        self.positions[pos_key] = {
            "position_id": str(uuid.uuid4()),
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": margin,
            "leverage": self.leverage,
            "entry_time": datetime.now(timezone.utc),
            "signal": signal,
            "trailing_stop_active": False,
            "peak_pnl_pct": 0.0,
            "stop_loss_price": stop_price,
            "take_profit_price": take_profit_price,
            "liquidation_price": liquidation_price,
            "strategy_name": strategy.name,
        }

        logger.info(
            f"📈 OPEN [{strategy.name}] {side.upper()} {symbol} @ ${price:,.2f} | "
            f"Margin: ${margin:,.2f} | Size: ${position_value:,.2f} ({self.leverage}x) | "
            f"Conf: {signal['confidence']:.0%}"
        )
        logger.info(
            f"   └─ SL: ${stop_price:,.2f} ({strategy.stop_loss_pct:.1%}) | "
            f"TP: ${take_profit_price:,.2f} ({strategy.take_profit_pct:.1%}) | "
            f"Liq: ${liquidation_price:,.2f}"
        )

        # Log to Supabase
        try:
            await self.trade_repo.log_trade({
                "position_id": self.positions[pos_key]["position_id"],
                "symbol": symbol,
                "exchange": "binance",
                "side": "buy" if side == "long" else "sell",
                "entry_price": price,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "quantity": quantity,
                "strategy_name": strategy.name,
                "signal_source": strategy.source_label,
                "signal_confidence": signal["confidence"],
                "entry_reasoning": signal["reasoning"],
                "indicators_at_entry": signal.get("indicators"),
            })
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    async def _close_position(self, pos_key: str, price: float, reason: str, strategy: StrategyConfig):
        """Close a leveraged position."""
        position = self.positions.pop(pos_key, None)
        if not position:
            return

        strategy_name, symbol = self._parse_pos_key(pos_key)
        stats = self.strategy_stats[strategy.name]

        # Calculate PnL on the leveraged position
        if position["side"] == "long":
            pnl = (price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - price) * position["quantity"]

        pnl_pct = self._calculate_pnl_pct(position, price)
        margin = position.get("margin", position["quantity"] * position["entry_price"] / self.leverage)
        roe = (pnl / margin) * 100 if margin > 0 else 0

        # Update per-strategy stats
        stats["capital"] += pnl
        stats["total_pnl"] += pnl
        stats["trade_count"] += 1
        if pnl > 0:
            stats["winning_trades"] += 1

        duration = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds()

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} CLOSE [{strategy.name}] {symbol} @ ${price:,.2f} | "
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
        for pos_key in list(self.positions.keys()):
            try:
                strategy_name, symbol = self._parse_pos_key(pos_key)
                strategy = self.strategy_map.get(strategy_name)
                if not strategy:
                    continue
                ticker = await collector.get_binance_ticker(symbol)
                price = ticker.get("price", self.positions[pos_key]["entry_price"]) if ticker else self.positions[pos_key]["entry_price"]
                await self._close_position(pos_key, price, "Session end", strategy)
            except Exception as e:
                logger.error(f"Error closing {pos_key}: {e}")

    async def _save_signal(self, symbol: str, signal: dict, price: float, strategy: StrategyConfig):
        """Save signal to Supabase."""
        try:
            # Convert indicators to JSON-safe format
            indicators = signal.get("indicators", {})
            if indicators:
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
                "source": strategy.source_label,
                "confidence": signal["confidence"],
                "entry_price": price,
                "reasoning": signal["reasoning"],
                "indicators": indicators,
                "status": "pending",
            })
        except Exception as e:
            logger.debug(f"Failed to save signal: {e}")

    async def _save_performance_snapshots(self):
        """Save performance snapshot for each strategy."""
        for strategy in self.strategies:
            try:
                stats = self.strategy_stats[strategy.name]
                positions_value = sum(
                    p["quantity"] * p["entry_price"]
                    for pk, p in self.positions.items()
                    if pk.startswith(f"{strategy.name}:")
                )

                win_rate = stats["winning_trades"] / stats["trade_count"] if stats["trade_count"] > 0 else 0

                await self.perf_repo.save_snapshot({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "strategy_name": strategy.name,
                    "total_equity": stats["capital"] + positions_value,
                    "cash_balance": stats["capital"],
                    "positions_value": positions_value,
                    "total_pnl": stats["total_pnl"],
                    "total_trades": stats["trade_count"],
                    "winning_trades": stats["winning_trades"],
                    "losing_trades": stats["trade_count"] - stats["winning_trades"],
                    "win_rate": win_rate,
                    "open_positions": {
                        pk.split(":", 1)[1]: {"side": p["side"], "entry": p["entry_price"]}
                        for pk, p in self.positions.items()
                        if pk.startswith(f"{strategy.name}:")
                    },
                })
            except Exception as e:
                logger.debug(f"Failed to save performance for {strategy.name}: {e}")

    def _print_status(self):
        """Print current status with per-strategy details."""
        total_cap = self.total_capital
        total_trades = self.total_trades
        total_winning = self.total_winning
        win_rate = total_winning / total_trades * 100 if total_trades > 0 else 0

        logger.info(
            f"💰 Total: ${total_cap:,.2f} | "
            f"Trades: {total_trades} | WR: {win_rate:.0f}% | "
            f"Open: {len(self.positions)}"
        )

        # Per-strategy summary
        for s in self.strategies:
            stats = self.strategy_stats[s.name]
            s_trades = stats["trade_count"]
            s_wr = stats["winning_trades"] / s_trades * 100 if s_trades > 0 else 0
            s_pnl = stats["total_pnl"]
            pnl_emoji = "🟢" if s_pnl >= 0 else "🔴"
            logger.info(
                f"   {pnl_emoji} {s.name}: ${stats['capital']:,.2f} | "
                f"PnL: ${s_pnl:+,.2f} | {s_trades} trades | WR: {s_wr:.0f}%"
            )

        # Open positions
        for pos_key, pos in self.positions.items():
            strategy_name, symbol = self._parse_pos_key(pos_key)
            trailing = "🔒" if pos.get("trailing_stop_active") else "⏳"
            peak = pos.get("peak_pnl_pct", 0)
            logger.info(
                f"   {trailing} [{strategy_name}] {symbol}: {pos['side']} @ ${pos['entry_price']:,.2f} | "
                f"Peak: {peak:.2%} | "
                f"SL: ${pos.get('stop_loss_price', 0):,.2f} | "
                f"TP: ${pos.get('take_profit_price', 0):,.2f}"
            )

    def _print_summary(self):
        """Print trading summary."""
        total_cap = self.total_capital
        total_return = (total_cap / self.initial_capital - 1) * 100 if self.initial_capital > 0 else 0
        total_trades = self.total_trades
        total_winning = self.total_winning
        win_rate = total_winning / total_trades * 100 if total_trades > 0 else 0

        print("\n" + "=" * 70)
        print("📊 MULTI-STRATEGY PAPER TRADING SUMMARY")
        print("=" * 70)
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Capital:    ${total_cap:,.2f}")
        print(f"Total Return:     {total_return:+.2f}%")
        print("-" * 70)

        for s in self.strategies:
            stats = self.strategy_stats[s.name]
            s_return = (stats["capital"] / stats["initial_capital"] - 1) * 100 if stats["initial_capital"] > 0 else 0
            s_trades = stats["trade_count"]
            s_wr = stats["winning_trades"] / s_trades * 100 if s_trades > 0 else 0
            print(f"  {s.name}:")
            print(f"    Capital: ${stats['initial_capital']:,.2f} → ${stats['capital']:,.2f} ({s_return:+.2f}%)")
            print(f"    Trades: {s_trades} | WR: {s_wr:.1f}% | PnL: ${stats['total_pnl']:+,.2f}")

        print("-" * 70)
        print(f"Total Trades:     {total_trades}")
        print(f"Winning Trades:   {total_winning}")
        print(f"Losing Trades:    {total_trades - total_winning}")
        print(f"Win Rate:         {win_rate:.1f}%")
        print(f"Total PnL:        ${self.total_pnl:,.2f}")
        print("=" * 70)


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Strategy Paper Trading")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "XRPUSDT"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=1000.0,
        help="Capital per strategy (each strategy gets this amount)"
    )
    parser.add_argument(
        "--leverage", type=int, default=10,
        help="Leverage multiplier (default: 10x)"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Check interval in seconds"
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=["agreement_classic", "agreement_mtf", "momentum"],
        choices=["agreement_classic", "agreement_mtf", "momentum"],
        help="Strategies to run (default: all three)"
    )
    # Exit params (applied to agreement strategies; momentum has own defaults)
    parser.add_argument("--stop-loss", type=float, default=0.012, help="Agreement SL %% (default: 1.2%%)")
    parser.add_argument("--take-profit", type=float, default=0.025, help="Agreement TP %% (default: 2.5%%)")
    parser.add_argument("--trailing-activation", type=float, default=0.015, help="Agreement trailing activation (default: 1.5%%)")
    parser.add_argument("--trailing-distance", type=float, default=0.008, help="Agreement trailing distance (default: 0.8%%)")
    parser.add_argument("--max-position-hours", type=float, default=4.0, help="Agreement max hold hours (default: 4.0)")

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

    # Build strategy configs
    selected = args.strategies
    capital_per_strategy = args.capital  # Each strategy gets full capital amount

    strategy_configs = []
    for name in selected:
        if name == "agreement_classic":
            strategy_configs.append(StrategyConfig(
                name="agreement_classic",
                strategy_type="agreement",
                generator=TechnicalSignalGenerator(),
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit,
                trailing_activation_pct=args.trailing_activation,
                trailing_distance_pct=args.trailing_distance,
                max_position_hours=args.max_position_hours,
                capital=capital_per_strategy,
            ))
        elif name == "agreement_mtf":
            strategy_configs.append(StrategyConfig(
                name="agreement_mtf",
                strategy_type="agreement_mtf",
                generator=MTFSignalGenerator(),
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit,
                trailing_activation_pct=args.trailing_activation,
                trailing_distance_pct=args.trailing_distance,
                max_position_hours=args.max_position_hours,
                capital=capital_per_strategy,
            ))
        elif name == "momentum":
            strategy_configs.append(StrategyConfig(
                name="momentum",
                strategy_type="momentum",
                generator=MomentumBreakoutGenerator(),
                stop_loss_pct=0.008,         # Tighter SL for momentum
                take_profit_pct=0.018,       # Faster TP
                trailing_activation_pct=0.010,
                trailing_distance_pct=0.005,
                max_position_hours=2.0,      # Shorter hold
                capital=capital_per_strategy,
            ))

    # Create trader
    trader = SimplePaperTrader(
        symbols=args.symbols,
        strategies=strategy_configs,
        initial_capital=args.capital,
        leverage=args.leverage,
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
