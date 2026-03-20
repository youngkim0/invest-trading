#!/usr/bin/env python
"""Multi-strategy paper trading script using Binance real-time data and Supabase.

v6.5 — Research-based short strategies (regime-filtered):
- All short strategies gated by 4h bearish regime (SMA50 slope negative OR price < SMA200)
- 3 new short strategies replacing v6.4's non-firing derivatives-only shorts:
  - regime_short_confluence: moderate thresholds on 4+ conditions (funding>0.03% + taker<0.95 +
    top traders >55% long + price<SMA20). Research: multi-condition confluence > single extreme.
  - failed_breakout_short: price action exhaustion (new 20-bar high + long upper wick +
    high volume + failure candle). No derivatives dependency.
  - refined_liq_cascade: derivatives-based with realistic thresholds (funding>0.05% +
    OI rising>1% 4h + sustained taker selling + RSI<45).
- Key changes from v6.4: wider SL (2.0x ATR), smaller position size (1.5% risk),
  time stops (8-12h), regime gate prevents shorting in bull markets.
- Research basis: BIS Working Paper 1087, QuantJourney funding analysis, PocketOption Z-score study

v6.4 — Short-only strategies (REPLACED by v6.5):
- liquidation_cascade: 0 trades ever (funding threshold 0.01% never reached)
- panic_momentum: 12 trades, -$22 (taker 5m<0.90 too extreme)
- breakdown_reversal: 1 trade, -$7 (volume 2.0x threshold too strict)

v6.3.5 — Long-only mode:
- Disabled all short/sell signals across all 4 strategies
- Evidence: 38 short trades had 18.4% WR, -$262 total. No strategy had short edge.
- Buys: 55.9% WR, +$698. Removing shorts eliminates $262 drag.

v6.3 — Reversal adaptation:
- Fast 15m reversal override: overrides lagging 1h HTF when 15m shows clear reversal
- Global circuit breaker: 3 SLs in 2h → pause all trading for 1h
- Fixes blind spot where all strategies kept buying into bearish reversal
"""

import asyncio
import signal
import sys
from dataclasses import dataclass, field
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


def calculate_rsi(prices: pd.Series, period: int = 14) -> dict:
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range. Returns ATR value or 0 if insufficient data."""
    if len(df) < period + 1:
        return 0.0
    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    tr = pd.concat([
        highs - lows,
        (highs - closes.shift(1)).abs(),
        (lows - closes.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    val = float(atr.iloc[-1])
    return val if not pd.isna(val) else 0.0


def determine_htf_trend(df_1h: pd.DataFrame) -> dict:
    """Determine higher-timeframe trend from 1h candles.

    Returns dict with {direction, strength, slope} instead of just a string.
    - direction: 'bullish', 'bearish', or 'neutral'
    - strength: 0.0-1.0 (SMA spread normalized)
    - slope: SMA20 slope (positive = rising)
    """
    if len(df_1h) < 50:
        return {"direction": "neutral", "strength": 0.0, "slope": 0.0}

    prices = df_1h["close"]
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    current_price = float(prices.iloc[-1])
    sma20_val = float(sma20.iloc[-1])
    sma50_val = float(sma50.iloc[-1])

    # SMA spread as strength indicator (0-1 scale, capped at 2%)
    sma_spread = abs(sma20_val - sma50_val) / sma50_val if sma50_val > 0 else 0
    strength = min(1.0, sma_spread / 0.02)  # 2% spread = max strength

    # SMA20 slope: change over last 5 bars
    if len(sma20) >= 5:
        sma20_5ago = float(sma20.iloc[-5])
        slope = (sma20_val - sma20_5ago) / sma20_5ago if sma20_5ago > 0 else 0
    else:
        slope = 0.0

    # Direction
    if sma_spread < 0.003:
        direction = "neutral"
        strength = strength * 0.5  # Reduce strength in ranging markets
    elif sma20_val > sma50_val and current_price > sma20_val:
        direction = "bullish"
    elif sma20_val < sma50_val and current_price < sma20_val:
        direction = "bearish"
    else:
        direction = "neutral"

    return {"direction": direction, "strength": round(strength, 3), "slope": round(slope, 5)}


def apply_fast_reversal_override(htf_trend: dict, df_15m: pd.DataFrame) -> dict:
    """Suppress HTF trend when 15m data conflicts (don't flip direction).

    When 15m shows the opposite of 1h, set direction to neutral to STOP
    trading rather than reversing into shorts. This prevents forced sells
    during normal pullbacks while still protecting against blind trend-following.
    """
    if df_15m is None or len(df_15m) < 25:
        return htf_trend

    htf_direction = htf_trend.get("direction", "neutral")
    if htf_direction == "neutral":
        return htf_trend

    prices = df_15m["close"]
    current_price = float(prices.iloc[-1])
    sma20 = float(prices.rolling(20).mean().iloc[-1])
    rsi_data = calculate_rsi(prices)
    rsi = rsi_data["value"]

    # Bearish conflict: HTF bullish but 15m bearish → suppress to neutral
    if htf_direction == "bullish" and current_price < sma20 and rsi < 40:
        logger.info(
            f"⚡ HTF conflict: bullish suppressed to NEUTRAL "
            f"(15m price ${current_price:,.2f} < SMA20 ${sma20:,.2f}, RSI {rsi:.0f})"
        )
        return {
            "direction": "neutral",
            "strength": htf_trend.get("strength", 0.0) * 0.3,
            "slope": htf_trend.get("slope", 0.0),
            "suppressed": True,
        }

    # Bullish conflict: HTF bearish but 15m bullish → suppress to neutral
    if htf_direction == "bearish" and current_price > sma20 and rsi > 60:
        logger.info(
            f"⚡ HTF conflict: bearish suppressed to NEUTRAL "
            f"(15m price ${current_price:,.2f} > SMA20 ${sma20:,.2f}, RSI {rsi:.0f})"
        )
        return {
            "direction": "neutral",
            "strength": htf_trend.get("strength", 0.0) * 0.3,
            "slope": htf_trend.get("slope", 0.0),
            "suppressed": True,
        }

    return htf_trend


def hold_signal(reasoning: str = "No signal", htf_trend: dict = None) -> dict:
    """Return a standard hold signal."""
    return {
        "signal": "hold",
        "confidence": 0.5,
        "reasoning": reasoning,
        "indicators": {"htf_trend": htf_trend.get("direction", "neutral") if htf_trend else "neutral"},
    }


def apply_htf_adjustment(signal_type: str, confidence: float, htf_trend: dict, reasons: list) -> float:
    """Adjust confidence based on HTF trend alignment.

    Instead of hard-blocking, HTF strength modulates confidence:
    - Aligned with trend: +5-15% confidence boost
    - Counter-trend: -10-25% confidence penalty (scaled by trend strength)
    - Neutral: no adjustment
    """
    direction = htf_trend.get("direction", "neutral")
    strength = htf_trend.get("strength", 0.0)

    is_buy = signal_type in ("buy", "strong_buy")
    is_sell = signal_type in ("sell", "strong_sell")

    if direction == "bullish":
        if is_buy:
            boost = 0.05 + 0.10 * strength
            confidence = min(0.90, confidence + boost)
            reasons.append(f"HTF bullish +{boost:.0%} conf (str={strength:.2f})")
        elif is_sell:
            penalty = 0.10 + 0.15 * strength
            confidence = max(0.45, confidence - penalty)
            reasons.append(f"HTF bullish -{penalty:.0%} conf for sell")
    elif direction == "bearish":
        if is_sell:
            boost = 0.05 + 0.10 * strength
            confidence = min(0.90, confidence + boost)
            reasons.append(f"HTF bearish +{boost:.0%} conf (str={strength:.2f})")
        elif is_buy:
            penalty = 0.10 + 0.15 * strength
            confidence = max(0.45, confidence - penalty)
            reasons.append(f"HTF bearish -{penalty:.0%} conf for buy")

    return confidence


def check_bearish_regime(df_4h: pd.DataFrame) -> dict:
    """Check if market is in a bearish regime suitable for shorting.

    Regime gate prevents shorting in bull markets (structural long bias = negative EV).
    Research: shorts only work when macro trend is bearish (BIS Working Paper 1087).

    Uses three fast-responding triggers (any one = bearish):
    1. Price below 4h SMA50 — sustained downtrend (~8 days)
    2. 4h SMA20 slope negative (last 3 candles) — trend change (~3 days)
    3. Price dropped >3% from 5-day high (30 candles) — catches sudden selloffs

    Returns dict with:
    - is_bearish: True if regime allows shorting
    - reason: Human-readable explanation
    - sma50_slope: 4h SMA50 slope (negative = bearish)
    - price_vs_sma200: price position relative to SMA200 (if available)
    """
    if df_4h is None or len(df_4h) < 30:
        return {"is_bearish": False, "reason": "Insufficient 4h data for regime check",
                "sma50_slope": 0, "price_vs_sma200": None}

    prices = df_4h["close"]
    highs = df_4h["high"]
    current_price = float(prices.iloc[-1])
    reasons = []

    # === Trigger 1: Price below 4h SMA50 ===
    sma50_val = None
    slope = 0
    if len(df_4h) >= 50:
        sma50 = prices.rolling(50).mean()
        sma50_val = float(sma50.iloc[-1])
        sma50_5ago = float(sma50.iloc[-5]) if len(sma50) >= 5 else sma50_val
        slope = (sma50_val - sma50_5ago) / sma50_5ago if sma50_5ago > 0 else 0
        if current_price < sma50_val:
            reasons.append(f"price below 4h SMA50 ({current_price:.2f} < {sma50_val:.2f})")

    # === Trigger 2: 4h SMA20 slope negative (last 3 candles) ===
    if len(df_4h) >= 23:
        sma20 = prices.rolling(20).mean()
        sma20_now = float(sma20.iloc[-1])
        sma20_3ago = float(sma20.iloc[-3])
        sma20_slope = (sma20_now - sma20_3ago) / sma20_3ago if sma20_3ago > 0 else 0
        if sma20_slope < 0:
            reasons.append(f"4h SMA20 slope negative ({sma20_slope*100:+.3f}%)")

    # === Trigger 3: Price dropped >3% from 5-day high (30 candles) ===
    lookback = min(30, len(highs))
    high_5d = float(highs.iloc[-lookback:].max())
    drop_pct = (current_price - high_5d) / high_5d * 100
    if drop_pct < -3.0:
        reasons.append(f"price dropped {drop_pct:.1f}% from 5-day high ({high_5d:.2f})")

    # Optional: compute SMA200 for metadata
    price_vs_sma200 = None
    if len(df_4h) >= 200:
        sma200_val = float(prices.rolling(200).mean().iloc[-1])
        price_vs_sma200 = (current_price - sma200_val) / sma200_val

    if reasons:
        return {
            "is_bearish": True,
            "reason": "Bearish regime: " + ", ".join(reasons),
            "sma50_slope": slope,
            "price_vs_sma200": price_vs_sma200,
        }

    return {
        "is_bearish": False,
        "reason": f"Bullish regime: price above SMA50, SMA20 rising, no recent drop",
        "sma50_slope": slope,
        "price_vs_sma200": price_vs_sma200,
    }


# =============================================================================
# Strategy 1: Funding Mean Reversion (LOW frequency, 0-2 trades/day)
# =============================================================================

class FundingMeanReversionGenerator:
    """Contrarian funding rate strategy — elevated funding precedes reversals.

    Entry requires ALL 3 conditions (boolean, no scoring):
    1. Funding rate elevated: > 0.03% (short) or < -0.02% (long)
    2. OI rising > 0.5% over last 30min (new money = fuel)
    3. Price hasn't already reversed > 0.5% in last hour
    """

    def __init__(self):
        self._last_funding_rate: float | None = None
        self._last_funding_time: str | None = None

    def generate_signal(self, df_1m: pd.DataFrame, htf_trend: dict,
                        derivatives: dict = None, **kwargs) -> dict:
        """Generate signal from funding rate mean reversion."""
        if not derivatives:
            return hold_signal("No derivatives data", htf_trend)

        prices = df_1m["close"] if not df_1m.empty else pd.Series()
        if len(prices) < 60:
            return hold_signal("Insufficient 1m data", htf_trend)

        current_price = float(prices.iloc[-1])
        reasons = []

        # === CONDITION 1: Extreme funding rate ===
        funding_data = derivatives.get("funding_rate", [])
        if not funding_data:
            return hold_signal("No funding data", htf_trend)

        latest = funding_data[-1]
        rate = latest["funding_rate"]
        funding_time = latest["funding_time"]

        # Cache funding rate
        if self._last_funding_time != funding_time:
            self._last_funding_rate = rate
            self._last_funding_time = funding_time
        else:
            rate = self._last_funding_rate if self._last_funding_rate is not None else rate

        # Determine direction from funding
        direction = None
        if rate > 0.0003:  # > 0.03% → short (contrarian)
            # Shorts disabled — 18.4% WR across 38 trades, no edge
            return hold_signal(f"Funding short disabled (rate {rate*100:.3f}%)", htf_trend)
        elif rate < -0.0002:  # < -0.02% → long (contrarian)
            direction = "buy"
            reasons.append(f"Funding elevated- {rate*100:.3f}% → long")
        else:
            return hold_signal(f"Funding normal ({rate*100:.3f}%)", htf_trend)

        # === CONDITION 2: OI rising > 0.5% over last 30min ===
        oi_history = derivatives.get("oi_history", [])
        if len(oi_history) < 2:
            return hold_signal("Insufficient OI data", htf_trend)

        current_oi = oi_history[-1]["sum_open_interest_value"]
        # Find OI ~30min ago (6 x 5min intervals)
        lookback_idx = max(0, len(oi_history) - 7)
        earlier_oi = oi_history[lookback_idx]["sum_open_interest_value"]
        oi_change_pct = (current_oi - earlier_oi) / earlier_oi * 100 if earlier_oi > 0 else 0

        if oi_change_pct < 0.5:
            return hold_signal(f"OI not rising enough ({oi_change_pct:.2f}%)", htf_trend)
        reasons.append(f"OI +{oi_change_pct:.2f}% (30min)")

        # === CONDITION 3: Price hasn't already reversed > 0.5% in last hour ===
        price_1h_ago = float(prices.iloc[-60]) if len(prices) >= 60 else float(prices.iloc[0])
        price_change_1h = (current_price - price_1h_ago) / price_1h_ago * 100

        if direction == "buy" and price_change_1h > 0.5:
            return hold_signal(f"Already reversed +{price_change_1h:.2f}% (too late)", htf_trend)
        elif direction == "sell" and price_change_1h < -0.5:
            return hold_signal(f"Already reversed {price_change_1h:.2f}% (too late)", htf_trend)
        reasons.append(f"Price 1h: {price_change_1h:+.2f}% (not reversed yet)")

        # All 3 conditions met — generate signal
        is_strong = abs(rate) > 0.001  # Very extreme funding = strong signal
        signal_type = f"strong_{direction}" if is_strong else direction
        confidence = 0.75 if is_strong else 0.68

        # Apply HTF adjustment
        confidence = apply_htf_adjustment(signal_type, confidence, htf_trend, reasons)

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "funding_rate": rate,
                "oi_change_pct": oi_change_pct,
                "price_change_1h": price_change_1h,
                "htf_trend": htf_trend.get("direction", "neutral"),
                "htf_strength": htf_trend.get("strength", 0),
            },
        }


# =============================================================================
# Strategy 2: Trend Breakout (MEDIUM frequency, 2-5 trades/day)
# =============================================================================

class TrendBreakoutGenerator:
    """Trade 15m breakouts, trend-aligned or range breakouts with volume.

    Entry requires ALL 3 conditions (boolean, no scoring):
    1. HTF trending (strength > 0.1) OR ranging (neutral) with higher volume bar
    2. 15m price closes above 10-bar high (long) or below 10-bar low (short)
    3. 15m volume > 1.2x avg (trending) or > 1.5x avg (ranging)
    """

    def generate_signal(self, df_15m: pd.DataFrame, htf_trend: dict,
                        **kwargs) -> dict:
        """Generate signal from trend-aligned or range breakouts on 15m."""
        if df_15m is None or len(df_15m) < 30:
            return hold_signal("Insufficient 15m data", htf_trend)

        direction = htf_trend.get("direction", "neutral")
        strength = htf_trend.get("strength", 0.0)
        reasons = []

        # === CONDITION 1: HTF trending OR ranging (neutral allows breakouts with higher vol) ===
        required_vol_ratio = 1.2
        if direction == "neutral":
            # Ranging market: allow breakouts in either direction, require higher volume
            required_vol_ratio = 1.5
            reasons.append("Ranging mode (vol 1.5x req)")
        elif strength < 0.1:
            return hold_signal(f"HTF {direction} too weak (str={strength:.2f})", htf_trend)
        else:
            reasons.append(f"HTF {direction} str={strength:.2f}")

        prices = df_15m["close"]
        highs = df_15m["high"]
        lows = df_15m["low"]
        volumes = df_15m["volume"]
        current_price = float(prices.iloc[-1])

        # === CONDITION 2: 15m breakout (10-bar high/low) ===
        high_10 = float(highs.iloc[-11:-1].max())  # 10 bars before current
        low_10 = float(lows.iloc[-11:-1].min())

        signal_direction = None
        if direction == "bullish" and current_price > high_10:
            signal_direction = "buy"
            reasons.append(f"15m break above 10-bar high ${high_10:,.2f}")
        elif direction == "bearish" and current_price < low_10:
            # Shorts disabled — 10% WR on breakout shorts, no edge
            return hold_signal(f"Bearish breakout short disabled", htf_trend)
        elif direction == "neutral":
            # Ranging: only trade long breakouts (shorts disabled)
            if current_price > high_10:
                signal_direction = "buy"
                reasons.append(f"15m range break above ${high_10:,.2f}")
            elif current_price < low_10:
                return hold_signal(f"Range short breakout disabled", htf_trend)
            else:
                return hold_signal(f"No breakout (${current_price:,.2f} in range ${low_10:,.2f}-${high_10:,.2f})", htf_trend)
        else:
            return hold_signal(f"No breakout (price ${current_price:,.2f}, H10=${high_10:,.2f}, L10=${low_10:,.2f})", htf_trend)

        # === CONDITION 3: 15m volume > required threshold ===
        avg_vol = float(volumes.iloc[-21:-1].mean())
        current_vol = float(volumes.iloc[-1])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        if vol_ratio < required_vol_ratio:
            return hold_signal(f"Volume too low ({vol_ratio:.2f}x, need {required_vol_ratio}x)", htf_trend)
        reasons.append(f"Volume {vol_ratio:.1f}x avg")

        # All 3 conditions met
        is_strong = vol_ratio > 2.0 and strength > 0.6
        signal_type = f"strong_{signal_direction}" if is_strong else signal_direction
        confidence = 0.75 if is_strong else 0.70

        # Reduce confidence for range breakouts (no HTF confirmation)
        if direction == "neutral":
            confidence -= 0.05
            reasons.append("Range breakout (-5% conf)")

        # HTF is already aligned by design, so just a small boost for high strength
        if strength > 0.6:
            confidence = min(0.85, confidence + 0.05)
            reasons.append(f"Strong trend boost")

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "high_10": high_10,
                "low_10": low_10,
                "vol_ratio": vol_ratio,
                "htf_trend": direction,
                "htf_strength": strength,
            },
        }


# =============================================================================
# Strategy 3: Trend Pullback (MEDIUM frequency, 2-5 trades/day)
# =============================================================================

class TrendPullbackGenerator:
    """Buy dips in uptrends, sell rallies in downtrends.

    Complementary to TrendBreakout: breakouts catch the START of moves,
    pullbacks catch CONTINUATION within established trends.

    Entry requires ALL 3 conditions (boolean, no scoring):
    1. HTF trend established: strength > 0.15 (moderate trend, lower bar than breakout's 0.3)
    2. 15m RSI shows pullback: 30-45 in uptrend (dip) or 55-70 in downtrend (rally)
       — NOT extremes (<30 or >70 = potential reversal, not pullback)
    3. Price within 1.0x ATR(15m) of SMA20(15m) — pulled back to the mean

    Direction follows HTF trend. Better entry prices than breakout = better R:R.
    """

    def generate_signal(self, df_15m: pd.DataFrame, htf_trend: dict,
                        **kwargs) -> dict:
        """Generate signal from trend pullback."""
        if df_15m is None or len(df_15m) < 30:
            return hold_signal("Insufficient 15m data", htf_trend)

        direction = htf_trend.get("direction", "neutral")
        strength = htf_trend.get("strength", 0.0)
        reasons = []

        # === CONDITION 1: HTF trend established (buys: 0.15, sells: 0.3) ===
        min_strength = 0.15 if direction == "bullish" else 0.3
        if direction == "neutral" or strength < min_strength:
            return hold_signal(f"HTF {direction} too weak for pullback (str={strength:.2f})", htf_trend)
        reasons.append(f"HTF {direction} str={strength:.2f}")

        prices = df_15m["close"]
        current_price = float(prices.iloc[-1])

        # === CONDITION 2: 15m RSI shows pullback (NOT extremes) ===
        rsi_data = calculate_rsi(prices)
        rsi = rsi_data["value"]

        signal_direction = None
        if direction == "bullish" and 30 <= rsi <= 45:
            signal_direction = "buy"
            reasons.append(f"RSI {rsi:.0f} (pullback in uptrend)")
        elif direction == "bearish" and 55 <= rsi <= 70:
            # Shorts disabled — 0% WR on pullback shorts, no edge
            return hold_signal(f"Pullback short disabled (RSI {rsi:.0f})", htf_trend)
        else:
            if direction == "bullish":
                return hold_signal(f"RSI {rsi:.0f} not in pullback zone (need 30-45 for uptrend)", htf_trend)
            else:
                return hold_signal(f"RSI {rsi:.0f} not in pullback zone (need 55-70 for downtrend)", htf_trend)

        # === CONDITION 3: Price within 1.0x ATR of SMA20 (pulled back to mean) ===
        atr_15m = calculate_atr(df_15m)
        if atr_15m <= 0:
            return hold_signal("No ATR data", htf_trend)

        sma20 = float(prices.rolling(20).mean().iloc[-1])
        distance_from_sma = abs(current_price - sma20)
        max_distance = 1.0 * atr_15m

        if distance_from_sma > max_distance:
            return hold_signal(f"Price too far from SMA20 ({distance_from_sma/atr_15m:.1f}x ATR, need <1.0x)", htf_trend)
        reasons.append(f"Price {distance_from_sma/atr_15m:.1f}x ATR from SMA20")

        # All 3 conditions met
        confidence = 0.70

        # Boost for strong trend
        if strength > 0.4:
            confidence = min(0.85, confidence + 0.05)
            reasons.append("Strong trend boost")

        return {
            "signal": signal_direction,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "rsi": rsi,
                "sma20": sma20,
                "distance_from_sma_atr": round(distance_from_sma / atr_15m, 2),
                "atr_15m": atr_15m,
                "htf_trend": direction,
                "htf_strength": strength,
            },
        }


# =============================================================================
# Strategy 4: Order Flow (MEDIUM frequency, 2-5 trades/day)
# =============================================================================

class OrderFlowGenerator:
    """Aggressive order flow + non-crowded positioning = smart money entering.

    Uses two unused derivatives data sources:
    - Taker buy/sell ratio (15m): aggressive buyer/seller imbalance
    - Top trader long/short ratio: positioning crowding filter

    Entry requires ALL 3 conditions (boolean, no scoring):
    1. HTF trend established: strength > 0.1 (any directional trend)
    2. 15m taker buy/sell ratio confirms direction:
       > 1.05 for buys (aggressive buying pressure)
       < 0.95 for sells (aggressive selling pressure)
    3. Top trader positioning NOT crowded in our direction:
       long_account < 0.58 when buying (room to run, not overcrowded)
       short_account < 0.58 when selling (room to run, not overcrowded)

    Academic evidence: order flow has permanent price impact (Sharpe 3.63),
    backtested at 142% vs 101% B&H with 1.05/0.95 thresholds.
    """

    def generate_signal(self, df_15m: pd.DataFrame, htf_trend: dict,
                        derivatives: dict = None, **kwargs) -> dict:
        """Generate signal from order flow + positioning data."""
        if df_15m is None or len(df_15m) < 20:
            return hold_signal("Insufficient 15m data", htf_trend)
        if not derivatives:
            return hold_signal("No derivatives data", htf_trend)

        direction = htf_trend.get("direction", "neutral")
        strength = htf_trend.get("strength", 0.0)
        reasons = []

        # === CONDITION 1: HTF trend established (buys: 0.1, sells: 0.3) ===
        min_strength = 0.1 if direction == "bullish" else 0.3
        if direction == "neutral" or strength < min_strength:
            return hold_signal(f"HTF {direction} too weak (str={strength:.2f})", htf_trend)
        reasons.append(f"HTF {direction} str={strength:.2f}")

        # === CONDITION 2: 15m taker ratio confirms direction ===
        taker_data = derivatives.get("taker_ratio_15m", [])
        if not taker_data:
            return hold_signal("No taker ratio data", htf_trend)

        taker_entry = taker_data[0] if isinstance(taker_data, list) else taker_data
        taker_ratio = taker_entry.get("buy_sell_ratio", 1.0)

        signal_direction = None
        if direction == "bullish" and taker_ratio > 1.05:
            signal_direction = "buy"
            reasons.append(f"Taker ratio {taker_ratio:.3f} (aggressive buying)")
        elif direction == "bearish" and taker_ratio < 0.95:
            # Shorts disabled — 14.3% WR on order flow shorts, no edge
            return hold_signal(f"Order flow short disabled (taker {taker_ratio:.3f})", htf_trend)
        else:
            return hold_signal(
                f"Taker ratio {taker_ratio:.3f} not confirming {direction} (need >1.05 or <0.95)",
                htf_trend,
            )

        # === CONDITION 3: Top traders NOT crowded in our direction ===
        ls_data = derivatives.get("top_long_short", [])
        if not ls_data:
            return hold_signal("No long/short ratio data", htf_trend)

        ls_entry = ls_data[0] if isinstance(ls_data, list) else ls_data
        long_account = ls_entry.get("long_account", 0.5)
        short_account = ls_entry.get("short_account", 0.5)

        if signal_direction == "buy" and long_account >= 0.58:
            return hold_signal(
                f"Top traders too crowded long ({long_account:.1%}, need <58%)",
                htf_trend,
            )
        elif signal_direction == "sell" and short_account >= 0.58:
            return hold_signal(
                f"Top traders too crowded short ({short_account:.1%}, need <58%)",
                htf_trend,
            )

        crowd_pct = long_account if signal_direction == "buy" else short_account
        reasons.append(f"Top traders {crowd_pct:.0%} {'long' if signal_direction == 'buy' else 'short'} (not crowded)")

        # All 3 conditions met
        confidence = 0.70

        # Boost for strong taker imbalance
        if taker_ratio > 1.10 or taker_ratio < 0.90:
            confidence = min(0.85, confidence + 0.05)
            reasons.append("Strong flow imbalance boost")

        # Boost for contrarian positioning alignment (crowd is on other side)
        if (signal_direction == "buy" and long_account < 0.48) or \
           (signal_direction == "sell" and short_account < 0.48):
            confidence = min(0.85, confidence + 0.05)
            reasons.append("Contrarian positioning boost")

        return {
            "signal": signal_direction,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": float(df_15m["close"].iloc[-1]),
                "taker_ratio": taker_ratio,
                "long_account": long_account,
                "short_account": short_account,
                "htf_trend": direction,
                "htf_strength": strength,
            },
        }


# =============================================================================
# Strategy 5: Regime Short Confluence (SHORT-ONLY, 2-5 trades/day)
# =============================================================================

class RegimeShortConfluenceGenerator:
    """Multi-condition short using moderate thresholds on 4+ signals.

    Research-based: no single derivatives indicator works alone. The edge is in
    moderate thresholds on multiple indicators firing simultaneously.
    (QuantJourney funding analysis, BIS Working Paper 1087)

    REGIME GATE: 4h bearish regime required (checked externally).

    Entry requires ALL 4 conditions (boolean, no scoring):
    1. Funding rate > +0.03% (longs paying, crowded but not extreme)
    2. Taker buy/sell ratio (15m) < 0.95 (active selling pressure)
    3. Top traders > 55% long accounts (crowd is long = contrarian short)
    4. 15m price < SMA20 (bearish price action confirmation)
    """

    def generate_signal(self, df_15m: pd.DataFrame, htf_trend: dict,
                        derivatives: dict = None, regime: dict = None, **kwargs) -> dict:
        """Generate short signal from multi-condition confluence."""
        if df_15m is None or len(df_15m) < 30:
            return hold_signal("Insufficient 15m data", htf_trend)
        if not derivatives:
            return hold_signal("No derivatives data", htf_trend)

        # === REGIME GATE ===
        if not regime or not regime.get("is_bearish"):
            reason = regime.get("reason", "No regime data") if regime else "No regime data"
            return hold_signal(f"Short blocked: {reason}", htf_trend)

        prices = df_15m["close"]
        current_price = float(prices.iloc[-1])
        reasons = [f"Regime: {regime['reason'][:60]}"]

        # === CONDITION 1: Funding rate > +0.03% ===
        funding_data = derivatives.get("funding_rate", [])
        if not funding_data:
            return hold_signal("No funding data", htf_trend)

        rate = funding_data[-1]["funding_rate"]
        if rate <= 0.0003:  # > 0.03%
            return hold_signal(f"Funding too low ({rate*100:.3f}%, need >0.03%)", htf_trend)
        reasons.append(f"Funding +{rate*100:.3f}%")

        # === CONDITION 2: 15m taker ratio < 0.95 ===
        taker_data = derivatives.get("taker_ratio_15m", [])
        if not taker_data:
            return hold_signal("No taker ratio data", htf_trend)

        taker_entry = taker_data[0] if isinstance(taker_data, list) else taker_data
        taker_ratio = float(taker_entry.get("buy_sell_ratio", 1.0))

        if taker_ratio >= 0.95:
            return hold_signal(f"Taker ratio not bearish ({taker_ratio:.3f}, need <0.95)", htf_trend)
        reasons.append(f"Taker {taker_ratio:.3f} (selling)")

        # === CONDITION 3: Top traders > 55% long (crowd is long) ===
        ls_data = derivatives.get("top_long_short", [])
        if not ls_data:
            return hold_signal("No L/S ratio data", htf_trend)

        ls_entry = ls_data[0] if isinstance(ls_data, list) else ls_data
        long_account = float(ls_entry.get("long_account", 0.5))

        if long_account <= 0.55:
            return hold_signal(f"Top traders not crowded long ({long_account:.0%}, need >55%)", htf_trend)
        reasons.append(f"Top traders {long_account:.0%} long (crowded)")

        # === CONDITION 4: Price below 15m SMA20 ===
        sma20 = float(prices.rolling(20).mean().iloc[-1])
        if current_price >= sma20:
            return hold_signal(f"Price above SMA20 ({current_price:.2f} >= {sma20:.2f})", htf_trend)
        reasons.append(f"Price below SMA20")

        # All 4 conditions met — short signal
        rsi_data = calculate_rsi(prices)
        rsi = rsi_data["value"]
        is_strong = rate > 0.0005 and taker_ratio < 0.90 and long_account > 0.60
        signal_type = "strong_sell" if is_strong else "sell"
        confidence = 0.75 if is_strong else 0.70

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "funding_rate": rate,
                "taker_ratio": taker_ratio,
                "long_account": long_account,
                "sma20": sma20,
                "rsi": rsi,
                "regime_slope": regime.get("sma50_slope", 0),
            },
        }


# =============================================================================
# Strategy 6: Failed Breakout Short (SHORT-ONLY, 1-3 trades/day)
# =============================================================================

class FailedBreakoutShortGenerator:
    """Price exhaustion pattern — breakout above resistance fails and reverses.

    Research: Volume exhaustion + failed breakout = liquidity sweep reversal.
    (PocketOption Z-score study: Z>2.0 + RSI>78 = 78% success rate)

    REGIME GATE: 4h bearish regime required (checked externally).

    Entry requires ALL 4 conditions (boolean, no scoring):
    1. 15m candle made new 20-bar high (breakout attempt)
    2. Volume on breakout candle > 1.5x average (exhaustion volume)
    3. Breakout candle has long upper wick (closes in lower 40% of range)
    4. Next candle (current) closes below breakout candle's midpoint (failure)
    """

    def generate_signal(self, df_15m: pd.DataFrame, htf_trend: dict,
                        derivatives: dict = None, regime: dict = None, **kwargs) -> dict:
        """Generate short signal from failed breakout exhaustion."""
        if df_15m is None or len(df_15m) < 25:
            return hold_signal("Insufficient 15m data", htf_trend)

        # === REGIME GATE ===
        if not regime or not regime.get("is_bearish"):
            reason = regime.get("reason", "No regime data") if regime else "No regime data"
            return hold_signal(f"Short blocked: {reason}", htf_trend)

        prices = df_15m["close"]
        highs = df_15m["high"]
        lows = df_15m["low"]
        opens = df_15m["open"]
        volumes = df_15m["volume"]
        reasons = [f"Regime bearish"]

        # We look at the PREVIOUS candle as the breakout candle, current as confirmation
        if len(prices) < 22:
            return hold_signal("Need 22+ candles", htf_trend)

        # Previous candle (potential breakout)
        prev_high = float(highs.iloc[-2])
        prev_low = float(lows.iloc[-2])
        prev_close = float(prices.iloc[-2])
        prev_open = float(opens.iloc[-2])
        prev_vol = float(volumes.iloc[-2])
        prev_range = prev_high - prev_low

        # Current candle (confirmation)
        current_price = float(prices.iloc[-1])

        # === CONDITION 1: Previous candle made new 20-bar high ===
        high_20 = float(highs.iloc[-22:-2].max())  # 20 bars before prev candle
        if prev_high <= high_20:
            return hold_signal(f"No breakout (prev high {prev_high:.2f} <= 20-bar {high_20:.2f})", htf_trend)
        reasons.append(f"Prev candle broke 20-bar high {high_20:.2f}")

        # === CONDITION 2: Volume > 1.5x average ===
        avg_vol = float(volumes.iloc[-22:-2].mean())
        vol_ratio = prev_vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio < 1.5:
            return hold_signal(f"Breakout volume too low ({vol_ratio:.2f}x, need 1.5x)", htf_trend)
        reasons.append(f"Breakout vol {vol_ratio:.1f}x")

        # === CONDITION 3: Long upper wick (close in lower 40% of range) ===
        if prev_range <= 0:
            return hold_signal("Zero range candle", htf_trend)
        close_position = (prev_close - prev_low) / prev_range  # 0=closed at low, 1=closed at high
        if close_position > 0.40:
            return hold_signal(f"No rejection wick (close at {close_position:.0%} of range, need <40%)", htf_trend)
        reasons.append(f"Rejection wick (close at {close_position:.0%} of range)")

        # === CONDITION 4: Current candle closes below prev candle midpoint ===
        prev_midpoint = (prev_high + prev_low) / 2
        if current_price >= prev_midpoint:
            return hold_signal(f"No follow-through ({current_price:.2f} >= midpoint {prev_midpoint:.2f})", htf_trend)
        reasons.append(f"Failed: price {current_price:.2f} < midpoint {prev_midpoint:.2f}")

        # All 4 conditions met — short signal
        # RSI divergence as bonus (not required)
        rsi_data = calculate_rsi(prices)
        rsi = rsi_data["value"]
        is_strong = vol_ratio > 2.0 and close_position < 0.20
        signal_type = "strong_sell" if is_strong else "sell"
        confidence = 0.75 if is_strong else 0.70

        # Boost if RSI was overbought on breakout candle
        if rsi > 65:
            confidence = min(0.85, confidence + 0.05)
            reasons.append(f"RSI {rsi:.0f} (overbought)")

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "breakout_high": prev_high,
                "high_20": high_20,
                "vol_ratio": vol_ratio,
                "close_position": close_position,
                "prev_midpoint": prev_midpoint,
                "rsi": rsi,
            },
        }


# =============================================================================
# Strategy 7: Refined Liquidation Cascade (SHORT-ONLY, 0-2 trades/day)
# =============================================================================

class RefinedLiqCascadeGenerator:
    """Derivatives-based short with realistic thresholds (v6.4 cascade redesigned).

    v6.4 had funding > 0.01% which never fired. Research shows 0.05%+ is the
    actionable threshold. Also uses longer OI lookback (4h vs 30min) and
    sustained taker selling instead of instant snapshot.

    REGIME GATE: 4h bearish regime required (checked externally).

    Entry requires ALL 4 conditions (boolean, no scoring):
    1. Funding rate > +0.05% (genuinely elevated, not normal noise)
    2. OI rising > 1% over last 4h while price flat/falling (trapped longs adding)
    3. Taker buy/sell ratio (15m) < 0.93 (sustained selling, not just a blip)
    4. 15m RSI < 45 (momentum confirms weakness, not buying the dip)
    """

    def generate_signal(self, df_15m: pd.DataFrame, htf_trend: dict,
                        derivatives: dict = None, regime: dict = None, **kwargs) -> dict:
        """Generate short signal from refined liquidation cascade conditions."""
        if df_15m is None or len(df_15m) < 30:
            return hold_signal("Insufficient 15m data", htf_trend)
        if not derivatives:
            return hold_signal("No derivatives data", htf_trend)

        # === REGIME GATE ===
        if not regime or not regime.get("is_bearish"):
            reason = regime.get("reason", "No regime data") if regime else "No regime data"
            return hold_signal(f"Short blocked: {reason}", htf_trend)

        prices = df_15m["close"]
        current_price = float(prices.iloc[-1])
        reasons = [f"Regime bearish"]

        # === CONDITION 1: Funding rate > +0.05% ===
        funding_data = derivatives.get("funding_rate", [])
        if not funding_data:
            return hold_signal("No funding data", htf_trend)

        rate = funding_data[-1]["funding_rate"]
        if rate <= 0.0005:  # > 0.05%
            return hold_signal(f"Funding not elevated ({rate*100:.3f}%, need >0.05%)", htf_trend)
        reasons.append(f"Funding +{rate*100:.3f}% (elevated)")

        # === CONDITION 2: OI rising > 1% (use full available history) ===
        oi_history = derivatives.get("oi_history", [])
        if len(oi_history) < 7:
            return hold_signal("Insufficient OI history", htf_trend)

        current_oi = oi_history[-1]["sum_open_interest_value"]
        earliest_oi = oi_history[0]["sum_open_interest_value"]  # Use full lookback
        oi_change_pct = (current_oi - earliest_oi) / earliest_oi * 100 if earliest_oi > 0 else 0

        if oi_change_pct <= 1.0:
            return hold_signal(f"OI not rising enough ({oi_change_pct:+.2f}%, need >1%)", htf_trend)

        # Price should be flat or falling while OI rises (divergence)
        price_start = float(prices.iloc[0]) if len(prices) > 0 else current_price
        price_change = (current_price - price_start) / price_start * 100
        if price_change > 0.5:
            return hold_signal(f"Price rising with OI ({price_change:+.2f}%), no divergence", htf_trend)
        reasons.append(f"OI +{oi_change_pct:.2f}% / price {price_change:+.2f}% (divergence)")

        # === CONDITION 3: Taker ratio < 0.93 (sustained selling) ===
        taker_data = derivatives.get("taker_ratio_15m", [])
        if not taker_data:
            return hold_signal("No taker ratio data", htf_trend)

        taker_entry = taker_data[0] if isinstance(taker_data, list) else taker_data
        taker_ratio = float(taker_entry.get("buy_sell_ratio", 1.0))

        if taker_ratio >= 0.93:
            return hold_signal(f"Taker not selling ({taker_ratio:.3f}, need <0.93)", htf_trend)
        reasons.append(f"Taker {taker_ratio:.3f} (sustained selling)")

        # === CONDITION 4: 15m RSI < 45 ===
        rsi_data = calculate_rsi(prices)
        rsi = rsi_data["value"]

        if rsi >= 45:
            return hold_signal(f"RSI too high ({rsi:.0f}, need <45)", htf_trend)
        reasons.append(f"RSI {rsi:.0f} (weak)")

        # All 4 conditions met — short signal
        is_strong = rate > 0.001 and oi_change_pct > 2.0 and taker_ratio < 0.88
        signal_type = "strong_sell" if is_strong else "sell"
        confidence = 0.75 if is_strong else 0.70

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "funding_rate": rate,
                "oi_change_pct": oi_change_pct,
                "price_change_pct": price_change,
                "taker_ratio": taker_ratio,
                "rsi": rsi,
            },
        }


# =============================================================================
# Strategy 8: Crash Momentum Short (SHORT-ONLY, 1-3 trades/day during crashes)
# =============================================================================

class CrashMomentumShortGenerator:
    """Trend continuation short during active selloffs — 1h price action.

    Designed for the scenario existing short strategies miss: when the crash is
    already underway, derivatives data flips (funding goes negative, OI drops,
    taker ratio rises as dip-buyers step in). This strategy uses ONLY 1h price
    action to catch continuation moves with less noise than 15m.

    Uses 1h candles instead of 15m because 15m is too noisy during crashes —
    small bounces trigger SL before the real move. 1h captures the macro trend.

    Research: momentum continuation has highest edge in first 24-48h of a
    selloff before mean reversion kicks in. Dead cat bounces fail ~70% of the
    time in the first 2 days of a trend break (Jegadeesh & Titman, 1993).

    REGIME GATE: 4h bearish regime required (checked externally).

    Entry requires ALL 6 conditions on 1h candles (boolean, no scoring):
    1. Price below 1h SMA20 by >0.3% (confirmed downtrend, not just touching)
    2. 1h SMA20 slope negative (crash still active, not consolidating)
    3. RSI between 25-45 (not at absolute bottom, but weak — room to fall)
    4. At least 2 of last 3 candles are red (selling momentum present)
    5. Price made a lower low vs 3 candles ago (trend continuing, not consolidating)
    """

    def generate_signal(self, df_1h: pd.DataFrame, htf_trend: dict,
                        derivatives: dict = None, regime: dict = None, **kwargs) -> dict:
        """Generate short signal from 1h crash momentum conditions."""
        if df_1h is None or len(df_1h) < 25:
            return hold_signal("Insufficient 1h data", htf_trend)

        # === REGIME GATE ===
        if not regime or not regime.get("is_bearish"):
            reason = regime.get("reason", "No regime data") if regime else "No regime data"
            return hold_signal(f"Short blocked: {reason}", htf_trend)

        prices = df_1h["close"]
        opens = df_1h["open"]
        lows = df_1h["low"]
        current_price = float(prices.iloc[-1])
        reasons = [f"Regime: {regime['reason'][:60]}"]

        # === CONDITION 1: Price below 1h SMA20 by >0.3% (not just barely touching) ===
        sma20_series = prices.rolling(20).mean()
        sma20 = float(sma20_series.iloc[-1])
        if current_price >= sma20:
            return hold_signal(f"Price above 1h SMA20 ({current_price:.2f} >= {sma20:.2f})", htf_trend)
        pct_below_sma = (sma20 - current_price) / sma20 * 100
        if pct_below_sma < 0.3:
            return hold_signal(f"Price only {pct_below_sma:.2f}% below SMA20 (need >0.3%)", htf_trend)
        reasons.append(f"Price {pct_below_sma:.1f}% below 1h SMA20")

        # === CONDITION 2: 1h SMA20 slope negative (crash still active) ===
        sma20_3ago = float(sma20_series.iloc[-3])
        sma20_slope = (sma20 - sma20_3ago) / sma20_3ago if sma20_3ago > 0 else 0
        if sma20_slope >= 0:
            return hold_signal(f"1h SMA20 not falling ({sma20_slope*100:+.3f}%), crash may be over", htf_trend)
        reasons.append(f"SMA20 falling ({sma20_slope*100:.3f}%)")

        # === CONDITION 2: RSI between 25-45 ===
        rsi_data = calculate_rsi(prices)
        rsi = rsi_data["value"]
        if rsi < 25:
            return hold_signal(f"RSI too low ({rsi:.0f}), oversold bounce risk", htf_trend)
        if rsi >= 45:
            return hold_signal(f"RSI too high ({rsi:.0f}, need <45)", htf_trend)
        reasons.append(f"RSI {rsi:.0f} (weak but not oversold)")

        # === CONDITION 3: At least 2 of last 3 1h candles are red ===
        red_count = 0
        for i in range(-3, 0):
            if float(prices.iloc[i]) < float(opens.iloc[i]):
                red_count += 1
        if red_count < 2:
            return hold_signal(f"Only {red_count}/3 red 1h candles (need 2+)", htf_trend)
        reasons.append(f"{red_count}/3 red 1h candles")

        # === CONDITION 4: Lower low vs 3 candles ago (3h lookback) ===
        low_3ago = float(lows.iloc[-3])
        current_low = float(lows.iloc[-1])
        if current_low >= low_3ago:
            return hold_signal(f"No lower low ({current_low:.2f} >= {low_3ago:.2f} from 3h ago)", htf_trend)
        reasons.append(f"Lower low ({current_low:.2f} < {low_3ago:.2f})")

        # All conditions met — short signal
        is_strong = pct_below_sma > 1.5 and rsi < 35 and red_count == 3
        signal_type = "strong_sell" if is_strong else "sell"
        confidence = 0.75 if is_strong else 0.70

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "sma20": sma20,
                "rsi": rsi,
                "red_count": red_count,
                "current_low": current_low,
                "low_3ago": low_3ago,
                "pct_below_sma": pct_below_sma,
                "sma20_slope": sma20_slope,
            },
        }


# =============================================================================
# Strategy Config + Position Sizing
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str                        # "funding_reversion", "trend_breakout", etc.
    strategy_type: str               # "funding", "breakout", "pullback", "flow", "regime_short", "failed_bkout_short", "refined_cascade", "crash_momentum"
    generator: object                # Signal generator instance
    sl_atr_mult: float               # SL as multiple of ATR
    tp_atr_mult: float               # TP as multiple of ATR
    trailing_atr_mult: float         # Trailing activation as multiple of ATR
    trailing_dist_atr_mult: float    # Trailing distance as multiple of ATR
    max_position_hours: float        # Max hold time
    risk_per_trade_pct: float        # Risk % of capital per trade (e.g. 0.02 = 2%)
    capital: float                   # Allocated capital
    atr_timeframe: str = "1h"        # Which timeframe to compute ATR on
    trailing_enabled: bool = False   # Whether trailing stops are active
    min_sl_pct: float = 0.0         # Floor for SL % (0 = no floor)

    @property
    def rr_ratio(self) -> float:
        return self.tp_atr_mult / self.sl_atr_mult if self.sl_atr_mult > 0 else 0

    @property
    def source_label(self) -> str:
        """Signal source label for DB."""
        return self.strategy_type


def calculate_position_size(capital: float, risk_pct: float, sl_distance_pct: float,
                            leverage: int, price: float, max_margin_pct: float = 0.30) -> tuple[float, float]:
    """Calculate position size based on risk per trade.

    Args:
        capital: Available capital
        risk_pct: Fraction of capital to risk (e.g. 0.02 = 2%)
        sl_distance_pct: Stop loss distance as fraction of price (e.g. 0.015 = 1.5%)
        leverage: Leverage multiplier
        price: Current price
        max_margin_pct: Maximum margin as fraction of capital (cap at 30%)

    Returns:
        (quantity, margin) tuple
    """
    risk_amount = capital * risk_pct  # e.g. $1000 * 0.02 = $20 risked
    # Position value needed so that SL distance = risk_amount
    # position_value * sl_distance_pct = risk_amount
    if sl_distance_pct <= 0:
        return 0.0, 0.0
    position_value = risk_amount / sl_distance_pct
    margin = position_value / leverage

    # Cap margin at max_margin_pct of capital
    max_margin = capital * max_margin_pct
    if margin > max_margin:
        margin = max_margin
        position_value = margin * leverage

    quantity = position_value / price
    return quantity, margin


# Strategy name mapping for backward compat with old trades
LEGACY_STRATEGY_MAP = {
    "paper_technical": "agreement_classic",
    "agreement_classic": "agreement_classic",
    "agreement_mtf": "agreement_mtf",
    "momentum": "momentum",
    "funding_sentiment": "funding_sentiment",
    "volatility_squeeze": "volatility_squeeze",
    "taker_flow": "taker_flow",
    "oi_momentum": "oi_momentum",
}


class SimplePaperTrader:
    """Multi-strategy paper trading engine."""

    def __init__(
        self,
        symbols: list[str] = ["BTCUSDT"],
        strategies: list[StrategyConfig] = None,
        initial_capital: float = 1000.0,
        leverage: int = 10,
        check_interval: int = 60,
        signal_exit_min_confidence: float = 0.70,
        signal_exit_min_profit_pct: float = 0.01,
        stale_position_min_profit_pct: float = 0.005,
    ):
        self.symbols = symbols
        self.leverage = leverage
        self.check_interval = check_interval
        self.signal_exit_min_confidence = signal_exit_min_confidence
        self.signal_exit_min_profit_pct = signal_exit_min_profit_pct
        self.stale_position_min_profit_pct = stale_position_min_profit_pct

        # Liquidation tracking
        self.liquidation_pct = 1.0 / leverage * 0.9  # ~9% for 10x (with 10% buffer)

        # Default strategy if none provided
        if strategies is None:
            strategies = [
                StrategyConfig(
                    name="funding_reversion",
                    strategy_type="funding",
                    generator=FundingMeanReversionGenerator(),
                    sl_atr_mult=2.0,
                    tp_atr_mult=4.0,
                    trailing_atr_mult=3.0,
                    trailing_dist_atr_mult=1.5,
                    max_position_hours=12.0,
                    risk_per_trade_pct=0.02,
                    capital=initial_capital,
                    atr_timeframe="1h",
                    trailing_enabled=False,
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
        # Cross-strategy symbol cooldown after any SL: {symbol: datetime}
        self.symbol_sl_cooldowns: dict[str, datetime] = {}
        self.symbol_sl_cooldown_minutes = 30
        # Hold signal logging: track last hold reason per strategy:symbol to avoid spam
        self.last_hold_reasons: dict[str, str] = {}  # {strategy:symbol: reasoning}
        self.last_hold_save_times: dict[str, datetime] = {}  # {strategy:symbol: last_save_time}
        self.hold_save_interval_minutes = 15  # Save hold to DB at most once per 15min per key
        # Pileup block: only block if existing position opened recently
        self.pileup_block_minutes = 30  # Positions older than this don't block new entries
        # Max consecutive stop losses per strategy:symbol per day
        self.daily_stop_losses: dict[str, int] = {}
        self.max_daily_stop_losses = 2
        self.last_reset_date: str = ""

        # Global circuit breaker: pause ALL strategies after N consecutive SLs
        self.global_sl_timestamps: list[datetime] = []  # timestamps of recent SLs
        self.circuit_breaker_window_hours = 2  # Look at SLs within this window
        self.circuit_breaker_threshold = 2     # N SLs to trigger (was 3, tightened: 13-loss streaks)
        self.circuit_breaker_pause_minutes = 60  # How long to pause
        self.circuit_breaker_until: datetime | None = None  # When the pause ends

        # Reversal close: close losing non-pullback positions when reversal persists 2+ cycles
        # Key: "symbol" → count of consecutive main-loop cycles with reversal override active
        self.reversal_override_counts: dict[str, int] = {}
        self.reversal_close_min_cycles = 2  # Must persist 2 consecutive cycles before closing
        self._reversal_counted_this_cycle: set[str] = set()  # Prevent multi-counting per cycle

    @property
    def total_capital(self):
        return sum(s["capital"] for s in self.strategy_stats.values())

    @property
    def total_pnl(self):
        return sum(s["total_pnl"] for s in self.strategy_stats.values())

    def _get_available_capital(self, strategy: 'StrategyConfig') -> float:
        """Get available capital for a strategy from the shared pool.

        Instead of fixed per-strategy buckets, all strategies share one pool.
        Each strategy is capped at its max allocation, but can use capital
        that idle strategies aren't using.

        Available = min(strategy_max_cap, total_pool - margin_in_use_by_others)
        """
        stats = self.strategy_stats[strategy.name]
        strategy_cap = stats["capital"]  # This strategy's max allocation (grows/shrinks with PnL)

        # Calculate total margin currently locked in open positions
        total_margin_in_use = 0.0
        my_margin_in_use = 0.0
        for pos_key, pos in self.positions.items():
            margin = pos.get("margin", 0)
            strat_name = pos_key.split(":")[0]
            total_margin_in_use += margin
            if strat_name == strategy.name:
                my_margin_in_use += margin

        # Pool available = total capital - margin used by ALL strategies
        pool_available = self.total_capital - total_margin_in_use

        # This strategy can use: min(its own cap, pool available)
        # But don't count margin it already has in use (it's already "spent")
        available = min(strategy_cap - my_margin_in_use, pool_available)
        return max(available, 0.0)

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

                    # Map strategy name (handle legacy names)
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

                    # Use stored SL/TP pct if available, otherwise use defaults
                    sl_pct = trade.get("indicators_at_entry", {}).get("sl_pct") if trade.get("indicators_at_entry") else None
                    tp_pct = trade.get("indicators_at_entry", {}).get("tp_pct") if trade.get("indicators_at_entry") else None
                    trailing_act_pct = trade.get("indicators_at_entry", {}).get("trailing_act_pct") if trade.get("indicators_at_entry") else None
                    trailing_dist_pct = trade.get("indicators_at_entry", {}).get("trailing_dist_pct") if trade.get("indicators_at_entry") else None

                    # Fallback to reasonable defaults if not stored
                    if sl_pct is None:
                        sl_pct = 0.02
                    if tp_pct is None:
                        tp_pct = 0.04
                    if trailing_act_pct is None:
                        trailing_act_pct = 0.03
                    if trailing_dist_pct is None:
                        trailing_dist_pct = 0.015

                    if side == "long":
                        stop_price = entry_price * (1 - sl_pct)
                        take_profit_price = entry_price * (1 + tp_pct)
                    else:
                        stop_price = entry_price * (1 + sl_pct)
                        take_profit_price = entry_price * (1 - tp_pct)

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
                        "sl_pct": sl_pct,
                        "tp_pct": tp_pct,
                        "trailing_act_pct": trailing_act_pct,
                        "trailing_dist_pct": trailing_dist_pct,
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
        logger.info("🚀 Starting v6.0 Evidence-Based Paper Trading")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Total Capital: ${self.initial_capital:,.2f} | Leverage: {self.leverage}x")
        logger.info("-" * 70)
        for s in self.strategies:
            logger.info(
                f"   📋 {s.name}: ${s.capital:,.2f} | "
                f"SL={s.sl_atr_mult}x ATR({s.atr_timeframe}) TP={s.tp_atr_mult}x ATR | "
                f"R:R={s.rr_ratio:.1f}:1 | Risk={s.risk_per_trade_pct:.1%}/trade | Max hold: {s.max_position_hours}h"
            )
        logger.info("-" * 70)
        logger.info(f"   SL cooldown: {self.stop_loss_cooldown_minutes}min | Max daily SL: {self.max_daily_stop_losses}")
        logger.info(f"   Circuit breaker: {self.circuit_breaker_threshold} SLs in {self.circuit_breaker_window_hours}h → pause {self.circuit_breaker_pause_minutes}min")
        logger.info("=" * 70)

        collector = MarketDataCollector()

        try:
            while self.running:
                self._reversal_counted_this_cycle.clear()
                for symbol in self.symbols:
                    # Fetch all market data once per symbol
                    market_data = await self._fetch_market_data(symbol, collector)
                    if market_data is None:
                        continue

                    # Run each strategy
                    for strategy in self.strategies:
                        await self._process_strategy_symbol(strategy, symbol, market_data)

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

    async def _fetch_market_data(self, symbol: str, collector: MarketDataCollector) -> dict | None:
        """Fetch all needed data for a symbol (candles + derivatives)."""
        try:
            # Parallel fetch: 1m, 5m, 15m, 1h candles + ticker + derivatives
            needs_derivatives = any(s.strategy_type in ("funding", "oi", "flow", "regime_short", "failed_bkout_short", "refined_cascade") for s in self.strategies)
            needs_4h = any(s.strategy_type in ("regime_short", "failed_bkout_short", "refined_cascade", "crash_momentum") for s in self.strategies)

            tasks = [
                collector.get_binance_klines(symbol, "1m", 100),
                collector.get_binance_klines(symbol, "5m", 100),
                collector.get_binance_klines(symbol, "15m", 100),
                collector.get_binance_klines(symbol, "1h", 100),
                collector.get_binance_ticker(symbol),
            ]
            if needs_derivatives:
                tasks.append(collector.get_derivatives_data(symbol))
            if needs_4h:
                tasks.append(collector.get_binance_klines(symbol, "4h", 210))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            df_1m = results[0] if not isinstance(results[0], Exception) else pd.DataFrame()
            df_5m = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()
            df_15m = results[2] if not isinstance(results[2], Exception) else pd.DataFrame()
            df_1h = results[3] if not isinstance(results[3], Exception) else pd.DataFrame()
            ticker = results[4] if not isinstance(results[4], Exception) else {}

            if df_1m.empty:
                return None

            current_price = ticker.get("price", 0) if ticker else 0
            if current_price <= 0:
                return None

            market_data = {
                "1m": df_1m,
                "5m": df_5m,
                "15m": df_15m,
                "1h": df_1h,
                "current_price": current_price,
            }

            idx = 5
            if needs_derivatives:
                market_data["derivatives"] = results[idx] if not isinstance(results[idx], Exception) else {}
                idx += 1
            if needs_4h:
                market_data["4h"] = results[idx] if not isinstance(results[idx], Exception) else pd.DataFrame()

            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    async def _process_strategy_symbol(self, strategy: StrategyConfig, symbol: str, market_data: dict):
        """Process a single strategy+symbol combination."""
        try:
            df_1m = market_data["1m"]
            df_1h = market_data["1h"]
            current_price = market_data["current_price"]

            # Determine HTF trend (shared 1h analysis) — now returns dict
            htf_trend = {"direction": "neutral", "strength": 0.0, "slope": 0.0}
            if not df_1h.empty:
                htf_trend = determine_htf_trend(df_1h)

            # Fast reversal override: use 15m data to catch reversals early
            df_15m = market_data.get("15m", pd.DataFrame())
            if not df_15m.empty:
                htf_trend = apply_fast_reversal_override(htf_trend, df_15m)

            # Reversal close: count once per symbol per main-loop cycle
            if symbol not in self._reversal_counted_this_cycle:
                self._reversal_counted_this_cycle.add(symbol)
                if htf_trend.get("suppressed"):
                    self.reversal_override_counts[symbol] = self.reversal_override_counts.get(symbol, 0) + 1
                else:
                    self.reversal_override_counts[symbol] = 0

            # Close losing non-pullback positions when reversal persists 2+ cycles
            if self.reversal_override_counts.get(symbol, 0) >= self.reversal_close_min_cycles:
                await self._close_on_reversal(strategy, symbol, current_price, htf_trend)

            # Compute ATR for this strategy's timeframe
            atr_df_map = {
                "1h": df_1h,
                "15m": market_data.get("15m", pd.DataFrame()),
                "5m": market_data.get("5m", pd.DataFrame()),
            }
            atr_df = atr_df_map.get(strategy.atr_timeframe, df_1h)
            atr_value = calculate_atr(atr_df) if not atr_df.empty else 0

            # Generate signal based on strategy type
            if strategy.strategy_type == "funding":
                signal_result = strategy.generator.generate_signal(
                    df_1m, htf_trend,
                    derivatives=market_data.get("derivatives"),
                )
            elif strategy.strategy_type == "breakout":
                signal_result = strategy.generator.generate_signal(
                    market_data.get("15m", pd.DataFrame()), htf_trend,
                )
            elif strategy.strategy_type == "pullback":
                signal_result = strategy.generator.generate_signal(
                    market_data.get("15m", pd.DataFrame()), htf_trend,
                )
            elif strategy.strategy_type == "flow":
                signal_result = strategy.generator.generate_signal(
                    market_data.get("15m", pd.DataFrame()), htf_trend,
                    derivatives=market_data.get("derivatives"),
                )
            elif strategy.strategy_type in ("regime_short", "failed_bkout_short", "refined_cascade"):
                # Compute regime gate from 4h data (shared across short strategies)
                df_4h = market_data.get("4h", pd.DataFrame())
                regime = check_bearish_regime(df_4h) if not df_4h.empty else {"is_bearish": False, "reason": "No 4h data"}
                signal_result = strategy.generator.generate_signal(
                    market_data.get("15m", pd.DataFrame()), htf_trend,
                    derivatives=market_data.get("derivatives"),
                    regime=regime,
                )
            elif strategy.strategy_type == "crash_momentum":
                # Crash momentum uses 1h candles (less noise than 15m during crashes)
                df_4h = market_data.get("4h", pd.DataFrame())
                regime = check_bearish_regime(df_4h) if not df_4h.empty else {"is_bearish": False, "reason": "No 4h data"}
                signal_result = strategy.generator.generate_signal(
                    market_data.get("1h", pd.DataFrame()), htf_trend,
                    regime=regime,
                )
            else:
                signal_result = {"signal": "hold", "confidence": 0.5,
                                 "reasoning": f"Unknown strategy type: {strategy.strategy_type}"}

            # Determine timeframe label for DB
            tf_map = {"funding": "1h", "breakout": "15m", "pullback": "15m", "flow": "15m",
                      "regime_short": "15m", "failed_bkout_short": "15m", "refined_cascade": "15m",
                      "crash_momentum": "1h"}
            timeframe = tf_map.get(strategy.strategy_type, "1m")

            # Save signals to DB (holds are throttled: once per 15min or on reason change)
            if signal_result.get("signal", "hold") != "hold":
                await self._save_signal(symbol, signal_result, current_price, strategy, timeframe)
            else:
                await self._maybe_save_hold(symbol, signal_result, current_price, strategy, timeframe)

            # Trading logic — pass ATR for position sizing
            pos_key = self._pos_key(strategy.name, symbol)
            if pos_key in self.positions:
                await self._check_exit(pos_key, signal_result, current_price, strategy)
            else:
                await self._check_entry(pos_key, symbol, signal_result, current_price, strategy, atr_value)

        except Exception as e:
            logger.error(f"Error processing [{strategy.name}] {symbol}: {e}")

    async def _close_on_reversal(self, strategy: StrategyConfig, symbol: str,
                                  current_price: float, htf_trend: dict):
        """Close losing non-pullback positions when reversal override persists.

        Conditions (all must be true):
        1. Reversal override active for 2+ consecutive cycles
        2. Position is in loss (don't cut winners)
        3. Strategy is NOT pullback (pullback expects counter-trend dips)
        4. Position direction conflicts with the reversal
        """
        if strategy.strategy_type in ("pullback", "crash_momentum"):
            return

        pos_key = self._pos_key(strategy.name, symbol)
        position = self.positions.get(pos_key)
        if not position:
            return

        # Check if position direction conflicts with the reversal
        # Suppressed from bullish → market turning bearish → longs are at risk
        # Suppressed from bearish → market turning bullish → shorts are at risk
        side = position.get("side")
        original_direction = "bullish" if side == "long" else "bearish"

        # The reversal override only fires when HTF was in the SAME direction as our position
        # and 15m shows the opposite. So if suppressed=True, it means our position's direction
        # is being challenged. We only need to check we're in loss.
        pnl_pct = self._calculate_pnl_pct(position, current_price)
        if pnl_pct >= 0:
            return  # Position is profitable — let it run

        cycles = self.reversal_override_counts.get(symbol, 0)
        logger.info(
            f"⚡ REVERSAL CLOSE [{strategy.name}] {symbol}: {side} in loss "
            f"({pnl_pct:.2%}), reversal persisted {cycles} cycles"
        )
        await self._close_position(pos_key, current_price, f"Reversal override ({cycles} cycles, {pnl_pct:.2%})", strategy)

    async def _check_entry(self, pos_key: str, symbol: str, signal: dict, price: float,
                           strategy: StrategyConfig, atr_value: float):
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

        # Check global circuit breaker
        if self.circuit_breaker_until:
            remaining = (self.circuit_breaker_until - datetime.now(timezone.utc)).total_seconds() / 60
            if remaining > 0:
                logger.info(f"   🔌 Circuit breaker active ({remaining:.0f}min left) — all entries paused")
                return
            else:
                logger.info("   🔌 Circuit breaker lifted — resuming trading")
                self.circuit_breaker_until = None

        # Reset daily stop loss counter at midnight UTC
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.daily_stop_losses = {}
            self.last_reset_date = today

        # Check cross-strategy symbol cooldown (any SL on this symbol blocks all strategies)
        if symbol in self.symbol_sl_cooldowns:
            cooldown_until = self.symbol_sl_cooldowns[symbol]
            remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
            if remaining > 0:
                logger.info(f"   [{strategy.name}] {symbol}: Symbol SL cooldown ({remaining:.0f}min left)")
                return
            else:
                del self.symbol_sl_cooldowns[symbol]

        # Check per-strategy stop loss cooldown
        cooldown_key = pos_key
        if cooldown_key in self.stop_loss_cooldowns:
            cooldown_until = self.stop_loss_cooldowns[cooldown_key]
            remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
            if remaining > 0:
                logger.info(f"   [{strategy.name}] {symbol}: SL cooldown ({remaining:.0f}min left)")
                return
            else:
                del self.stop_loss_cooldowns[cooldown_key]

        # Check max daily stop losses (crash_momentum gets 3, others get 2)
        max_sl = 3 if strategy.strategy_type == "crash_momentum" else self.max_daily_stop_losses
        if self.daily_stop_losses.get(cooldown_key, 0) >= max_sl:
            logger.info(f"   [{strategy.name}] {symbol}: Max daily SL reached ({max_sl})")
            return

        # Check if another strategy recently entered same direction on this symbol
        # Only block if the existing position was opened within the last 30 minutes
        # (prevents simultaneous pileup but allows independent sequential entries)
        proposed_side = "long" if signal["signal"] in ["buy", "strong_buy"] else "short"
        now = datetime.now(timezone.utc)
        for existing_key, existing_pos in self.positions.items():
            if existing_key == pos_key:
                continue
            _, existing_symbol = self._parse_pos_key(existing_key)
            if existing_symbol == symbol and existing_pos.get("side") == proposed_side:
                entry_time = existing_pos.get("entry_time")
                if entry_time and (now - entry_time).total_seconds() < self.pileup_block_minutes * 60:
                    age_min = (now - entry_time).total_seconds() / 60
                    logger.info(f"   [{strategy.name}] {symbol}: Blocked — {existing_key} already {proposed_side} ({age_min:.0f}min ago)")
                    return

        if signal["signal"] in ["buy", "strong_buy"]:
            await self._open_position(pos_key, symbol, "long", price, signal, strategy, atr_value)
        elif signal["signal"] in ["sell", "strong_sell"]:
            await self._open_position(pos_key, symbol, "short", price, signal, strategy, atr_value)

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
        _, symbol = self._parse_pos_key(pos_key)
        position = self.positions.get(pos_key)
        if not position:
            return

        pnl_pct = self._calculate_pnl_pct(position, price)

        # Use position-stored SL/TP pct (computed at entry from ATR)
        sl_pct = position.get("sl_pct", 0.02)
        tp_pct = position.get("tp_pct", 0.04)
        trailing_act_pct = position.get("trailing_act_pct", 0.03)
        trailing_dist_pct = position.get("trailing_dist_pct", 0.015)

        # Update peak profit and trailing stop (only if trailing enabled)
        if position.get("trailing_enabled", True):
            self._update_trailing_stop(pos_key, pnl_pct, trailing_act_pct)

        should_exit = False
        exit_reason = ""

        # 0. LIQUIDATION CHECK
        if pnl_pct <= -self.liquidation_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"💀 LIQUIDATED ({pnl_pct:.2%} = {roe:.0f}% ROE)"
            logger.warning(f"⚠️ Position liquidated! [{strategy.name}] {pos_key}")

        # 1. TAKE PROFIT
        elif pnl_pct >= tp_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"Take profit ({pnl_pct:.2%} = +{roe:.0f}% ROE)"

        # 2. TRAILING STOP (only if trailing enabled for this strategy)
        elif position.get("trailing_enabled", True) and position.get("trailing_stop_active", False):
            peak_pnl = position.get("peak_pnl_pct", 0)
            trailing_stop_level = peak_pnl - trailing_dist_pct
            if pnl_pct <= trailing_stop_level:
                should_exit = True
                exit_reason = f"Trailing stop (peak: {peak_pnl:.2%}, current: {pnl_pct:.2%})"

        # 3. STOP LOSS
        elif pnl_pct <= -sl_pct:
            should_exit = True
            roe = pnl_pct * self.leverage * 100
            exit_reason = f"Stop loss ({pnl_pct:.2%} = {roe:.0f}% ROE)"
            # Activate per-strategy cooldown
            self.stop_loss_cooldowns[pos_key] = datetime.now(timezone.utc) + timedelta(minutes=self.stop_loss_cooldown_minutes)
            # Activate cross-strategy symbol cooldown
            self.symbol_sl_cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(minutes=self.symbol_sl_cooldown_minutes)
            self.daily_stop_losses[pos_key] = self.daily_stop_losses.get(pos_key, 0) + 1
            logger.info(f"   [{strategy.name}] SL cooldown activated. Daily: {self.daily_stop_losses[pos_key]}/{self.max_daily_stop_losses}")

            # Global circuit breaker: track SL timestamps across all strategies
            now = datetime.now(timezone.utc)
            self.global_sl_timestamps.append(now)
            # Prune old timestamps outside the window
            cutoff = now - timedelta(hours=self.circuit_breaker_window_hours)
            self.global_sl_timestamps = [t for t in self.global_sl_timestamps if t > cutoff]
            # Check if threshold reached
            if len(self.global_sl_timestamps) >= self.circuit_breaker_threshold:
                self.circuit_breaker_until = now + timedelta(minutes=self.circuit_breaker_pause_minutes)
                logger.warning(
                    f"🔌 CIRCUIT BREAKER: {len(self.global_sl_timestamps)} SLs in {self.circuit_breaker_window_hours}h "
                    f"— ALL trading paused for {self.circuit_breaker_pause_minutes}min"
                )

        # 4. TIME-BASED EXIT
        elif self._is_stale_position(position, pnl_pct, strategy):
            should_exit = True
            hours_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 3600
            exit_reason = f"Stale position ({hours_open:.1f}h, only {pnl_pct:.2%} profit)"

        # 5. RSI-BASED PROFIT TAKING (using signal indicators)
        elif self._should_exit_on_rsi(position, signal, pnl_pct, trailing_act_pct):
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
        """Check if position is stale (old with minimal profit).
        Uses 25% of designed TP as minimum profit threshold (strategy-aware)."""
        hours_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 3600
        if hours_open >= strategy.max_position_hours:
            min_profit = position.get("tp_pct", 0.04) * 0.25
            if pnl_pct < min_profit:
                return True
        return False

    def _should_exit_on_rsi(self, position: dict, signal: dict, pnl_pct: float,
                            trailing_act_pct: float) -> bool:
        """Exit profitable positions when RSI indicates overbought/oversold reversal."""
        if pnl_pct < trailing_act_pct:
            return False

        indicators = signal.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        side = position.get("side")

        if side == "long" and rsi > 80:
            return True
        if side == "short" and rsi < 20:
            return True
        return False

    def _update_trailing_stop(self, pos_key: str, current_pnl_pct: float, trailing_act_pct: float):
        """Update trailing stop based on current P&L."""
        position = self.positions.get(pos_key)
        if not position:
            return

        if current_pnl_pct >= trailing_act_pct:
            if not position.get("trailing_stop_active", False):
                position["trailing_stop_active"] = True
                position["peak_pnl_pct"] = current_pnl_pct
                _, symbol = self._parse_pos_key(pos_key)
                strategy_name = position.get("strategy_name", "")
                logger.info(
                    f"🔒 Trailing stop activated [{strategy_name}] {symbol} at {current_pnl_pct:.2%}"
                )
            elif current_pnl_pct > position.get("peak_pnl_pct", 0):
                position["peak_pnl_pct"] = current_pnl_pct
                _, symbol = self._parse_pos_key(pos_key)
                strategy_name = position.get("strategy_name", "")
                logger.debug(f"📈 [{strategy_name}] {symbol} new peak: {current_pnl_pct:.2%}")

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
                              signal: dict, strategy: StrategyConfig, atr_value: float):
        """Open a leveraged position with ATR-based stops and risk-based sizing."""
        stats = self.strategy_stats[strategy.name]

        # Compute ATR-based SL/TP as percentages of price
        if atr_value <= 0:
            # Fallback: use 1.5% SL if ATR unavailable
            atr_value = price * 0.015

        sl_distance = atr_value * strategy.sl_atr_mult
        tp_distance = atr_value * strategy.tp_atr_mult
        trailing_act_distance = atr_value * strategy.trailing_atr_mult
        trailing_dist_distance = atr_value * strategy.trailing_dist_atr_mult

        sl_pct = sl_distance / price
        tp_pct = tp_distance / price
        trailing_act_pct = trailing_act_distance / price
        trailing_dist_pct = trailing_dist_distance / price

        # Apply SL floor: if ATR-based SL is below minimum, scale all stops proportionally
        if strategy.min_sl_pct > 0 and sl_pct < strategy.min_sl_pct:
            scale = strategy.min_sl_pct / sl_pct
            logger.info(f"   [{strategy.name}] Applied SL floor: {sl_pct:.3%} → {strategy.min_sl_pct:.3%} (scale {scale:.2f}x)")
            sl_pct = strategy.min_sl_pct
            tp_pct *= scale
            trailing_act_pct *= scale
            trailing_dist_pct *= scale

        # Risk-based position sizing (shared capital pool)
        available_capital = self._get_available_capital(strategy)
        if available_capital <= 0:
            logger.info(f"   [{strategy.name}] {symbol}: No capital available (pool fully allocated)")
            return
        quantity, margin = calculate_position_size(
            capital=available_capital,
            risk_pct=strategy.risk_per_trade_pct,
            sl_distance_pct=sl_pct,
            leverage=self.leverage,
            price=price,
        )

        if quantity <= 0:
            logger.warning(f"   [{strategy.name}] {symbol}: Position size too small, skipping")
            return

        position_value = quantity * price

        # Calculate stop loss, take profit, and liquidation prices
        if side == "long":
            stop_price = price * (1 - sl_pct)
            take_profit_price = price * (1 + tp_pct)
            liquidation_price = price * (1 - self.liquidation_pct)
        else:
            stop_price = price * (1 + sl_pct)
            take_profit_price = price * (1 - tp_pct)
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
            # Store computed pct on position for exit checks
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "trailing_act_pct": trailing_act_pct,
            "trailing_dist_pct": trailing_dist_pct,
            "trailing_enabled": strategy.trailing_enabled,
        }

        logger.info(
            f"📈 OPEN [{strategy.name}] {side.upper()} {symbol} @ ${price:,.2f} | "
            f"Margin: ${margin:,.2f} | Size: ${position_value:,.2f} ({self.leverage}x) | "
            f"Conf: {signal['confidence']:.0%}"
        )
        logger.info(
            f"   └─ SL: ${stop_price:,.2f} ({sl_pct:.2%}) | "
            f"TP: ${take_profit_price:,.2f} ({tp_pct:.2%}) | "
            f"ATR: ${atr_value:,.2f} | Risk: ${available_capital * strategy.risk_per_trade_pct:,.2f} | Pool: ${available_capital:,.0f}"
        )

        # Log to Supabase
        try:
            indicators = signal.get("indicators", {})
            # Store ATR-computed params for position reload
            indicators["sl_pct"] = sl_pct
            indicators["tp_pct"] = tp_pct
            indicators["trailing_act_pct"] = trailing_act_pct
            indicators["trailing_dist_pct"] = trailing_dist_pct
            indicators["atr_value"] = atr_value

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
                "indicators_at_entry": indicators,
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
            position_id = position["position_id"]
            exit_data = {
                "exit_price": price,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "gross_pnl": pnl,
                "net_pnl": pnl * 0.999,  # 0.1% fee estimate
                "return_pct": pnl_pct * 100,
                "duration_seconds": int(duration),
                "exit_reasoning": reason,
            }
            await asyncio.to_thread(
                lambda: self.trade_repo.table.update(exit_data)
                .eq("position_id", position_id)
                .execute()
            )
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

    async def _save_signal(self, symbol: str, signal: dict, price: float,
                           strategy: StrategyConfig, timeframe: str = "1m"):
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
                "timeframe": timeframe,
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

    async def _maybe_save_hold(self, symbol: str, signal: dict, price: float,
                               strategy: StrategyConfig, timeframe: str):
        """Save hold signal to DB, throttled: only on reason change or every 15 min."""
        hold_key = f"{strategy.name}:{symbol}"
        reasoning = signal.get("reasoning", "")
        now = datetime.now(timezone.utc)

        last_reason = self.last_hold_reasons.get(hold_key)
        last_save = self.last_hold_save_times.get(hold_key)

        reason_changed = (last_reason != reasoning)
        interval_elapsed = (
            last_save is None or
            (now - last_save).total_seconds() >= self.hold_save_interval_minutes * 60
        )

        if reason_changed or interval_elapsed:
            self.last_hold_reasons[hold_key] = reasoning
            self.last_hold_save_times[hold_key] = now
            await self._save_signal(symbol, signal, price, strategy, timeframe)

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

    parser = argparse.ArgumentParser(description="v6.0 Evidence-Based Paper Trading")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "XRPUSDT"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=1000.0,
        help="Base capital unit (long: funding 0.5x, breakout 1.0x, pullback 0.75x, flow 0.75x; short: regime 0.5x, failed_bkout 0.4x, cascade 0.45x)"
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
        default=["funding_reversion", "trend_breakout", "trend_pullback", "order_flow",
                 "regime_short", "failed_breakout_short", "refined_liq_cascade", "crash_momentum"],
        choices=["funding_reversion", "trend_breakout", "trend_pullback", "order_flow",
                 "regime_short", "failed_breakout_short", "refined_liq_cascade", "crash_momentum"],
        help="Strategies to run (default: all 8)"
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

    # Build strategy configs
    selected = args.strategies
    base_capital = args.capital

    # Capital allocation: more to proven strategies, less to new/experimental
    # Short strategies sized at 50-70% of long equivalents (research: asymmetric risk)
    capital_allocation = {
        "funding_reversion": base_capital * 0.5,       # $500 — rare-event, mostly idle
        "trend_breakout": base_capital * 1.0,          # $1000 — proven profitable strategy
        "trend_pullback": base_capital * 0.75,         # $750 — proven, complementary to breakout
        "order_flow": base_capital * 0.75,             # $750 — proven, uses taker ratio + L/S data
        "regime_short": base_capital * 0.4,            # $400 — multi-condition confluence short (pre-crash)
        "failed_breakout_short": base_capital * 0.35,  # $350 — price action exhaustion short (pre-crash)
        "refined_liq_cascade": base_capital * 0.4,     # $400 — derivatives-based, rare events (pre-crash)
        "crash_momentum": base_capital * 0.5,          # $500 — price action continuation during crashes
    }

    strategy_configs = []
    for name in selected:
        if name == "funding_reversion":
            strategy_configs.append(StrategyConfig(
                name="funding_reversion",
                strategy_type="funding",
                generator=FundingMeanReversionGenerator(),
                sl_atr_mult=2.0,
                tp_atr_mult=4.0,
                trailing_atr_mult=3.0,
                trailing_dist_atr_mult=1.5,
                max_position_hours=12.0,
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="1h",
                trailing_enabled=False,
            ))
        elif name == "trend_breakout":
            strategy_configs.append(StrategyConfig(
                name="trend_breakout",
                strategy_type="breakout",
                generator=TrendBreakoutGenerator(),
                sl_atr_mult=1.5,
                tp_atr_mult=3.0,
                trailing_atr_mult=2.5,
                trailing_dist_atr_mult=1.0,
                max_position_hours=6.0,
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m",
                trailing_enabled=False,
            ))
        elif name == "trend_pullback":
            strategy_configs.append(StrategyConfig(
                name="trend_pullback",
                strategy_type="pullback",
                generator=TrendPullbackGenerator(),
                sl_atr_mult=1.5,
                tp_atr_mult=3.0,
                trailing_atr_mult=2.5,
                trailing_dist_atr_mult=1.0,
                max_position_hours=8.0,
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m",
                trailing_enabled=False,
            ))
        elif name == "order_flow":
            strategy_configs.append(StrategyConfig(
                name="order_flow",
                strategy_type="flow",
                generator=OrderFlowGenerator(),
                sl_atr_mult=1.5,
                tp_atr_mult=3.0,
                trailing_atr_mult=2.5,
                trailing_dist_atr_mult=1.0,
                max_position_hours=6.0,
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m",
                trailing_enabled=False,
            ))
        elif name == "regime_short":
            strategy_configs.append(StrategyConfig(
                name="regime_short",
                strategy_type="regime_short",
                generator=RegimeShortConfluenceGenerator(),
                sl_atr_mult=2.0,       # Wider SL for shorts (research: violent bounces)
                tp_atr_mult=3.0,       # 1.5:1 R:R
                trailing_atr_mult=2.5, # Start trailing at 2.5x ATR (was 1.5 — let winners run)
                trailing_dist_atr_mult=1.5,  # Trail at 1.5x ATR (was 1.0 — wider to avoid noise clips)
                max_position_hours=12.0,  # Time stop: 12h max
                risk_per_trade_pct=0.015,  # 1.5% risk (smaller than longs)
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m",
                trailing_enabled=True,
            ))
        elif name == "failed_breakout_short":
            strategy_configs.append(StrategyConfig(
                name="failed_breakout_short",
                strategy_type="failed_bkout_short",
                generator=FailedBreakoutShortGenerator(),
                sl_atr_mult=2.0,       # SL above the failed breakout high
                tp_atr_mult=2.5,       # 1.25:1 R:R (tighter since pattern is quick)
                trailing_atr_mult=2.0, # Was 1.5 — let winners run
                trailing_dist_atr_mult=1.0, # Was 0.8 — wider to avoid noise clips
                max_position_hours=6.0,  # Time stop: 6h (failed breakouts resolve fast)
                risk_per_trade_pct=0.015,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m",
                trailing_enabled=True,
            ))
        elif name == "refined_liq_cascade":
            strategy_configs.append(StrategyConfig(
                name="refined_liq_cascade",
                strategy_type="refined_cascade",
                generator=RefinedLiqCascadeGenerator(),
                sl_atr_mult=2.0,       # Wide SL
                tp_atr_mult=4.0,       # 2:1 R:R (rare but high conviction)
                trailing_atr_mult=2.5, # Was 2.0 — let winners run
                trailing_dist_atr_mult=1.5, # Was 1.0 — wider to avoid noise clips
                max_position_hours=8.0,  # Time stop: 8h
                risk_per_trade_pct=0.015,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m",
                trailing_enabled=True,
            ))
        elif name == "crash_momentum":
            strategy_configs.append(StrategyConfig(
                name="crash_momentum",
                strategy_type="crash_momentum",
                generator=CrashMomentumShortGenerator(),
                sl_atr_mult=2.0,       # Wide SL — crash bounces are violent on 1h
                tp_atr_mult=4.0,       # 2:1 R:R (bigger moves on 1h)
                trailing_atr_mult=2.5, # Start trailing at 2.5x ATR (was 2.0 — let crash profits run)
                trailing_dist_atr_mult=1.5,  # Trail at 1.5x ATR (was 1.0 — wider for crash volatility)
                max_position_hours=12.0,  # Time stop: 12h (1h candles need more time)
                risk_per_trade_pct=0.015,  # 1.5% risk
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="1h",
                trailing_enabled=True,
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
