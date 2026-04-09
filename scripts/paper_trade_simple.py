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
import os
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
from data.collectors.hyperliquid_whales import HyperliquidWhaleTracker
from data.storage.supabase_client import (
    TradeLogRepository,
    SignalRepository,
    PerformanceRepository,
    TradeAnalysisRepository,
    AIReviewRepository,
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
        return {
            "direction": "neutral",
            "strength": htf_trend.get("strength", 0.0) * 0.3,
            "slope": htf_trend.get("slope", 0.0),
            "suppressed": True,
            "conflict_detail": f"bullish→NEUTRAL (15m ${current_price:,.2f} < SMA20 ${sma20:,.2f}, RSI {rsi:.0f})",
        }

    # Bullish conflict: HTF bearish but 15m bullish → suppress to neutral
    if htf_direction == "bearish" and current_price > sma20 and rsi > 60:
        return {
            "direction": "neutral",
            "strength": htf_trend.get("strength", 0.0) * 0.3,
            "slope": htf_trend.get("slope", 0.0),
            "suppressed": True,
            "conflict_detail": f"bearish→NEUTRAL (15m ${current_price:,.2f} > SMA20 ${sma20:,.2f}, RSI {rsi:.0f})",
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


def check_4h_uptrend(df_4h: pd.DataFrame) -> dict:
    """Check if 4h structure confirms a genuine uptrend for long entries.

    The 1h HTF trend flips 'bullish' on 2-3 day bear market rallies.
    This 4h check filters those out by requiring structural uptrend evidence.

    Requires at least 2 of 3 confirmations:
    1. Price above 4h SMA50 — sustained uptrend (~8 days)
    2. 4h SMA20 slope positive (last 3 candles) — trend direction
    3. Higher lows: last 10-candle low > previous 10-candle low — trend structure
    """
    if df_4h is None or len(df_4h) < 50:
        return {"confirmed": False, "reason": "Insufficient 4h data for uptrend check",
                "confirmations": 0}

    prices = df_4h["close"]
    lows = df_4h["low"]
    current_price = float(prices.iloc[-1])
    confirmations = []

    # === Confirmation 1: Price above 4h SMA50 ===
    sma50 = prices.rolling(50).mean()
    sma50_val = float(sma50.iloc[-1])
    if current_price > sma50_val:
        confirmations.append(f"price above 4h SMA50 ({current_price:.2f} > {sma50_val:.2f})")

    # === Confirmation 2: 4h SMA20 slope positive (last 3 candles) ===
    if len(df_4h) >= 23:
        sma20 = prices.rolling(20).mean()
        sma20_now = float(sma20.iloc[-1])
        sma20_3ago = float(sma20.iloc[-3])
        sma20_slope = (sma20_now - sma20_3ago) / sma20_3ago if sma20_3ago > 0 else 0
        if sma20_slope > 0:
            confirmations.append(f"4h SMA20 slope positive ({sma20_slope*100:+.3f}%)")

    # === Confirmation 3: Higher lows (last 10 candles vs previous 10) ===
    if len(lows) >= 20:
        recent_low = float(lows.iloc[-10:].min())
        prev_low = float(lows.iloc[-20:-10].min())
        if recent_low > prev_low:
            confirmations.append(f"higher lows ({recent_low:.2f} > {prev_low:.2f})")

    confirmed = len(confirmations) >= 2
    if confirmed:
        reason = "4h uptrend confirmed: " + ", ".join(confirmations)
    else:
        reason = f"4h uptrend NOT confirmed ({len(confirmations)}/2 needed): " + (", ".join(confirmations) if confirmations else "no bullish signals")

    return {"confirmed": confirmed, "reason": reason, "confirmations": len(confirmations)}


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

        if oi_change_pct < 0.2:
            return hold_signal(f"OI not rising enough ({oi_change_pct:.2f}%, need >0.2%)", htf_trend)
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
# Strategy 2a: Uptrend Pullback (PROVEN — backtested +$439/6mo, 44.7% WR)
# =============================================================================

class UptrendPullbackGenerator:
    """Buy pullbacks to 1h SMA20 during confirmed uptrends with volume confirmation.

    Backtested over 6 months (Oct 2025 - Apr 2026), 6 coins, 215 trades:
    - PnL: +$439 (+$73/month)
    - Win rate: 44.7%
    - Max drawdown: $243

    Entry requires ALL 4 conditions:
    1. Uptrend: 1h SMA20 > SMA50 and price > SMA50
    2. Pullback: Price within 0.5% of SMA20 (touching support)
    3. Volume: Current volume > 1.5x 20-bar average (smart money buying the dip)
    4. Not oversold crash: Price must be above SMA50 (still in trend structure)

    Why this works:
    - SMA20 acts as dynamic support in uptrends
    - Volume spike at support = institutional buying
    - Tight SL below SMA20 = small risk, defined invalidation
    - Catches trend continuation, not breakout (lower false positive rate)
    """

    def generate_signal(self, df_1h: pd.DataFrame, htf_trend: dict,
                        **kwargs) -> dict:
        """Generate signal from pullback to SMA20 in uptrend."""
        if df_1h is None or len(df_1h) < 50:
            return hold_signal("Insufficient 1h data for pullback", htf_trend)

        prices = df_1h["close"]
        volumes = df_1h["volume"]
        current_price = float(prices.iloc[-1])
        reasons = []

        # === CONDITION 1: Uptrend — SMA20 > SMA50, price > SMA50 ===
        sma20 = float(prices.rolling(20).mean().iloc[-1])
        sma50 = float(prices.rolling(50).mean().iloc[-1])

        if sma20 <= sma50:
            return hold_signal(f"Not in uptrend (SMA20 {sma20:.2f} <= SMA50 {sma50:.2f})", htf_trend)
        if current_price <= sma50:
            return hold_signal(f"Price below SMA50 ({current_price:.2f} <= {sma50:.2f})", htf_trend)
        reasons.append(f"Uptrend (SMA20 {sma20:.2f} > SMA50 {sma50:.2f})")

        # === CONDITION 2: Pullback — price within 0.5% of SMA20 ===
        dist_pct = abs(current_price - sma20) / sma20
        # Price should be at or slightly below SMA20 (touching support)
        if dist_pct > 0.005:
            return hold_signal(f"Price too far from SMA20 ({dist_pct:.2%}, need <0.5%)", htf_trend)
        if current_price < sma20 * 0.995:
            return hold_signal(f"Price too far below SMA20 ({current_price:.2f} < {sma20*0.995:.2f})", htf_trend)
        reasons.append(f"Pullback to SMA20 (dist={dist_pct:.2%})")

        # === CONDITION 3: Volume confirmation — > 1.5x average ===
        avg_vol = float(volumes.iloc[-21:-1].mean())
        current_vol = float(volumes.iloc[-1])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        if vol_ratio < 1.5:
            return hold_signal(f"Volume too low ({vol_ratio:.2f}x, need 1.5x)", htf_trend)
        reasons.append(f"Volume {vol_ratio:.1f}x avg")

        # All conditions met
        is_strong = vol_ratio > 2.5 and dist_pct < 0.002
        confidence = 0.75 if is_strong else 0.70

        # SMA spread as strength indicator
        sma_spread = (sma20 - sma50) / sma50
        if sma_spread > 0.02:  # Strong trend
            confidence = min(0.85, confidence + 0.05)
            reasons.append("Strong trend spread")

        return {
            "signal": "strong_buy" if is_strong else "buy",
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "sma20": sma20,
                "sma50": sma50,
                "dist_pct": dist_pct,
                "vol_ratio": vol_ratio,
                "sma_spread": sma_spread,
                "htf_trend": htf_trend.get("direction", "neutral"),
                "htf_strength": htf_trend.get("strength", 0),
            },
        }


# =============================================================================
# Strategy 2b: Multi-TF RSI Momentum (PROVEN — +$1,302/6mo, 64.3% WR, 7/7 months)
# =============================================================================

class RSIMomentumGenerator:
    """Buy 1h RSI dips in 4h uptrends — multi-timeframe momentum continuation.

    Backtested over 6 months (Oct 2025 - Apr 2026), 6 coins, 224 trades:
    - PnL: +$1,302 (+$217/month)
    - Win rate: 64.3%
    - Max drawdown: $113
    - Profitable on ALL 6 coins, ALL 7 months

    Entry requires ALL 4 conditions:
    1. 4H RSI > 45 (macro uptrend confirmed)
    2. 1H RSI was below 40 last candle (pullback/dip happened)
    3. 1H RSI is now rising (recovery starting)
    4. Volume > 1.3x average (buyers stepping in)

    Why this works:
    - 4H RSI filters out bear markets (no entries when 4H weak)
    - 1H RSI < 40 catches oversold dips within uptrends
    - Rising RSI confirms the dip is being bought, not continuing
    - Volume confirms institutional participation
    """

    def generate_signal(self, df_1h: pd.DataFrame, htf_trend: dict,
                        df_4h: pd.DataFrame = None, **kwargs) -> dict:
        """Generate signal from multi-timeframe RSI momentum."""
        if df_1h is None or len(df_1h) < 20:
            return hold_signal("Insufficient 1h data", htf_trend)
        if df_4h is None or len(df_4h) < 20:
            return hold_signal("Insufficient 4h data for RSI momentum", htf_trend)

        # === CONDITION 1: 4H RSI > 45 (macro uptrend) ===
        rsi_4h = self._calc_rsi(df_4h["close"])
        if rsi_4h is None:
            return hold_signal("Cannot calculate 4H RSI", htf_trend)
        r4h = float(rsi_4h.iloc[-1])
        if r4h < 45:
            return hold_signal(f"4H RSI too low ({r4h:.0f}, need >45)", htf_trend)

        # === CONDITION 2: 1H RSI was below 40 (dip happened) ===
        rsi_1h = self._calc_rsi(df_1h["close"])
        if rsi_1h is None or len(rsi_1h) < 3:
            return hold_signal("Cannot calculate 1H RSI", htf_trend)
        r1h = float(rsi_1h.iloc[-1])
        r1h_prev = float(rsi_1h.iloc[-2])
        if r1h_prev >= 40:
            return hold_signal(f"1H RSI wasn't oversold (prev={r1h_prev:.0f}, need <40)", htf_trend)

        # === CONDITION 3: 1H RSI is rising (recovery) ===
        if r1h <= r1h_prev:
            return hold_signal(f"1H RSI not rising ({r1h:.0f} <= {r1h_prev:.0f})", htf_trend)
        if r1h >= 55:
            return hold_signal(f"1H RSI too high ({r1h:.0f}, entry window passed)", htf_trend)

        # === CONDITION 4: Volume confirmation ===
        volumes = df_1h["volume"]
        vol_avg = float(volumes.iloc[-21:-1].mean())
        vol_now = float(volumes.iloc[-1])
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
        if vol_ratio < 1.3:
            return hold_signal(f"Volume too low ({vol_ratio:.2f}x, need 1.3x)", htf_trend)

        reasons = [
            f"4H RSI {r4h:.0f} (uptrend)",
            f"1H RSI {r1h_prev:.0f}→{r1h:.0f} (dip recovery)",
            f"Volume {vol_ratio:.1f}x avg",
        ]

        is_strong = r4h > 55 and vol_ratio > 2.0
        confidence = 0.75 if is_strong else 0.70

        return {
            "signal": "strong_buy" if is_strong else "buy",
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": float(df_1h["close"].iloc[-1]),
                "rsi_4h": r4h,
                "rsi_1h": r1h,
                "rsi_1h_prev": r1h_prev,
                "vol_ratio": vol_ratio,
            },
        }

    @staticmethod
    def _calc_rsi(series, period=14):
        if len(series) < period + 1:
            return None
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)


# =============================================================================
# Strategy 2c: Bollinger Squeeze Expansion (PROVEN — +$1,358/6mo, 82.4% WR, 7/7 months)
# =============================================================================

class BollingerSqueezeGenerator:
    """Enter when Bollinger Bands squeeze then expand with strong directional candle.

    Backtested over 6 months (Oct 2025 - Apr 2026), 6 coins, 74 trades:
    - PnL: +$1,358 (+$226/month)
    - Win rate: 82.4%
    - Max drawdown: $20
    - Profitable on ALL 6 coins, ALL 7 months

    Entry requires ALL 4 conditions:
    1. Squeeze: 4H Bollinger bandwidth was below average (compressed volatility)
    2. Expansion: Current bandwidth is expanding (breakout starting)
    3. Strong candle: Body > 60% of range (momentum, not indecision)
    4. Direction: Price above middle band (bullish breakout)

    Why this works:
    - Volatility compression → expansion is one of the most reliable patterns
    - 4H timeframe filters out 15m/1h noise squeezes
    - Strong candle body confirms real momentum, not a wick trap
    - 82% WR because squeezes are rare but high-conviction setups
    """

    def generate_signal(self, df_4h: pd.DataFrame, htf_trend: dict,
                        **kwargs) -> dict:
        """Generate signal from Bollinger Band squeeze expansion on 4H."""
        if df_4h is None or len(df_4h) < 25:
            return hold_signal("Insufficient 4h data for BB squeeze", htf_trend)

        prices = df_4h["close"]
        current_price = float(prices.iloc[-1])

        # Calculate Bollinger Bands
        sma20 = prices.rolling(20).mean()
        std20 = prices.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20

        # Bandwidth = (upper - lower) / middle
        bandwidth = (upper - lower) / sma20

        if len(bandwidth) < 12:
            return hold_signal("Insufficient bandwidth history", htf_trend)

        curr_bw = float(bandwidth.iloc[-1])
        prev_bw = float(bandwidth.iloc[-2])
        avg_bw = float(bandwidth.rolling(10).mean().iloc[-2])

        # === CONDITION 1: Was squeezed (bandwidth below 80% of average) ===
        if prev_bw >= avg_bw * 0.8:
            return hold_signal(f"No squeeze detected (bw={prev_bw:.4f} >= {avg_bw*0.8:.4f})", htf_trend)

        # === CONDITION 2: Now expanding ===
        if curr_bw <= prev_bw:
            return hold_signal(f"Bandwidth not expanding ({curr_bw:.4f} <= {prev_bw:.4f})", htf_trend)

        # === CONDITION 3: Strong candle (body > 60% of range) ===
        candle_open = float(df_4h["open"].iloc[-1])
        candle_high = float(df_4h["high"].iloc[-1])
        candle_low = float(df_4h["low"].iloc[-1])
        candle_range = candle_high - candle_low
        candle_body = abs(current_price - candle_open)

        if candle_range <= 0:
            return hold_signal("Zero-range candle", htf_trend)

        body_pct = candle_body / candle_range
        if body_pct < 0.6:
            return hold_signal(f"Weak candle (body={body_pct:.0%}, need >60%)", htf_trend)

        # === CONDITION 4: Direction — price above middle band ===
        middle = float(sma20.iloc[-1])
        if current_price <= middle:
            return hold_signal(f"Price below middle band ({current_price:.2f} <= {middle:.2f})", htf_trend)

        reasons = [
            f"BB squeeze expanded (bw {prev_bw:.4f}→{curr_bw:.4f})",
            f"Strong candle ({body_pct:.0%} body)",
            f"Above middle band",
        ]

        # Higher confidence for very tight squeezes
        squeeze_depth = prev_bw / avg_bw if avg_bw > 0 else 1
        is_strong = squeeze_depth < 0.6 and body_pct > 0.75
        confidence = 0.80 if is_strong else 0.72

        return {
            "signal": "strong_buy" if is_strong else "buy",
            "confidence": confidence,
            "reasoning": ", ".join(reasons),
            "indicators": {
                "price": current_price,
                "bandwidth": curr_bw,
                "prev_bandwidth": prev_bw,
                "avg_bandwidth": avg_bw,
                "squeeze_depth": squeeze_depth,
                "body_pct": body_pct,
                "middle_band": middle,
            },
        }


# =============================================================================
# Strategy 2d: Trend Breakout (DISABLED — backtested -$1,451/6mo)
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

        # === CONDITION 1: HTF must be trending ===
        # v6.9.4: Restored to original strength threshold 0.1 (was 0.3 in v6.9.2).
        # 30-day data: trend_breakout is #1 earner at +$230. The 0.3 filter + 4h filter
        # stacked to block ALL longs including during real rallies. Losing weeks (-$64)
        # are covered by winning weeks (+$271). Neutral mode stays disabled (solid evidence).
        required_vol_ratio = 1.5
        if direction == "neutral":
            return hold_signal("Neutral-mode breakouts disabled (false break rate too high)", htf_trend)
        elif strength < 0.1:
            return hold_signal(f"HTF {direction} too weak (str={strength:.2f}, need 0.1)", htf_trend)
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

        # === CONDITION 1: Price below 1h SMA20 by >0.5% (confirmed downtrend) ===
        # v6.9.4: Raised from 0.3% to 0.5%. crash_momentum fired 131 trades in 30 days
        # for only +$9 total. 48 exits were "no momentum" — too many marginal entries.
        # 0.5% ensures we only enter when there's real selling pressure, not just noise.
        sma20_series = prices.rolling(20).mean()
        sma20 = float(sma20_series.iloc[-1])
        if current_price >= sma20:
            return hold_signal(f"Price above 1h SMA20 ({current_price:.2f} >= {sma20:.2f})", htf_trend)
        pct_below_sma = (sma20 - current_price) / sma20 * 100
        if pct_below_sma < 0.5:
            return hold_signal(f"Price only {pct_below_sma:.2f}% below SMA20 (need >0.5%)", htf_trend)
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


class SmartMoneyFlowGenerator:
    """Smart money composite signal — tracks whale positioning across multiple data sources.

    Combines 6 data sources into a single Smart Money Score (-1.0 to +1.0):
    1. Top trader position ratio (Binance, capital-weighted)
    2. Top trader vs retail divergence (contrarian retail signal)
    3. Hyperliquid whale consensus (actual whale positions)
    4. Taker buy/sell ratio (aggressive order flow)
    5. Fear & Greed Index (contrarian extreme sentiment)
    6. Funding rate extreme (contrarian crowd positioning)
    """

    def generate_signal(self, df_15m, htf_trend, derivatives=None,
                        whale_consensus=None, fear_greed=None):
        if df_15m is None or len(df_15m) < 20:
            return hold_signal("Insufficient 15m data", htf_trend)
        if not derivatives:
            return hold_signal("No derivatives data", htf_trend)

        direction = htf_trend.get("direction", "neutral")
        strength = htf_trend.get("strength", 0.0)

        # Don't trade against a strong HTF trend
        # (if HTF strongly bullish, don't take short signals and vice versa)

        components = []  # (name, score, weight) — score is -1 to +1
        reasons = []

        # === 1. Top trader POSITION ratio (capital-weighted whale direction) ===
        top_pos = derivatives.get("top_position_ratio", [])
        if top_pos:
            long_pct = top_pos[0].get("long_account", 0.5)
            if long_pct > 0.55:
                score = min(1.0, (long_pct - 0.5) / 0.15)  # 0.55→0.33, 0.65→1.0
                components.append(("top_position", score, 0.25))
                reasons.append(f"Top traders {long_pct:.0%} long")
            elif long_pct < 0.45:
                score = max(-1.0, (long_pct - 0.5) / 0.15)
                components.append(("top_position", score, 0.25))
                reasons.append(f"Top traders {1-long_pct:.0%} short")
            else:
                components.append(("top_position", 0.0, 0.25))

        # === 2. Top trader vs retail divergence ===
        global_ls = derivatives.get("global_long_short", [])
        top_ls = derivatives.get("top_long_short", [])
        if global_ls and top_ls:
            retail_long = global_ls[0].get("long_account", 0.5)
            pro_long = top_ls[0].get("long_account", 0.5)
            divergence = pro_long - retail_long  # +ve = pros more bullish than retail
            if abs(divergence) > 0.03:  # Meaningful divergence (>3%)
                score = max(-1.0, min(1.0, divergence / 0.10))  # Scale: 10% div = full score
                components.append(("pro_retail_div", score, 0.25))
                reasons.append(f"Pro-retail divergence {divergence:+.1%}")
            else:
                components.append(("pro_retail_div", 0.0, 0.25))

        # === 3. Hyperliquid whale consensus ===
        if whale_consensus:
            wc = whale_consensus.get("consensus", 0.0)
            total_whales = whale_consensus.get("total", 0)
            if total_whales >= 3 and abs(wc) > 0.2:  # Need at least 3 whales with signal
                components.append(("hl_whales", wc, 0.20))
                side = "long" if wc > 0 else "short"
                reasons.append(f"HL whales {side} ({whale_consensus.get('long_count',0)}L/{whale_consensus.get('short_count',0)}S)")
            else:
                components.append(("hl_whales", 0.0, 0.20))

        # === 4. Taker ratio (15m aggressive flow) ===
        taker_15m = derivatives.get("taker_ratio_15m", [])
        if taker_15m:
            ratio = taker_15m[0].get("buy_sell_ratio", 1.0)
            if ratio > 1.05:
                score = min(1.0, (ratio - 1.0) / 0.15)  # 1.05→0.33, 1.15→1.0
                components.append(("taker_flow", score, 0.15))
                reasons.append(f"Taker ratio {ratio:.2f} (buying)")
            elif ratio < 0.95:
                score = max(-1.0, (ratio - 1.0) / 0.15)
                components.append(("taker_flow", score, 0.15))
                reasons.append(f"Taker ratio {ratio:.2f} (selling)")
            else:
                components.append(("taker_flow", 0.0, 0.15))

        # === 5. Fear & Greed extreme (contrarian) ===
        if fear_greed:
            fg_value = fear_greed.get("value", 50)
            if fg_value < 20:  # Extreme fear → contrarian long
                score = min(1.0, (20 - fg_value) / 15)  # 20→0, 5→1.0
                components.append(("fear_greed", score, 0.10))
                reasons.append(f"Fear&Greed {fg_value} (extreme fear)")
            elif fg_value > 75:  # Extreme greed → contrarian short
                score = max(-1.0, (75 - fg_value) / 15)
                components.append(("fear_greed", score, 0.10))
                reasons.append(f"Fear&Greed {fg_value} (extreme greed)")
            else:
                components.append(("fear_greed", 0.0, 0.10))

        # === 6. Funding rate extreme (contrarian) ===
        funding = derivatives.get("funding_rate", [])
        if funding:
            rate = funding[0].get("funding_rate", 0)
            if rate < -0.0002:  # Shorts paying → contrarian long
                score = min(1.0, abs(rate) / 0.001)
                components.append(("funding", score, 0.05))
                reasons.append(f"Funding {rate*100:.3f}% (shorts paying)")
            elif rate > 0.0003:  # Longs paying → contrarian short
                score = max(-1.0, -rate / 0.001)
                components.append(("funding", score, 0.05))
                reasons.append(f"Funding {rate*100:.3f}% (longs paying)")
            else:
                components.append(("funding", 0.0, 0.05))

        # === Compute weighted score ===
        if not components:
            return hold_signal("No smart money data available", htf_trend)

        total_weight = sum(w for _, _, w in components)
        if total_weight <= 0:
            return hold_signal("No smart money components", htf_trend)

        smart_score = sum(s * w for _, s, w in components) / total_weight
        agreeing = sum(1 for _, s, _ in components if abs(s) > 0.1 and (
            (smart_score > 0 and s > 0) or (smart_score < 0 and s < 0)
        ))

        # === Entry decision ===
        min_score = 0.45  # was 0.55 (killed strategy — 0 trades in 6 days). Was 0.4 before that. 0.45 filters weakest entries while still firing.
        min_agreeing = 3

        if abs(smart_score) < min_score:
            return hold_signal(
                f"Smart money score {smart_score:+.2f} too weak (need ±{min_score}), "
                f"{agreeing} agreeing", htf_trend)

        if agreeing < min_agreeing:
            return hold_signal(
                f"Only {agreeing}/{len(components)} components agree (need {min_agreeing})", htf_trend)

        # Require at least one of whale or funding to be active (non-zero)
        # AI analysis: all 5 consecutive losses had hl_whales=0.0 and funding=0.0
        component_scores = {name: s for name, s, _ in components}
        whale_score = abs(component_scores.get("hl_whales", 0.0))
        funding_score = abs(component_scores.get("funding", 0.0))
        if whale_score < 0.1 and funding_score < 0.1:
            return hold_signal(
                f"Smart money {smart_score:+.2f} but no whale/funding confirmation "
                f"(whales={whale_score:.2f}, funding={funding_score:.2f})", htf_trend)

        # Check HTF not strongly opposing
        if smart_score > 0 and direction == "bearish" and strength > 0.3:
            return hold_signal(f"Smart money bullish but HTF strongly bearish (str={strength:.2f})", htf_trend)
        if smart_score < 0 and direction == "bullish" and strength > 0.3:
            return hold_signal(f"Smart money bearish but HTF strongly bullish (str={strength:.2f})", htf_trend)

        # Generate signal
        if smart_score > 0:
            signal_type = "strong_buy" if smart_score > 0.6 else "buy"
        else:
            signal_type = "strong_sell" if smart_score < -0.6 else "sell"

        # Confidence: map score magnitude to 0.65-0.85 range
        confidence = min(0.85, 0.65 + abs(smart_score) * 0.3)

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": f"Smart money score {smart_score:+.2f} ({agreeing}/{len(components)} agree): " + ", ".join(reasons),
            "indicators": {
                "smart_score": round(smart_score, 3),
                "components_agreeing": agreeing,
                "total_components": len(components),
                "component_scores": {name: round(s, 3) for name, s, _ in components},
                "htf_trend": direction,
                "htf_strength": strength,
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
    max_concurrent_positions: int = 0  # 0 = unlimited, >0 = cap across all symbols
    # Adaptive sizing (opt-in, default preserves current behavior)
    adaptive_sizing: bool = False    # Master toggle
    min_risk_pct: float = 0.005     # Floor: 0.5% risk minimum
    max_risk_pct: float = 0.025     # Ceiling: 2.5% risk maximum (v7.0: reduced from 3%)

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


def calculate_kelly_risk_pct(
    base_risk_pct: float,
    recent_results: list[float],
    min_trades: int = 15,
    kelly_fraction: float = 0.25,
    min_risk_pct: float = 0.005,
    max_risk_pct: float = 0.025,
) -> float:
    """Calculate position size using quarter-Kelly criterion from actual trade results.

    v7.0: Replaces calculate_adaptive_risk_pct which had broken scaling:
    - Win streak boost (gambler's fallacy)
    - Regime boost (increases risk when drawdowns compound)
    - Multi-strategy boost (correlated signals aren't independent)

    Kelly formula: f* = (p * b - q) / b
    where p = win rate, q = loss rate, b = avg_win / avg_loss

    Uses quarter-Kelly for safety (captures 75% of growth with 50% less drawdown).
    Falls back to base_risk_pct if insufficient trade history.
    """
    if len(recent_results) < min_trades:
        return base_risk_pct

    wins = [r for r in recent_results if r > 0]
    losses = [r for r in recent_results if r <= 0]

    if not wins or not losses:
        return base_risk_pct

    win_rate = len(wins) / len(recent_results)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))

    if avg_loss == 0:
        return base_risk_pct

    b = avg_win / avg_loss  # Win/loss ratio
    kelly = (win_rate * b - (1 - win_rate)) / b

    if kelly <= 0:
        return min_risk_pct  # No edge — minimum sizing

    risk = kelly * kelly_fraction
    return max(min_risk_pct, min(max_risk_pct, risk))


def calculate_vol_adjusted_risk(
    base_risk_pct: float,
    atr_pct: float,
    target_vol_pct: float = 0.015,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
) -> float:
    """Scale risk inversely with realized volatility (v7.0).

    When ATR is 3% (high vol), scale = 1.5/3.0 = 0.5 (half size).
    When ATR is 0.75% (low vol), scale = 1.5/0.75 = 2.0, capped at 1.5.
    """
    if atr_pct <= 0:
        return base_risk_pct
    vol_scale = max(min_scale, min(max_scale, target_vol_pct / atr_pct))
    return base_risk_pct * vol_scale


# v7.0: Correlation groups — coins within a group are 80-95% correlated
# Max 2 same-direction positions per group (6 coins ≈ 2 real bets, not 6)
CORRELATION_GROUPS = {
    "large_cap": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "alt_cap": ["XRPUSDT", "DOGEUSDT", "AVAXUSDT"],
}
MAX_PER_CORRELATION_GROUP = 3  # v7.0.1: relaxed from 2 — allow full group in strong trends


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
                # Rolling performance for adaptive sizing
                "recent_results": [],
                "consecutive_losses": 0,
                "consecutive_wins": 0,
            }
            self.initial_capital += s.capital

        # Positions keyed by "strategy_name:symbol"
        self.positions: dict[str, dict] = {}

        # Supabase repositories
        self.trade_repo = TradeLogRepository()
        self.signal_repo = SignalRepository()
        self.perf_repo = PerformanceRepository()
        self.review_repo = AIReviewRepository()
        self.analysis_repo = TradeAnalysisRepository()

        # AI-powered analysis (Claude API)
        self.ai_analyzer = None
        self.ai_post_trade_enabled = os.environ.get("AI_POST_TRADE_ENABLED", "true").lower() == "true"
        self.ai_signal_gate_enabled = os.environ.get("AI_SIGNAL_GATE_ENABLED", "false").lower() == "true"
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key and (self.ai_post_trade_enabled or self.ai_signal_gate_enabled):
            try:
                from core.ai.claude_client import ClaudeAnalyzer
                self.ai_analyzer = ClaudeAnalyzer(api_key=api_key)
                features = []
                if self.ai_post_trade_enabled:
                    features.append("post-trade")
                if self.ai_signal_gate_enabled:
                    features.append("signal-gate")
                logger.info(f"[AI] Claude analyzer initialized ({', '.join(features)})")
            except Exception as e:
                logger.warning(f"[AI] Could not initialize Claude analyzer: {e}")

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

        # Directional SL tracking: prevents re-entering same direction after repeated SLs
        # Key: "strategy_name:direction", Value: list of SL timestamps
        self.directional_sl_history: dict[str, list[datetime]] = {}
        self.directional_sl_max = 2           # N SLs in same direction to trigger block
        self.directional_sl_window_hours = 2  # Time window for counting

        # Per-strategy circuit breaker: pause a strategy after N SLs (not global — don't punish winners for losers)
        self.strategy_sl_timestamps: dict[str, list[datetime]] = {}  # strategy_name -> SL timestamps
        self.circuit_breaker_window_hours = 2  # Look at SLs within this window
        self.circuit_breaker_threshold = 2     # N SLs to trigger
        self.circuit_breaker_pause_minutes = 60  # How long to pause
        self.circuit_breaker_until: dict[str, datetime] = {}  # strategy_name -> pause end time

        # Reversal close: close losing non-pullback positions when reversal persists 2+ cycles
        # Key: "symbol" → count of consecutive main-loop cycles with reversal override active
        self.reversal_override_counts: dict[str, int] = {}
        self.reversal_close_min_cycles = 2  # Must persist 2 consecutive cycles before closing
        self._reversal_counted_this_cycle: set[str] = set()  # Prevent multi-counting per cycle

        # === v7.0 Portfolio-level risk controls ===
        # Daily loss limit: stop opening new trades after 5% daily drawdown
        self.daily_pnl = 0.0
        self.daily_loss_limit_pct = 0.03  # 3% of initial capital (v7.0.3: was 5%, tightened after -$146 day)
        self.daily_loss_halt = False
        self.daily_pnl_reset_date = ""

        # Max directional exposure: cap same-direction positions across all strategies
        # v7.0.1: Relaxed from 3→6 longs, 4 shorts. Data showed 3 longs blocked $248 of
        # profit in the best week, turning +$181 into -$68. Crypto profits come from riding
        # correlated rallies. Daily loss limit + portfolio heat handle the downside.
        self.max_long_positions = 6
        self.max_short_positions = 4

        # Portfolio heat: max total open risk (sum of all SL distances × position values)
        # v7.0.1: Relaxed from 8%→12%. With Kelly sizing keeping per-trade risk small,
        # 8% heat was too tight for 6 concurrent positions. 12% allows 6 positions × 2% each.
        self.max_portfolio_heat_pct = 0.12  # 12% max total risk if all stops hit

    @property
    def total_capital(self):
        return sum(s["capital"] for s in self.strategy_stats.values())

    @property
    def total_pnl(self):
        return sum(s["total_pnl"] for s in self.strategy_stats.values())

    # === v7.0 Portfolio risk check methods ===

    def _check_portfolio_exposure(self, proposed_side: str) -> bool:
        """Check if adding a position would exceed max directional exposure."""
        count = sum(1 for p in self.positions.values() if p.get("side") == proposed_side)
        limit = self.max_long_positions if proposed_side == "long" else self.max_short_positions
        return count < limit

    def _check_correlation_limit(self, symbol: str, proposed_side: str) -> bool:
        """Check if adding a position would exceed correlation group limits."""
        for group_name, group_symbols in CORRELATION_GROUPS.items():
            if symbol in group_symbols:
                count = sum(
                    1 for k, v in self.positions.items()
                    if self._parse_pos_key(k)[1] in group_symbols
                    and v.get("side") == proposed_side
                )
                if count >= MAX_PER_CORRELATION_GROUP:
                    return False
        return True

    def _calculate_portfolio_heat(self) -> float:
        """Total portfolio risk: sum of (position_value × sl_distance) / initial_capital."""
        total_risk = 0.0
        for pos in self.positions.values():
            pos_value = pos.get("quantity", 0) * pos.get("entry_price", 0)
            sl_pct = pos.get("sl_pct", 0.02)
            total_risk += pos_value * sl_pct
        return total_risk / self.initial_capital if self.initial_capital > 0 else 0

    def _check_portfolio_heat(self, strategy: 'StrategyConfig') -> bool:
        """Check if adding a new position would exceed max portfolio heat."""
        estimated_risk = strategy.risk_per_trade_pct * strategy.capital
        current_heat = self._calculate_portfolio_heat()
        return (current_heat + estimated_risk / self.initial_capital) <= self.max_portfolio_heat_pct

    async def _auto_rebalance_capital(self):
        """Auto-rebalance capital every 12 hours using AI analysis of strategy performance.

        Uses Claude AI to analyze 7-day trade data and recommend allocations.
        Falls back to PnL/hr math if AI is unavailable.

        Constraints: min $200 per strategy, max 50% for any single strategy.
        Only rebalances strategies with no open positions.
        """
        try:
            # Fetch last 7 days of trades from DB
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            result = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("strategy_name,net_pnl,entry_time,exit_time,duration_seconds")
                .gte("exit_time", cutoff)
                .not_.is_("exit_time", "null")
                .execute()
            )
            trades = result.data or []

            if len(trades) < 10:
                logger.info("📊 [Rebalance] <10 trades in 7 days, skipping rebalance")
                return

            total_capital = self.total_capital
            new_allocs = None

            # Try AI-powered rebalance first
            if self.ai_analyzer:
                logger.info("📊 [Rebalance] Asking AI for capital allocation recommendation...")
                new_allocs = await self.ai_analyzer.recommend_capital_allocation(
                    strategy_stats=self.strategy_stats,
                    total_capital=total_capital,
                    recent_trades=trades,
                )

            # Fallback: PnL/hr math if AI unavailable
            if not new_allocs:
                logger.info("📊 [Rebalance] AI unavailable, using PnL/hr math fallback")
                from collections import defaultdict
                strat_perf = defaultdict(lambda: {"pnl": 0.0, "time_mins": 0.0, "trades": 0})

                for t in trades:
                    strat = t.get("strategy_name", "unknown")
                    pnl = t.get("net_pnl") or 0
                    dur = (t.get("duration_seconds") or 0) / 60
                    strat_perf[strat]["pnl"] += pnl
                    strat_perf[strat]["time_mins"] += dur
                    strat_perf[strat]["trades"] += 1

                scores = {}
                for strat, perf in strat_perf.items():
                    if strat not in self.strategy_stats:
                        continue
                    pnl_per_hr = perf["pnl"] / (perf["time_mins"] / 60) if perf["time_mins"] > 60 else 0
                    scores[strat] = max(0.0, pnl_per_hr)

                total_score = sum(scores.values())
                if total_score <= 0:
                    logger.info("📊 [Rebalance] No positive-edge strategies, keeping current allocation")
                    return

                # v7.0.4: Min $500 per strategy. $225 was too small to take meaningful positions.
                # Order_flow and smart_money got gutted to $225 and couldn't trade during +4.2% rally.
                min_alloc = max(500.0, total_capital * 0.08)
                max_alloc = total_capital * 0.50
                new_allocs = {}
                for s in self.strategies:
                    score = scores.get(s.name, 0)
                    raw = (score / total_score) * total_capital if score > 0 else min_alloc
                    new_allocs[s.name] = max(min_alloc, min(max_alloc, raw))

                alloc_sum = sum(new_allocs.values())
                if alloc_sum > 0:
                    scale = total_capital / alloc_sum
                    new_allocs = {k: v * scale for k, v in new_allocs.items()}

            # Apply new allocations (only if strategy has no open positions)
            changes = []
            for s in self.strategies:
                if s.name not in new_allocs:
                    continue
                stats = self.strategy_stats[s.name]
                old_cap = stats["capital"]
                new_cap = new_allocs[s.name]

                has_open = any(k.startswith(f"{s.name}:") for k in self.positions)
                if has_open:
                    continue

                if abs(new_cap - old_cap) > 10:
                    changes.append(f"{s.name}: ${old_cap:.0f} → ${new_cap:.0f}")
                    stats["capital"] = new_cap

            if changes:
                source = "AI" if self.ai_analyzer else "PnL/hr math"
                logger.info(f"📊 [Rebalance] Capital rebalanced ({source}, 7-day data):")
                for c in changes:
                    logger.info(f"   {c}")
            else:
                logger.info("📊 [Rebalance] No changes needed (positions open or allocations stable)")

        except Exception as e:
            logger.warning(f"📊 [Rebalance] Error during auto-rebalance: {e}")

    async def _ai_regime_check(self, market_data_btc: dict):
        """#2+#5: Hourly AI regime detection from BTC market data."""
        if not self.ai_analyzer:
            return
        try:
            prices_1h = market_data_btc.get("1h")
            if prices_1h is None or prices_1h.empty:
                return

            close = prices_1h["close"]
            current = float(close.iloc[-1])
            sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else current
            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current
            atr = calculate_atr(prices_1h) if len(prices_1h) >= 14 else 0
            atr_pct = atr / current * 100 if current > 0 else 0

            sma_status = "SMA20 above SMA50" if sma20 > sma50 else "SMA20 below SMA50"
            fg = self._fear_greed_cache

            # 4h trend
            df_4h = market_data_btc.get("4h")
            trend_4h = "unknown"
            if df_4h is not None and not df_4h.empty and len(df_4h) >= 50:
                sma50_4h = float(df_4h["close"].rolling(50).mean().iloc[-1])
                trend_4h = "bullish" if current > sma50_4h else "bearish"

            derivatives = market_data_btc.get("derivatives", {})
            taker = derivatives.get("taker_ratio_15m", [{}])
            taker_ratio = taker[0].get("buy_sell_ratio", 1.0) if isinstance(taker, list) and taker else 1.0
            top_ls = derivatives.get("top_long_short", [{}])
            top_long = top_ls[0].get("long_account", 0.5) if isinstance(top_ls, list) and top_ls else 0.5
            funding = derivatives.get("funding_rate", [{}])
            fund_rate = funding[0].get("rate", 0) if isinstance(funding, list) and funding else 0

            data = {
                "btc_price": f"{current:,.0f}",
                "btc_sma_status": f"{sma_status} (spread: {abs(sma20-sma50)/sma50*100:.2f}%)",
                "btc_4h_trend": trend_4h,
                "fear_greed": f"{fg.get('value', '?')} ({fg.get('classification', '?')})",
                "btc_funding": f"{fund_rate*100:.4f}%",
                "taker_ratio": f"{taker_ratio:.3f}",
                "top_long_pct": f"{top_long:.1%}",
                "atr_pct": f"{atr_pct:.2f}%",
            }

            result = await self.ai_analyzer.detect_regime(data)
            if result:
                old_regime = self._ai_regime.get("regime", "unknown")
                old_bias = self._ai_regime.get("bias", "neutral")
                self._ai_regime = result
                new_regime = result.get("regime", "unknown")
                new_bias = result.get("bias", "neutral")
                self._log_ai_action("regime_detection", {
                    "regime": new_regime, "bias": new_bias,
                    "confidence": result.get("confidence", 0),
                    "risk_level": result.get("risk_level", "?"),
                    "reasoning": result.get("reasoning", "")[:100],
                    "shifted": old_regime != new_regime or old_bias != new_bias,
                })

                # Regime SHIFT detected — take immediate action
                if old_regime != new_regime or old_bias != new_bias:
                    logger.info(f"🧠 [AI Regime] SHIFT: {old_regime}/{old_bias} → {new_regime}/{new_bias}")
                    await self._on_regime_shift(result)
        except Exception as e:
            logger.warning(f"🧠 [AI Regime] Error: {e}")

    async def _on_regime_shift(self, regime: dict):
        """React immediately when AI detects a regime change.

        Actions:
        1. Tighten stops on positions conflicting with new regime
        2. Close losing positions that conflict with high-confidence regime
        """
        bias = regime.get("bias", "neutral")
        confidence = regime.get("confidence", 0)
        risk_level = regime.get("risk_level", "medium")

        if confidence < 0.65 or bias == "neutral":
            return  # Not confident enough to act

        conflicting_side = "long" if bias == "short" else "short" if bias == "long" else None
        if not conflicting_side:
            return

        actions_taken = 0
        for pos_key, position in list(self.positions.items()):
            if position.get("side") != conflicting_side:
                continue

            pnl_pct = self._calculate_pnl_pct(position, position.get("entry_price", 0))
            strategy_name = pos_key.split(":")[0]
            strategy = self.strategy_map.get(strategy_name)
            if not strategy:
                continue

            # Get current price for this symbol
            _, symbol = self._parse_pos_key(pos_key)

            if confidence >= 0.8 and pnl_pct < -0.005 and risk_level == "high":
                # High confidence regime shift + position losing → close
                logger.info(f"🧠 [Regime Shift] Closing {pos_key}: {conflicting_side} in loss ({pnl_pct:.2%}) vs {bias} regime (conf={confidence})")
                self._log_ai_action("regime_shift_close", {"position": pos_key, "side": conflicting_side, "pnl_pct": f"{pnl_pct:.2%}", "regime_bias": bias})
                current_price = position.get("entry_price", 0) * (1 + pnl_pct) if position.get("side") == "long" else position.get("entry_price", 0) * (1 - pnl_pct)
                await self._close_position(pos_key, current_price, f"AI regime shift: {bias} bias (conf={confidence:.0%})", strategy)
                actions_taken += 1
            elif confidence >= 0.7:
                # Moderate confidence → tighten to breakeven if profitable
                entry = position["entry_price"]
                if pnl_pct > 0.002:  # Slightly profitable
                    if conflicting_side == "long":
                        position["stop_loss_price"] = entry * 1.001
                    else:
                        position["stop_loss_price"] = entry * 0.999
                    logger.info(f"🧠 [Regime Shift] Tightened {pos_key} SL to breakeven (regime: {bias}, conf={confidence})")
                    actions_taken += 1

        if actions_taken:
            logger.info(f"🧠 [Regime Shift] Took {actions_taken} actions on regime change to {bias}")

    async def _ai_pattern_learning(self):
        """#1: Daily AI pattern learning from all historical trades."""
        if not self.ai_analyzer:
            return
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            result = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("strategy_name,symbol,side,net_pnl,signal_confidence,indicators_at_entry,entry_time,exit_time,exit_reasoning,duration_seconds")
                .gte("exit_time", cutoff)
                .not_.is_("exit_time", "null")
                .order("exit_time", desc=True)
                .limit(500)
                .execute()
            )
            trades = result.data or []
            if len(trades) < 20:
                logger.info("🧠 [AI Pattern] <20 trades, skipping learning")
                return

            strategy_names = [s.name for s in self.strategies]
            rules = await self.ai_analyzer.learn_trade_patterns(trades, strategy_names)
            if rules:
                self._ai_learned_rules = rules
                logger.info(f"🧠 [AI Pattern] Updated learned rules for {len(rules)} strategies")
                for strat, rule in rules.items():
                    if rule.get("notes"):
                        logger.info(f"   {strat}: {rule['notes']}")
                self._log_ai_action("pattern_learning", {
                    "strategies": len(rules),
                    "rules": {k: v.get("notes", "") for k, v in rules.items()},
                    "trade_count": len(trades),
                })
        except Exception as e:
            logger.warning(f"🧠 [AI Pattern] Error: {e}")

    async def _ai_parameter_tuning(self):
        """#4: Daily AI parameter tuning recommendations."""
        if not self.ai_analyzer:
            return
        try:
            # v7.0.3: Only use trades from CURRENT system version, not old data.
            # Using old v6.9.x data caused the AI to widen SL for a $2,500 strategy
            # based on results from a $1,250 strategy — amplifying losses.
            # Use dashboard cutoff date as the system start marker.
            system_start = "2026-04-06T04:50:00Z"  # v7.0 deploy
            result = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("strategy_name,net_pnl,exit_reasoning,duration_seconds")
                .gte("exit_time", system_start)
                .not_.is_("exit_time", "null")
                .execute()
            )
            trades = result.data or []
            if len(trades) < 30:  # Need 30+ trades under current system before tuning
                logger.info(f"🔧 [AI Tuning] Only {len(trades)} trades since v7.0 deploy, need 30+ before tuning")
                return

            current_params = {s.name: {"sl_atr_mult": s.sl_atr_mult, "tp_atr_mult": s.tp_atr_mult,
                                        "max_position_hours": s.max_position_hours} for s in self.strategies}

            recommendations = await self.ai_analyzer.tune_parameters(
                self.strategy_stats, trades, current_params
            )
            if not recommendations:
                return

            # Apply parameter changes with GRADUAL stepping
            # v7.0.3: Max 10% change per tuning cycle to prevent sudden large shifts.
            # AI tuner jumped SL 1.5→2.0 (33%) in one step, which amplified losses by $20/trade.
            MAX_STEP_PCT = 0.10  # Max 10% change per cycle
            for s in self.strategies:
                rec = recommendations.get(s.name)
                if not rec:
                    continue
                changes = []
                if rec.get("sl_atr_mult") and rec["sl_atr_mult"] != s.sl_atr_mult:
                    old = s.sl_atr_mult
                    target = max(1.0, min(3.0, rec["sl_atr_mult"]))
                    max_delta = old * MAX_STEP_PCT
                    new_val = max(old - max_delta, min(old + max_delta, target))
                    s.sl_atr_mult = round(new_val, 2)
                    changes.append(f"SL {old}→{s.sl_atr_mult}x ATR (target {target})")
                if rec.get("tp_atr_mult") and rec["tp_atr_mult"] != s.tp_atr_mult:
                    old = s.tp_atr_mult
                    target = max(1.5, min(5.0, rec["tp_atr_mult"]))
                    max_delta = old * MAX_STEP_PCT
                    new_val = max(old - max_delta, min(old + max_delta, target))
                    s.tp_atr_mult = round(new_val, 2)
                    changes.append(f"TP {old}→{s.tp_atr_mult}x ATR (target {target})")
                if changes:
                    logger.info(f"🔧 [AI Tuning] {s.name}: {', '.join(changes)} — {rec.get('notes', '')}")
                    self._log_ai_action("parameter_tuning", {
                        "strategy": s.name, "changes": changes,
                        "notes": rec.get("notes", ""),
                    })
        except Exception as e:
            logger.warning(f"🔧 [AI Tuning] Error: {e}")

    async def _ai_symbol_selection(self):
        """#6: AI symbol selection every 12h."""
        if not self.ai_analyzer:
            return
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            result = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("symbol,net_pnl")
                .gte("exit_time", cutoff)
                .not_.is_("exit_time", "null")
                .execute()
            )
            trades = result.data or []
            if len(trades) < 15:
                return

            from collections import defaultdict
            sym_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
            for t in trades:
                sym = t.get("symbol", "?")
                pnl = t.get("net_pnl") or 0
                sym_stats[sym]["trades"] += 1
                sym_stats[sym]["pnl"] += pnl
                if pnl > 0:
                    sym_stats[sym]["wins"] += 1

            recommendation = await self.ai_analyzer.select_symbols(dict(sym_stats), self.symbols)
            if recommendation:
                self._ai_avoid_long = set(recommendation.get("avoid_long") or [])
                self._ai_avoid_short = set(recommendation.get("avoid_short") or [])
                logger.info(f"🎯 [AI Symbols] Avoid long: {self._ai_avoid_long}, Avoid short: {self._ai_avoid_short}")
                self._log_ai_action("symbol_selection", {
                    "avoid_long": list(self._ai_avoid_long),
                    "avoid_short": list(self._ai_avoid_short),
                    "reasoning": recommendation.get("reasoning", ""),
                })
        except Exception as e:
            logger.warning(f"🎯 [AI Symbols] Error: {e}")

    async def _ai_check_exit(self, pos_key: str, position: dict, current_price: float,
                              strategy: StrategyConfig) -> str | None:
        """#3: AI exit optimization for open positions. Returns exit reason or None."""
        if not self.ai_analyzer:
            return None
        try:
            mins_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 60
            if mins_open < 30:
                return None  # Too early for AI exit check

            pnl_pct = self._calculate_pnl_pct(position, current_price)
            _, symbol = self._parse_pos_key(pos_key)

            result = await self.ai_analyzer.optimize_exit(
                position={
                    "side": position.get("side", "?"),
                    "pnl_pct": pnl_pct,
                    "mins_open": mins_open,
                    "sl_pct": position.get("sl_pct", 0.02),
                    "tp_pct": position.get("tp_pct", 0.04),
                    "strategy": strategy.name,
                    "symbol": symbol,
                },
                current_price=current_price,
                market_context=self._ai_regime,
            )

            if not result:
                return None

            action = result.get("action", "hold")
            reasoning = result.get("reasoning", "")

            if action == "close_now":
                logger.info(f"🧠 [AI Exit] {pos_key}: CLOSE — {reasoning}")
                self._log_ai_action("exit_close", {"position": pos_key, "reasoning": reasoning})
                return f"AI exit: {reasoning}"
            elif action == "tighten_stop":
                # Move SL to breakeven
                entry = position["entry_price"]
                if position["side"] == "long" and current_price > entry:
                    position["stop_loss_price"] = entry * 1.001  # Tiny buffer above entry
                    logger.info(f"🧠 [AI Exit] {pos_key}: Tightened SL to breakeven — {reasoning}")
                elif position["side"] == "short" and current_price < entry:
                    position["stop_loss_price"] = entry * 0.999
                    logger.info(f"🧠 [AI Exit] {pos_key}: Tightened SL to breakeven — {reasoning}")
            return None
        except Exception as e:
            logger.warning(f"🧠 [AI Exit] Error for {pos_key}: {e}")
            return None

    def _log_ai_action(self, action_type: str, details: dict):
        """Record an AI action to the buffer. Flushed to DB every 15 min."""
        self._ai_action_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action_type,
            **details,
        })

    async def _flush_ai_action_log(self):
        """Save buffered AI actions to ai_reviews table and clear buffer."""
        if not self._ai_action_log:
            return
        try:
            import json
            # Build summary of all actions in this period
            actions = self._ai_action_log.copy()
            self._ai_action_log.clear()

            # Group by type for readable summary
            by_type = {}
            for a in actions:
                t = a.get("action", "unknown")
                by_type.setdefault(t, []).append(a)

            summary_parts = []
            for action_type, items in by_type.items():
                summary_parts.append(f"**{action_type}** ({len(items)}x)")
                for item in items[-3:]:  # Last 3 per type
                    detail = {k: v for k, v in item.items() if k not in ("timestamp", "action")}
                    summary_parts.append(f"  {item['timestamp'][11:19]} — {json.dumps(detail, default=str)[:200]}")

            summary = "\n".join(summary_parts)

            # Current state snapshot
            state = {
                "regime": self._ai_regime,
                "learned_rules": {k: v.get("notes", "") for k, v in self._ai_learned_rules.items()} if self._ai_learned_rules else {},
                "avoid_long": list(self._ai_avoid_long),
                "avoid_short": list(self._ai_avoid_short),
                "open_positions": len(self.positions),
                "total_capital": self.total_capital,
                "daily_pnl": round(self.daily_pnl, 2),
                "strategy_capitals": {s.name: round(self.strategy_stats[s.name]["capital"], 0) for s in self.strategies},
            }

            await asyncio.to_thread(
                lambda: self.review_repo.table.insert({
                    "review_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "period": "ai_action_log",
                    "summary": summary[:3000],
                    "strategy_insights": json.dumps(state),
                    "suggestions": json.dumps(actions[-20:], default=str),  # Last 20 actions
                    "model_used": "system",
                    "tokens_used": 0,
                }).execute()
            )
            logger.info(f"📝 [AI Log] Flushed {len(actions)} actions to DB")
        except Exception as e:
            logger.warning(f"📝 [AI Log] Flush failed: {e}")

    async def _save_market_snapshot(self, collector):
        """Save market data snapshot to DB every 5 minutes for AI learning.

        Stores derivatives data (funding, OI, taker ratios, top trader positions),
        external data (news, BTC dominance, DXY proxy, Fear & Greed), and whale data.
        All from free APIs. Stored in ai_reviews with period='market_snapshot'.
        """
        try:
            import json

            # Fetch BTC derivatives (already fetched in main loop, but snapshot may be stale)
            btc_derivs = await collector.get_derivatives_data("BTCUSDT")

            # Fetch new external data sources (all free)
            news, btc_dom, dxy, funding_hist = await asyncio.gather(
                collector.get_crypto_news(5),
                collector.get_btc_dominance(),
                collector.get_macro_dxy(),
                collector.get_funding_rate_history("BTCUSDT", 3),
                return_exceptions=True,
            )

            # Build snapshot
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Derivatives
                "btc_funding_rate": btc_derivs.get("funding_rate", [{}])[0].get("rate", 0) if btc_derivs.get("funding_rate") else 0,
                "btc_open_interest": btc_derivs.get("open_interest", {}).get("openInterest", 0),
                "btc_taker_ratio": btc_derivs.get("taker_ratio_15m", [{}])[0].get("buy_sell_ratio", 1.0) if btc_derivs.get("taker_ratio_15m") else 1.0,
                "btc_top_long_pct": btc_derivs.get("top_long_short", [{}])[0].get("long_account", 0.5) if btc_derivs.get("top_long_short") else 0.5,
                "btc_global_long_pct": btc_derivs.get("global_long_short", [{}])[0].get("long_account", 0.5) if btc_derivs.get("global_long_short") else 0.5,
                # External - handle exceptions from gather
                "fear_greed": self._fear_greed_cache.get("value", 50),
                "fear_greed_class": self._fear_greed_cache.get("classification", "Neutral"),
                "btc_dominance": btc_dom.get("btc_dominance", 0) if not isinstance(btc_dom, Exception) else 0,
                "market_cap_change_24h": btc_dom.get("market_cap_change_24h", 0) if not isinstance(btc_dom, Exception) else 0,
                "dxy_approx": dxy.get("dxy_approx", 100) if not isinstance(dxy, Exception) else 100,
                "eur_usd": dxy.get("eur_usd", 1.0) if not isinstance(dxy, Exception) else 1.0,
                # News headlines
                "news_headlines": [n.get("title", "") for n in news[:5]] if not isinstance(news, Exception) else [],
                # Funding rate history (last 3 = 24h)
                "funding_history": funding_hist if not isinstance(funding_hist, Exception) else [],
                # Whale data
                "whale_consensus": {k: v.get("consensus", "neutral") for k, v in self._whale_consensus_cache.items()} if self._whale_consensus_cache else {},
                # AI state
                "ai_regime": self._ai_regime,
            }

            await asyncio.to_thread(
                lambda: self.review_repo.table.insert({
                    "review_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "period": "market_snapshot",
                    "summary": f"BTC funding={snapshot['btc_funding_rate']:.4%} | OI={snapshot['btc_open_interest']} | "
                               f"taker={snapshot['btc_taker_ratio']:.3f} | F&G={snapshot['fear_greed']} | "
                               f"BTC.D={snapshot['btc_dominance']:.1f}% | DXY≈{snapshot['dxy_approx']}",
                    "strategy_insights": json.dumps(snapshot, default=str),
                    "suggestions": json.dumps(snapshot.get("news_headlines", []), default=str),
                    "model_used": "data_collector",
                    "tokens_used": 0,
                }).execute()
            )
            logger.debug(f"📸 [Snapshot] Market data saved (F&G={snapshot['fear_greed']}, BTC.D={snapshot['btc_dominance']:.1f}%)")

        except Exception as e:
            logger.warning(f"📸 [Snapshot] Failed to save market data: {e}")

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
        whale_tracker = HyperliquidWhaleTracker()

        # Shared state for data fetched once per cycle (not per symbol)
        self._fear_greed_cache = {"value": 50, "classification": "Neutral", "last_fetch": None}
        self._whale_consensus_cache = {}  # {symbol: {consensus, long_count, short_count, ...}}
        # v7.0.2: AI Intelligence Suite timers and state
        self._last_rebalance_time = datetime.now(timezone.utc)
        self._last_regime_check = datetime.min.replace(tzinfo=timezone.utc)
        self._last_pattern_learn = datetime.min.replace(tzinfo=timezone.utc)
        self._last_param_tune = datetime.min.replace(tzinfo=timezone.utc)
        self._last_symbol_select = datetime.min.replace(tzinfo=timezone.utc)
        self._ai_regime = {"regime": "unknown", "confidence": 0, "risk_level": "medium", "bias": "neutral"}
        self._ai_learned_rules = {}  # {strategy_name: {avoid_symbols, min_vol_ratio, ...}}
        self._ai_avoid_long = set()   # Symbols AI says to avoid for longs
        self._ai_avoid_short = set()  # Symbols AI says to avoid for shorts
        self._ai_action_log = []  # Buffer of AI actions to flush to DB every 15min
        self._last_ai_log_flush = datetime.min.replace(tzinfo=timezone.utc)
        self._last_market_snapshot = datetime.min.replace(tzinfo=timezone.utc)  # v7.0.3: market data storage
        self._market_data_cache = {}  # Store last fetched market data for snapshot

        try:
            while self.running:
                self._reversal_counted_this_cycle.clear()

                # Fetch shared data once per cycle (not per symbol)
                has_smart_money = any(s.strategy_type == "smart_money" for s in self.strategies)
                if has_smart_money:
                    try:
                        self._whale_consensus_cache = await whale_tracker.get_whale_consensus()
                    except Exception as e:
                        logger.warning(f"Whale tracker failed: {e}")
                    # Fear & Greed: refresh every 15 minutes
                    now_ts = datetime.now(timezone.utc)
                    last_fg = self._fear_greed_cache.get("last_fetch")
                    if not last_fg or (now_ts - last_fg).total_seconds() > 900:
                        try:
                            fg = await collector.get_fear_greed_index()
                            self._fear_greed_cache.update(fg)
                            self._fear_greed_cache["last_fetch"] = now_ts
                        except Exception as e:
                            logger.warning(f"Fear & Greed fetch failed: {e}")

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

                # === v7.1: Simplified AI — observe and collect only ===
                now_utc = datetime.now(timezone.utc)

                # Every 15min: AI regime detection (logging only, no entry blocking)
                if (now_utc - self._last_regime_check).total_seconds() >= 900:
                    await self._flush_ai_action_log()
                    btc_data = await self._fetch_market_data("BTCUSDT", collector)
                    if btc_data:
                        await self._ai_regime_check(btc_data)
                    self._last_regime_check = now_utc

                # Every 5min: Save market data snapshot to DB for future analysis
                if (now_utc - self._last_market_snapshot).total_seconds() >= 300:
                    await self._save_market_snapshot(collector)
                    self._last_market_snapshot = now_utc

                # REMOVED: AI capital rebalance (was starving winners, feeding churners)
                # REMOVED: AI symbol selection (was blocking BTC/ETH longs during rallies)
                # REMOVED: AI parameter tuning (was widening SL from old data, amplifying losses)
                # REMOVED: AI pattern learning entry blocks (was blocking crash_momentum on AVAX)
                # Capital allocation is now FIXED at startup. Strategies manage their own trades.

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
            needs_derivatives = any(s.strategy_type in ("funding", "oi", "flow", "regime_short", "failed_bkout_short", "refined_cascade", "smart_money") for s in self.strategies)
            needs_4h = any(s.strategy_type in ("regime_short", "failed_bkout_short", "refined_cascade", "crash_momentum", "rsi_momentum", "bb_squeeze") for s in self.strategies)

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
                    conflict_detail = htf_trend.get("conflict_detail", "suppressed")
                    logger.info(f"⚡ [{symbol}] HTF conflict: {conflict_detail}")
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
            elif strategy.strategy_type == "uptrend_pullback":
                signal_result = strategy.generator.generate_signal(
                    df_1h, htf_trend,
                )
            elif strategy.strategy_type == "rsi_momentum":
                df_4h = market_data.get("4h", pd.DataFrame())
                signal_result = strategy.generator.generate_signal(
                    df_1h, htf_trend, df_4h=df_4h,
                )
            elif strategy.strategy_type == "bb_squeeze":
                df_4h = market_data.get("4h", pd.DataFrame())
                signal_result = strategy.generator.generate_signal(
                    df_4h, htf_trend,
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
            elif strategy.strategy_type == "smart_money":
                signal_result = strategy.generator.generate_signal(
                    market_data.get("15m", pd.DataFrame()), htf_trend,
                    derivatives=market_data.get("derivatives"),
                    whale_consensus=self._whale_consensus_cache.get(symbol),
                    fear_greed=self._fear_greed_cache,
                )
            else:
                signal_result = {"signal": "hold", "confidence": 0.5,
                                 "reasoning": f"Unknown strategy type: {strategy.strategy_type}"}

            # Determine timeframe label for DB
            tf_map = {"funding": "1h", "uptrend_pullback": "1h", "rsi_momentum": "1h", "bb_squeeze": "4h",
                      "breakout": "15m", "pullback": "15m", "flow": "15m",
                      "regime_short": "15m", "failed_bkout_short": "15m", "refined_cascade": "15m",
                      "crash_momentum": "1h", "smart_money": "15m"}
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
                await self._check_entry(pos_key, symbol, signal_result, current_price, strategy, atr_value, htf_trend)

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
                           strategy: StrategyConfig, atr_value: float, htf_trend: dict = None):
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

        # === v7.0: Daily loss limit ===
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_pnl_reset_date:
            self.daily_pnl = 0.0
            self.daily_loss_halt = False
            self.daily_pnl_reset_date = today
        if self.daily_loss_halt:
            return

        # Check per-strategy circuit breaker
        cb_until = self.circuit_breaker_until.get(strategy.name)
        if cb_until:
            remaining = (cb_until - datetime.now(timezone.utc)).total_seconds() / 60
            if remaining > 0:
                logger.info(f"   🔌 [{strategy.name}] Circuit breaker active ({remaining:.0f}min left)")
                return
            else:
                logger.info(f"   🔌 [{strategy.name}] Circuit breaker lifted — resuming")
                del self.circuit_breaker_until[strategy.name]

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

        # === v7.0: Portfolio-level risk gates ===
        if not self._check_portfolio_exposure(proposed_side):
            logger.info(f"   [{strategy.name}] {symbol}: Max {proposed_side} positions reached ({self.max_long_positions if proposed_side == 'long' else self.max_short_positions})")
            return

        if not self._check_correlation_limit(symbol, proposed_side):
            logger.info(f"   [{strategy.name}] {symbol}: Correlation group limit reached for {proposed_side}")
            return

        if not self._check_portfolio_heat(strategy):
            heat = self._calculate_portfolio_heat()
            logger.info(f"   [{strategy.name}] {symbol}: Portfolio heat limit ({heat:.1%} >= {self.max_portfolio_heat_pct:.0%})")
            return

        # === v7.1: AI gates REMOVED ===
        # AI symbol avoidance, learned rules, and regime entry blocking all REMOVED.
        # These were blocking BTC/ETH longs during a 4.2% rally, starving strategies
        # of capital, and paralyzing the system to 4 trades/day.
        # AI now only: detects regime (for logging), collects data, and does post-trade analysis.
        # The strategies themselves handle all entry/exit decisions.

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

        # Check directional SL guard: block if too many recent SLs in same direction
        dir_key = f"{strategy.name}:{proposed_side}"
        if dir_key in self.directional_sl_history:
            cutoff = now - timedelta(hours=self.directional_sl_window_hours)
            recent_sls = [t for t in self.directional_sl_history[dir_key] if t > cutoff]
            self.directional_sl_history[dir_key] = recent_sls  # prune old entries
            if len(recent_sls) >= self.directional_sl_max:
                logger.info(f"   [{strategy.name}] {symbol}: Directional SL guard — {len(recent_sls)} {proposed_side} SLs in {self.directional_sl_window_hours}h, skipping")
                return

        # Check max concurrent positions for this strategy (across all symbols)
        if strategy.max_concurrent_positions > 0:
            open_count = sum(1 for k in self.positions if k.startswith(f"{strategy.name}:"))
            if open_count >= strategy.max_concurrent_positions:
                logger.info(f"   [{strategy.name}] {symbol}: Max concurrent positions reached ({open_count}/{strategy.max_concurrent_positions})")
                return

        # AI signal gate: evaluate signal before execution
        if self.ai_signal_gate_enabled and self.ai_analyzer:
            try:
                direction = "long" if signal["signal"] in ["buy", "strong_buy"] else "short"
                open_pos_list = [
                    {
                        "side": v.get("side", "unknown"),
                        "symbol": self._parse_pos_key(k)[1],
                        "quantity": v.get("quantity", 0),
                        "entry_price": v.get("entry_price", 0),
                        "unrealized_pnl": 0,
                    }
                    for k, v in self.positions.items()
                ]

                # Get recent trades for context
                recent = await asyncio.to_thread(
                    lambda: self.trade_repo.table.select("*")
                    .eq("strategy_name", strategy.name)
                    .order("entry_time", desc=True)
                    .limit(20)
                    .execute()
                )
                recent_trades = recent.data or []

                evaluation = await self.ai_analyzer.evaluate_signal(
                    strategy_name=strategy.name,
                    symbol=symbol,
                    direction=direction,
                    confidence=signal["confidence"],
                    signal_reasoning=signal.get("reasoning", ""),
                    indicators=signal.get("indicators", {}),
                    recent_trades=recent_trades,
                    open_positions=open_pos_list,
                )
                if evaluation:
                    signal["original_confidence"] = signal["confidence"]
                    signal["confidence"] = evaluation.adjusted_confidence
                    signal["ai_reasoning"] = evaluation.reasoning
                    signal["ai_risk_flags"] = evaluation.risk_flags
                    # Re-check confidence threshold after AI adjustment
                    if signal["confidence"] < 0.65:
                        logger.info(
                            f"   [{strategy.name}] {symbol}: AI gate BLOCKED "
                            f"({signal['original_confidence']:.2f} -> {signal['confidence']:.2f}): "
                            f"{evaluation.reasoning}"
                        )
                        return
            except Exception as e:
                logger.debug(f"[AI] Signal gate error (proceeding with original): {e}")

        if signal["signal"] in ["buy", "strong_buy"]:
            await self._open_position(pos_key, symbol, "long", price, signal, strategy, atr_value, htf_trend)
        elif signal["signal"] in ["sell", "strong_sell"]:
            await self._open_position(pos_key, symbol, "short", price, signal, strategy, atr_value, htf_trend)

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
            # Track directional SL for consecutive-loss guard
            dir_key = f"{strategy.name}:{position.get('side', 'unknown')}"
            if dir_key not in self.directional_sl_history:
                self.directional_sl_history[dir_key] = []
            self.directional_sl_history[dir_key].append(datetime.now(timezone.utc))
            # Activate per-strategy cooldown
            self.stop_loss_cooldowns[pos_key] = datetime.now(timezone.utc) + timedelta(minutes=self.stop_loss_cooldown_minutes)
            # Activate cross-strategy symbol cooldown
            self.symbol_sl_cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(minutes=self.symbol_sl_cooldown_minutes)
            self.daily_stop_losses[pos_key] = self.daily_stop_losses.get(pos_key, 0) + 1
            logger.info(f"   [{strategy.name}] SL cooldown activated. Daily: {self.daily_stop_losses[pos_key]}/{self.max_daily_stop_losses}")

            # Per-strategy circuit breaker: track SL timestamps per strategy
            now = datetime.now(timezone.utc)
            strat_name = strategy.name
            if strat_name not in self.strategy_sl_timestamps:
                self.strategy_sl_timestamps[strat_name] = []
            self.strategy_sl_timestamps[strat_name].append(now)
            cutoff = now - timedelta(hours=self.circuit_breaker_window_hours)
            self.strategy_sl_timestamps[strat_name] = [t for t in self.strategy_sl_timestamps[strat_name] if t > cutoff]
            if len(self.strategy_sl_timestamps[strat_name]) >= self.circuit_breaker_threshold:
                self.circuit_breaker_until[strat_name] = now + timedelta(minutes=self.circuit_breaker_pause_minutes)
                logger.warning(
                    f"🔌 CIRCUIT BREAKER [{strat_name}]: {len(self.strategy_sl_timestamps[strat_name])} SLs in "
                    f"{self.circuit_breaker_window_hours}h — {strat_name} paused for {self.circuit_breaker_pause_minutes}min"
                )

        # 3.5. EARLY MOMENTUM CHECK — crash_momentum: close if no move after 30min
        # AI analysis: 11/30 trades closed as stale/session-end with tiny PnL.
        # Real winners move >0.3% within 30min; if it hasn't moved, exit early.
        elif strategy.strategy_type == "crash_momentum":
            mins_open = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds() / 60
            if 30 <= mins_open <= 35 and abs(pnl_pct) < 0.003:
                should_exit = True
                exit_reason = f"No momentum ({pnl_pct:.2%} after {mins_open:.0f}min, need ±0.3%)"

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

        # 7. AI EXIT OPTIMIZATION — DISABLED
        # Was closing profitable positions before they could run. Killed the only
        # trend_breakout long 10 hours before a +4.2% BTC rally. Let strategies
        # manage their own exits via SL/TP/trailing/stale mechanisms.

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

    def _count_agreeing_strategies(self, symbol: str, side: str) -> int:
        """Count how many strategies have open positions on this symbol in the same direction."""
        count = 0
        for pos_key, pos in self.positions.items():
            if pos_key.endswith(f":{symbol}") and pos.get("side") == side:
                count += 1
        return count

    def _check_regime_alignment(self, side: str, htf_trend: dict | None) -> bool:
        """Check if trade direction aligns with HTF regime."""
        if not htf_trend:
            return True  # Assume aligned if no data
        direction = htf_trend.get("direction", "neutral")
        if side == "long" and direction == "bullish":
            return True
        if side == "short" and direction == "bearish":
            return True
        if direction == "neutral":
            return True  # Neutral doesn't penalize
        return False

    def _get_effective_risk_pct(self, strategy: StrategyConfig, signal: dict,
                                 symbol: str, side: str, htf_trend: dict | None) -> float:
        """Calculate effective risk percentage using Kelly criterion (v7.0)."""
        if not strategy.adaptive_sizing:
            return strategy.risk_per_trade_pct

        stats = self.strategy_stats[strategy.name]
        effective_risk = calculate_kelly_risk_pct(
            base_risk_pct=strategy.risk_per_trade_pct,
            recent_results=stats["recent_results"],
            min_trades=15,
            kelly_fraction=0.25,
            min_risk_pct=strategy.min_risk_pct,
            max_risk_pct=strategy.max_risk_pct,
        )
        # Log Kelly sizing info
        n = len(stats["recent_results"])
        wins = sum(1 for r in stats["recent_results"] if r > 0)
        wr = wins / n * 100 if n > 0 else 0
        logger.info(
            f"   [{strategy.name}] Kelly risk: {strategy.risk_per_trade_pct:.1%} → {effective_risk:.1%} "
            f"(WR={wr:.0f}% over {n} trades)"
        )
        return effective_risk

    async def _open_position(self, pos_key: str, symbol: str, side: str, price: float,
                              signal: dict, strategy: StrategyConfig, atr_value: float,
                              htf_trend: dict = None):
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
        effective_risk = self._get_effective_risk_pct(strategy, signal, symbol, side, htf_trend)
        # v7.0: Volatility-adjusted sizing — size smaller when market is volatile
        atr_pct = atr_value / price if price > 0 else 0
        effective_risk = calculate_vol_adjusted_risk(effective_risk, atr_pct)
        quantity, margin = calculate_position_size(
            capital=available_capital,
            risk_pct=effective_risk,
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
            indicators["effective_risk_pct"] = effective_risk

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
        # v7.0: Track daily PnL for daily loss limit
        self.daily_pnl += pnl
        if self.initial_capital > 0 and self.daily_pnl / self.initial_capital < -self.daily_loss_limit_pct:
            self.daily_loss_halt = True
            logger.warning(f"🛑 DAILY LOSS LIMIT: ${self.daily_pnl:.2f} ({self.daily_pnl/self.initial_capital:.1%}). No new entries today.")
        stats["trade_count"] += 1
        if pnl > 0:
            stats["winning_trades"] += 1
            stats["consecutive_wins"] += 1
            stats["consecutive_losses"] = 0
        else:
            stats["consecutive_losses"] += 1
            stats["consecutive_wins"] = 0
        stats["recent_results"].append(pnl)
        if len(stats["recent_results"]) > 30:  # v7.0: increased from 20 for Kelly sizing
            stats["recent_results"].pop(0)

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
            exit_time_str = datetime.now(timezone.utc).isoformat()
            exit_data = {
                "exit_price": price,
                "exit_time": exit_time_str,
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

            # Fire-and-forget AI post-trade analysis
            if self.ai_post_trade_enabled and self.ai_analyzer:
                asyncio.create_task(self._ai_analyze_trade(
                    position_id=position_id,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    side=position.get("side", "unknown"),
                    entry_price=position["entry_price"],
                    exit_price=price,
                    entry_time=position["entry_time"].isoformat(),
                    exit_time=exit_time_str,
                    pnl=pnl,
                    pnl_pct=pnl_pct * self.leverage * 100,  # ROE %
                    duration_seconds=int(duration),
                    exit_reason=reason,
                    entry_indicators=position.get("signal", {}).get("indicators", {}),
                    strategy_stats=dict(stats),
                ))
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

    async def _ai_analyze_trade(self, **kwargs):
        """Run AI post-trade analysis in background. Errors are logged, never raised."""
        try:
            # Collect active positions snapshot for context
            active_positions = [
                {
                    "strategy": self._parse_pos_key(k)[0],
                    "symbol": self._parse_pos_key(k)[1],
                    "side": v.get("side", "unknown"),
                    "entry_price": v.get("entry_price", 0),
                }
                for k, v in self.positions.items()
            ]

            analysis = await self.ai_analyzer.analyze_trade(
                active_positions=active_positions,
                **kwargs,
            )

            if analysis:
                await asyncio.to_thread(
                    lambda: self.analysis_repo.table.insert({
                        "position_id": analysis.position_id,
                        "analysis_text": analysis.analysis_text,
                        "patterns_identified": analysis.patterns_identified,
                        "suggestion": analysis.suggestion,
                        "model_used": analysis.model_used,
                        "tokens_used": analysis.tokens_used,
                    }).execute()
                )
        except Exception as e:
            logger.debug(f"[AI] Post-trade analysis error: {e}")

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
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"],
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
        default=["funding_reversion", "uptrend_pullback", "rsi_momentum", "bb_squeeze", "order_flow", "smart_money"],
        choices=["funding_reversion", "uptrend_pullback", "rsi_momentum", "bb_squeeze",
                 "trend_breakout", "trend_pullback", "order_flow",
                 "regime_short", "failed_breakout_short", "refined_liq_cascade", "crash_momentum",
                 "smart_money"],
        help="Strategies to run (default: 4 — proven pullback + unproven derivatives)"
    )
    parser.add_argument(
        "--adaptive-sizing", action="store_true", default=False,
        help="Enable adaptive position sizing based on confidence/performance/regime"
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

    # v7.1: FIXED capital allocation. No more AI rebalancing.
    # AI rebalancer gave crash_momentum $2,361 and trend_breakout $728 — the opposite
    # of what works. Capital allocation is now locked based on 30-day proven PnL/hr.
    # Shared pool still lets idle capital flow to active strategies.
    # v8.0: 4 strategies based on 6-month backtest evidence.
    # uptrend_pullback: PROVEN +$439/6mo, 44.7% WR — gets the most capital.
    # order_flow: unproven (needs live derivatives data), +$61/30d live — promising.
    # smart_money: unproven, +$6/30d live — whale/sentiment composite.
    # funding_reversion: rare event, idle capital flows to others via shared pool.
    # All others DISABLED — backtest-proven losers.
    # v8.0: Capital based on 6-month backtest results.
    # 3 proven strategies + 3 unproven with live-data edge.
    # Shared pool lets idle capital flow to active strategies.
    capital_allocation = {
        "funding_reversion": base_capital * 0.75,      # $750 — rare event, idle capital shared
        "uptrend_pullback": base_capital * 1.25,       # $1250 — proven +$439/6mo, 44.7% WR
        "rsi_momentum": base_capital * 1.5,            # $1500 — proven +$1302/6mo, 64.3% WR
        "bb_squeeze": base_capital * 1.5,              # $1500 — proven +$1358/6mo, 82.4% WR
        "trend_breakout": base_capital * 0.5,          # disabled by default
        "trend_pullback": base_capital * 0.5,          # disabled by default
        "order_flow": base_capital * 1.0,              # $1000 — promising +$61/30d live
        "regime_short": base_capital * 0.5,            # disabled
        "failed_breakout_short": base_capital * 0.5,   # disabled
        "refined_liq_cascade": base_capital * 0.5,     # disabled
        "crash_momentum": base_capital * 0.5,          # disabled
        "smart_money": base_capital * 1.0,             # $1000 — whale/sentiment
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
        elif name == "uptrend_pullback":
            strategy_configs.append(StrategyConfig(
                name="uptrend_pullback",
                strategy_type="uptrend_pullback",
                generator=UptrendPullbackGenerator(),
                sl_atr_mult=1.5,       # Tight SL below SMA20
                tp_atr_mult=2.5,       # 1.67:1 R:R (backtest-optimized)
                trailing_atr_mult=2.0,
                trailing_dist_atr_mult=1.0,
                max_position_hours=8.0,
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="1h",    # Uses 1h candles (not 15m)
                trailing_enabled=False,
                max_concurrent_positions=1,
            ))
        elif name == "rsi_momentum":
            strategy_configs.append(StrategyConfig(
                name="rsi_momentum",
                strategy_type="rsi_momentum",
                generator=RSIMomentumGenerator(),
                sl_atr_mult=2.0,       # Wider SL — RSI dips can be volatile
                tp_atr_mult=2.5,       # 1.25:1 R:R (compensated by 64% WR)
                trailing_atr_mult=2.0,
                trailing_dist_atr_mult=1.0,
                max_position_hours=8.0,
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="1h",
                trailing_enabled=False,
                max_concurrent_positions=1,
            ))
        elif name == "bb_squeeze":
            strategy_configs.append(StrategyConfig(
                name="bb_squeeze",
                strategy_type="bb_squeeze",
                generator=BollingerSqueezeGenerator(),
                sl_atr_mult=2.0,       # Wide SL — squeeze breakouts need room
                tp_atr_mult=3.0,       # 1.5:1 R:R (82% WR = can afford wider TP)
                trailing_atr_mult=2.5,
                trailing_dist_atr_mult=1.0,
                max_position_hours=12.0,  # Longer hold — 4H timeframe
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="1h",    # ATR from 1h for finer granularity
                trailing_enabled=False,
                max_concurrent_positions=1,
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
                max_concurrent_positions=1,  # v7.0.3: 1 at a time — prevents correlated blowups (2 breakout SLs = $124 loss)
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
                max_concurrent_positions=1,  # v7.0.3: 1 at a time — same as breakout
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
                max_concurrent_positions=2,  # Crypto correlated — cap exposure
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
                max_concurrent_positions=2,  # Crypto correlated — cap like crash_momentum
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
                tp_atr_mult=3.0,       # 1.5:1 R:R (was 4.0 — too many stale exits before TP hit)
                trailing_atr_mult=2.5, # Start trailing at 2.5x ATR (was 2.0 — let crash profits run)
                trailing_dist_atr_mult=1.5,  # Trail at 1.5x ATR (was 1.0 — wider for crash volatility)
                max_position_hours=12.0,  # Time stop: 12h (1h candles need more time)
                risk_per_trade_pct=0.015,  # 1.5% risk
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="1h",
                trailing_enabled=True,
                max_concurrent_positions=2,  # Crypto is correlated — cap exposure
            ))

        elif name == "smart_money":
            strategy_configs.append(StrategyConfig(
                name="smart_money",
                strategy_type="smart_money",
                generator=SmartMoneyFlowGenerator(),
                sl_atr_mult=2.0,       # Wider SL — composite signals need room
                tp_atr_mult=3.0,       # 1.5:1 R:R
                trailing_atr_mult=2.5,
                trailing_dist_atr_mult=1.0,
                max_position_hours=8.0,  # Time stop: 8h
                risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m",
                trailing_enabled=True,
                max_concurrent_positions=1,  # v7.0.3: 1 at a time
            ))

    # Apply adaptive sizing if enabled
    if args.adaptive_sizing:
        for cfg in strategy_configs:
            cfg.adaptive_sizing = True
        logger.info("Adaptive position sizing ENABLED (risk range: 0.5% – 3.0%)")

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
