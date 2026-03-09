#!/usr/bin/env python
"""Multi-strategy paper trading script using Binance real-time data and Supabase.

v5.0 — Derivatives-Data Signal Strategies:
- funding_sentiment: Funding rate + OI divergence + basis scoring
- volatility_squeeze: Bollinger Band squeeze + Keltner breakout on 4h candles
- taker_flow: Taker volume + order book imbalance + L/S ratio
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


def determine_htf_trend(df_1h: pd.DataFrame) -> str:
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


# =============================================================================
# Strategy 1: Funding Sentiment (Funding Rate + OI Divergence + Basis)
# =============================================================================

class FundingSentimentGenerator:
    """Generate signals from funding rate, open interest divergence, and basis.

    Funding rate = cost of leverage. Extreme funding precedes reversals.
    OI divergence (OI rising while price falls) = leading indicator of regime change.
    """

    def __init__(self):
        # Cache last known funding rate (only updates every 8h)
        self._last_funding_rate: float | None = None
        self._last_funding_time: str | None = None

    def generate_signal(self, df: pd.DataFrame, htf_trend: str = "neutral",
                        derivatives: dict = None, **kwargs) -> dict:
        """Generate trading signal from derivatives sentiment data."""
        if derivatives is None or not derivatives:
            return {"signal": "hold", "confidence": 0.5, "reasoning": "No derivatives data",
                    "indicators": {"htf_trend": htf_trend}}

        prices = df["close"] if not df.empty else pd.Series()
        current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0

        total_score = 0.0
        reasons = []

        # === FUNDING RATE SCORE (max ±3.0) ===
        funding_score = self._score_funding(derivatives, reasons)
        total_score += funding_score

        # === OI DIVERGENCE SCORE (max ±3.0) ===
        oi_score = self._score_oi_divergence(derivatives, prices, reasons)
        total_score += oi_score

        # === BASIS SCORE (max ±2.0) ===
        basis_score = self._score_basis(derivatives, reasons)
        total_score += basis_score

        # Determine signal
        signal_type = "hold"
        confidence = 0.5

        if total_score >= 4.0:
            signal_type = "strong_buy"
            confidence = min(0.85, 0.60 + abs(total_score) * 0.03)
        elif total_score >= 2.5:
            signal_type = "buy"
            confidence = min(0.75, 0.55 + abs(total_score) * 0.03)
        elif total_score <= -4.0:
            signal_type = "strong_sell"
            confidence = min(0.85, 0.60 + abs(total_score) * 0.03)
        elif total_score <= -2.5:
            signal_type = "sell"
            confidence = min(0.75, 0.55 + abs(total_score) * 0.03)

        # === HTF TREND HARD FILTER ===
        if htf_trend == "bearish" and signal_type in ("buy", "strong_buy"):
            reasons = [f"HTF bearish - blocked {signal_type}"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend != "bearish" and signal_type in ("sell", "strong_sell"):
            reasons = [f"HTF {htf_trend} - blocked {signal_type} (need bearish)"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend == "neutral" and signal_type in ("buy", "strong_buy"):
            confidence = max(0.5, confidence - 0.10)
            reasons.append("HTF neutral (-10% conf)")

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons) if reasons else "No strong signals",
            "indicators": {
                "price": current_price,
                "funding_score": funding_score,
                "oi_score": oi_score,
                "basis_score": basis_score,
                "total_score": total_score,
                "htf_trend": htf_trend,
            },
        }

    def _score_funding(self, derivatives: dict, reasons: list) -> float:
        """Score funding rate (max ±3.0). Contrarian: extreme funding → reversal."""
        score = 0.0
        funding_data = derivatives.get("funding_rate", [])
        if not funding_data:
            return 0.0

        # Update cache if we have new data
        latest = funding_data[-1]
        rate = latest["funding_rate"]
        funding_time = latest["funding_time"]

        if self._last_funding_time != funding_time:
            self._last_funding_rate = rate
            self._last_funding_time = funding_time
        else:
            rate = self._last_funding_rate if self._last_funding_rate is not None else rate

        # Extreme positive funding → contrarian short
        if rate > 0.0005:  # >0.05%
            score = -2.0
            reasons.append(f"Funding extreme+ ({rate*100:.3f}%) → short")
        elif rate > 0.0003:  # >0.03%
            score = -1.0
            reasons.append(f"Funding high ({rate*100:.3f}%)")
        # Extreme negative funding → contrarian long
        elif rate < -0.0003:  # <-0.03%
            score = 2.0
            reasons.append(f"Funding extreme- ({rate*100:.3f}%) → long")
        elif rate < -0.0001:  # <-0.01%
            score = 1.0
            reasons.append(f"Funding negative ({rate*100:.3f}%)")

        # Funding acceleration: compare recent rates
        if len(funding_data) >= 2:
            prev_rate = funding_data[-2]["funding_rate"]
            acceleration = rate - prev_rate
            if abs(acceleration) > 0.0002:  # >0.02% change
                accel_factor = 1.0 if (acceleration > 0 and score < 0) or (acceleration < 0 and score > 0) else 0.5
                score *= (1 + accel_factor * 0.5)
                score = max(-3.0, min(3.0, score))

        return score

    def _score_oi_divergence(self, derivatives: dict, prices: pd.Series, reasons: list) -> float:
        """Score OI divergence (max ±3.0). OI vs price divergence = leading signal."""
        score = 0.0
        oi_history = derivatives.get("oi_history", [])
        if len(oi_history) < 2 or len(prices) < 2:
            return 0.0

        # Current vs earlier OI
        current_oi = oi_history[-1]["sum_open_interest_value"]
        earlier_oi = oi_history[0]["sum_open_interest_value"]
        oi_change_pct = (current_oi - earlier_oi) / earlier_oi * 100 if earlier_oi > 0 else 0

        # Price change over same period
        price_now = float(prices.iloc[-1])
        price_earlier = float(prices.iloc[0]) if len(prices) > len(oi_history) else float(prices.iloc[-min(len(prices), len(oi_history))])
        price_change_pct = (price_now - price_earlier) / price_earlier * 100 if price_earlier > 0 else 0

        # OI↑ + price↓ = shorts building (bearish)
        if oi_change_pct > 2.0 and price_change_pct < -0.5:
            score = -2.0
            reasons.append(f"OI↑{oi_change_pct:.1f}% + price↓{price_change_pct:.1f}% (shorts building)")

        # OI↓ + price↑ = short squeeze (bullish)
        elif oi_change_pct < -2.0 and price_change_pct > 0.5:
            score = 2.0
            reasons.append(f"OI↓{oi_change_pct:.1f}% + price↑{price_change_pct:.1f}% (short squeeze)")

        # OI↑ + price↑ = trend confirmation (bullish)
        elif oi_change_pct > 2.0 and price_change_pct > 0.5:
            score = 1.5
            reasons.append(f"OI↑{oi_change_pct:.1f}% + price↑{price_change_pct:.1f}% (trend confirm)")

        # OI↑ + price flat = accumulation (mild bullish)
        elif oi_change_pct > 3.0 and abs(price_change_pct) < 0.5:
            score = 0.5
            reasons.append(f"OI↑{oi_change_pct:.1f}% + flat price (accumulation)")

        # OI elevated vs 7d avg amplifies
        if len(oi_history) >= 12:  # ~1h of 5m data minimum
            avg_oi = sum(r["sum_open_interest_value"] for r in oi_history) / len(oi_history)
            if current_oi > avg_oi * 1.1:  # 10%+ above average
                score *= 1.3
                score = max(-3.0, min(3.0, score))

        return score

    def _score_basis(self, derivatives: dict, reasons: list) -> float:
        """Score futures basis (max ±2.0). Premium = bullish, discount = bearish."""
        score = 0.0
        premium = derivatives.get("premium_index", {})
        if not premium:
            return 0.0

        mark_price = premium.get("mark_price", 0)
        index_price = premium.get("index_price", 0)
        if index_price <= 0 or mark_price <= 0:
            return 0.0

        basis_pct = (mark_price - index_price) / index_price * 100

        # Futures premium > 0.2% = bullish
        if basis_pct > 0.5:
            score = -1.0  # Contrarian: extreme premium → short
            reasons.append(f"Basis +{basis_pct:.3f}% (extreme premium → contrarian)")
        elif basis_pct > 0.2:
            score = 1.0
            reasons.append(f"Basis +{basis_pct:.3f}% (bullish premium)")

        # Futures discount < -0.1% = bearish
        elif basis_pct < -0.3:
            score = 1.0  # Contrarian: extreme discount → long
            reasons.append(f"Basis {basis_pct:.3f}% (extreme discount → contrarian)")
        elif basis_pct < -0.1:
            score = -1.0
            reasons.append(f"Basis {basis_pct:.3f}% (bearish discount)")

        return score


# =============================================================================
# Strategy 2: Volatility Squeeze (BB inside KC → breakout)
# =============================================================================

class VolatilitySqueezeGenerator:
    """Generate signals from Bollinger Band squeeze + Keltner Channel breakout.

    Volatility is mean-reverting. BB inside KC = squeeze.
    Breakout from squeeze produces asymmetric payoff on 4h timeframes.
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 kc_period: int = 20, kc_atr_mult: float = 1.5):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_mult = kc_atr_mult

    def generate_signal(self, df: pd.DataFrame, htf_trend: str = "neutral",
                        df_4h: pd.DataFrame = None, df_1m: pd.DataFrame = None,
                        **kwargs) -> dict:
        """Generate signal from BB/KC squeeze on 4h candles."""
        if df_4h is None or len(df_4h) < self.bb_period + 10:
            return {"signal": "hold", "confidence": 0.5, "reasoning": "Insufficient 4h data",
                    "indicators": {"htf_trend": htf_trend}}

        # Check if last 4h candle is complete (avoid acting on incomplete candles)
        # 4h candles close at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        now = datetime.now(timezone.utc)
        last_candle_time = df_4h.index[-1]
        if hasattr(last_candle_time, 'to_pydatetime'):
            last_candle_time = last_candle_time.to_pydatetime()
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.replace(tzinfo=timezone.utc)

        # If the latest candle started less than 4h ago, it's still forming — use previous
        candle_age = (now - last_candle_time).total_seconds()
        if candle_age < 4 * 3600:
            # Use second-to-last candle as the "completed" one
            if len(df_4h) < self.bb_period + 11:
                return {"signal": "hold", "confidence": 0.5, "reasoning": "Waiting for 4h candle close",
                        "indicators": {"htf_trend": htf_trend}}
            df_4h = df_4h.iloc[:-1]

        prices = df_4h["close"]
        highs = df_4h["high"]
        lows = df_4h["low"]
        current_price = float(prices.iloc[-1])

        # Bollinger Bands
        bb_sma = prices.rolling(self.bb_period).mean()
        bb_std = prices.rolling(self.bb_period).std()
        bb_upper = bb_sma + self.bb_std * bb_std
        bb_lower = bb_sma - self.bb_std * bb_std
        bb_width = (bb_upper - bb_lower) / bb_sma

        # Keltner Channels (ATR-based)
        tr = pd.concat([
            highs - lows,
            (highs - prices.shift(1)).abs(),
            (lows - prices.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.kc_period).mean()
        kc_mid = prices.rolling(self.kc_period).mean()
        kc_upper = kc_mid + self.kc_atr_mult * atr
        kc_lower = kc_mid - self.kc_atr_mult * atr

        # Squeeze detection: BB inside KC
        in_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        squeeze_history = in_squeeze.iloc[-(self.bb_period):]

        current_squeeze = bool(in_squeeze.iloc[-1])
        prev_squeeze = bool(in_squeeze.iloc[-2]) if len(in_squeeze) > 1 else False

        total_score = 0.0
        reasons = []

        # Squeeze must exist (current or just fired) for any signal
        squeeze_bars = 0
        for val in reversed(squeeze_history.values):
            if val:
                squeeze_bars += 1
            else:
                break
        # If squeeze just fired, count the bars that WERE in squeeze
        if not current_squeeze and prev_squeeze:
            squeeze_bars = 0
            for val in reversed(squeeze_history.values[:-1]):
                if val:
                    squeeze_bars += 1
                else:
                    break

        if not current_squeeze and not prev_squeeze:
            return {
                "signal": "hold", "confidence": 0.5,
                "reasoning": "No squeeze detected",
                "indicators": {
                    "price": current_price,
                    "in_squeeze": False,
                    "squeeze_bars": 0,
                    "bb_width": float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0,
                    "htf_trend": htf_trend,
                },
            }

        # === DURATION SCORE (max 2.0) ===
        if squeeze_bars >= 6:  # 24h+ of squeeze
            total_score += 2.0
            reasons.append(f"Long squeeze ({squeeze_bars} bars)")
        elif squeeze_bars >= 3:  # 12h+
            total_score += 1.0
            reasons.append(f"Squeeze ({squeeze_bars} bars)")
        else:
            total_score += 0.5

        # === INTENSITY: BB width in bottom 25th percentile ===
        bb_w_current = float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0
        bb_w_percentile = float((bb_width.dropna() < bb_w_current).mean()) if len(bb_width.dropna()) > 0 else 0.5
        if bb_w_percentile <= 0.25:
            total_score += 1.0
            reasons.append(f"Tight squeeze (BB width {bb_w_percentile:.0%}ile)")

        # === BREAKOUT DIRECTION (max 2.0) ===
        direction = 0  # +1 bullish, -1 bearish
        squeeze_fired = prev_squeeze and not current_squeeze

        if squeeze_fired:
            # Squeeze just fired — determine direction from momentum
            total_score += 1.0  # Bonus for breakout event
            mom = float(prices.iloc[-1]) - float(prices.iloc[-3]) if len(prices) > 2 else 0
            if mom > 0:
                direction = 1
                total_score += 1.0
                reasons.append("Squeeze fired ↑ (bullish breakout)")
            else:
                direction = -1
                total_score += 1.0
                reasons.append("Squeeze fired ↓ (bearish breakout)")
        else:
            # Still loading in squeeze — use momentum direction
            sma_val = float(bb_sma.iloc[-1]) if not pd.isna(bb_sma.iloc[-1]) else current_price
            if current_price > sma_val:
                direction = 1
                reasons.append("Squeeze loading (bullish bias)")
            else:
                direction = -1
                reasons.append("Squeeze loading (bearish bias)")

        # === VOLUME CONFIRMATION (1m data) ===
        if df_1m is not None and len(df_1m) > 20 and squeeze_fired:
            vol = df_1m["volume"]
            avg_vol = float(vol.iloc[-21:-1].mean())
            current_vol = float(vol.iloc[-1])
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
            if vol_ratio >= 1.5:
                total_score += 1.5
                reasons.append(f"Volume spike {vol_ratio:.1f}x")

        # === EXPANSION RATE ===
        if len(bb_width) > 1:
            prev_bw = float(bb_width.iloc[-2]) if not pd.isna(bb_width.iloc[-2]) else 0
            if prev_bw > 0 and bb_w_current > 0:
                expansion = (bb_w_current - prev_bw) / prev_bw
                if expansion > 0.10:
                    total_score += 1.0
                    reasons.append(f"BB expanding {expansion:.0%}")

        # Determine signal
        signal_type = "hold"
        confidence = 0.5

        if direction > 0:
            if total_score >= 4.0:
                signal_type = "strong_buy"
                confidence = min(0.85, 0.60 + total_score * 0.03)
            elif total_score >= 2.5:
                signal_type = "buy"
                confidence = min(0.75, 0.55 + total_score * 0.03)
        elif direction < 0:
            if total_score >= 4.0:
                signal_type = "strong_sell"
                confidence = min(0.85, 0.60 + total_score * 0.03)
            elif total_score >= 2.5:
                signal_type = "sell"
                confidence = min(0.75, 0.55 + total_score * 0.03)

        # === HTF TREND HARD FILTER ===
        if htf_trend == "bearish" and signal_type in ("buy", "strong_buy"):
            reasons = [f"HTF bearish - blocked {signal_type}"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend != "bearish" and signal_type in ("sell", "strong_sell"):
            reasons = [f"HTF {htf_trend} - blocked {signal_type} (need bearish)"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend == "neutral" and signal_type in ("buy", "strong_buy"):
            confidence = max(0.5, confidence - 0.10)
            reasons.append("HTF neutral (-10% conf)")

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons) if reasons else "No squeeze signals",
            "indicators": {
                "price": current_price,
                "in_squeeze": current_squeeze,
                "squeeze_fired": squeeze_fired,
                "squeeze_bars": squeeze_bars,
                "bb_width": bb_w_current,
                "bb_width_percentile": bb_w_percentile,
                "direction": direction,
                "total_score": total_score,
                "htf_trend": htf_trend,
            },
        }


# =============================================================================
# Strategy 3: Taker Flow (Taker Volume + Order Book Imbalance + L/S Ratio)
# =============================================================================

class TakerFlowGenerator:
    """Generate signals from taker volume, order book imbalance, and L/S ratio.

    Taker volume = real-time aggression (who is market-buying/selling).
    Order book imbalance = passive liquidity positioning.
    This is the short-term (1m) strategy — flow data is inherently short-lived.
    """

    def generate_signal(self, df: pd.DataFrame, htf_trend: str = "neutral",
                        derivatives: dict = None, orderbook: dict = None,
                        **kwargs) -> dict:
        """Generate signal from taker flow and order book data."""
        if derivatives is None or not derivatives:
            return {"signal": "hold", "confidence": 0.5, "reasoning": "No derivatives data",
                    "indicators": {"htf_trend": htf_trend}}

        prices = df["close"] if not df.empty else pd.Series()
        current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0

        total_score = 0.0
        reasons = []

        # === TAKER RATIO SCORE (max ±3.0) ===
        taker_score = self._score_taker_ratio(derivatives, reasons)
        total_score += taker_score

        # === ORDER BOOK SCORE (max ±2.0) ===
        ob_score = self._score_orderbook(orderbook, current_price, reasons)
        total_score += ob_score

        # === LONG-SHORT RATIO SCORE (max ±1.5, contrarian) ===
        ls_score = self._score_long_short(derivatives, reasons)
        total_score += ls_score

        # Determine signal
        signal_type = "hold"
        confidence = 0.5

        if total_score >= 3.5:
            signal_type = "strong_buy"
            confidence = min(0.85, 0.60 + abs(total_score) * 0.03)
        elif total_score >= 2.0:
            signal_type = "buy"
            confidence = min(0.75, 0.55 + abs(total_score) * 0.03)
        elif total_score <= -3.5:
            signal_type = "strong_sell"
            confidence = min(0.85, 0.60 + abs(total_score) * 0.03)
        elif total_score <= -2.0:
            signal_type = "sell"
            confidence = min(0.75, 0.55 + abs(total_score) * 0.03)

        # === HTF TREND HARD FILTER ===
        if htf_trend == "bearish" and signal_type in ("buy", "strong_buy"):
            reasons = [f"HTF bearish - blocked {signal_type}"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend != "bearish" and signal_type in ("sell", "strong_sell"):
            reasons = [f"HTF {htf_trend} - blocked {signal_type} (need bearish)"] + reasons
            signal_type = "hold"
            confidence = 0.5
        elif htf_trend == "neutral" and signal_type in ("buy", "strong_buy"):
            confidence = max(0.5, confidence - 0.10)
            reasons.append("HTF neutral (-10% conf)")

        return {
            "signal": signal_type,
            "confidence": confidence,
            "reasoning": ", ".join(reasons) if reasons else "No flow signals",
            "indicators": {
                "price": current_price,
                "taker_score": taker_score,
                "ob_score": ob_score,
                "ls_score": ls_score,
                "total_score": total_score,
                "htf_trend": htf_trend,
            },
        }

    def _score_taker_ratio(self, derivatives: dict, reasons: list) -> float:
        """Score taker buy/sell ratio (max ±3.0)."""
        score = 0.0

        # 5m taker ratio (primary)
        taker_5m = derivatives.get("taker_ratio_5m", [])
        if taker_5m:
            ratio = taker_5m[0]["buy_sell_ratio"]
            if ratio > 1.3:
                score = 2.0
                reasons.append(f"5m taker ratio {ratio:.2f} (aggressive buying)")
            elif ratio > 1.1:
                score = 1.0
                reasons.append(f"5m taker ratio {ratio:.2f} (buying)")
            elif ratio < 0.77:
                score = -2.0
                reasons.append(f"5m taker ratio {ratio:.2f} (aggressive selling)")
            elif ratio < 0.91:
                score = -1.0
                reasons.append(f"5m taker ratio {ratio:.2f} (selling)")

        # 15m confirmation adds ±1.0
        taker_15m = derivatives.get("taker_ratio_15m", [])
        if taker_15m and abs(score) > 0:
            ratio_15m = taker_15m[0]["buy_sell_ratio"]
            if score > 0 and ratio_15m > 1.1:
                score += 1.0
                reasons.append(f"15m confirms buying ({ratio_15m:.2f})")
            elif score < 0 and ratio_15m < 0.91:
                score += -1.0
                reasons.append(f"15m confirms selling ({ratio_15m:.2f})")

        return max(-3.0, min(3.0, score))

    def _score_orderbook(self, orderbook: dict, current_price: float, reasons: list) -> float:
        """Score order book imbalance (max ±2.0)."""
        score = 0.0
        if not orderbook or current_price <= 0:
            return 0.0

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return 0.0

        # Calculate volume within 0.5% of current price
        price_range = current_price * 0.005
        bid_vol = sum(q for p, q in bids if p >= current_price - price_range)
        ask_vol = sum(q for p, q in asks if p <= current_price + price_range)
        total_vol = bid_vol + ask_vol

        if total_vol <= 0:
            return 0.0

        bid_pct = bid_vol / total_vol

        if bid_pct > 0.65:
            score = 1.5
            reasons.append(f"OB bid heavy ({bid_pct:.0%}) → support")
        elif bid_pct < 0.35:
            score = -1.5
            reasons.append(f"OB ask heavy ({1-bid_pct:.0%}) → resistance")

        # Whale order detection: any single order > 5x average
        all_orders = bids + asks
        if all_orders:
            avg_size = sum(q for _, q in all_orders) / len(all_orders)
            for p, q in bids:
                if q > avg_size * 5 and p >= current_price - price_range:
                    score += 0.5
                    reasons.append(f"Whale bid ${p:,.0f}")
                    break
            for p, q in asks:
                if q > avg_size * 5 and p <= current_price + price_range:
                    score -= 0.5
                    reasons.append(f"Whale ask ${p:,.0f}")
                    break

        return max(-2.0, min(2.0, score))

    def _score_long_short(self, derivatives: dict, reasons: list) -> float:
        """Score top trader L/S ratio (max ±1.5, contrarian)."""
        score = 0.0
        ls_data = derivatives.get("top_long_short", [])
        if not ls_data:
            return 0.0

        long_pct = ls_data[0]["long_account"]  # Already 0-1

        # Contrarian: crowded longs = bearish, crowded shorts = bullish
        if long_pct > 0.71:
            score = -1.5
            reasons.append(f"Top traders {long_pct:.0%} long (crowded → contrarian short)")
        elif long_pct > 0.62:
            score = -0.5
            reasons.append(f"Top traders {long_pct:.0%} long (leaning long)")
        elif long_pct < 0.38:  # i.e. >62% short
            score = 1.5
            reasons.append(f"Top traders {1-long_pct:.0%} short (crowded → contrarian long)")
        elif long_pct < 0.48:  # i.e. >52% short
            score = 0.5
            reasons.append(f"Top traders {1-long_pct:.0%} short (leaning short)")

        return score


# =============================================================================
# Strategy Config + Paper Trader
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str                        # "funding_sentiment", "volatility_squeeze", "taker_flow"
    strategy_type: str               # "funding", "squeeze", "taker"
    generator: object                # Signal generator instance
    stop_loss_pct: float             # e.g., 0.015
    take_profit_pct: float           # e.g., 0.040
    trailing_activation_pct: float   # e.g., 0.020
    trailing_distance_pct: float     # e.g., 0.010
    max_position_hours: float        # e.g., 8.0
    capital: float                   # Allocated capital

    @property
    def rr_ratio(self) -> float:
        return self.take_profit_pct / self.stop_loss_pct if self.stop_loss_pct > 0 else 0

    @property
    def source_label(self) -> str:
        """Signal source label for DB."""
        return self.strategy_type  # "funding", "squeeze", "taker"


# Strategy name mapping for backward compat with old trades
LEGACY_STRATEGY_MAP = {
    "paper_technical": "agreement_classic",
    "agreement_classic": "agreement_classic",
    "agreement_mtf": "agreement_mtf",
    "momentum": "momentum",
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

        # Default strategy if none provided
        if strategies is None:
            strategies = [
                StrategyConfig(
                    name="funding_sentiment",
                    strategy_type="funding",
                    generator=FundingSentimentGenerator(),
                    stop_loss_pct=0.015,
                    take_profit_pct=0.040,
                    trailing_activation_pct=0.020,
                    trailing_distance_pct=0.010,
                    max_position_hours=8.0,
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
        logger.info("🚀 Starting v5.0 Derivatives-Data Paper Trading")
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
            # Parallel fetch: 1m candles, 1h candles, ticker, derivatives, orderbook
            # Also fetch 4h candles if squeeze strategy is active
            fetch_4h = any(s.strategy_type == "squeeze" for s in self.strategies)
            fetch_derivatives = any(s.strategy_type in ("funding", "taker") for s in self.strategies)
            fetch_orderbook = any(s.strategy_type == "taker" for s in self.strategies)

            tasks = [
                collector.get_binance_klines(symbol, "1m", 100),
                collector.get_binance_klines(symbol, "1h", 100),
                collector.get_binance_ticker(symbol),
            ]
            if fetch_4h:
                tasks.append(collector.get_binance_klines(symbol, "4h", 50))
            if fetch_derivatives:
                tasks.append(collector.get_derivatives_data(symbol))
            if fetch_orderbook:
                tasks.append(collector.get_orderbook(symbol, 100))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            df_1m = results[0] if not isinstance(results[0], Exception) else pd.DataFrame()
            df_1h = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()
            ticker = results[2] if not isinstance(results[2], Exception) else {}

            if df_1m.empty:
                return None

            current_price = ticker.get("price", 0) if ticker else 0
            if current_price <= 0:
                return None

            market_data = {
                "1m": df_1m,
                "1h": df_1h,
                "current_price": current_price,
            }

            idx = 3
            if fetch_4h:
                market_data["4h"] = results[idx] if not isinstance(results[idx], Exception) else pd.DataFrame()
                idx += 1
            if fetch_derivatives:
                market_data["derivatives"] = results[idx] if not isinstance(results[idx], Exception) else {}
                idx += 1
            if fetch_orderbook:
                market_data["orderbook"] = results[idx] if not isinstance(results[idx], Exception) else {}
                idx += 1

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

            # Determine HTF trend (shared 1h analysis)
            htf_trend = "neutral"
            if not df_1h.empty:
                htf_trend = determine_htf_trend(df_1h)

            # Generate signal based on strategy type
            if strategy.strategy_type == "funding":
                signal_result = strategy.generator.generate_signal(
                    df_1m, htf_trend,
                    derivatives=market_data.get("derivatives"),
                )
            elif strategy.strategy_type == "squeeze":
                signal_result = strategy.generator.generate_signal(
                    df_1m, htf_trend,
                    df_4h=market_data.get("4h"),
                    df_1m=df_1m,
                )
            elif strategy.strategy_type == "taker":
                signal_result = strategy.generator.generate_signal(
                    df_1m, htf_trend,
                    derivatives=market_data.get("derivatives"),
                    orderbook=market_data.get("orderbook"),
                )
            else:
                signal_result = {"signal": "hold", "confidence": 0.5,
                                 "reasoning": f"Unknown strategy type: {strategy.strategy_type}"}

            # Determine timeframe label for DB
            timeframe = "4h" if strategy.strategy_type == "squeeze" else "1m"

            # Save signal to DB
            await self._save_signal(symbol, signal_result, current_price, strategy, timeframe)

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

        # 5. RSI-BASED PROFIT TAKING (using 1m data if available)
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

    parser = argparse.ArgumentParser(description="v5.0 Derivatives-Data Paper Trading")
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
        default=["funding_sentiment", "volatility_squeeze", "taker_flow"],
        choices=["funding_sentiment", "volatility_squeeze", "taker_flow"],
        help="Strategies to run (default: all three)"
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
    capital_per_strategy = args.capital

    strategy_configs = []
    for name in selected:
        if name == "funding_sentiment":
            strategy_configs.append(StrategyConfig(
                name="funding_sentiment",
                strategy_type="funding",
                generator=FundingSentimentGenerator(),
                stop_loss_pct=0.015,
                take_profit_pct=0.040,
                trailing_activation_pct=0.020,
                trailing_distance_pct=0.010,
                max_position_hours=8.0,
                capital=capital_per_strategy,
            ))
        elif name == "volatility_squeeze":
            strategy_configs.append(StrategyConfig(
                name="volatility_squeeze",
                strategy_type="squeeze",
                generator=VolatilitySqueezeGenerator(),
                stop_loss_pct=0.010,
                take_profit_pct=0.030,
                trailing_activation_pct=0.015,
                trailing_distance_pct=0.007,
                max_position_hours=12.0,
                capital=capital_per_strategy,
            ))
        elif name == "taker_flow":
            strategy_configs.append(StrategyConfig(
                name="taker_flow",
                strategy_type="taker",
                generator=TakerFlowGenerator(),
                stop_loss_pct=0.008,
                take_profit_pct=0.016,
                trailing_activation_pct=0.008,
                trailing_distance_pct=0.004,
                max_position_hours=2.0,
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
