"""SMC Pattern Detector.

Detects Smart Money Concepts patterns in OHLCV data including:
- Order Blocks
- Fair Value Gaps
- Liquidity Sweeps
- Channels
- Market Structure (CHoCH, BOS)
"""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from data.features.smc.zones import (
    Channel,
    FairValueGap,
    LiquiditySweep,
    MarketStructure,
    OrderBlock,
    PriceZone,
    ZoneDirection,
    ZoneStatus,
)


class SMCDetector:
    """Detect Smart Money Concepts patterns in price data.

    Attributes:
        lookback: Number of candles to analyze (default 100)
        atr_period: Period for ATR calculation (default 14)
        swing_lookback: Candles on each side for swing detection (default 5)
        min_impulse_atr: Minimum impulse size in ATR multiples (default 2.0)
    """

    def __init__(
        self,
        lookback: int = 100,
        atr_period: int = 14,
        swing_lookback: int = 5,
        min_impulse_atr: float = 2.0,
    ):
        """Initialize SMC detector.

        Args:
            lookback: Number of candles to analyze
            atr_period: Period for ATR calculation
            swing_lookback: Candles on each side for swing detection
            min_impulse_atr: Minimum impulse size in ATR multiples
        """
        self.lookback = lookback
        self.atr_period = atr_period
        self.swing_lookback = swing_lookback
        self.min_impulse_atr = min_impulse_atr

        # Cached results
        self._atr: pd.Series | None = None
        self._swing_highs: list[tuple[int, float]] = []
        self._swing_lows: list[tuple[int, float]] = []

    def analyze(
        self,
        candles: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run full SMC analysis on candle data.

        Args:
            candles: OHLCV DataFrame

        Returns:
            Dictionary containing all detected patterns
        """
        if len(candles) < self.lookback:
            logger.warning(f"Not enough candles for analysis: {len(candles)} < {self.lookback}")
            return {}

        # Use most recent candles
        df = candles.tail(self.lookback).copy()
        df = df.reset_index(drop=True)

        # Calculate ATR
        self._atr = self._calculate_atr(df)

        # Detect swing points
        self._swing_highs, self._swing_lows = self._detect_swing_points(df)

        # Run all detections
        result = {
            "order_blocks": self.detect_order_blocks(df),
            "fair_value_gaps": self.detect_fair_value_gaps(df),
            "liquidity_sweeps": self.detect_liquidity_sweeps(df),
            "channels": self.detect_channels(df),
            "market_structure": self.get_market_structure(df),
            "atr": float(self._atr.iloc[-1]) if self._atr is not None else 0.0,
        }

        return result

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()

        return atr

    def _detect_swing_points(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Detect swing highs and lows.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (swing_highs, swing_lows) as lists of (index, price)
        """
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        n = len(df)

        swing_highs = []
        swing_lows = []

        for i in range(self.swing_lookback, n - self.swing_lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(1, self.swing_lookback + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append((i, float(high[i])))

            # Check for swing low
            is_swing_low = True
            for j in range(1, self.swing_lookback + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append((i, float(low[i])))

        return swing_highs, swing_lows

    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        max_blocks: int = 10,
    ) -> list[OrderBlock]:
        """Detect Order Blocks in price data.

        An Order Block is the last opposite candle before a strong impulse move.

        Args:
            df: OHLCV DataFrame
            max_blocks: Maximum number of OBs to return

        Returns:
            List of detected Order Blocks
        """
        if self._atr is None:
            self._atr = self._calculate_atr(df)

        order_blocks: list[OrderBlock] = []

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        open_ = df["open"].astype(float).values
        close = df["close"].astype(float).values
        volume = df["volume"].astype(float).values if "volume" in df.columns else np.ones(len(df))
        atr = self._atr.values

        avg_volume = np.mean(volume) if len(volume) > 0 else 1.0

        n = len(df)
        min_candles_after = 3  # Need at least 3 candles after OB

        for i in range(1, n - min_candles_after):
            current_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            if current_atr <= 0:
                continue

            # Check for bullish impulse (OB is bearish candle before)
            impulse_move = close[i + min_candles_after] - close[i]
            if impulse_move > self.min_impulse_atr * current_atr:
                # Look for last bearish candle
                for j in range(i, max(0, i - 5), -1):
                    if close[j] < open_[j]:  # Bearish candle
                        ob = OrderBlock(
                            upper=float(high[j]),
                            lower=float(low[j]),
                            direction=ZoneDirection.BULLISH,
                            status=ZoneStatus.FRESH,
                            created_at=datetime.now(timezone.utc),
                            candle_index=j,
                            impulse_strength=float(impulse_move / current_atr),
                            displacement=float(high[i + 1] - low[i]) if i + 1 < n else 0.0,
                            volume_ratio=float(volume[j] / avg_volume) if avg_volume > 0 else 1.0,
                            body_type="bearish",
                        )
                        order_blocks.append(ob)
                        break

            # Check for bearish impulse (OB is bullish candle before)
            impulse_move = close[i] - close[i + min_candles_after]
            if impulse_move > self.min_impulse_atr * current_atr:
                # Look for last bullish candle
                for j in range(i, max(0, i - 5), -1):
                    if close[j] > open_[j]:  # Bullish candle
                        ob = OrderBlock(
                            upper=float(high[j]),
                            lower=float(low[j]),
                            direction=ZoneDirection.BEARISH,
                            status=ZoneStatus.FRESH,
                            created_at=datetime.now(timezone.utc),
                            candle_index=j,
                            impulse_strength=float(impulse_move / current_atr),
                            displacement=float(high[i] - low[i + 1]) if i + 1 < n else 0.0,
                            volume_ratio=float(volume[j] / avg_volume) if avg_volume > 0 else 1.0,
                            body_type="bullish",
                        )
                        order_blocks.append(ob)
                        break

        # Update status based on current price
        if len(order_blocks) > 0:
            current_price = float(close[-1])
            current_low = float(low[-1])
            current_high = float(high[-1])

            for ob in order_blocks:
                self._update_zone_status(ob, current_price, current_low, current_high)

        # Sort by recency and strength, return top N
        order_blocks.sort(key=lambda x: (x.candle_index, x.strength), reverse=True)
        return order_blocks[:max_blocks]

    def detect_fair_value_gaps(
        self,
        df: pd.DataFrame,
        max_gaps: int = 10,
    ) -> list[FairValueGap]:
        """Detect Fair Value Gaps (imbalances) in price data.

        FVG is a 3-candle pattern where:
        - Bullish: candle1.low > candle3.high (gap up)
        - Bearish: candle1.high < candle3.low (gap down)

        Args:
            df: OHLCV DataFrame
            max_gaps: Maximum number of FVGs to return

        Returns:
            List of detected Fair Value Gaps
        """
        if self._atr is None:
            self._atr = self._calculate_atr(df)

        fair_value_gaps: list[FairValueGap] = []

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        atr = self._atr.values

        n = len(df)

        for i in range(2, n):
            current_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            if current_atr <= 0:
                continue

            # Bullish FVG: candle[i-2].low > candle[i].high
            if low[i - 2] > high[i]:
                gap_upper = float(low[i - 2])
                gap_lower = float(high[i])
                gap_size = gap_upper - gap_lower

                fvg = FairValueGap(
                    upper=gap_upper,
                    lower=gap_lower,
                    direction=ZoneDirection.BULLISH,
                    status=ZoneStatus.FRESH,
                    created_at=datetime.now(timezone.utc),
                    candle_index=i - 1,  # Middle candle
                    candle1_index=i - 2,
                    candle2_index=i - 1,
                    candle3_index=i,
                    gap_size_atr=float(gap_size / current_atr),
                )
                fair_value_gaps.append(fvg)

            # Bearish FVG: candle[i-2].high < candle[i].low
            if high[i - 2] < low[i]:
                gap_upper = float(low[i])
                gap_lower = float(high[i - 2])
                gap_size = gap_upper - gap_lower

                fvg = FairValueGap(
                    upper=gap_upper,
                    lower=gap_lower,
                    direction=ZoneDirection.BEARISH,
                    status=ZoneStatus.FRESH,
                    created_at=datetime.now(timezone.utc),
                    candle_index=i - 1,
                    candle1_index=i - 2,
                    candle2_index=i - 1,
                    candle3_index=i,
                    gap_size_atr=float(gap_size / current_atr),
                )
                fair_value_gaps.append(fvg)

        # Update fill percentages
        if len(fair_value_gaps) > 0:
            current_low = float(low[-1])
            current_high = float(high[-1])

            for fvg in fair_value_gaps:
                fvg.update_fill_percent(current_low, current_high)

        # Filter out invalid and sort
        fair_value_gaps = [fvg for fvg in fair_value_gaps if fvg.is_valid()]
        fair_value_gaps.sort(key=lambda x: (x.candle_index, x.strength), reverse=True)

        return fair_value_gaps[:max_gaps]

    def detect_liquidity_sweeps(
        self,
        df: pd.DataFrame,
        max_sweeps: int = 5,
    ) -> list[LiquiditySweep]:
        """Detect Liquidity Sweeps (fakeouts/stop hunts).

        A sweep occurs when price briefly breaks a swing high/low
        then quickly reverses, indicating stop hunt activity.

        Args:
            df: OHLCV DataFrame
            max_sweeps: Maximum number of sweeps to return

        Returns:
            List of detected Liquidity Sweeps
        """
        if self._atr is None:
            self._atr = self._calculate_atr(df)

        if not self._swing_highs or not self._swing_lows:
            self._swing_highs, self._swing_lows = self._detect_swing_points(df)

        sweeps: list[LiquiditySweep] = []

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        open_ = df["open"].astype(float).values
        close = df["close"].astype(float).values
        atr = self._atr.values

        n = len(df)

        # Check recent candles for sweeps
        for i in range(max(self.swing_lookback + 5, 10), n):
            current_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            if current_atr <= 0:
                continue

            candle_body = abs(close[i] - open_[i])
            candle_range = high[i] - low[i]
            body_percent = (candle_body / candle_range * 100) if candle_range > 0 else 50

            # Check for sweep of swing highs (bearish sweep)
            for sh_idx, sh_price in self._swing_highs:
                if sh_idx >= i - 3:  # Too recent
                    continue
                if sh_idx < i - 20:  # Too old
                    continue

                # Check if we swept the high then closed below
                if high[i] > sh_price and close[i] < sh_price:
                    wick_size = high[i] - max(open_[i], close[i])
                    reversal = high[i] - close[i]

                    sweep = LiquiditySweep(
                        upper=float(high[i]),
                        lower=float(min(open_[i], close[i])),
                        direction=ZoneDirection.BEARISH,
                        status=ZoneStatus.FRESH,
                        created_at=datetime.now(timezone.utc),
                        candle_index=i,
                        sweep_level=float(sh_price),
                        wick_size=float(wick_size / current_atr),
                        reversal_strength=float(reversal / current_atr),
                        swing_type="high",
                        candle_body_percent=float(body_percent),
                    )
                    sweeps.append(sweep)

            # Check for sweep of swing lows (bullish sweep)
            for sl_idx, sl_price in self._swing_lows:
                if sl_idx >= i - 3:
                    continue
                if sl_idx < i - 20:
                    continue

                # Check if we swept the low then closed above
                if low[i] < sl_price and close[i] > sl_price:
                    wick_size = min(open_[i], close[i]) - low[i]
                    reversal = close[i] - low[i]

                    sweep = LiquiditySweep(
                        upper=float(max(open_[i], close[i])),
                        lower=float(low[i]),
                        direction=ZoneDirection.BULLISH,
                        status=ZoneStatus.FRESH,
                        created_at=datetime.now(timezone.utc),
                        candle_index=i,
                        sweep_level=float(sl_price),
                        wick_size=float(wick_size / current_atr),
                        reversal_strength=float(reversal / current_atr),
                        swing_type="low",
                        candle_body_percent=float(body_percent),
                    )
                    sweeps.append(sweep)

        # Sort by recency and strength
        sweeps.sort(key=lambda x: (x.candle_index, x.strength), reverse=True)
        return sweeps[:max_sweeps]

    def detect_channels(
        self,
        df: pd.DataFrame,
        min_touches: int = 4,
    ) -> list[Channel]:
        """Detect price channels (parallel trendlines).

        Args:
            df: OHLCV DataFrame
            min_touches: Minimum touches required for valid channel

        Returns:
            List of detected Channels
        """
        if not self._swing_highs or not self._swing_lows:
            self._swing_highs, self._swing_lows = self._detect_swing_points(df)

        channels: list[Channel] = []

        # Need at least 2 points for each trendline
        if len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return channels

        # Try to fit ascending channel (higher lows + higher highs)
        ascending = self._fit_channel(
            self._swing_highs[-4:],
            self._swing_lows[-4:],
            ZoneDirection.BULLISH,
            min_touches,
        )
        if ascending:
            channels.append(ascending)

        # Try to fit descending channel (lower lows + lower highs)
        descending = self._fit_channel(
            self._swing_highs[-4:],
            self._swing_lows[-4:],
            ZoneDirection.BEARISH,
            min_touches,
        )
        if descending:
            channels.append(descending)

        return channels

    def _fit_channel(
        self,
        swing_highs: list[tuple[int, float]],
        swing_lows: list[tuple[int, float]],
        direction: ZoneDirection,
        min_touches: int,
    ) -> Channel | None:
        """Attempt to fit a channel to swing points."""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Simple linear regression for trendlines
        try:
            # Fit upper line through swing highs
            h_x = np.array([sh[0] for sh in swing_highs])
            h_y = np.array([sh[1] for sh in swing_highs])
            h_slope, h_intercept = np.polyfit(h_x, h_y, 1)

            # Fit lower line through swing lows
            l_x = np.array([sl[0] for sl in swing_lows])
            l_y = np.array([sl[1] for sl in swing_lows])
            l_slope, l_intercept = np.polyfit(l_x, l_y, 1)

            # Check if lines are roughly parallel (similar slope)
            slope_diff = abs(h_slope - l_slope)
            avg_slope = (abs(h_slope) + abs(l_slope)) / 2

            if avg_slope > 0 and slope_diff / avg_slope > 0.5:
                return None  # Lines not parallel enough

            # Check direction matches
            if direction == ZoneDirection.BULLISH and avg_slope <= 0:
                return None
            if direction == ZoneDirection.BEARISH and avg_slope >= 0:
                return None

            # Calculate channel width
            mid_x = (h_x.mean() + l_x.mean()) / 2
            width = (h_slope * mid_x + h_intercept) - (l_slope * mid_x + l_intercept)

            if width <= 0:
                return None

            # Count touches
            touch_count = len(swing_highs) + len(swing_lows)

            if touch_count < min_touches:
                return None

            # Calculate strength
            strength = min(touch_count / 8, 1.0) * 0.5 + 0.5

            return Channel(
                upper_line=(float(h_slope), float(h_intercept)),
                lower_line=(float(l_slope), float(l_intercept)),
                direction=direction,
                touch_count=touch_count,
                width=float(width),
                start_index=min(h_x.min(), l_x.min()),
                end_index=max(h_x.max(), l_x.max()),
                strength=strength,
            )

        except Exception as e:
            logger.debug(f"Channel fitting failed: {e}")
            return None

    def get_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """Analyze market structure (trend, CHoCH, BOS).

        Args:
            df: OHLCV DataFrame

        Returns:
            MarketStructure analysis
        """
        if not self._swing_highs or not self._swing_lows:
            self._swing_highs, self._swing_lows = self._detect_swing_points(df)

        # Count consecutive higher highs and lower lows
        higher_highs = 0
        lower_lows = 0

        # Check swing highs
        for i in range(1, min(5, len(self._swing_highs))):
            if self._swing_highs[-i][1] > self._swing_highs[-i - 1][1]:
                higher_highs += 1
            else:
                break

        # Check swing lows
        for i in range(1, min(5, len(self._swing_lows))):
            if self._swing_lows[-i][1] < self._swing_lows[-i - 1][1]:
                lower_lows += 1
            else:
                break

        # Determine trend
        if higher_highs >= 2 and len(self._swing_lows) >= 2:
            # Check for higher lows too
            higher_lows = 0
            for i in range(1, min(5, len(self._swing_lows))):
                if self._swing_lows[-i][1] > self._swing_lows[-i - 1][1]:
                    higher_lows += 1
                else:
                    break
            if higher_lows >= 1:
                trend = ZoneDirection.BULLISH
            else:
                trend = ZoneDirection.BULLISH
        elif lower_lows >= 2:
            trend = ZoneDirection.BEARISH
        else:
            # Default to direction of recent price action
            close = df["close"].astype(float)
            if len(close) > 10:
                trend = ZoneDirection.BULLISH if close.iloc[-1] > close.iloc[-10] else ZoneDirection.BEARISH
            else:
                trend = ZoneDirection.BULLISH

        # Detect CHoCH (Change of Character)
        last_choch = None
        if len(self._swing_lows) >= 2 and trend == ZoneDirection.BEARISH:
            # Bullish CHoCH: price breaks above last swing high in downtrend
            close = df["close"].astype(float).values
            for i in range(len(close) - 1, max(0, len(close) - 20), -1):
                for sh_idx, sh_price in reversed(self._swing_highs):
                    if sh_idx < i and close[i] > sh_price:
                        last_choch = i
                        break
                if last_choch:
                    break

        # Detect BOS (Break of Structure)
        last_bos = None
        if len(self._swing_highs) >= 1:
            close = df["close"].astype(float).values
            if trend == ZoneDirection.BULLISH:
                # Bullish BOS: new higher high
                for i in range(len(close) - 1, max(0, len(close) - 10), -1):
                    if close[i] > self._swing_highs[-1][1]:
                        last_bos = i
                        break

        # Calculate trend strength
        strength = 0.5
        if higher_highs >= 3 or lower_lows >= 3:
            strength = 0.9
        elif higher_highs >= 2 or lower_lows >= 2:
            strength = 0.7

        return MarketStructure(
            trend=trend,
            swing_highs=self._swing_highs,
            swing_lows=self._swing_lows,
            last_choch=last_choch,
            last_bos=last_bos,
            higher_highs=higher_highs,
            lower_lows=lower_lows,
            strength=strength,
        )

    def _update_zone_status(
        self,
        zone: PriceZone,
        current_price: float,
        current_low: float,
        current_high: float,
    ) -> None:
        """Update zone status based on current price action."""
        if zone.status == ZoneStatus.INVALIDATED:
            return

        if zone.direction == ZoneDirection.BULLISH:
            # Bullish zone (support): invalidated if price closes below
            if current_price < zone.lower:
                zone.status = ZoneStatus.INVALIDATED
            elif zone.contains_price(current_low) or zone.contains_price(current_price):
                zone.touches += 1
                if zone.touches > 3:
                    zone.status = ZoneStatus.MITIGATED
                elif zone.touches > 0:
                    zone.status = ZoneStatus.TESTED
        else:
            # Bearish zone (resistance): invalidated if price closes above
            if current_price > zone.upper:
                zone.status = ZoneStatus.INVALIDATED
            elif zone.contains_price(current_high) or zone.contains_price(current_price):
                zone.touches += 1
                if zone.touches > 3:
                    zone.status = ZoneStatus.MITIGATED
                elif zone.touches > 0:
                    zone.status = ZoneStatus.TESTED

    def get_valid_zones(
        self,
        order_blocks: list[OrderBlock],
        fair_value_gaps: list[FairValueGap],
        direction: ZoneDirection | None = None,
    ) -> list[PriceZone]:
        """Get all valid zones, optionally filtered by direction.

        Args:
            order_blocks: List of order blocks
            fair_value_gaps: List of FVGs
            direction: Optional direction filter

        Returns:
            List of valid zones sorted by strength
        """
        zones: list[PriceZone] = []

        for ob in order_blocks:
            if ob.is_valid():
                if direction is None or ob.direction == direction:
                    zones.append(ob)

        for fvg in fair_value_gaps:
            if fvg.is_valid():
                if direction is None or fvg.direction == direction:
                    zones.append(fvg)

        # Sort by strength descending
        zones.sort(key=lambda x: x.strength, reverse=True)

        return zones

    def get_nearest_zone(
        self,
        zones: list[PriceZone],
        current_price: float,
        direction: ZoneDirection | None = None,
    ) -> PriceZone | None:
        """Find the nearest valid zone to current price.

        Args:
            zones: List of zones to search
            current_price: Current price
            direction: Optional direction filter

        Returns:
            Nearest zone or None
        """
        valid_zones = [z for z in zones if z.is_valid()]

        if direction:
            valid_zones = [z for z in valid_zones if z.direction == direction]

        if not valid_zones:
            return None

        # Sort by distance to current price
        valid_zones.sort(key=lambda z: z.distance_to_price(current_price))

        return valid_zones[0]
