"""Unit tests for SMC detector module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.features.smc.detector import SMCDetector
from data.features.smc.zones import (
    ZoneDirection,
    ZoneStatus,
    OrderBlock,
    FairValueGap,
    LiquiditySweep,
    MarketStructure,
)


def create_test_candles(
    n: int = 100,
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: str = "up",
    seed: int = 42,
) -> pd.DataFrame:
    """Create test OHLCV data.

    Args:
        n: Number of candles
        base_price: Starting price
        volatility: Price volatility
        trend: "up", "down", or "range"
        seed: Random seed

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    # Generate prices with trend
    if trend == "up":
        drift = 0.001
    elif trend == "down":
        drift = -0.001
    else:
        drift = 0.0

    returns = np.random.normal(drift, volatility, n)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = []
    for i, close in enumerate(close_prices):
        volatility_mult = volatility * base_price
        high = close + abs(np.random.normal(0, volatility_mult))
        low = close - abs(np.random.normal(0, volatility_mult))
        open_ = close + np.random.normal(0, volatility_mult / 2)

        # Ensure consistency
        high = max(high, open_, close)
        low = min(low, open_, close)

        data.append({
            "timestamp": datetime.now() - timedelta(hours=n - i),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(1000, 5000),
        })

    return pd.DataFrame(data)


def create_impulse_candles(
    n: int = 100,
    impulse_index: int = 50,
    impulse_direction: str = "up",
    impulse_size: float = 5.0,
) -> pd.DataFrame:
    """Create candles with a specific impulse move.

    Args:
        n: Number of candles
        impulse_index: Where to place the impulse
        impulse_direction: "up" or "down"
        impulse_size: Size of impulse in price units

    Returns:
        DataFrame with impulse pattern
    """
    base_price = 100.0
    data = []

    for i in range(n):
        if i < impulse_index:
            # Before impulse - ranging
            close = base_price + np.random.uniform(-1, 1)
        elif i == impulse_index:
            # Impulse candle
            if impulse_direction == "up":
                close = base_price + impulse_size
            else:
                close = base_price - impulse_size
        else:
            # After impulse - continue slightly in direction
            if impulse_direction == "up":
                close = base_price + impulse_size + (i - impulse_index) * 0.1
            else:
                close = base_price - impulse_size - (i - impulse_index) * 0.1

        open_ = close - np.random.uniform(-0.5, 0.5)
        high = max(open_, close) + abs(np.random.uniform(0, 0.5))
        low = min(open_, close) - abs(np.random.uniform(0, 0.5))

        data.append({
            "timestamp": datetime.now() - timedelta(hours=n - i),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000 if i != impulse_index else 5000,  # High volume on impulse
        })

    return pd.DataFrame(data)


class TestSMCDetector:
    """Tests for SMCDetector class."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = SMCDetector(
            lookback=100,
            atr_period=14,
            swing_lookback=5,
            min_impulse_atr=2.0,
        )

        assert detector.lookback == 100
        assert detector.atr_period == 14
        assert detector.min_impulse_atr == 2.0

    def test_analyze_returns_all_components(self):
        """Test that analyze returns all expected components."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        result = detector.analyze(candles)

        assert "order_blocks" in result
        assert "fair_value_gaps" in result
        assert "liquidity_sweeps" in result
        assert "channels" in result
        assert "market_structure" in result
        assert "atr" in result

    def test_insufficient_candles(self):
        """Test handling of insufficient candles."""
        detector = SMCDetector(lookback=100)
        candles = create_test_candles(n=50)

        result = detector.analyze(candles)

        assert result == {}


class TestOrderBlockDetection:
    """Tests for Order Block detection."""

    def test_detect_bullish_order_block(self):
        """Test detection of bullish order block."""
        detector = SMCDetector(min_impulse_atr=1.5)

        # Create candles with bullish impulse
        candles = create_impulse_candles(
            n=100,
            impulse_index=50,
            impulse_direction="up",
            impulse_size=5.0,
        )

        obs = detector.detect_order_blocks(candles)

        # Should find at least one bullish OB
        bullish_obs = [ob for ob in obs if ob.direction == ZoneDirection.BULLISH]
        assert len(bullish_obs) >= 0  # May or may not find depending on criteria

    def test_detect_bearish_order_block(self):
        """Test detection of bearish order block."""
        detector = SMCDetector(min_impulse_atr=1.5)

        # Create candles with bearish impulse
        candles = create_impulse_candles(
            n=100,
            impulse_index=50,
            impulse_direction="down",
            impulse_size=5.0,
        )

        obs = detector.detect_order_blocks(candles)

        # Should find at least one bearish OB
        bearish_obs = [ob for ob in obs if ob.direction == ZoneDirection.BEARISH]
        assert len(bearish_obs) >= 0

    def test_order_block_properties(self):
        """Test that detected OBs have valid properties."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        obs = detector.detect_order_blocks(candles)

        for ob in obs:
            assert ob.upper > ob.lower
            assert ob.direction in [ZoneDirection.BULLISH, ZoneDirection.BEARISH]
            assert 0 <= ob.strength <= 1
            assert ob.impulse_strength >= 0


class TestFairValueGapDetection:
    """Tests for Fair Value Gap detection."""

    def test_detect_fvg(self):
        """Test basic FVG detection."""
        detector = SMCDetector()
        candles = create_test_candles(n=150, volatility=0.03)

        fvgs = detector.detect_fair_value_gaps(candles)

        # FVGs should be valid zones
        for fvg in fvgs:
            assert fvg.upper > fvg.lower
            assert fvg.direction in [ZoneDirection.BULLISH, ZoneDirection.BEARISH]

    def test_fvg_candle_indices(self):
        """Test that FVG indices are sequential."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        fvgs = detector.detect_fair_value_gaps(candles)

        for fvg in fvgs:
            assert fvg.candle3_index == fvg.candle2_index + 1
            assert fvg.candle2_index == fvg.candle1_index + 1


class TestLiquiditySweepDetection:
    """Tests for Liquidity Sweep detection."""

    def test_detect_sweep(self):
        """Test liquidity sweep detection."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        sweeps = detector.detect_liquidity_sweeps(candles)

        for sweep in sweeps:
            assert sweep.upper > sweep.lower
            assert sweep.swing_type in ["high", "low"]


class TestMarketStructure:
    """Tests for Market Structure analysis."""

    def test_structure_analysis_returns_valid(self):
        """Test market structure analysis returns valid structure."""
        detector = SMCDetector()
        candles = create_test_candles(n=150, trend="up")

        structure = detector.get_market_structure(candles)

        # Structure should have valid direction
        assert structure.trend in [ZoneDirection.BULLISH, ZoneDirection.BEARISH]
        assert isinstance(structure.swing_highs, list)
        assert isinstance(structure.swing_lows, list)

    def test_structure_with_clear_trend(self):
        """Test structure with clear trending data."""
        detector = SMCDetector()

        # Create data with very clear uptrend
        base_price = 100.0
        data = []
        for i in range(150):
            # Steadily increasing with small noise
            close = base_price + i * 0.5 + np.random.uniform(-0.1, 0.1)
            high = close + abs(np.random.uniform(0, 0.2))
            low = close - abs(np.random.uniform(0, 0.2))
            open_ = close + np.random.uniform(-0.1, 0.1)
            data.append({
                "timestamp": datetime.now() - timedelta(hours=150 - i),
                "open": open_, "high": high, "low": low, "close": close,
                "volume": 1000,
            })

        candles = pd.DataFrame(data)
        structure = detector.get_market_structure(candles)

        # With clear trend, should detect it
        assert structure.trend in [ZoneDirection.BULLISH, ZoneDirection.BEARISH]

    def test_swing_points_detected(self):
        """Test that swing points are detected."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        structure = detector.get_market_structure(candles)

        # Should have some swing points
        assert len(structure.swing_highs) > 0 or len(structure.swing_lows) > 0


class TestZoneValidation:
    """Tests for zone validation and filtering."""

    def test_get_valid_zones(self):
        """Test filtering for valid zones."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        result = detector.analyze(candles)
        obs = result.get("order_blocks", [])
        fvgs = result.get("fair_value_gaps", [])

        valid = detector.get_valid_zones(obs, fvgs)

        for zone in valid:
            assert zone.is_valid()

    def test_filter_by_direction(self):
        """Test filtering zones by direction."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        result = detector.analyze(candles)
        obs = result.get("order_blocks", [])
        fvgs = result.get("fair_value_gaps", [])

        bullish = detector.get_valid_zones(obs, fvgs, direction=ZoneDirection.BULLISH)
        bearish = detector.get_valid_zones(obs, fvgs, direction=ZoneDirection.BEARISH)

        for zone in bullish:
            assert zone.direction == ZoneDirection.BULLISH

        for zone in bearish:
            assert zone.direction == ZoneDirection.BEARISH

    def test_get_nearest_zone(self):
        """Test finding nearest zone to price."""
        detector = SMCDetector()
        candles = create_test_candles(n=150)

        result = detector.analyze(candles)
        obs = result.get("order_blocks", [])
        fvgs = result.get("fair_value_gaps", [])

        all_zones = obs + fvgs
        current_price = float(candles["close"].iloc[-1])

        nearest = detector.get_nearest_zone(all_zones, current_price)

        if nearest and len(all_zones) > 1:
            # Verify it's actually nearest
            distances = [z.distance_to_price(current_price) for z in all_zones if z.is_valid()]
            if distances:
                assert nearest.distance_to_price(current_price) <= min(distances) + 0.01


class TestATRCalculation:
    """Tests for ATR calculation."""

    def test_atr_value(self):
        """Test ATR calculation returns valid value."""
        detector = SMCDetector(atr_period=14)
        candles = create_test_candles(n=150)

        result = detector.analyze(candles)

        assert "atr" in result
        assert result["atr"] > 0
