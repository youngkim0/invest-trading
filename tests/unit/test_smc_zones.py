"""Unit tests for SMC zones module."""

import pytest
from datetime import datetime

from data.features.smc.zones import (
    ZoneDirection,
    ZoneStatus,
    ZoneType,
    PriceZone,
    OrderBlock,
    FairValueGap,
    LiquiditySweep,
    Channel,
    MarketStructure,
)


class TestPriceZone:
    """Tests for PriceZone base class."""

    def test_price_zone_creation(self):
        """Test basic zone creation."""
        zone = PriceZone(
            upper=100.0,
            lower=95.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.FRESH,
        )

        assert zone.upper == 100.0
        assert zone.lower == 95.0
        assert zone.direction == ZoneDirection.BULLISH
        assert zone.status == ZoneStatus.FRESH

    def test_midpoint_calculation(self):
        """Test zone midpoint calculation."""
        zone = PriceZone(upper=100.0, lower=90.0, direction=ZoneDirection.BULLISH)
        assert zone.midpoint == 95.0

    def test_height_calculation(self):
        """Test zone height calculation."""
        zone = PriceZone(upper=100.0, lower=90.0, direction=ZoneDirection.BULLISH)
        assert zone.height == 10.0

    def test_contains_price(self):
        """Test price containment check."""
        zone = PriceZone(upper=100.0, lower=90.0, direction=ZoneDirection.BULLISH)

        assert zone.contains_price(95.0)  # Inside
        assert zone.contains_price(100.0)  # On upper boundary
        assert zone.contains_price(90.0)  # On lower boundary
        assert not zone.contains_price(101.0)  # Above
        assert not zone.contains_price(89.0)  # Below

    def test_distance_to_price(self):
        """Test distance calculation."""
        zone = PriceZone(upper=100.0, lower=90.0, direction=ZoneDirection.BULLISH)

        assert zone.distance_to_price(95.0) == 0.0  # Inside
        assert zone.distance_to_price(105.0) == 5.0  # Above
        assert zone.distance_to_price(85.0) == 5.0  # Below

    def test_is_valid(self):
        """Test validity check."""
        fresh = PriceZone(
            upper=100.0, lower=90.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.FRESH
        )
        tested = PriceZone(
            upper=100.0, lower=90.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.TESTED
        )
        invalid = PriceZone(
            upper=100.0, lower=90.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.INVALIDATED
        )

        assert fresh.is_valid()
        assert tested.is_valid()
        assert not invalid.is_valid()

    def test_to_dict(self):
        """Test dictionary conversion."""
        zone = PriceZone(
            upper=100.0,
            lower=90.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.FRESH,
            strength=0.8,
        )
        d = zone.to_dict()

        assert d["upper"] == 100.0
        assert d["lower"] == 90.0
        assert d["direction"] == "bullish"
        assert d["status"] == "fresh"
        assert d["strength"] == 0.8
        assert d["midpoint"] == 95.0
        assert d["height"] == 10.0


class TestOrderBlock:
    """Tests for OrderBlock class."""

    def test_order_block_creation(self):
        """Test order block creation."""
        ob = OrderBlock(
            upper=100.0,
            lower=95.0,
            direction=ZoneDirection.BULLISH,
            impulse_strength=3.0,
            volume_ratio=1.8,
        )

        assert ob.impulse_strength == 3.0
        assert ob.volume_ratio == 1.8
        assert ob.direction == ZoneDirection.BULLISH

    def test_strength_calculation(self):
        """Test automatic strength calculation."""
        # Strong OB: high impulse + high volume
        strong_ob = OrderBlock(
            upper=100.0, lower=95.0,
            direction=ZoneDirection.BULLISH,
            impulse_strength=4.0,
            volume_ratio=2.0,
            status=ZoneStatus.FRESH,
        )

        # Weak OB: low impulse + low volume
        weak_ob = OrderBlock(
            upper=100.0, lower=95.0,
            direction=ZoneDirection.BULLISH,
            impulse_strength=1.0,
            volume_ratio=0.8,
            status=ZoneStatus.TESTED,
        )

        assert strong_ob.strength > weak_ob.strength

    def test_order_block_to_dict(self):
        """Test OB dictionary conversion."""
        ob = OrderBlock(
            upper=100.0,
            lower=95.0,
            direction=ZoneDirection.BULLISH,
            impulse_strength=3.0,
            displacement=5.0,
            volume_ratio=1.5,
            body_type="bearish",
        )
        d = ob.to_dict()

        assert d["type"] == ZoneType.ORDER_BLOCK.value
        assert d["impulse_strength"] == 3.0
        assert d["displacement"] == 5.0
        assert d["volume_ratio"] == 1.5
        assert d["body_type"] == "bearish"


class TestFairValueGap:
    """Tests for FairValueGap class."""

    def test_fvg_creation(self):
        """Test FVG creation."""
        fvg = FairValueGap(
            upper=100.0,
            lower=98.0,
            direction=ZoneDirection.BULLISH,
            candle1_index=10,
            candle2_index=11,
            candle3_index=12,
            gap_size_atr=1.5,
        )

        assert fvg.candle2_index == 11
        assert fvg.gap_size_atr == 1.5

    def test_fill_percent_update_bullish(self):
        """Test FVG fill percentage update for bullish gap."""
        fvg = FairValueGap(
            upper=100.0,  # Bottom of candle 1
            lower=95.0,   # Top of candle 3
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.FRESH,
        )

        # Price drops to top of FVG
        fvg.update_fill_percent(current_low=99.0, current_high=105.0)
        assert fvg.gap_filled_percent == pytest.approx(20.0, rel=0.1)
        assert fvg.status == ZoneStatus.TESTED

        # Price fills >50%
        fvg2 = FairValueGap(
            upper=100.0, lower=95.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.FRESH,
        )
        fvg2.update_fill_percent(current_low=97.0, current_high=105.0)
        assert fvg2.gap_filled_percent == pytest.approx(60.0, rel=0.1)
        assert fvg2.status == ZoneStatus.MITIGATED

    def test_fvg_strength_calculation(self):
        """Test FVG strength based on characteristics."""
        # Fresh, large gap
        strong_fvg = FairValueGap(
            upper=100.0, lower=95.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.FRESH,
            gap_size_atr=2.0,
            gap_filled_percent=0.0,
        )

        # Partially filled small gap
        weak_fvg = FairValueGap(
            upper=100.0, lower=99.0,
            direction=ZoneDirection.BULLISH,
            status=ZoneStatus.TESTED,
            gap_size_atr=0.5,
            gap_filled_percent=60.0,
        )

        assert strong_fvg.strength > weak_fvg.strength


class TestLiquiditySweep:
    """Tests for LiquiditySweep class."""

    def test_sweep_creation(self):
        """Test liquidity sweep creation."""
        sweep = LiquiditySweep(
            upper=102.0,
            lower=95.0,
            direction=ZoneDirection.BEARISH,
            sweep_level=101.0,
            wick_size=2.0,
            reversal_strength=1.5,
            swing_type="high",
        )

        assert sweep.sweep_level == 101.0
        assert sweep.swing_type == "high"

    def test_sweep_strength_calculation(self):
        """Test sweep strength calculation."""
        # Strong sweep: large wick, strong reversal
        strong = LiquiditySweep(
            upper=105.0, lower=95.0,
            direction=ZoneDirection.BEARISH,
            wick_size=3.0,
            reversal_strength=2.0,
            candle_body_percent=20.0,
        )

        # Weak sweep: small wick, weak reversal
        weak = LiquiditySweep(
            upper=101.0, lower=99.0,
            direction=ZoneDirection.BEARISH,
            wick_size=0.5,
            reversal_strength=0.3,
            candle_body_percent=80.0,
        )

        assert strong.strength > weak.strength


class TestChannel:
    """Tests for Channel class."""

    def test_channel_creation(self):
        """Test channel creation."""
        channel = Channel(
            upper_line=(0.01, 100.0),
            lower_line=(0.01, 95.0),
            direction=ZoneDirection.BULLISH,
            touch_count=6,
            width=5.0,
        )

        assert channel.touch_count == 6
        assert channel.direction == ZoneDirection.BULLISH

    def test_get_line_values(self):
        """Test trendline value calculation."""
        channel = Channel(
            upper_line=(0.1, 100.0),  # y = 0.1x + 100
            lower_line=(0.1, 90.0),   # y = 0.1x + 90
            direction=ZoneDirection.BULLISH,
        )

        # At index 10
        assert channel.get_upper_at_index(10) == pytest.approx(101.0)
        assert channel.get_lower_at_index(10) == pytest.approx(91.0)

    def test_price_position(self):
        """Test price position in channel."""
        channel = Channel(
            upper_line=(0.0, 100.0),  # Flat at 100
            lower_line=(0.0, 90.0),   # Flat at 90
            direction=ZoneDirection.BULLISH,
        )

        assert channel.price_position(99.0, 0) == "near_upper"
        assert channel.price_position(91.0, 0) == "near_lower"
        assert channel.price_position(95.0, 0) == "middle"


class TestMarketStructure:
    """Tests for MarketStructure class."""

    def test_structure_creation(self):
        """Test market structure creation."""
        structure = MarketStructure(
            trend=ZoneDirection.BULLISH,
            swing_highs=[(10, 100.0), (20, 105.0)],
            swing_lows=[(5, 90.0), (15, 92.0)],
            higher_highs=2,
            lower_lows=0,
        )

        assert structure.trend == ZoneDirection.BULLISH
        assert structure.is_trending
        assert structure.trend_type == "uptrend"

    def test_trend_type_detection(self):
        """Test trend type classification."""
        strong_up = MarketStructure(
            trend=ZoneDirection.BULLISH,
            higher_highs=3,
        )
        assert strong_up.trend_type == "strong_uptrend"

        strong_down = MarketStructure(
            trend=ZoneDirection.BEARISH,
            lower_lows=3,
        )
        assert strong_down.trend_type == "strong_downtrend"

        ranging = MarketStructure(
            trend=ZoneDirection.BULLISH,
            higher_highs=1,
            lower_lows=1,
        )
        assert ranging.trend_type == "ranging"
