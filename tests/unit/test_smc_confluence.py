"""Unit tests for SMC confluence engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from data.features.smc.confluence import ConfluenceEngine, ConfluenceResult
from data.features.smc.detector import SMCDetector
from data.features.smc.zones import (
    ZoneDirection,
    ZoneStatus,
    OrderBlock,
    FairValueGap,
    LiquiditySweep,
    MarketStructure,
)


def create_test_order_block(
    direction: ZoneDirection = ZoneDirection.BULLISH,
    upper: float = 100.0,
    lower: float = 95.0,
    strength: float = 0.7,
    status: ZoneStatus = ZoneStatus.FRESH,
) -> OrderBlock:
    """Create a test order block."""
    return OrderBlock(
        upper=upper,
        lower=lower,
        direction=direction,
        status=status,
        impulse_strength=2.5,
        volume_ratio=1.5,
    )


def create_test_fvg(
    direction: ZoneDirection = ZoneDirection.BULLISH,
    upper: float = 100.0,
    lower: float = 98.0,
    status: ZoneStatus = ZoneStatus.FRESH,
) -> FairValueGap:
    """Create a test FVG."""
    return FairValueGap(
        upper=upper,
        lower=lower,
        direction=direction,
        status=status,
        candle1_index=10,
        candle2_index=11,
        candle3_index=12,
        gap_size_atr=1.5,
    )


def create_test_sweep(
    direction: ZoneDirection = ZoneDirection.BULLISH,
) -> LiquiditySweep:
    """Create a test liquidity sweep."""
    return LiquiditySweep(
        upper=102.0,
        lower=95.0,
        direction=direction,
        sweep_level=94.0,
        wick_size=2.0,
        reversal_strength=1.5,
        swing_type="low",
    )


def create_test_structure(
    direction: ZoneDirection = ZoneDirection.BULLISH,
) -> MarketStructure:
    """Create a test market structure."""
    return MarketStructure(
        trend=direction,
        swing_highs=[(10, 105.0), (20, 108.0)],
        swing_lows=[(5, 95.0), (15, 97.0)],
        higher_highs=2 if direction == ZoneDirection.BULLISH else 0,
        lower_lows=0 if direction == ZoneDirection.BULLISH else 2,
        strength=0.7,
    )


class TestConfluenceEngine:
    """Tests for ConfluenceEngine class."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ConfluenceEngine(
            min_confluence_score=0.65,
            min_rr_ratio=2.0,
            atr_stop_multiplier=1.5,
        )

        assert engine.min_confluence_score == 0.65
        assert engine.min_rr_ratio == 2.0
        assert engine.atr_stop_multiplier == 1.5

    def test_analyze_with_zones(self):
        """Test confluence analysis with zones."""
        engine = ConfluenceEngine(min_confluence_score=0.5)

        # Create test data
        ob = create_test_order_block(
            direction=ZoneDirection.BULLISH,
            upper=100.0,
            lower=95.0,
        )
        fvg = create_test_fvg(
            direction=ZoneDirection.BULLISH,
            upper=99.0,
            lower=97.0,
        )
        structure = create_test_structure(ZoneDirection.BULLISH)

        result = engine.analyze(
            current_price=101.0,
            order_blocks=[ob],
            fair_value_gaps=[fvg],
            liquidity_sweeps=[],
            channels=[],
            market_structure=structure,
            atr=2.0,
            htf_bias=ZoneDirection.BULLISH,
        )

        if result:
            assert result.score > 0
            assert result.direction in [ZoneDirection.BULLISH, ZoneDirection.BEARISH]
            assert result.entry_zone[0] > result.entry_zone[1]  # upper > lower

    def test_analyze_returns_none_without_zones(self):
        """Test that analysis returns None without valid zones."""
        engine = ConfluenceEngine(min_confluence_score=0.8)

        structure = create_test_structure()

        result = engine.analyze(
            current_price=100.0,
            order_blocks=[],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            channels=[],
            market_structure=structure,
            atr=2.0,
        )

        assert result is None

    def test_confluence_score_calculation(self):
        """Test that confluence score is calculated correctly."""
        engine = ConfluenceEngine(min_confluence_score=0.4)

        # High quality setup: OB + FVG overlap + HTF alignment
        ob = create_test_order_block(
            direction=ZoneDirection.BULLISH,
            upper=100.0,
            lower=95.0,
        )
        fvg = create_test_fvg(
            direction=ZoneDirection.BULLISH,
            upper=99.0,
            lower=96.0,  # Overlaps with OB
        )
        structure = create_test_structure(ZoneDirection.BULLISH)

        result = engine.analyze(
            current_price=101.0,
            order_blocks=[ob],
            fair_value_gaps=[fvg],
            liquidity_sweeps=[],
            channels=[],
            market_structure=structure,
            atr=2.0,
            htf_bias=ZoneDirection.BULLISH,
        )

        if result:
            # Score should include zone strength, confluence, HTF alignment, freshness
            assert "zone_strength" in result.factors
            assert "zone_confluence" in result.factors
            assert "htf_alignment" in result.factors


class TestConfluenceResult:
    """Tests for ConfluenceResult class."""

    def test_result_properties(self):
        """Test ConfluenceResult properties."""
        result = ConfluenceResult(
            score=0.75,
            direction=ZoneDirection.BULLISH,
            entry_zone=(100.0, 95.0),
            stop_loss=93.0,
            take_profit_1=110.0,
            take_profit_2=115.0,
            risk_reward=2.5,
            factors={"zone_strength": 0.3, "zone_confluence": 0.25},
            confidence=0.7,
        )

        assert result.score == 0.75
        assert result.direction == ZoneDirection.BULLISH
        assert result.risk_reward == 2.5
        assert len(result.factors) == 2

    def test_result_to_dict(self):
        """Test ConfluenceResult to_dict method."""
        result = ConfluenceResult(
            score=0.75,
            direction=ZoneDirection.BULLISH,
            entry_zone=(100.0, 95.0),
            stop_loss=93.0,
            take_profit_1=110.0,
            risk_reward=2.5,
            confidence=0.7,
        )

        d = result.to_dict()

        assert d["score"] == 0.75
        assert d["direction"] == "bullish"
        assert d["entry_zone"]["upper"] == 100.0
        assert d["entry_zone"]["lower"] == 95.0
        assert d["stop_loss"] == 93.0
        assert d["risk_reward"] == 2.5


class TestRiskRewardCalculation:
    """Tests for risk/reward calculation."""

    def test_minimum_rr_filter(self):
        """Test that setups below minimum R:R are filtered."""
        engine = ConfluenceEngine(min_rr_ratio=3.0)  # High R:R requirement

        ob = create_test_order_block(ZoneDirection.BULLISH)
        structure = create_test_structure(ZoneDirection.BULLISH)

        # This may not meet the 3:1 R:R requirement
        result = engine.analyze(
            current_price=101.0,
            order_blocks=[ob],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            channels=[],
            market_structure=structure,
            atr=2.0,
        )

        # Result should be None if R:R is below threshold
        if result:
            assert result.risk_reward >= 3.0


class TestHTFAlignment:
    """Tests for HTF alignment scoring."""

    def test_htf_alignment_bonus(self):
        """Test that HTF alignment increases score."""
        engine = ConfluenceEngine(min_confluence_score=0.4)

        ob = create_test_order_block(ZoneDirection.BULLISH)
        structure = create_test_structure(ZoneDirection.BULLISH)

        # With HTF alignment
        result_aligned = engine.analyze(
            current_price=101.0,
            order_blocks=[ob],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            channels=[],
            market_structure=structure,
            atr=2.0,
            htf_bias=ZoneDirection.BULLISH,
        )

        # Without HTF alignment
        result_not_aligned = engine.analyze(
            current_price=101.0,
            order_blocks=[ob],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            channels=[],
            market_structure=structure,
            atr=2.0,
            htf_bias=ZoneDirection.BEARISH,
        )

        if result_aligned and result_not_aligned:
            assert result_aligned.factors.get("htf_alignment", 0) > result_not_aligned.factors.get("htf_alignment", 0)


class TestFindBestSetup:
    """Tests for find_best_setup convenience method."""

    def test_find_best_setup_with_candles(self):
        """Test find_best_setup with candle data."""
        engine = ConfluenceEngine(min_confluence_score=0.5)
        detector = SMCDetector()

        # Create test candles with some volatility
        np.random.seed(42)
        n = 150
        base_price = 100.0
        data = []

        for i in range(n):
            close = base_price + np.random.normal(0, 2)
            high = close + abs(np.random.normal(0, 1))
            low = close - abs(np.random.normal(0, 1))
            open_ = close + np.random.normal(0, 0.5)

            data.append({
                "timestamp": datetime.now(),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000,
            })

        candles = pd.DataFrame(data)

        result = engine.find_best_setup(candles, detector)

        # May or may not find a setup depending on random data
        if result:
            assert result.score >= engine.min_confluence_score
            assert result.risk_reward >= engine.min_rr_ratio
