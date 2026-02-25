"""Unit tests for SMC strategy."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from config.strategies import StrategyConfig, StrategyType, TimeFrame
from core.strategies.smc_strategy import SMCStrategy, SMCStrategyConfig, create_smc_strategy
from data.storage.models import OrderSide, SignalSource, SignalType


def create_test_candles(
    n: int = 100,
    base_price: float = 100.0,
    trend: str = "up",
    seed: int = 42,
) -> pd.DataFrame:
    """Create test OHLCV data."""
    np.random.seed(seed)

    if trend == "up":
        drift = 0.002
    elif trend == "down":
        drift = -0.002
    else:
        drift = 0.0

    data = []
    price = base_price

    for i in range(n):
        price = price * (1 + np.random.normal(drift, 0.02))
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_ = price + np.random.normal(0, price * 0.005)

        data.append({
            "timestamp": datetime.now() - timedelta(hours=n - i),
            "open": open_,
            "high": high,
            "low": low,
            "close": price,
            "volume": np.random.uniform(1000, 5000),
        })

    return pd.DataFrame(data)


class TestSMCStrategyConfig:
    """Tests for SMCStrategyConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SMCStrategyConfig()

        assert config.min_confluence_score == 0.65
        assert config.require_htf_alignment == True
        assert config.min_rr_ratio == 2.0
        assert config.use_mtf_analysis == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SMCStrategyConfig(
            min_confluence_score=0.7,
            require_htf_alignment=False,
            min_rr_ratio=3.0,
        )

        assert config.min_confluence_score == 0.7
        assert config.require_htf_alignment == False
        assert config.min_rr_ratio == 3.0


class TestSMCStrategy:
    """Tests for SMCStrategy class."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = StrategyConfig(strategy_type=StrategyType.SMC)
        strategy = SMCStrategy(config=config)

        assert strategy.name == "smc_strategy"
        assert strategy.detector is not None
        assert strategy.confluence_engine is not None
        assert strategy.mtf_coordinator is not None

    def test_strategy_with_custom_smc_config(self):
        """Test strategy with custom SMC config."""
        config = StrategyConfig()
        smc_config = SMCStrategyConfig(
            min_confluence_score=0.8,
            min_rr_ratio=3.0,
        )
        strategy = SMCStrategy(config=config, smc_config=smc_config)

        assert strategy.smc_config.min_confluence_score == 0.8
        assert strategy.smc_config.min_rr_ratio == 3.0

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        config = StrategyConfig()
        strategy = SMCStrategy(config=config)

        candles = create_test_candles(n=30)  # Too few candles
        signals = strategy.generate_signals("BTC/USDT", candles)

        assert len(signals) == 0

    def test_generate_signals_with_data(self):
        """Test signal generation with sufficient data."""
        config = StrategyConfig()
        smc_config = SMCStrategyConfig(
            min_confluence_score=0.4,  # Lower threshold for testing
            min_rr_ratio=1.5,
        )
        strategy = SMCStrategy(config=config, smc_config=smc_config)

        candles = create_test_candles(n=150, trend="up")
        signals = strategy.generate_signals("BTC/USDT", candles)

        # May or may not generate signals depending on random data
        for signal in signals:
            assert signal.symbol == "BTC/USDT"
            assert signal.source == SignalSource.TECHNICAL
            assert 0 <= signal.confidence <= 1

    def test_update_htf_candles(self):
        """Test HTF candle update."""
        config = StrategyConfig()
        strategy = SMCStrategy(config=config)

        htf_candles = create_test_candles(n=100)
        strategy.update_htf_candles("BTC/USDT", htf_candles)

        assert "BTC/USDT" in strategy._htf_candles

    def test_update_ltf_candles(self):
        """Test LTF candle update."""
        config = StrategyConfig()
        strategy = SMCStrategy(config=config)

        ltf_candles = create_test_candles(n=100)
        strategy.update_ltf_candles("BTC/USDT", ltf_candles)

        assert "BTC/USDT" in strategy._ltf_candles


class TestShouldEnter:
    """Tests for should_enter method."""

    def test_should_not_enter_with_position(self):
        """Test that should_enter returns None with existing position."""
        config = StrategyConfig()
        strategy = SMCStrategy(config=config)

        candles = create_test_candles(n=150)

        # Mock position
        class MockPosition:
            side = OrderSide.BUY

        signal = strategy.should_enter("BTC/USDT", candles, MockPosition())
        assert signal is None

    def test_should_enter_without_position(self):
        """Test should_enter without existing position."""
        config = StrategyConfig()
        smc_config = SMCStrategyConfig(
            min_confluence_score=0.3,  # Very low for testing
            min_rr_ratio=1.0,
        )
        strategy = SMCStrategy(config=config, smc_config=smc_config)

        candles = create_test_candles(n=150, trend="up")

        signal = strategy.should_enter("BTC/USDT", candles, None)

        # May or may not generate signal
        if signal:
            assert signal.signal_type in [
                SignalType.BUY, SignalType.STRONG_BUY,
                SignalType.SELL, SignalType.STRONG_SELL,
            ]


class TestShouldExit:
    """Tests for should_exit method."""

    def test_should_exit_long_on_sell_signal(self):
        """Test exit for long position on sell signal."""
        config = StrategyConfig()
        smc_config = SMCStrategyConfig(
            min_confluence_score=0.3,
            min_rr_ratio=1.0,
        )
        strategy = SMCStrategy(config=config, smc_config=smc_config)

        # Create downtrending data more likely to generate sell signal
        candles = create_test_candles(n=150, trend="down")

        class MockPosition:
            side = OrderSide.BUY
            entry_price = 100.0

        signal = strategy.should_exit("BTC/USDT", candles, MockPosition())

        # May or may not generate exit signal
        if signal and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            assert "Exit long" in signal.reasoning


class TestSignalHistory:
    """Tests for signal history tracking."""

    def test_signal_history_tracking(self):
        """Test that signals are tracked in history."""
        config = StrategyConfig()
        smc_config = SMCStrategyConfig(
            min_confluence_score=0.3,
            min_rr_ratio=1.0,
        )
        strategy = SMCStrategy(config=config, smc_config=smc_config)

        candles = create_test_candles(n=150)
        strategy.generate_signals("BTC/USDT", candles)

        history = strategy.get_signal_history()
        assert isinstance(history, list)

    def test_get_analysis(self):
        """Test getting analysis for symbol."""
        config = StrategyConfig()
        strategy = SMCStrategy(config=config)

        # Enable MTF by providing HTF data
        htf_candles = create_test_candles(n=100)
        mtf_candles = create_test_candles(n=150)

        strategy.update_htf_candles("BTC/USDT", htf_candles)
        strategy.generate_signals("BTC/USDT", mtf_candles)

        analysis = strategy.get_analysis("BTC/USDT")
        # May or may not have analysis depending on data


class TestCreateSMCStrategy:
    """Tests for create_smc_strategy factory function."""

    def test_create_with_defaults(self):
        """Test creation with default config."""
        strategy = create_smc_strategy()

        assert isinstance(strategy, SMCStrategy)
        assert strategy.config is not None

    def test_create_with_custom_config(self):
        """Test creation with custom configs."""
        config = StrategyConfig(
            name="custom_smc",
            primary_timeframe=TimeFrame.H4,
        )
        smc_config = SMCStrategyConfig(
            min_confluence_score=0.7,
        )

        strategy = create_smc_strategy(config=config, smc_config=smc_config)

        assert strategy.config.name == "custom_smc"
        assert strategy.config.primary_timeframe == TimeFrame.H4
        assert strategy.smc_config.min_confluence_score == 0.7


class TestStrategyStatus:
    """Tests for strategy status method."""

    def test_get_status(self):
        """Test getting strategy status."""
        config = StrategyConfig()
        strategy = SMCStrategy(config=config)

        status = strategy.get_status()

        assert "name" in status
        assert "smc_config" in status
        assert "state" in status
