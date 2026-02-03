"""Basic tests to verify imports and configurations."""

import pytest


class TestImports:
    """Test that all major modules can be imported."""

    def test_config_imports(self):
        """Test config module imports."""
        from config import get_settings
        from config.exchanges import BinanceConfig, AlpacaConfig
        from config.strategies import StrategyConfig, StrategyType

        settings = get_settings()
        assert settings is not None

    def test_data_storage_imports(self):
        """Test data storage imports."""
        from data.storage.models import (
            OHLCV,
            Order,
            Position,
            Signal,
            TradeLog,
            PerformanceSnapshot,
        )
        from data.storage.repository import DatabaseManager

        assert OHLCV is not None
        assert DatabaseManager is not None

    def test_feature_imports(self):
        """Test feature engineering imports."""
        from data.features.technical import TechnicalIndicators

        assert TechnicalIndicators is not None

    def test_rl_imports(self):
        """Test RL module imports."""
        from ai.rl.environment import TradingEnv, TradingEnvConfig
        from ai.rl.rewards import SharpeReward, SortinoReward

        assert TradingEnv is not None
        assert SharpeReward is not None

    def test_rl_agent_imports(self):
        """Test RL agent imports (requires stable-baselines3)."""
        pytest.importorskip("stable_baselines3", reason="stable-baselines3 not installed")

        from ai.rl.agents.ppo_agent import PPOTradingAgent
        from ai.rl.agents.dqn_agent import DQNTradingAgent

        assert PPOTradingAgent is not None
        assert DQNTradingAgent is not None

    def test_llm_imports(self):
        """Test LLM module imports."""
        pytest.importorskip("langchain_core", reason="langchain not installed")

        from ai.llm.market_analyst import MarketAnalyst
        from ai.llm.news_analyzer import NewsAnalyzer
        from ai.llm.graph import TradingWorkflow

        assert MarketAnalyst is not None
        assert TradingWorkflow is not None

    def test_strategy_imports(self):
        """Test strategy imports."""
        from core.strategies.base_strategy import BaseStrategy, TradeSignal

        assert BaseStrategy is not None
        assert TradeSignal is not None

    def test_hybrid_strategy_imports(self):
        """Test hybrid strategy imports (requires langchain)."""
        pytest.importorskip("langchain_core", reason="langchain not installed")

        from core.strategies.hybrid_strategy import HybridStrategy

        assert HybridStrategy is not None

    def test_journal_imports(self):
        """Test journal imports."""
        from journal.trade_logger import TradeLogger, TradeEntry
        from journal.performance import PerformanceAnalyzer

        assert TradeLogger is not None
        assert PerformanceAnalyzer is not None

    def test_feedback_loop_imports(self):
        """Test feedback loop imports (requires langchain)."""
        pytest.importorskip("langchain_core", reason="langchain not installed")

        from journal.feedback_loop import FeedbackLoop

        assert FeedbackLoop is not None


class TestTechnicalIndicators:
    """Test technical indicators."""

    def test_indicator_calculation(self):
        """Test basic indicator calculations."""
        import pandas as pd
        import numpy as np
        from data.features.technical import TechnicalIndicators

        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame({
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 101,
            "low": np.random.randn(100).cumsum() + 99,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

        # Ensure high > low > 0
        data["high"] = data[["open", "high", "low", "close"]].max(axis=1)
        data["low"] = data[["open", "high", "low", "close"]].min(axis=1)
        data = data.abs() + 1  # Ensure positive values

        ti = TechnicalIndicators(data)

        # Test SMA
        sma = ti.sma(20)
        assert len(sma) == len(data)
        assert not sma.iloc[20:].isna().all()

        # Test RSI
        rsi = ti.rsi(14)
        assert len(rsi) == len(data)

        # Test MACD
        macd = ti.macd()
        assert "macd" in macd
        assert "signal" in macd


class TestRLEnvironment:
    """Test RL environment."""

    def test_environment_creation(self):
        """Test creating RL environment."""
        import numpy as np
        import pandas as pd
        from ai.rl.environment import TradingEnv, TradingEnvConfig

        # Create sample data first
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame({
            "open": np.abs(np.random.randn(100).cumsum()) + 100,
            "high": np.abs(np.random.randn(100).cumsum()) + 101,
            "low": np.abs(np.random.randn(100).cumsum()) + 99,
            "close": np.abs(np.random.randn(100).cumsum()) + 100,
            "volume": np.random.randint(1000, 10000, 100).astype(float),
        }, index=dates)

        # Ensure high > low
        data["high"] = data[["open", "high", "low", "close"]].max(axis=1)
        data["low"] = data[["open", "high", "low", "close"]].min(axis=1)

        config = TradingEnvConfig(
            initial_balance=10000.0,
            lookback_window=30,
        )

        env = TradingEnv(df=data, config=config)

        # Test reset
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))


class TestRewardFunctions:
    """Test reward functions."""

    def test_sharpe_reward(self):
        """Test Sharpe ratio reward."""
        from ai.rl.rewards import SharpeReward, RewardMetrics

        reward_fn = SharpeReward(window=20)

        # Create metrics with sample data
        metrics = RewardMetrics(
            returns=[0.01] * 25,
            portfolio_values=[10000 + i * 100 for i in range(25)],
            positions=[1.0] * 25,
            transaction_count=5,
            drawdown=0.02,
            max_drawdown=0.05,
        )

        r = reward_fn.calculate(metrics)
        assert isinstance(r, float)

    def test_sortino_reward(self):
        """Test Sortino ratio reward."""
        from ai.rl.rewards import SortinoReward, RewardMetrics

        reward_fn = SortinoReward(window=20)

        # Create metrics with some negative returns
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 5
        metrics = RewardMetrics(
            returns=returns,
            portfolio_values=[10000 + i * 50 for i in range(25)],
            positions=[1.0] * 25,
            transaction_count=5,
            drawdown=0.03,
            max_drawdown=0.08,
        )

        r = reward_fn.calculate(metrics)
        assert isinstance(r, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
