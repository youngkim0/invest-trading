"""Hybrid RL + LLM + SMC trading strategy.

Combines reinforcement learning signals with LLM market analysis
and Smart Money Concepts for comprehensive market analysis.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ai.llm.graph import TradingWorkflow
from ai.rl.agents.ppo_agent import PPOTradingAgent
from ai.rl.environment import TradingEnvConfig
from config.strategies import HybridStrategyConfig, HybridSMCStrategyConfig, StrategyConfig, StrategyType
from core.strategies.base_strategy import BaseStrategy, TradeSignal
from data.features.smc.confluence import ConfluenceEngine
from data.features.smc.detector import SMCDetector
from data.features.smc.zones import ZoneDirection
from data.features.technical import TechnicalIndicators
from data.storage.models import OrderSide, SignalSource, SignalType


class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining RL agent with LLM analysis and SMC."""

    def __init__(
        self,
        config: StrategyConfig,
        rl_model_path: str | None = None,
        name: str = "hybrid_strategy",
        enable_smc: bool = False,
    ):
        """Initialize hybrid strategy.

        Args:
            config: Strategy configuration
            rl_model_path: Path to trained RL model
            name: Strategy name
            enable_smc: Enable SMC signal integration
        """
        super().__init__(config, name)

        # Determine if SMC should be enabled
        self.enable_smc = enable_smc or config.strategy_type == StrategyType.HYBRID_SMC

        # Use appropriate config based on SMC enablement
        if self.enable_smc:
            self.hybrid_smc_config: HybridSMCStrategyConfig = config.hybrid_smc_config
            self.hybrid_config = HybridStrategyConfig(
                rl_config=self.hybrid_smc_config.rl_config,
                llm_config=self.hybrid_smc_config.llm_config,
                rl_weight=self.hybrid_smc_config.rl_weight,
                llm_weight=self.hybrid_smc_config.llm_weight,
            )
        else:
            self.hybrid_config: HybridStrategyConfig = config.hybrid_config
            self.hybrid_smc_config = None

        # Initialize RL agent
        self.rl_agent: PPOTradingAgent | None = None
        self.rl_model_path = rl_model_path

        if rl_model_path:
            self._load_rl_agent(rl_model_path)

        # Initialize LLM workflow
        self.llm_workflow = TradingWorkflow()

        # Initialize SMC components if enabled
        self.smc_detector: SMCDetector | None = None
        self.confluence_engine: ConfluenceEngine | None = None

        if self.enable_smc:
            smc_cfg = self.hybrid_smc_config.smc_config
            self.smc_detector = SMCDetector(
                lookback=100,
                atr_period=config.indicators.atr_period,
                min_impulse_atr=smc_cfg.min_impulse_atr,
                swing_lookback=smc_cfg.swing_lookback,
            )
            self.confluence_engine = ConfluenceEngine(
                min_confluence_score=smc_cfg.min_confluence_score,
                min_rr_ratio=smc_cfg.min_rr_ratio,
                atr_stop_multiplier=smc_cfg.atr_stop_multiplier,
            )
            logger.info("SMC integration enabled for hybrid strategy")

        # Signal history
        self.rl_signals: list[dict[str, Any]] = []
        self.llm_signals: list[dict[str, Any]] = []
        self.smc_signals: list[dict[str, Any]] = []
        self.combined_signals: list[TradeSignal] = []

        # Feature cache
        self._indicator_cache: dict[str, dict[str, Any]] = {}
        self._smc_cache: dict[str, dict[str, Any]] = {}

    def _load_rl_agent(self, model_path: str) -> None:
        """Load trained RL agent."""
        try:
            env_config = TradingEnvConfig(
                lookback_window=self.config.rl_config.include_technical_indicators and 60 or 30,
                action_type=self.config.rl_config.action_type,
            )

            self.rl_agent = PPOTradingAgent(env_config=env_config)
            self.rl_agent.load(model_path)
            logger.info(f"RL agent loaded from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
            self.rl_agent = None

    def generate_signals(
        self,
        symbol: str,
        candles: pd.DataFrame,
    ) -> list[TradeSignal]:
        """Generate trading signals combining RL, LLM, and optionally SMC.

        Args:
            symbol: Trading symbol
            candles: OHLCV DataFrame

        Returns:
            List of trade signals
        """
        if len(candles) < 60:
            return []

        signals = []

        # Calculate indicators
        indicators = self._calculate_indicators(symbol, candles)
        self._indicator_cache[symbol] = indicators

        # Get RL signal
        rl_signal = self._get_rl_signal(symbol, candles, indicators)

        # Get LLM signal (async - simplified for sync context)
        llm_signal = self._get_llm_signal_sync(symbol, candles, indicators)

        # Get SMC signal if enabled
        smc_signal = None
        if self.enable_smc:
            smc_signal = self._get_smc_signal(symbol, candles, indicators)

        # Combine signals
        combined = self._combine_signals(symbol, rl_signal, llm_signal, candles, smc_signal)

        if combined:
            signals.append(combined)
            self.combined_signals.append(combined)

        return signals

    def _calculate_indicators(
        self,
        symbol: str,
        candles: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate technical indicators."""
        ti = TechnicalIndicators(candles)

        indicators = {}

        try:
            # Trend
            indicators["sma_20"] = float(ti.sma(20).iloc[-1])
            indicators["sma_50"] = float(ti.sma(50).iloc[-1])
            indicators["ema_12"] = float(ti.ema(12).iloc[-1])
            indicators["ema_26"] = float(ti.ema(26).iloc[-1])

            macd = ti.macd()
            indicators["macd"] = float(macd["macd"].iloc[-1])
            indicators["macd_signal"] = float(macd["signal"].iloc[-1])
            indicators["macd_histogram"] = float(macd["histogram"].iloc[-1])

            # Momentum
            indicators["rsi"] = float(ti.rsi().iloc[-1])

            stoch = ti.stochastic()
            indicators["stoch_k"] = float(stoch["stoch_k"].iloc[-1])
            indicators["stoch_d"] = float(stoch["stoch_d"].iloc[-1])

            # Volatility
            indicators["atr"] = float(ti.atr().iloc[-1])

            bb = ti.bollinger_bands()
            indicators["bb_upper"] = float(bb["upper"].iloc[-1])
            indicators["bb_lower"] = float(bb["lower"].iloc[-1])
            indicators["bb_pband"] = float(bb["pband"].iloc[-1])

            # Current price
            indicators["close"] = float(candles["close"].iloc[-1])

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")

        return indicators

    def _get_rl_signal(
        self,
        symbol: str,
        candles: pd.DataFrame,
        indicators: dict[str, Any],
    ) -> dict[str, Any]:
        """Get signal from RL agent."""
        if not self.rl_agent:
            return {"action": "hold", "confidence": 0.5}

        try:
            # Prepare observation (simplified - in production match environment exactly)
            obs = self._prepare_rl_observation(candles, indicators)

            action, info = self.rl_agent.predict(obs, deterministic=True)

            # Convert action to signal
            action_map = {0: "hold", 1: "buy", 2: "sell"}
            action_str = action_map.get(int(action[0]), "hold")

            # Get Q-values for confidence if available
            confidence = 0.7  # Default confidence
            if "q_values" in info and info["q_values"] is not None:
                q_values = info["q_values"]
                # Softmax for confidence
                exp_q = np.exp(q_values - np.max(q_values))
                probs = exp_q / exp_q.sum()
                confidence = float(probs[int(action[0])])

            signal = {
                "action": action_str,
                "confidence": confidence,
                "raw_action": int(action[0]),
            }

            self.rl_signals.append({
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                **signal,
            })

            return signal

        except Exception as e:
            logger.error(f"RL signal generation failed: {e}")
            return {"action": "hold", "confidence": 0.5}

    def _prepare_rl_observation(
        self,
        candles: pd.DataFrame,
        indicators: dict[str, Any],
    ) -> np.ndarray:
        """Prepare observation for RL agent."""
        # Get last N candles
        lookback = 60
        recent = candles.tail(lookback)

        # Normalize prices relative to current
        current_price = float(recent["close"].iloc[-1])

        features = []

        # OHLCV normalized
        for col in ["open", "high", "low", "close"]:
            normalized = (recent[col].astype(float) - current_price) / current_price
            features.append(normalized.values)

        # Volume normalized
        vol_mean = recent["volume"].mean()
        vol_std = recent["volume"].std() + 1e-8
        vol_norm = (recent["volume"].astype(float) - vol_mean) / vol_std
        features.append(vol_norm.values)

        # Add indicators (broadcast to all timesteps)
        ind_features = []
        if indicators.get("rsi") is not None:
            ind_features.append(indicators["rsi"] / 100)
        if indicators.get("macd") is not None:
            ind_features.append(indicators["macd"] / current_price)
        if indicators.get("bb_pband") is not None:
            ind_features.append(indicators["bb_pband"] / 100)

        # Stack features
        obs = np.column_stack(features)

        # Add indicator columns
        if ind_features:
            ind_array = np.tile(ind_features, (len(obs), 1))
            obs = np.column_stack([obs, ind_array])

        # Pad if needed
        if len(obs) < lookback:
            padding = np.zeros((lookback - len(obs), obs.shape[1]))
            obs = np.vstack([padding, obs])

        return obs.astype(np.float32)

    def _get_llm_signal_sync(
        self,
        symbol: str,
        candles: pd.DataFrame,
        indicators: dict[str, Any],
    ) -> dict[str, Any]:
        """Get LLM signal (synchronous wrapper).

        In production, this should be async.
        """
        import asyncio

        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run async workflow
            decision = loop.run_until_complete(
                self.llm_workflow.run(
                    symbol=symbol,
                    current_price=indicators.get("close", 0.0),
                    technical_indicators=indicators,
                    news_articles=[],  # Would add news here
                )
            )

            signal_map = {
                "strong_buy": "buy",
                "buy": "buy",
                "hold": "hold",
                "sell": "sell",
                "strong_sell": "sell",
            }

            signal = {
                "action": signal_map.get(decision.signal.value, "hold"),
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "risk_level": decision.risk_level,
            }

            self.llm_signals.append({
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                **signal,
            })

            return signal

        except Exception as e:
            logger.error(f"LLM signal generation failed: {e}")
            return {"action": "hold", "confidence": 0.5}

    def _get_smc_signal(
        self,
        symbol: str,
        candles: pd.DataFrame,
        indicators: dict[str, Any],
    ) -> dict[str, Any]:
        """Get signal from SMC analysis.

        Args:
            symbol: Trading symbol
            candles: OHLCV DataFrame
            indicators: Technical indicators

        Returns:
            SMC signal dict with action, confidence, and zone info
        """
        if not self.smc_detector or not self.confluence_engine:
            return {"action": "hold", "confidence": 0.5}

        try:
            # Run SMC detection
            analysis = self.smc_detector.analyze(candles)

            if not analysis:
                return {"action": "hold", "confidence": 0.5}

            current_price = float(candles["close"].iloc[-1])
            atr = analysis.get("atr", 0.0)

            if atr <= 0:
                return {"action": "hold", "confidence": 0.5}

            # Get confluence
            confluence = self.confluence_engine.analyze(
                current_price=current_price,
                order_blocks=analysis.get("order_blocks", []),
                fair_value_gaps=analysis.get("fair_value_gaps", []),
                liquidity_sweeps=analysis.get("liquidity_sweeps", []),
                channels=analysis.get("channels", []),
                market_structure=analysis.get("market_structure"),
                atr=atr,
                htf_bias=None,  # Could integrate with MTF here
            )

            if not confluence:
                return {"action": "hold", "confidence": 0.5}

            # Convert confluence to signal
            action = "buy" if confluence.direction == ZoneDirection.BULLISH else "sell"

            signal = {
                "action": action,
                "confidence": confluence.confidence,
                "confluence_score": confluence.score,
                "risk_reward": confluence.risk_reward,
                "entry_zone": confluence.entry_zone,
                "stop_loss": confluence.stop_loss,
                "take_profit": confluence.take_profit_1,
                "zones_count": len(confluence.zones),
                "factors": confluence.factors,
            }

            # Cache for later use
            self._smc_cache[symbol] = {
                "analysis": analysis,
                "confluence": confluence,
            }

            self.smc_signals.append({
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                **signal,
            })

            return signal

        except Exception as e:
            logger.error(f"SMC signal generation failed: {e}")
            return {"action": "hold", "confidence": 0.5}

    def _combine_signals(
        self,
        symbol: str,
        rl_signal: dict[str, Any],
        llm_signal: dict[str, Any],
        candles: pd.DataFrame,
        smc_signal: dict[str, Any] | None = None,
    ) -> TradeSignal | None:
        """Combine RL, LLM, and optionally SMC signals."""
        rl_action = rl_signal.get("action", "hold")
        rl_confidence = rl_signal.get("confidence", 0.5)

        llm_action = llm_signal.get("action", "hold")
        llm_confidence = llm_signal.get("confidence", 0.5)

        # Get SMC signal data if available
        smc_action = "hold"
        smc_confidence = 0.5
        smc_confluence_score = 0.0

        if smc_signal and self.enable_smc:
            smc_action = smc_signal.get("action", "hold")
            smc_confidence = smc_signal.get("confidence", 0.5)
            smc_confluence_score = smc_signal.get("confluence_score", 0.0)

        # Weights from config
        if self.enable_smc and self.hybrid_smc_config:
            rl_weight = self.hybrid_smc_config.rl_weight
            llm_weight = self.hybrid_smc_config.llm_weight
            smc_weight = self.hybrid_smc_config.smc_weight
        else:
            rl_weight = self.hybrid_config.rl_weight
            llm_weight = self.hybrid_config.llm_weight
            smc_weight = 0.0

        # Convert actions to numeric
        action_values = {"sell": -1, "hold": 0, "buy": 1}
        rl_value = action_values.get(rl_action, 0)
        llm_value = action_values.get(llm_action, 0)
        smc_value = action_values.get(smc_action, 0)

        # LLM veto check
        if self.hybrid_config.llm_veto_threshold:
            if llm_confidence > self.hybrid_config.llm_veto_threshold:
                if llm_value * rl_value < 0:  # Conflicting signals
                    logger.info(f"LLM veto activated for {symbol}: LLM={llm_action}, RL={rl_action}")
                    if self.enable_smc:
                        rl_weight = 0.25
                        llm_weight = 0.50
                        smc_weight = 0.25
                    else:
                        rl_weight = 0.3
                        llm_weight = 0.7

        # SMC veto check (high confluence can influence decision)
        if self.enable_smc and self.hybrid_smc_config:
            if smc_confluence_score > self.hybrid_smc_config.smc_veto_threshold:
                # SMC has high confluence, increase its weight
                if smc_value != 0:  # SMC has a directional signal
                    logger.info(f"SMC confluence boost for {symbol}: score={smc_confluence_score:.2f}")
                    rl_weight = 0.35
                    llm_weight = 0.20
                    smc_weight = 0.45

            # Zone filter: if enabled, only trade when SMC agrees
            if self.hybrid_smc_config.smc_zone_filter and smc_value == 0:
                # SMC is neutral (no valid zone), reduce signal strength
                rl_weight *= 0.5
                llm_weight *= 0.5

        # Weighted combination
        if self.enable_smc:
            combined_value = rl_weight * rl_value + llm_weight * llm_value + smc_weight * smc_value
            combined_confidence = rl_weight * rl_confidence + llm_weight * llm_confidence + smc_weight * smc_confidence
        else:
            combined_value = rl_weight * rl_value + llm_weight * llm_value
            combined_confidence = rl_weight * rl_confidence + llm_weight * llm_confidence

        # Determine final action
        if combined_value > 0.3:
            final_action = "buy"
            signal_type = SignalType.BUY if combined_value < 0.7 else SignalType.STRONG_BUY
            side = OrderSide.BUY
        elif combined_value < -0.3:
            final_action = "sell"
            signal_type = SignalType.SELL if combined_value > -0.7 else SignalType.STRONG_SELL
            side = OrderSide.SELL
        else:
            # Hold - don't generate signal
            return None

        # Get current price
        current_price = Decimal(str(candles["close"].iloc[-1]))

        # Use SMC zones for stop loss/take profit if available
        stop_loss = None
        take_profit = None

        if smc_signal and self.enable_smc and smc_action == final_action:
            # Use SMC-calculated levels
            if "stop_loss" in smc_signal and smc_signal["stop_loss"]:
                stop_loss = Decimal(str(smc_signal["stop_loss"]))
            if "take_profit" in smc_signal and smc_signal["take_profit"]:
                take_profit = Decimal(str(smc_signal["take_profit"]))

        # Fallback to ATR-based calculation
        if stop_loss is None:
            atr = Decimal(str(self._indicator_cache.get(symbol, {}).get("atr", 0)))
            stop_loss = self.calculate_stop_loss(current_price, side, atr if atr > 0 else None)

        if take_profit is None:
            take_profit = self.calculate_take_profit(current_price, stop_loss, side)

        # Build reasoning string
        reasoning_parts = [
            f"RL: {rl_action} ({rl_confidence:.2f})",
            f"LLM: {llm_action} ({llm_confidence:.2f})",
        ]
        if self.enable_smc:
            reasoning_parts.append(f"SMC: {smc_action} ({smc_confidence:.2f}, confluence={smc_confluence_score:.2f})")

        # Create signal
        signal = TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.HYBRID,
            confidence=combined_confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            side=side,
            timeframe=self.config.primary_timeframe,
            indicators=self._indicator_cache.get(symbol, {}),
            reasoning=", ".join(reasoning_parts),
            metadata={
                "rl_signal": rl_signal,
                "llm_signal": llm_signal,
                "smc_signal": smc_signal,
                "combined_value": combined_value,
                "weights": {
                    "rl": rl_weight,
                    "llm": llm_weight,
                    "smc": smc_weight if self.enable_smc else 0,
                },
                "smc_enabled": self.enable_smc,
            },
        )

        return signal

    def should_enter(
        self,
        symbol: str,
        candles: pd.DataFrame,
        current_position: Any | None = None,
    ) -> TradeSignal | None:
        """Check if should enter a position."""
        if current_position is not None:
            return None  # Already have position

        signals = self.generate_signals(symbol, candles)

        # Filter for entry signals
        for signal in signals:
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY,
                                      SignalType.SELL, SignalType.STRONG_SELL]:
                if signal.confidence >= self.config.llm_config.confidence_threshold:
                    return signal

        return None

    def should_exit(
        self,
        symbol: str,
        candles: pd.DataFrame,
        position: Any,
    ) -> TradeSignal | None:
        """Check if should exit a position."""
        signals = self.generate_signals(symbol, candles)

        # Get position side
        position_side = getattr(position, "side", None)

        for signal in signals:
            # Exit long on sell signal
            if position_side == OrderSide.BUY:
                if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    signal.reasoning = f"Exit long: {signal.reasoning}"
                    return signal

            # Exit short on buy signal
            elif position_side == OrderSide.SELL:
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    signal.reasoning = f"Exit short: {signal.reasoning}"
                    return signal

        return None

    def get_signal_history(self) -> dict[str, Any]:
        """Get signal history for analysis."""
        result = {
            "rl_signals": self.rl_signals[-100:],
            "llm_signals": self.llm_signals[-100:],
            "combined_signals": [s.to_dict() for s in self.combined_signals[-100:]],
        }

        if self.enable_smc:
            result["smc_signals"] = self.smc_signals[-100:]

        return result

    def get_smc_analysis(self, symbol: str) -> dict[str, Any] | None:
        """Get cached SMC analysis for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            SMC analysis dict or None
        """
        if not self.enable_smc:
            return None
        return self._smc_cache.get(symbol)

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        status = super().get_status()
        status["smc_enabled"] = self.enable_smc

        if self.enable_smc and self.hybrid_smc_config:
            status["weights"] = {
                "rl": self.hybrid_smc_config.rl_weight,
                "llm": self.hybrid_smc_config.llm_weight,
                "smc": self.hybrid_smc_config.smc_weight,
            }

        return status


def create_hybrid_strategy(
    config: StrategyConfig | None = None,
    rl_model_path: str | None = None,
    enable_smc: bool = False,
) -> HybridStrategy:
    """Create hybrid strategy instance.

    Args:
        config: Strategy configuration
        rl_model_path: Path to trained RL model
        enable_smc: Enable SMC signal integration

    Returns:
        Configured hybrid strategy
    """
    if config is None:
        config = StrategyConfig()

    return HybridStrategy(
        config=config,
        rl_model_path=rl_model_path,
        enable_smc=enable_smc,
    )


def create_hybrid_smc_strategy(
    config: StrategyConfig | None = None,
    rl_model_path: str | None = None,
) -> HybridStrategy:
    """Create hybrid strategy with SMC enabled.

    Convenience function that creates a hybrid strategy with SMC integration.

    Args:
        config: Strategy configuration
        rl_model_path: Path to trained RL model

    Returns:
        Configured hybrid strategy with SMC
    """
    if config is None:
        config = StrategyConfig(strategy_type=StrategyType.HYBRID_SMC)
    elif config.strategy_type != StrategyType.HYBRID_SMC:
        config.strategy_type = StrategyType.HYBRID_SMC

    return HybridStrategy(
        config=config,
        rl_model_path=rl_model_path,
        enable_smc=True,
    )
