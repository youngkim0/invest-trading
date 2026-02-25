"""SMC (Smart Money Concepts) trading strategy.

Generates trading signals based on Smart Money Concepts analysis:
- Order Blocks
- Fair Value Gaps
- Liquidity Sweeps
- Multi-Timeframe Analysis
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger

from config.strategies import StrategyConfig, TimeFrame
from core.strategies.base_strategy import BaseStrategy, TradeSignal
from data.features.smc.confluence import ConfluenceEngine, ConfluenceResult
from data.features.smc.detector import SMCDetector
from data.features.smc.mtf import MTFCoordinator, MTFAnalysis
from data.features.smc.zones import ZoneDirection
from data.storage.models import OrderSide, SignalSource, SignalType


class SMCStrategyConfig:
    """Configuration for SMC strategy.

    Attributes:
        min_confluence_score: Minimum confluence score for entry (default 0.65)
        require_htf_alignment: Require HTF bias alignment (default True)
        min_rr_ratio: Minimum risk/reward ratio (default 2.0)
        use_mtf_analysis: Use multi-timeframe analysis (default True)
        entry_on_retest: Wait for zone retest (default True)
        max_zones_per_direction: Max zones to track per direction (default 5)
        atr_stop_multiplier: ATR multiplier for stop loss (default 1.5)
        trailing_stop_threshold: Profit % to activate trailing (default 1.5)
    """

    def __init__(
        self,
        min_confluence_score: float = 0.65,
        require_htf_alignment: bool = True,
        min_rr_ratio: float = 2.0,
        use_mtf_analysis: bool = True,
        entry_on_retest: bool = True,
        max_zones_per_direction: int = 5,
        atr_stop_multiplier: float = 1.5,
        trailing_stop_threshold: float = 1.5,
    ):
        self.min_confluence_score = min_confluence_score
        self.require_htf_alignment = require_htf_alignment
        self.min_rr_ratio = min_rr_ratio
        self.use_mtf_analysis = use_mtf_analysis
        self.entry_on_retest = entry_on_retest
        self.max_zones_per_direction = max_zones_per_direction
        self.atr_stop_multiplier = atr_stop_multiplier
        self.trailing_stop_threshold = trailing_stop_threshold


class SMCStrategy(BaseStrategy):
    """Smart Money Concepts trading strategy.

    Uses SMC patterns (Order Blocks, FVGs, Liquidity Sweeps) combined with
    multi-timeframe analysis to generate high-probability trade signals.
    """

    def __init__(
        self,
        config: StrategyConfig,
        smc_config: SMCStrategyConfig | None = None,
        name: str = "smc_strategy",
    ):
        """Initialize SMC strategy.

        Args:
            config: Base strategy configuration
            smc_config: SMC-specific configuration
            name: Strategy name
        """
        super().__init__(config, name)

        self.smc_config = smc_config or SMCStrategyConfig()

        # Initialize SMC components
        self.detector = SMCDetector(
            lookback=100,
            atr_period=config.indicators.atr_period,
            min_impulse_atr=2.0,
        )

        self.confluence_engine = ConfluenceEngine(
            min_confluence_score=self.smc_config.min_confluence_score,
            min_rr_ratio=self.smc_config.min_rr_ratio,
            atr_stop_multiplier=self.smc_config.atr_stop_multiplier,
        )

        self.mtf_coordinator = MTFCoordinator(
            min_confluence=self.smc_config.min_confluence_score,
            min_rr_ratio=self.smc_config.min_rr_ratio,
        )

        # Multi-timeframe data storage
        self._htf_candles: dict[str, pd.DataFrame] = {}
        self._ltf_candles: dict[str, pd.DataFrame] = {}

        # Analysis cache
        self._last_analysis: dict[str, MTFAnalysis] = {}
        self._smc_signals: list[dict[str, Any]] = []

        logger.info(f"SMC Strategy initialized with config: {self.smc_config.__dict__}")

    def update_htf_candles(self, symbol: str, candles: pd.DataFrame) -> None:
        """Update higher timeframe candles for a symbol.

        Args:
            symbol: Trading symbol
            candles: HTF OHLCV DataFrame
        """
        self._htf_candles[symbol] = candles

    def update_ltf_candles(self, symbol: str, candles: pd.DataFrame) -> None:
        """Update lower timeframe candles for a symbol.

        Args:
            symbol: Trading symbol
            candles: LTF OHLCV DataFrame
        """
        self._ltf_candles[symbol] = candles

    def generate_signals(
        self,
        symbol: str,
        candles: pd.DataFrame,
    ) -> list[TradeSignal]:
        """Generate trading signals from SMC analysis.

        Args:
            symbol: Trading symbol
            candles: OHLCV DataFrame (primary timeframe)

        Returns:
            List of trade signals
        """
        if len(candles) < 60:
            return []

        signals = []

        # Get multi-timeframe data if available
        htf_candles = self._htf_candles.get(symbol)
        ltf_candles = self._ltf_candles.get(symbol)

        # Run analysis
        if self.smc_config.use_mtf_analysis and htf_candles is not None:
            analysis = self.mtf_coordinator.analyze(
                htf_candles=htf_candles,
                mtf_candles=candles,
                ltf_candles=ltf_candles,
            )
            self._last_analysis[symbol] = analysis

            if analysis.is_valid_setup and analysis.confluence:
                signal = self._create_signal_from_mtf(symbol, candles, analysis)
                if signal:
                    signals.append(signal)
        else:
            # Single timeframe analysis
            signal = self._analyze_single_timeframe(symbol, candles)
            if signal:
                signals.append(signal)

        # Log signals
        for signal in signals:
            self._smc_signals.append({
                "timestamp": datetime.now(timezone.utc),
                "symbol": symbol,
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "entry_price": str(signal.entry_price),
            })

        return signals

    def _analyze_single_timeframe(
        self,
        symbol: str,
        candles: pd.DataFrame,
    ) -> TradeSignal | None:
        """Analyze single timeframe for SMC setups.

        Args:
            symbol: Trading symbol
            candles: OHLCV DataFrame

        Returns:
            Trade signal or None
        """
        # Run SMC detection
        analysis = self.detector.analyze(candles)

        if not analysis:
            return None

        current_price = float(candles["close"].iloc[-1])
        atr = analysis.get("atr", 0.0)

        if atr <= 0:
            return None

        # Find confluence
        confluence = self.confluence_engine.analyze(
            current_price=current_price,
            order_blocks=analysis.get("order_blocks", []),
            fair_value_gaps=analysis.get("fair_value_gaps", []),
            liquidity_sweeps=analysis.get("liquidity_sweeps", []),
            channels=analysis.get("channels", []),
            market_structure=analysis.get("market_structure"),
            atr=atr,
            htf_bias=None,  # No HTF in single TF mode
        )

        if not confluence:
            return None

        return self._create_signal_from_confluence(symbol, candles, confluence, atr)

    def _create_signal_from_mtf(
        self,
        symbol: str,
        candles: pd.DataFrame,
        analysis: MTFAnalysis,
    ) -> TradeSignal | None:
        """Create trade signal from MTF analysis.

        Args:
            symbol: Trading symbol
            candles: Primary timeframe candles
            analysis: MTF analysis result

        Returns:
            Trade signal or None
        """
        if not analysis.confluence:
            return None

        # Check HTF alignment requirement
        if self.smc_config.require_htf_alignment and not analysis.aligned:
            logger.debug(f"Skipping {symbol}: HTF not aligned")
            return None

        confluence = analysis.confluence
        current_price = float(candles["close"].iloc[-1])

        # Determine signal type
        if confluence.direction == ZoneDirection.BULLISH:
            if confluence.score >= 0.8:
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY
            side = OrderSide.BUY
        else:
            if confluence.score >= 0.8:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
            side = OrderSide.SELL

        # Entry zone handling
        if self.smc_config.entry_on_retest:
            # Use zone midpoint as entry
            entry_price = Decimal(str((confluence.entry_zone[0] + confluence.entry_zone[1]) / 2))
        else:
            # Use current price
            entry_price = Decimal(str(current_price))

        # Build reasoning
        factors_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in confluence.factors.items()
        )
        reasoning = (
            f"SMC MTF: {confluence.direction.value}, "
            f"score={confluence.score:.2f}, "
            f"aligned={analysis.aligned}, "
            f"R:R={confluence.risk_reward:.1f}, "
            f"factors=[{factors_str}]"
        )

        return TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.TECHNICAL,  # SMC is a form of technical analysis
            confidence=confluence.confidence,
            entry_price=entry_price,
            stop_loss=Decimal(str(confluence.stop_loss)),
            take_profit=Decimal(str(confluence.take_profit_1)),
            side=side,
            timeframe=self.config.primary_timeframe,
            indicators={
                "confluence_score": confluence.score,
                "entry_quality": analysis.entry_quality,
                "htf_aligned": analysis.aligned,
                "zones_count": len(confluence.zones),
            },
            reasoning=reasoning,
            metadata={
                "smc_analysis": {
                    "direction": confluence.direction.value,
                    "confluence_score": confluence.score,
                    "risk_reward": confluence.risk_reward,
                    "entry_zone": confluence.entry_zone,
                    "factors": confluence.factors,
                    "take_profit_2": confluence.take_profit_2,
                    "take_profit_3": confluence.take_profit_3,
                },
                "mtf_aligned": analysis.aligned,
                "entry_quality": analysis.entry_quality,
                "recommendations": analysis.recommendations,
            },
        )

    def _create_signal_from_confluence(
        self,
        symbol: str,
        candles: pd.DataFrame,
        confluence: ConfluenceResult,
        atr: float,
    ) -> TradeSignal | None:
        """Create trade signal from confluence result.

        Args:
            symbol: Trading symbol
            candles: OHLCV DataFrame
            confluence: Confluence analysis result
            atr: ATR value

        Returns:
            Trade signal or None
        """
        current_price = float(candles["close"].iloc[-1])

        # Determine signal type
        if confluence.direction == ZoneDirection.BULLISH:
            if confluence.score >= 0.8:
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY
            side = OrderSide.BUY
        else:
            if confluence.score >= 0.8:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
            side = OrderSide.SELL

        # Entry price
        if self.smc_config.entry_on_retest:
            entry_price = Decimal(str((confluence.entry_zone[0] + confluence.entry_zone[1]) / 2))
        else:
            entry_price = Decimal(str(current_price))

        # Build reasoning
        factors_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in confluence.factors.items()
        )
        reasoning = (
            f"SMC: {confluence.direction.value}, "
            f"score={confluence.score:.2f}, "
            f"R:R={confluence.risk_reward:.1f}, "
            f"factors=[{factors_str}]"
        )

        return TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.TECHNICAL,
            confidence=confluence.confidence,
            entry_price=entry_price,
            stop_loss=Decimal(str(confluence.stop_loss)),
            take_profit=Decimal(str(confluence.take_profit_1)),
            side=side,
            timeframe=self.config.primary_timeframe,
            indicators={
                "confluence_score": confluence.score,
                "atr": atr,
                "zones_count": len(confluence.zones),
            },
            reasoning=reasoning,
            metadata={
                "smc_analysis": {
                    "direction": confluence.direction.value,
                    "confluence_score": confluence.score,
                    "risk_reward": confluence.risk_reward,
                    "entry_zone": confluence.entry_zone,
                    "factors": confluence.factors,
                },
            },
        )

    def should_enter(
        self,
        symbol: str,
        candles: pd.DataFrame,
        current_position: Any | None = None,
    ) -> TradeSignal | None:
        """Check if should enter a position.

        Args:
            symbol: Trading symbol
            candles: OHLCV DataFrame
            current_position: Current position if any

        Returns:
            Entry signal or None
        """
        if current_position is not None:
            return None

        signals = self.generate_signals(symbol, candles)

        for signal in signals:
            # Check minimum confidence
            if signal.confidence < self.smc_config.min_confluence_score:
                continue

            # Check signal type
            if signal.signal_type in [
                SignalType.BUY, SignalType.STRONG_BUY,
                SignalType.SELL, SignalType.STRONG_SELL
            ]:
                return signal

        return None

    def should_exit(
        self,
        symbol: str,
        candles: pd.DataFrame,
        position: Any,
    ) -> TradeSignal | None:
        """Check if should exit a position.

        SMC exit logic:
        1. Price reaches take profit target
        2. Zone invalidated (price closes beyond)
        3. Opposite signal generated

        Args:
            symbol: Trading symbol
            candles: OHLCV DataFrame
            position: Current position

        Returns:
            Exit signal or None
        """
        signals = self.generate_signals(symbol, candles)

        position_side = getattr(position, "side", None)
        current_price = float(candles["close"].iloc[-1])
        entry_price = float(getattr(position, "entry_price", current_price))

        for signal in signals:
            # Exit long on sell signal
            if position_side == OrderSide.BUY:
                if signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    # Check if signal is strong enough
                    if signal.confidence >= self.smc_config.min_confluence_score * 0.8:
                        signal.reasoning = f"Exit long: {signal.reasoning}"
                        return signal

            # Exit short on buy signal
            elif position_side == OrderSide.SELL:
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    if signal.confidence >= self.smc_config.min_confluence_score * 0.8:
                        signal.reasoning = f"Exit short: {signal.reasoning}"
                        return signal

        # Check for trailing stop activation
        if position_side == OrderSide.BUY:
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct >= self.smc_config.trailing_stop_threshold:
                # Could implement trailing stop here
                pass
        elif position_side == OrderSide.SELL:
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= self.smc_config.trailing_stop_threshold:
                pass

        return None

    def get_analysis(self, symbol: str) -> MTFAnalysis | None:
        """Get last analysis for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Last MTF analysis or None
        """
        return self._last_analysis.get(symbol)

    def get_signal_history(self) -> list[dict[str, Any]]:
        """Get SMC signal history.

        Returns:
            List of recent signals
        """
        return self._smc_signals[-100:]

    def get_status(self) -> dict[str, Any]:
        """Get strategy status."""
        status = super().get_status()

        # Add SMC-specific info
        status["smc_config"] = self.smc_config.__dict__
        status["recent_signals"] = len(self._smc_signals)
        status["symbols_analyzed"] = list(self._last_analysis.keys())

        return status


def create_smc_strategy(
    config: StrategyConfig | None = None,
    smc_config: SMCStrategyConfig | None = None,
) -> SMCStrategy:
    """Create SMC strategy instance.

    Args:
        config: Base strategy configuration
        smc_config: SMC-specific configuration

    Returns:
        Configured SMC strategy
    """
    if config is None:
        config = StrategyConfig()

    return SMCStrategy(config=config, smc_config=smc_config)
