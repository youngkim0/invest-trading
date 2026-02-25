"""Multi-Timeframe (MTF) SMC Coordinator.

Coordinates SMC analysis across multiple timeframes:
- HTF (4H): Overall bias and major zones
- MTF (1H): Entry zones and structure
- LTF (15m): Entry timing and precision
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from data.features.smc.confluence import ConfluenceEngine, ConfluenceResult
from data.features.smc.detector import SMCDetector
from data.features.smc.zones import (
    FairValueGap,
    MarketStructure,
    OrderBlock,
    PriceZone,
    ZoneDirection,
)


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe.

    Attributes:
        timeframe: Timeframe label (e.g., "4h", "1h", "15m")
        bias: Market bias for this timeframe
        structure: Market structure analysis
        order_blocks: Detected order blocks
        fair_value_gaps: Detected FVGs
        key_levels: Important price levels
        atr: ATR value for this timeframe
        timestamp: Analysis timestamp
    """

    timeframe: str
    bias: ZoneDirection
    structure: MarketStructure
    order_blocks: list[OrderBlock] = field(default_factory=list)
    fair_value_gaps: list[FairValueGap] = field(default_factory=list)
    key_levels: list[float] = field(default_factory=list)
    atr: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeframe": self.timeframe,
            "bias": self.bias.value,
            "structure": self.structure.to_dict(),
            "order_blocks_count": len(self.order_blocks),
            "fair_value_gaps_count": len(self.fair_value_gaps),
            "key_levels": self.key_levels,
            "atr": self.atr,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MTFAnalysis:
    """Complete multi-timeframe analysis.

    Attributes:
        htf: Higher timeframe analysis (4H)
        mtf: Medium timeframe analysis (1H)
        ltf: Lower timeframe analysis (15m)
        aligned: Whether all timeframes are aligned
        overall_bias: Dominant market bias
        confluence: Best confluence result
        entry_quality: Quality score for entry timing (0.0 to 1.0)
        recommendations: Trading recommendations
    """

    htf: TimeframeAnalysis | None = None
    mtf: TimeframeAnalysis | None = None
    ltf: TimeframeAnalysis | None = None
    aligned: bool = False
    overall_bias: ZoneDirection | None = None
    confluence: ConfluenceResult | None = None
    entry_quality: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_valid_setup(self) -> bool:
        """Check if this represents a valid trading setup."""
        return (
            self.aligned
            and self.confluence is not None
            and self.confluence.score >= 0.65
            and self.entry_quality >= 0.5
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "htf": self.htf.to_dict() if self.htf else None,
            "mtf": self.mtf.to_dict() if self.mtf else None,
            "ltf": self.ltf.to_dict() if self.ltf else None,
            "aligned": self.aligned,
            "overall_bias": self.overall_bias.value if self.overall_bias else None,
            "confluence": self.confluence.to_dict() if self.confluence else None,
            "entry_quality": self.entry_quality,
            "recommendations": self.recommendations,
            "is_valid_setup": self.is_valid_setup,
            "timestamp": self.timestamp.isoformat(),
        }


class MTFCoordinator:
    """Coordinate SMC analysis across multiple timeframes.

    Typical usage:
        coordinator = MTFCoordinator()
        analysis = coordinator.analyze(
            htf_candles=candles_4h,
            mtf_candles=candles_1h,
            ltf_candles=candles_15m,
        )

    Attributes:
        htf_label: Label for HTF (default "4h")
        mtf_label: Label for MTF (default "1h")
        ltf_label: Label for LTF (default "15m")
        detector_config: Configuration for SMC detectors
    """

    def __init__(
        self,
        htf_label: str = "4h",
        mtf_label: str = "1h",
        ltf_label: str = "15m",
        min_confluence: float = 0.65,
        min_rr_ratio: float = 2.0,
    ):
        """Initialize MTF coordinator.

        Args:
            htf_label: Higher timeframe label
            mtf_label: Medium timeframe label
            ltf_label: Lower timeframe label
            min_confluence: Minimum confluence score
            min_rr_ratio: Minimum risk/reward ratio
        """
        self.htf_label = htf_label
        self.mtf_label = mtf_label
        self.ltf_label = ltf_label

        # Create detectors for each timeframe
        self.htf_detector = SMCDetector(
            lookback=100,
            atr_period=14,
            swing_lookback=5,
            min_impulse_atr=2.5,  # Higher threshold for HTF
        )
        self.mtf_detector = SMCDetector(
            lookback=100,
            atr_period=14,
            swing_lookback=5,
            min_impulse_atr=2.0,
        )
        self.ltf_detector = SMCDetector(
            lookback=100,
            atr_period=14,
            swing_lookback=3,  # Smaller for LTF
            min_impulse_atr=1.5,  # Lower threshold for LTF
        )

        self.confluence_engine = ConfluenceEngine(
            min_confluence_score=min_confluence,
            min_rr_ratio=min_rr_ratio,
        )

    def analyze(
        self,
        htf_candles: pd.DataFrame | None = None,
        mtf_candles: pd.DataFrame | None = None,
        ltf_candles: pd.DataFrame | None = None,
    ) -> MTFAnalysis:
        """Run multi-timeframe analysis.

        Args:
            htf_candles: Higher timeframe OHLCV data (4H)
            mtf_candles: Medium timeframe OHLCV data (1H)
            ltf_candles: Lower timeframe OHLCV data (15m)

        Returns:
            MTFAnalysis with complete analysis
        """
        result = MTFAnalysis()

        # Analyze each available timeframe
        if htf_candles is not None and len(htf_candles) >= 50:
            result.htf = self._analyze_timeframe(
                htf_candles, self.htf_detector, self.htf_label
            )

        if mtf_candles is not None and len(mtf_candles) >= 50:
            result.mtf = self._analyze_timeframe(
                mtf_candles, self.mtf_detector, self.mtf_label
            )

        if ltf_candles is not None and len(ltf_candles) >= 50:
            result.ltf = self._analyze_timeframe(
                ltf_candles, self.ltf_detector, self.ltf_label
            )

        # Determine overall bias and alignment
        result.overall_bias, result.aligned = self._determine_bias_alignment(result)

        # Find confluence using MTF data with HTF bias
        if result.mtf is not None:
            htf_bias = result.htf.bias if result.htf else None

            # Get SMC data from MTF
            mtf_analysis = self.mtf_detector.analyze(mtf_candles)

            if mtf_analysis:
                current_price = float(mtf_candles["close"].iloc[-1])

                result.confluence = self.confluence_engine.analyze(
                    current_price=current_price,
                    order_blocks=mtf_analysis.get("order_blocks", []),
                    fair_value_gaps=mtf_analysis.get("fair_value_gaps", []),
                    liquidity_sweeps=mtf_analysis.get("liquidity_sweeps", []),
                    channels=mtf_analysis.get("channels", []),
                    market_structure=mtf_analysis.get("market_structure"),
                    atr=mtf_analysis.get("atr", 0.0),
                    htf_bias=htf_bias,
                )

        # Calculate entry quality using LTF
        result.entry_quality = self._calculate_entry_quality(result, ltf_candles)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def _analyze_timeframe(
        self,
        candles: pd.DataFrame,
        detector: SMCDetector,
        label: str,
    ) -> TimeframeAnalysis:
        """Analyze a single timeframe.

        Args:
            candles: OHLCV DataFrame
            detector: SMC detector for this timeframe
            label: Timeframe label

        Returns:
            TimeframeAnalysis for this timeframe
        """
        analysis = detector.analyze(candles)

        if not analysis:
            return TimeframeAnalysis(
                timeframe=label,
                bias=ZoneDirection.BULLISH,
                structure=MarketStructure(ZoneDirection.BULLISH),
            )

        structure = analysis.get("market_structure", MarketStructure(ZoneDirection.BULLISH))

        # Extract key levels from zones
        key_levels = self._extract_key_levels(
            analysis.get("order_blocks", []),
            analysis.get("fair_value_gaps", []),
        )

        return TimeframeAnalysis(
            timeframe=label,
            bias=structure.trend,
            structure=structure,
            order_blocks=analysis.get("order_blocks", []),
            fair_value_gaps=analysis.get("fair_value_gaps", []),
            key_levels=key_levels,
            atr=analysis.get("atr", 0.0),
        )

    def _extract_key_levels(
        self,
        order_blocks: list[OrderBlock],
        fair_value_gaps: list[FairValueGap],
        max_levels: int = 5,
    ) -> list[float]:
        """Extract key price levels from zones.

        Args:
            order_blocks: Order blocks
            fair_value_gaps: FVGs
            max_levels: Maximum levels to return

        Returns:
            List of key price levels
        """
        levels: list[float] = []

        # Add OB levels (midpoints of strongest)
        for ob in sorted(order_blocks, key=lambda x: x.strength, reverse=True)[:3]:
            levels.append(ob.midpoint)

        # Add FVG levels
        for fvg in sorted(fair_value_gaps, key=lambda x: x.strength, reverse=True)[:3]:
            levels.append(fvg.midpoint)

        # Sort and deduplicate
        levels = sorted(set(levels))

        return levels[:max_levels]

    def _determine_bias_alignment(
        self,
        analysis: MTFAnalysis,
    ) -> tuple[ZoneDirection | None, bool]:
        """Determine overall bias and check alignment.

        Args:
            analysis: MTF analysis

        Returns:
            Tuple of (overall_bias, aligned)
        """
        biases: list[ZoneDirection] = []

        if analysis.htf:
            biases.append(analysis.htf.bias)
        if analysis.mtf:
            biases.append(analysis.mtf.bias)
        if analysis.ltf:
            biases.append(analysis.ltf.bias)

        if not biases:
            return None, False

        # Count bullish vs bearish
        bullish_count = sum(1 for b in biases if b == ZoneDirection.BULLISH)
        bearish_count = len(biases) - bullish_count

        # Determine overall bias (majority wins)
        if bullish_count > bearish_count:
            overall = ZoneDirection.BULLISH
        elif bearish_count > bullish_count:
            overall = ZoneDirection.BEARISH
        else:
            # Tie - use HTF if available
            overall = analysis.htf.bias if analysis.htf else biases[0]

        # Check alignment (all same direction)
        aligned = bullish_count == len(biases) or bearish_count == len(biases)

        return overall, aligned

    def _calculate_entry_quality(
        self,
        analysis: MTFAnalysis,
        ltf_candles: pd.DataFrame | None,
    ) -> float:
        """Calculate entry timing quality using LTF.

        Factors:
        - Price at key level (0.30)
        - LTF structure alignment (0.25)
        - Recent sweep/reversal (0.25)
        - Momentum alignment (0.20)

        Args:
            analysis: MTF analysis
            ltf_candles: LTF candle data

        Returns:
            Entry quality score (0.0 to 1.0)
        """
        if not analysis.ltf or ltf_candles is None or len(ltf_candles) < 20:
            return 0.5  # Neutral if no LTF data

        quality = 0.0

        current_price = float(ltf_candles["close"].iloc[-1])

        # 1. Price at key level (0.30)
        # Check if price is near any key level from MTF
        if analysis.mtf and analysis.mtf.key_levels:
            min_distance = min(
                abs(current_price - level) / current_price
                for level in analysis.mtf.key_levels
            )
            if min_distance < 0.005:  # Within 0.5%
                quality += 0.30
            elif min_distance < 0.01:  # Within 1%
                quality += 0.20
            elif min_distance < 0.02:  # Within 2%
                quality += 0.10

        # 2. LTF structure alignment (0.25)
        if analysis.overall_bias and analysis.ltf.bias == analysis.overall_bias:
            quality += 0.25
        elif analysis.ltf.structure.is_trending:
            quality += 0.10

        # 3. Recent sweep/reversal on LTF (0.25)
        ltf_analysis = self.ltf_detector.analyze(ltf_candles)
        if ltf_analysis:
            sweeps = ltf_analysis.get("liquidity_sweeps", [])
            recent_aligned_sweep = any(
                s.direction == analysis.overall_bias
                for s in sweeps
                if s.candle_index > len(ltf_candles) - 10
            )
            if recent_aligned_sweep:
                quality += 0.25

        # 4. Momentum alignment (0.20)
        # Simple momentum check using recent candles
        close_prices = ltf_candles["close"].astype(float).tail(10)
        if len(close_prices) >= 10:
            momentum = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]

            if analysis.overall_bias == ZoneDirection.BULLISH and momentum > 0:
                quality += 0.20 * min(abs(momentum) * 50, 1.0)
            elif analysis.overall_bias == ZoneDirection.BEARISH and momentum < 0:
                quality += 0.20 * min(abs(momentum) * 50, 1.0)

        return min(quality, 1.0)

    def _generate_recommendations(
        self,
        analysis: MTFAnalysis,
    ) -> list[str]:
        """Generate trading recommendations.

        Args:
            analysis: MTF analysis

        Returns:
            List of recommendation strings
        """
        recommendations: list[str] = []

        # Alignment check
        if analysis.aligned:
            recommendations.append(
                f"✅ All timeframes aligned {analysis.overall_bias.value if analysis.overall_bias else 'neutral'}"
            )
        else:
            recommendations.append("⚠️ Timeframes not aligned - consider waiting")

        # Confluence check
        if analysis.confluence:
            score = analysis.confluence.score
            if score >= 0.8:
                recommendations.append(f"✅ Strong confluence ({score:.2f})")
            elif score >= 0.65:
                recommendations.append(f"✅ Good confluence ({score:.2f})")
            else:
                recommendations.append(f"⚠️ Weak confluence ({score:.2f})")

            # Risk/reward
            rr = analysis.confluence.risk_reward
            if rr >= 3.0:
                recommendations.append(f"✅ Excellent R:R ({rr:.1f}:1)")
            elif rr >= 2.0:
                recommendations.append(f"✅ Good R:R ({rr:.1f}:1)")
            else:
                recommendations.append(f"⚠️ Low R:R ({rr:.1f}:1)")
        else:
            recommendations.append("❌ No valid confluence found")

        # Entry quality
        if analysis.entry_quality >= 0.7:
            recommendations.append("✅ Good entry timing")
        elif analysis.entry_quality >= 0.5:
            recommendations.append("⚠️ Average entry timing")
        else:
            recommendations.append("⚠️ Poor entry timing - wait for better setup")

        # HTF structure
        if analysis.htf and analysis.htf.structure.is_trending:
            recommendations.append(
                f"📈 HTF in {analysis.htf.structure.trend_type.replace('_', ' ')}"
            )

        # Valid setup summary
        if analysis.is_valid_setup:
            recommendations.append("🎯 VALID SETUP - Consider entry")
        else:
            recommendations.append("⏳ WAIT - Setup not complete")

        return recommendations

    def get_entry_signal(
        self,
        htf_candles: pd.DataFrame | None = None,
        mtf_candles: pd.DataFrame | None = None,
        ltf_candles: pd.DataFrame | None = None,
    ) -> dict[str, Any] | None:
        """Get entry signal if valid setup exists.

        Convenience method that returns a simplified signal dict.

        Args:
            htf_candles: HTF data
            mtf_candles: MTF data
            ltf_candles: LTF data

        Returns:
            Signal dict or None if no valid setup
        """
        analysis = self.analyze(htf_candles, mtf_candles, ltf_candles)

        if not analysis.is_valid_setup or not analysis.confluence:
            return None

        return {
            "direction": analysis.overall_bias.value if analysis.overall_bias else None,
            "entry_zone": analysis.confluence.entry_zone,
            "stop_loss": analysis.confluence.stop_loss,
            "take_profit_1": analysis.confluence.take_profit_1,
            "take_profit_2": analysis.confluence.take_profit_2,
            "risk_reward": analysis.confluence.risk_reward,
            "confluence_score": analysis.confluence.score,
            "entry_quality": analysis.entry_quality,
            "confidence": analysis.confluence.confidence,
            "aligned": analysis.aligned,
            "recommendations": analysis.recommendations,
        }
