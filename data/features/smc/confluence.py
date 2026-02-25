"""SMC Confluence Engine.

Scores multiple SMC factors at a price level to determine
entry quality and generate trading zones with stop loss and targets.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from data.features.smc.detector import SMCDetector
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


@dataclass
class ConfluenceResult:
    """Result of confluence analysis at a price level.

    Attributes:
        score: Overall confluence score (0.0 to 1.0)
        direction: Trade direction (bullish/bearish)
        entry_zone: Recommended entry zone (upper, lower)
        stop_loss: Recommended stop loss price
        take_profit_1: First take profit target
        take_profit_2: Second take profit target
        take_profit_3: Third take profit target
        risk_reward: Risk/reward ratio to first target
        factors: Contributing factors and their scores
        zones: List of zones contributing to confluence
        confidence: Overall confidence level
    """

    score: float
    direction: ZoneDirection
    entry_zone: tuple[float, float]  # (upper, lower)
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None = None
    take_profit_3: float | None = None
    risk_reward: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)
    zones: list[PriceZone] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "direction": self.direction.value,
            "entry_zone": {"upper": self.entry_zone[0], "lower": self.entry_zone[1]},
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "risk_reward": self.risk_reward,
            "factors": self.factors,
            "zones_count": len(self.zones),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class ConfluenceEngine:
    """Score SMC confluence at price levels.

    Combines multiple SMC factors to determine high-probability
    trading zones with proper risk management levels.

    Attributes:
        min_confluence_score: Minimum score for valid setup (default 0.65)
        min_rr_ratio: Minimum risk/reward ratio (default 2.0)
        atr_stop_multiplier: ATR multiplier for stop loss (default 1.5)
    """

    def __init__(
        self,
        min_confluence_score: float = 0.65,
        min_rr_ratio: float = 2.0,
        atr_stop_multiplier: float = 1.5,
    ):
        """Initialize confluence engine.

        Args:
            min_confluence_score: Minimum score for valid setup
            min_rr_ratio: Minimum risk/reward ratio
            atr_stop_multiplier: ATR multiplier for stop loss
        """
        self.min_confluence_score = min_confluence_score
        self.min_rr_ratio = min_rr_ratio
        self.atr_stop_multiplier = atr_stop_multiplier

    def analyze(
        self,
        current_price: float,
        order_blocks: list[OrderBlock],
        fair_value_gaps: list[FairValueGap],
        liquidity_sweeps: list[LiquiditySweep],
        channels: list[Channel],
        market_structure: MarketStructure,
        atr: float,
        htf_bias: ZoneDirection | None = None,
    ) -> ConfluenceResult | None:
        """Analyze confluence at current price level.

        Args:
            current_price: Current market price
            order_blocks: Detected order blocks
            fair_value_gaps: Detected FVGs
            liquidity_sweeps: Detected liquidity sweeps
            channels: Detected channels
            market_structure: Market structure analysis
            atr: Current ATR value
            htf_bias: Higher timeframe bias (optional)

        Returns:
            ConfluenceResult if valid setup found, None otherwise
        """
        # Find nearby zones
        nearby_bullish = self._find_nearby_zones(
            current_price, order_blocks, fair_value_gaps, ZoneDirection.BULLISH, atr
        )
        nearby_bearish = self._find_nearby_zones(
            current_price, order_blocks, fair_value_gaps, ZoneDirection.BEARISH, atr
        )

        # Calculate scores for both directions
        bullish_score, bullish_factors = self._calculate_confluence_score(
            ZoneDirection.BULLISH,
            nearby_bullish,
            liquidity_sweeps,
            channels,
            market_structure,
            htf_bias,
            current_price,
        )

        bearish_score, bearish_factors = self._calculate_confluence_score(
            ZoneDirection.BEARISH,
            nearby_bearish,
            liquidity_sweeps,
            channels,
            market_structure,
            htf_bias,
            current_price,
        )

        # Determine best direction
        if bullish_score > bearish_score and bullish_score >= self.min_confluence_score:
            direction = ZoneDirection.BULLISH
            score = bullish_score
            factors = bullish_factors
            zones = nearby_bullish
        elif bearish_score >= self.min_confluence_score:
            direction = ZoneDirection.BEARISH
            score = bearish_score
            factors = bearish_factors
            zones = nearby_bearish
        else:
            return None

        # Calculate entry zone, stop loss, and targets
        entry_zone, stop_loss, targets = self._calculate_trade_levels(
            direction, zones, current_price, atr, order_blocks, fair_value_gaps
        )

        # Calculate risk/reward
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        risk = abs(entry_mid - stop_loss)
        reward = abs(targets[0] - entry_mid) if targets else 0

        if risk <= 0:
            return None

        risk_reward = reward / risk

        if risk_reward < self.min_rr_ratio:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            score, risk_reward, market_structure, htf_bias, direction
        )

        return ConfluenceResult(
            score=score,
            direction=direction,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit_1=targets[0] if len(targets) > 0 else entry_mid + risk * 2,
            take_profit_2=targets[1] if len(targets) > 1 else None,
            take_profit_3=targets[2] if len(targets) > 2 else None,
            risk_reward=risk_reward,
            factors=factors,
            zones=zones,
            confidence=confidence,
        )

    def _find_nearby_zones(
        self,
        current_price: float,
        order_blocks: list[OrderBlock],
        fair_value_gaps: list[FairValueGap],
        direction: ZoneDirection,
        atr: float,
        max_distance_atr: float = 3.0,
    ) -> list[PriceZone]:
        """Find zones near current price in specified direction.

        Args:
            current_price: Current price
            order_blocks: Order blocks
            fair_value_gaps: FVGs
            direction: Zone direction
            atr: ATR value
            max_distance_atr: Maximum distance in ATR multiples

        Returns:
            List of nearby valid zones
        """
        nearby: list[PriceZone] = []
        max_distance = atr * max_distance_atr

        for ob in order_blocks:
            if ob.direction != direction or not ob.is_valid():
                continue

            distance = ob.distance_to_price(current_price)
            if distance <= max_distance:
                # For bullish zones, price should be above
                if direction == ZoneDirection.BULLISH and current_price >= ob.lower:
                    nearby.append(ob)
                # For bearish zones, price should be below
                elif direction == ZoneDirection.BEARISH and current_price <= ob.upper:
                    nearby.append(ob)

        for fvg in fair_value_gaps:
            if fvg.direction != direction or not fvg.is_valid():
                continue

            distance = fvg.distance_to_price(current_price)
            if distance <= max_distance:
                if direction == ZoneDirection.BULLISH and current_price >= fvg.lower:
                    nearby.append(fvg)
                elif direction == ZoneDirection.BEARISH and current_price <= fvg.upper:
                    nearby.append(fvg)

        return nearby

    def _calculate_confluence_score(
        self,
        direction: ZoneDirection,
        zones: list[PriceZone],
        liquidity_sweeps: list[LiquiditySweep],
        channels: list[Channel],
        market_structure: MarketStructure,
        htf_bias: ZoneDirection | None,
        current_price: float,
    ) -> tuple[float, dict[str, float]]:
        """Calculate confluence score for a direction.

        Scoring weights:
        - Zone strength: 0.30
        - Zone confluence (OB+FVG overlap): 0.25
        - HTF alignment: 0.20
        - Zone freshness: 0.15
        - Volume confirmation: 0.10

        Args:
            direction: Trade direction
            zones: Nearby zones
            liquidity_sweeps: Detected sweeps
            channels: Detected channels
            market_structure: Market structure
            htf_bias: HTF bias
            current_price: Current price

        Returns:
            Tuple of (score, factors_dict)
        """
        factors: dict[str, float] = {}

        if not zones:
            return 0.0, factors

        # 1. Zone strength (0.30)
        max_zone_strength = max(z.strength for z in zones)
        factors["zone_strength"] = max_zone_strength * 0.30

        # 2. Zone confluence - OB + FVG overlap (0.25)
        ob_count = sum(1 for z in zones if isinstance(z, OrderBlock))
        fvg_count = sum(1 for z in zones if isinstance(z, FairValueGap))

        confluence_score = 0.0
        if ob_count > 0 and fvg_count > 0:
            # OB + FVG overlap is highest quality
            confluence_score = 1.0
        elif ob_count > 1 or fvg_count > 1:
            # Multiple same-type zones
            confluence_score = 0.7
        elif ob_count > 0 or fvg_count > 0:
            # Single zone
            confluence_score = 0.4

        factors["zone_confluence"] = confluence_score * 0.25

        # 3. HTF alignment (0.20)
        if htf_bias is not None:
            if htf_bias == direction:
                factors["htf_alignment"] = 0.20
            else:
                factors["htf_alignment"] = 0.0  # Counter-trend
        else:
            # Use market structure as proxy for HTF
            if market_structure.trend == direction:
                factors["htf_alignment"] = 0.15
            else:
                factors["htf_alignment"] = 0.05

        # 4. Zone freshness (0.15)
        fresh_zones = sum(1 for z in zones if z.status == ZoneStatus.FRESH)
        freshness_ratio = fresh_zones / len(zones) if zones else 0
        factors["zone_freshness"] = freshness_ratio * 0.15

        # 5. Volume/sweep confirmation (0.10)
        # Check for recent liquidity sweep in trade direction
        recent_sweep = None
        for sweep in liquidity_sweeps:
            if sweep.direction == direction:
                recent_sweep = sweep
                break

        if recent_sweep:
            factors["volume_confirmation"] = min(recent_sweep.strength, 1.0) * 0.10
        else:
            # Check OB volume ratio
            ob_volumes = [z.volume_ratio for z in zones if isinstance(z, OrderBlock)]
            if ob_volumes and max(ob_volumes) > 1.5:
                factors["volume_confirmation"] = 0.07
            else:
                factors["volume_confirmation"] = 0.0

        # Calculate total score
        total_score = sum(factors.values())

        return min(total_score, 1.0), factors

    def _calculate_trade_levels(
        self,
        direction: ZoneDirection,
        zones: list[PriceZone],
        current_price: float,
        atr: float,
        order_blocks: list[OrderBlock],
        fair_value_gaps: list[FairValueGap],
    ) -> tuple[tuple[float, float], float, list[float]]:
        """Calculate entry zone, stop loss, and take profit levels.

        Args:
            direction: Trade direction
            zones: Contributing zones
            current_price: Current price
            atr: ATR value
            order_blocks: All order blocks
            fair_value_gaps: All FVGs

        Returns:
            Tuple of (entry_zone, stop_loss, [targets])
        """
        if not zones:
            # Fallback to ATR-based levels
            if direction == ZoneDirection.BULLISH:
                entry_zone = (current_price, current_price - atr * 0.5)
                stop_loss = current_price - atr * self.atr_stop_multiplier
                targets = [current_price + atr * 2, current_price + atr * 3]
            else:
                entry_zone = (current_price + atr * 0.5, current_price)
                stop_loss = current_price + atr * self.atr_stop_multiplier
                targets = [current_price - atr * 2, current_price - atr * 3]
            return entry_zone, stop_loss, targets

        # Find best entry zone from overlapping zones
        if direction == ZoneDirection.BULLISH:
            # Entry at top of support zone, stop below
            zone_upper = max(z.upper for z in zones)
            zone_lower = min(z.lower for z in zones)
            entry_zone = (zone_upper, zone_lower)

            # Stop loss below zone + buffer
            stop_loss = zone_lower - atr * self.atr_stop_multiplier

            # Targets at next resistance zones
            targets = self._find_opposite_targets(
                current_price, order_blocks, fair_value_gaps,
                ZoneDirection.BEARISH, atr, 3
            )
        else:
            # Entry at bottom of resistance zone, stop above
            zone_upper = max(z.upper for z in zones)
            zone_lower = min(z.lower for z in zones)
            entry_zone = (zone_upper, zone_lower)

            # Stop loss above zone + buffer
            stop_loss = zone_upper + atr * self.atr_stop_multiplier

            # Targets at next support zones
            targets = self._find_opposite_targets(
                current_price, order_blocks, fair_value_gaps,
                ZoneDirection.BULLISH, atr, 3
            )

        return entry_zone, stop_loss, targets

    def _find_opposite_targets(
        self,
        current_price: float,
        order_blocks: list[OrderBlock],
        fair_value_gaps: list[FairValueGap],
        direction: ZoneDirection,
        atr: float,
        count: int = 3,
    ) -> list[float]:
        """Find take profit targets at opposite zones.

        Args:
            current_price: Current price
            order_blocks: All order blocks
            fair_value_gaps: All FVGs
            direction: Direction of target zones
            atr: ATR value
            count: Number of targets to find

        Returns:
            List of target prices
        """
        targets: list[float] = []

        # Collect opposite zones
        opposite_zones: list[PriceZone] = []

        for ob in order_blocks:
            if ob.direction == direction and ob.is_valid():
                opposite_zones.append(ob)

        for fvg in fair_value_gaps:
            if fvg.direction == direction and fvg.is_valid():
                opposite_zones.append(fvg)

        if direction == ZoneDirection.BEARISH:
            # Looking for resistance above current price
            above_zones = [z for z in opposite_zones if z.lower > current_price]
            above_zones.sort(key=lambda z: z.lower)

            for zone in above_zones[:count]:
                targets.append(zone.lower)  # Target bottom of resistance
        else:
            # Looking for support below current price
            below_zones = [z for z in opposite_zones if z.upper < current_price]
            below_zones.sort(key=lambda z: z.upper, reverse=True)

            for zone in below_zones[:count]:
                targets.append(zone.upper)  # Target top of support

        # Fill with ATR-based targets if not enough zones found
        while len(targets) < count:
            if direction == ZoneDirection.BEARISH:
                last_target = targets[-1] if targets else current_price
                targets.append(last_target + atr * 2)
            else:
                last_target = targets[-1] if targets else current_price
                targets.append(last_target - atr * 2)

        return targets[:count]

    def _calculate_confidence(
        self,
        score: float,
        risk_reward: float,
        market_structure: MarketStructure,
        htf_bias: ZoneDirection | None,
        direction: ZoneDirection,
    ) -> float:
        """Calculate overall confidence level.

        Args:
            score: Confluence score
            risk_reward: Risk/reward ratio
            market_structure: Market structure
            htf_bias: HTF bias
            direction: Trade direction

        Returns:
            Confidence value (0.0 to 1.0)
        """
        # Base confidence from score
        confidence = score * 0.5

        # Risk/reward bonus (0.2 max)
        rr_bonus = min(risk_reward / 5.0, 1.0) * 0.2
        confidence += rr_bonus

        # Market structure alignment (0.15 max)
        if market_structure.trend == direction:
            if market_structure.is_trending:
                confidence += 0.15
            else:
                confidence += 0.10
        else:
            confidence += 0.0  # Counter-trend

        # HTF alignment (0.15 max)
        if htf_bias is not None:
            if htf_bias == direction:
                confidence += 0.15
            # else: no bonus for counter-HTF

        return min(confidence, 1.0)

    def find_best_setup(
        self,
        candles: pd.DataFrame,
        detector: SMCDetector,
        htf_bias: ZoneDirection | None = None,
    ) -> ConfluenceResult | None:
        """Find best trading setup from candle data.

        Convenience method that runs full detection and confluence analysis.

        Args:
            candles: OHLCV DataFrame
            detector: SMC detector instance
            htf_bias: Higher timeframe bias

        Returns:
            Best confluence result or None
        """
        # Run detection
        analysis = detector.analyze(candles)

        if not analysis:
            return None

        current_price = float(candles["close"].iloc[-1])

        return self.analyze(
            current_price=current_price,
            order_blocks=analysis.get("order_blocks", []),
            fair_value_gaps=analysis.get("fair_value_gaps", []),
            liquidity_sweeps=analysis.get("liquidity_sweeps", []),
            channels=analysis.get("channels", []),
            market_structure=analysis.get("market_structure", MarketStructure(ZoneDirection.BULLISH)),
            atr=analysis.get("atr", 0.0),
            htf_bias=htf_bias,
        )
