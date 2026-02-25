"""SMC Zone dataclasses.

Defines data structures for Smart Money Concepts zones including
Order Blocks, Fair Value Gaps, Liquidity Sweeps, and Channels.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class ZoneDirection(str, Enum):
    """Zone direction (bullish/bearish)."""

    BULLISH = "bullish"
    BEARISH = "bearish"


class ZoneStatus(str, Enum):
    """Zone validity status."""

    FRESH = "fresh"  # Never tested
    TESTED = "tested"  # Price returned but held
    MITIGATED = "mitigated"  # Partially filled
    INVALIDATED = "invalidated"  # Broken through


class ZoneType(str, Enum):
    """Type of SMC zone."""

    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    CHANNEL = "channel"
    BREAKER_BLOCK = "breaker_block"


@dataclass
class PriceZone:
    """Base class for all SMC price zones.

    Attributes:
        upper: Upper boundary of the zone
        lower: Lower boundary of the zone
        direction: Bullish or bearish zone
        status: Current validity status
        strength: Zone strength score (0.0 to 1.0)
        created_at: Timestamp when zone was created
        candle_index: Index of the candle that created the zone
        touches: Number of times price has touched the zone
        metadata: Additional zone-specific data
    """

    upper: float
    lower: float
    direction: ZoneDirection
    status: ZoneStatus = ZoneStatus.FRESH
    strength: float = 0.5
    created_at: datetime | None = None
    candle_index: int = -1
    touches: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def midpoint(self) -> float:
        """Get zone midpoint."""
        return (self.upper + self.lower) / 2

    @property
    def height(self) -> float:
        """Get zone height (size)."""
        return abs(self.upper - self.lower)

    def contains_price(self, price: float) -> bool:
        """Check if price is within the zone."""
        return self.lower <= price <= self.upper

    def distance_to_price(self, price: float) -> float:
        """Get distance from zone to price (0 if inside)."""
        if self.contains_price(price):
            return 0.0
        return min(abs(price - self.upper), abs(price - self.lower))

    def is_valid(self) -> bool:
        """Check if zone is still valid for trading."""
        return self.status in [ZoneStatus.FRESH, ZoneStatus.TESTED]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "upper": self.upper,
            "lower": self.lower,
            "direction": self.direction.value,
            "status": self.status.value,
            "strength": self.strength,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "candle_index": self.candle_index,
            "touches": self.touches,
            "midpoint": self.midpoint,
            "height": self.height,
        }


@dataclass
class OrderBlock(PriceZone):
    """Order Block - Last opposite candle before strong impulse move.

    Represents institutional order flow where large players have placed
    orders. The last bearish candle before a bullish impulse (or vice versa)
    often acts as support/resistance.

    Attributes:
        impulse_strength: Strength of the impulse move (in ATR multiples)
        displacement: Size of the displacement candle
        volume_ratio: Volume relative to average (> 1.5 indicates strong OB)
        body_type: Whether OB candle was bullish/bearish
    """

    impulse_strength: float = 0.0
    displacement: float = 0.0
    volume_ratio: float = 1.0
    body_type: str = "unknown"  # "bullish" or "bearish"

    def __post_init__(self) -> None:
        """Calculate strength based on impulse and volume."""
        # Base strength from impulse (0.3 weight)
        impulse_score = min(self.impulse_strength / 4.0, 1.0) * 0.30

        # Volume confirmation (0.2 weight)
        volume_score = min(self.volume_ratio / 2.0, 1.0) * 0.20 if self.volume_ratio > 1.0 else 0.0

        # Freshness bonus (0.15 weight for fresh zones)
        freshness_score = 0.15 if self.status == ZoneStatus.FRESH else 0.0

        # Base score
        base_score = 0.35

        self.strength = min(base_score + impulse_score + volume_score + freshness_score, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "type": ZoneType.ORDER_BLOCK.value,
            "impulse_strength": self.impulse_strength,
            "displacement": self.displacement,
            "volume_ratio": self.volume_ratio,
            "body_type": self.body_type,
        })
        return data


@dataclass
class FairValueGap(PriceZone):
    """Fair Value Gap - 3-candle imbalance in price action.

    FVG occurs when:
    - Bullish: candle1.low > candle3.high (gap up)
    - Bearish: candle1.high < candle3.low (gap down)

    Price often returns to fill these gaps as they represent
    areas of inefficient price discovery.

    Attributes:
        gap_filled_percent: How much of the gap has been filled (0-100)
        candle1_index: Index of first candle in the pattern
        candle2_index: Index of middle (impulse) candle
        candle3_index: Index of third candle in the pattern
        gap_size_atr: Gap size relative to ATR
    """

    gap_filled_percent: float = 0.0
    candle1_index: int = -1
    candle2_index: int = -1
    candle3_index: int = -1
    gap_size_atr: float = 0.0

    def __post_init__(self) -> None:
        """Calculate strength based on gap characteristics."""
        # Larger gaps are stronger (0.25 weight)
        size_score = min(self.gap_size_atr / 2.0, 1.0) * 0.25

        # Unfilled gaps are stronger (0.25 weight)
        fill_score = (1.0 - self.gap_filled_percent / 100) * 0.25

        # Freshness (0.15 weight)
        freshness_score = 0.15 if self.status == ZoneStatus.FRESH else 0.0

        # Base score
        base_score = 0.35

        self.strength = min(base_score + size_score + fill_score + freshness_score, 1.0)

    def update_fill_percent(self, current_low: float, current_high: float) -> None:
        """Update gap fill percentage based on current price."""
        if self.direction == ZoneDirection.BULLISH:
            # Bullish FVG: check if price dropped into gap
            if current_low < self.upper:
                filled = (self.upper - max(current_low, self.lower)) / self.height * 100
                self.gap_filled_percent = min(filled, 100.0)
        else:
            # Bearish FVG: check if price rose into gap
            if current_high > self.lower:
                filled = (min(current_high, self.upper) - self.lower) / self.height * 100
                self.gap_filled_percent = min(filled, 100.0)

        # Update status
        if self.gap_filled_percent >= 100:
            self.status = ZoneStatus.INVALIDATED
        elif self.gap_filled_percent > 50:
            self.status = ZoneStatus.MITIGATED
        elif self.gap_filled_percent > 0:
            self.status = ZoneStatus.TESTED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "type": ZoneType.FAIR_VALUE_GAP.value,
            "gap_filled_percent": self.gap_filled_percent,
            "candle1_index": self.candle1_index,
            "candle2_index": self.candle2_index,
            "candle3_index": self.candle3_index,
            "gap_size_atr": self.gap_size_atr,
        })
        return data


@dataclass
class LiquiditySweep(PriceZone):
    """Liquidity Sweep - Price briefly breaks structure then reverses.

    Occurs when price sweeps above/below a swing point (grabbing liquidity)
    then quickly reverses, indicating smart money activity.

    Attributes:
        sweep_level: The price level that was swept
        wick_size: Size of the sweep wick relative to body
        reversal_strength: Strength of the reversal (in ATR)
        swing_type: Type of swing swept ("high" or "low")
        candle_body_percent: Percentage of candle that is body (vs wick)
    """

    sweep_level: float = 0.0
    wick_size: float = 0.0
    reversal_strength: float = 0.0
    swing_type: str = "unknown"  # "high" or "low"
    candle_body_percent: float = 0.0

    def __post_init__(self) -> None:
        """Calculate strength based on sweep characteristics."""
        # Larger wick relative to body (0.30 weight)
        wick_score = min(self.wick_size / 3.0, 1.0) * 0.30

        # Strong reversal (0.25 weight)
        reversal_score = min(self.reversal_strength / 2.0, 1.0) * 0.25

        # Small body means stronger rejection (0.10 weight)
        body_score = (1.0 - min(self.candle_body_percent / 100, 1.0)) * 0.10

        # Base score
        base_score = 0.35

        self.strength = min(base_score + wick_score + reversal_score + body_score, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "type": ZoneType.LIQUIDITY_SWEEP.value,
            "sweep_level": self.sweep_level,
            "wick_size": self.wick_size,
            "reversal_strength": self.reversal_strength,
            "swing_type": self.swing_type,
            "candle_body_percent": self.candle_body_percent,
        })
        return data


@dataclass
class Channel:
    """Price channel defined by parallel trendlines.

    Attributes:
        upper_line: Tuple of (slope, intercept) for upper trendline
        lower_line: Tuple of (slope, intercept) for lower trendline
        direction: Channel direction (bullish=ascending, bearish=descending)
        touch_count: Number of touches on both trendlines
        width: Average channel width
        start_index: Index where channel starts
        end_index: Index where channel ends (or current)
        strength: Channel strength (0.0 to 1.0)
    """

    upper_line: tuple[float, float]  # (slope, intercept)
    lower_line: tuple[float, float]
    direction: ZoneDirection
    touch_count: int = 0
    width: float = 0.0
    start_index: int = 0
    end_index: int = -1
    strength: float = 0.5

    def get_upper_at_index(self, index: int) -> float:
        """Get upper trendline value at given index."""
        return self.upper_line[0] * index + self.upper_line[1]

    def get_lower_at_index(self, index: int) -> float:
        """Get lower trendline value at given index."""
        return self.lower_line[0] * index + self.lower_line[1]

    def price_position(self, price: float, index: int) -> str:
        """Determine if price is near top, bottom, or middle of channel."""
        upper = self.get_upper_at_index(index)
        lower = self.get_lower_at_index(index)
        relative_pos = (price - lower) / (upper - lower) if upper != lower else 0.5

        if relative_pos > 0.8:
            return "near_upper"
        elif relative_pos < 0.2:
            return "near_lower"
        return "middle"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": ZoneType.CHANNEL.value,
            "upper_line": {"slope": self.upper_line[0], "intercept": self.upper_line[1]},
            "lower_line": {"slope": self.lower_line[0], "intercept": self.lower_line[1]},
            "direction": self.direction.value,
            "touch_count": self.touch_count,
            "width": self.width,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "strength": self.strength,
        }


@dataclass
class MarketStructure:
    """Market structure analysis (trend, CHoCH, BOS).

    Attributes:
        trend: Current trend direction
        swing_highs: List of (index, price) for swing highs
        swing_lows: List of (index, price) for swing lows
        last_choch: Index of last Change of Character
        last_bos: Index of last Break of Structure
        higher_highs: Count of consecutive higher highs
        lower_lows: Count of consecutive lower lows
    """

    trend: ZoneDirection
    swing_highs: list[tuple[int, float]] = field(default_factory=list)
    swing_lows: list[tuple[int, float]] = field(default_factory=list)
    last_choch: int | None = None
    last_bos: int | None = None
    higher_highs: int = 0
    lower_lows: int = 0
    strength: float = 0.5  # Trend strength

    @property
    def is_trending(self) -> bool:
        """Check if market is in a clear trend."""
        return self.higher_highs >= 2 or self.lower_lows >= 2

    @property
    def trend_type(self) -> str:
        """Get detailed trend type."""
        if self.higher_highs >= 3:
            return "strong_uptrend"
        elif self.higher_highs >= 2:
            return "uptrend"
        elif self.lower_lows >= 3:
            return "strong_downtrend"
        elif self.lower_lows >= 2:
            return "downtrend"
        return "ranging"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trend": self.trend.value,
            "trend_type": self.trend_type,
            "swing_highs": self.swing_highs[-5:],  # Last 5
            "swing_lows": self.swing_lows[-5:],
            "last_choch": self.last_choch,
            "last_bos": self.last_bos,
            "higher_highs": self.higher_highs,
            "lower_lows": self.lower_lows,
            "is_trending": self.is_trending,
            "strength": self.strength,
        }
