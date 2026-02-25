"""Smart Money Concepts (SMC) module.

Provides SMC-based technical analysis including:
- Order Blocks (OB)
- Fair Value Gaps (FVG)
- Liquidity Sweeps
- Market Structure (CHoCH, BOS)
- Multi-Timeframe Analysis
"""

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
from data.features.smc.detector import SMCDetector
from data.features.smc.confluence import ConfluenceEngine, ConfluenceResult
from data.features.smc.mtf import MTFCoordinator, MTFAnalysis

__all__ = [
    # Enums
    "ZoneDirection",
    "ZoneStatus",
    "ZoneType",
    # Zone types
    "PriceZone",
    "OrderBlock",
    "FairValueGap",
    "LiquiditySweep",
    "Channel",
    "MarketStructure",
    # Detection
    "SMCDetector",
    # Confluence
    "ConfluenceEngine",
    "ConfluenceResult",
    # MTF
    "MTFCoordinator",
    "MTFAnalysis",
]
