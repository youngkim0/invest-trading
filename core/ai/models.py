"""Data models for AI analysis results."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeAnalysis:
    """Result of post-trade AI analysis."""
    position_id: str
    analysis_text: str
    patterns_identified: list[dict] = field(default_factory=list)
    suggestion: str = ""
    model_used: str = ""
    tokens_used: int = 0


@dataclass
class SignalEvaluation:
    """Result of pre-trade signal evaluation."""
    adjusted_confidence: float
    reasoning: str
    risk_flags: list[str] = field(default_factory=list)
    model_used: str = ""
    tokens_used: int = 0


@dataclass
class PerformanceReview:
    """Result of periodic performance review."""
    review_date: str
    period: str  # 'daily' or 'weekly'
    summary: str
    strategy_insights: dict = field(default_factory=dict)
    suggestions: list[dict] = field(default_factory=list)
    model_used: str = ""
    tokens_used: int = 0
