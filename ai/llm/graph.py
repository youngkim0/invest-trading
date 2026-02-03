"""LangGraph workflow for multi-agent market analysis.

Orchestrates multiple LLM agents for comprehensive market analysis.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel, Field

from ai.llm.market_analyst import MarketAnalyst, MarketContext, TradingSignal
from ai.llm.news_analyzer import NewsAnalyzer, NewsArticle


class AgentState(BaseModel):
    """State passed between agents in the graph."""

    # Input
    symbol: str
    current_price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Technical data
    technical_indicators: dict[str, Any] = Field(default_factory=dict)

    # News data
    news_articles: list[dict[str, Any]] = Field(default_factory=list)

    # Agent outputs
    technical_analysis: dict[str, Any] | None = None
    news_analysis: dict[str, Any] | None = None
    sentiment_score: float | None = None

    # Final output
    final_signal: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    recommendations: list[str] = Field(default_factory=list)

    # Messages for tracking
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    # Control flow
    errors: list[str] = Field(default_factory=list)


class TradingDecision(BaseModel):
    """Final trading decision output."""

    symbol: str
    timestamp: datetime
    signal: TradingSignal
    confidence: float

    # Analysis summary
    technical_sentiment: str
    news_sentiment: str
    combined_sentiment: float

    # Price targets
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    # Risk assessment
    risk_level: str
    position_size_recommendation: float  # As fraction of portfolio

    # Reasoning
    reasoning: str
    key_factors: list[str]


class TradingWorkflow:
    """LangGraph workflow for trading analysis."""

    def __init__(self):
        """Initialize workflow."""
        self.market_analyst = MarketAnalyst()
        self.news_analyzer = NewsAnalyzer()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_technical", self._analyze_technical)
        workflow.add_node("analyze_news", self._analyze_news)
        workflow.add_node("combine_signals", self._combine_signals)
        workflow.add_node("generate_decision", self._generate_decision)

        # Add edges
        workflow.add_edge("analyze_technical", "analyze_news")
        workflow.add_edge("analyze_news", "combine_signals")
        workflow.add_edge("combine_signals", "generate_decision")
        workflow.add_edge("generate_decision", END)

        # Set entry point
        workflow.set_entry_point("analyze_technical")

        return workflow.compile()

    async def _analyze_technical(self, state: AgentState) -> dict[str, Any]:
        """Analyze technical indicators."""
        logger.info(f"Analyzing technical indicators for {state.symbol}")

        try:
            context = MarketContext(
                symbol=state.symbol,
                current_price=Decimal(str(state.current_price)),
                rsi=state.technical_indicators.get("rsi"),
                macd=state.technical_indicators.get("macd"),
                macd_signal=state.technical_indicators.get("macd_signal"),
                sma_20=Decimal(str(state.technical_indicators.get("sma_20", 0))) if state.technical_indicators.get("sma_20") else None,
                sma_50=Decimal(str(state.technical_indicators.get("sma_50", 0))) if state.technical_indicators.get("sma_50") else None,
                bb_upper=Decimal(str(state.technical_indicators.get("bb_upper", 0))) if state.technical_indicators.get("bb_upper") else None,
                bb_lower=Decimal(str(state.technical_indicators.get("bb_lower", 0))) if state.technical_indicators.get("bb_lower") else None,
            )

            analysis = await self.market_analyst.analyze(context)

            return {
                "technical_analysis": {
                    "sentiment": analysis.sentiment.value,
                    "signal": analysis.signal.value,
                    "confidence": analysis.confidence,
                    "summary": analysis.technical_summary,
                    "support": analysis.support_level,
                    "resistance": analysis.resistance_level,
                    "risk_level": analysis.risk_level,
                }
            }

        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {
                "technical_analysis": {
                    "sentiment": "neutral",
                    "signal": "hold",
                    "confidence": 0.5,
                    "summary": "Analysis unavailable",
                    "risk_level": "medium",
                },
                "errors": state.errors + [f"Technical analysis error: {e}"],
            }

    async def _analyze_news(self, state: AgentState) -> dict[str, Any]:
        """Analyze news and sentiment."""
        logger.info(f"Analyzing news for {state.symbol}")

        try:
            articles = [
                NewsArticle(
                    headline=a.get("headline", a.get("title", "")),
                    body=a.get("body", ""),
                    source=a.get("source"),
                )
                for a in state.news_articles
            ]

            if articles:
                summary = await self.news_analyzer.analyze_batch(articles, state.symbol)

                return {
                    "news_analysis": {
                        "overall_sentiment": summary.overall_sentiment,
                        "sentiment_distribution": summary.sentiment_distribution,
                        "market_impact": summary.overall_market_impact,
                        "bullish_factors": summary.bullish_factors,
                        "bearish_factors": summary.bearish_factors,
                        "high_impact_count": len(summary.high_impact_news),
                    },
                    "sentiment_score": summary.overall_sentiment,
                }
            else:
                return {
                    "news_analysis": {
                        "overall_sentiment": 0.0,
                        "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                        "market_impact": "No news data",
                        "bullish_factors": [],
                        "bearish_factors": [],
                        "high_impact_count": 0,
                    },
                    "sentiment_score": 0.0,
                }

        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return {
                "news_analysis": {
                    "overall_sentiment": 0.0,
                    "market_impact": "Analysis unavailable",
                },
                "sentiment_score": 0.0,
                "errors": state.errors + [f"News analysis error: {e}"],
            }

    async def _combine_signals(self, state: AgentState) -> dict[str, Any]:
        """Combine signals from different analyses."""
        logger.info("Combining analysis signals")

        # Get technical signal
        tech_analysis = state.technical_analysis or {}
        tech_signal = tech_analysis.get("signal", "hold")
        tech_confidence = tech_analysis.get("confidence", 0.5)

        # Get news sentiment
        news_sentiment = state.sentiment_score or 0.0

        # Convert signals to numeric
        signal_map = {
            "strong_buy": 2,
            "buy": 1,
            "hold": 0,
            "sell": -1,
            "strong_sell": -2,
        }
        tech_numeric = signal_map.get(tech_signal, 0)

        # Combine with weights
        tech_weight = 0.7
        news_weight = 0.3

        # Normalize news sentiment to similar scale
        news_numeric = news_sentiment * 2  # Scale -1,1 to -2,2

        combined_score = tech_weight * tech_numeric + news_weight * news_numeric

        # Convert back to signal
        if combined_score >= 1.5:
            final_signal = "strong_buy"
        elif combined_score >= 0.5:
            final_signal = "buy"
        elif combined_score >= -0.5:
            final_signal = "hold"
        elif combined_score >= -1.5:
            final_signal = "sell"
        else:
            final_signal = "strong_sell"

        # Combined confidence
        combined_confidence = tech_confidence * 0.7 + 0.3 * (0.5 + abs(news_sentiment) * 0.5)

        return {
            "final_signal": final_signal,
            "confidence": combined_confidence,
        }

    async def _generate_decision(self, state: AgentState) -> dict[str, Any]:
        """Generate final trading decision with recommendations."""
        logger.info("Generating final decision")

        tech_analysis = state.technical_analysis or {}
        news_analysis = state.news_analysis or {}

        # Generate reasoning
        reasoning_parts = []

        # Technical reasoning
        tech_sentiment = tech_analysis.get("sentiment", "neutral")
        reasoning_parts.append(f"Technical analysis indicates {tech_sentiment} sentiment")

        if tech_analysis.get("summary"):
            reasoning_parts.append(f"Technical summary: {tech_analysis['summary']}")

        # News reasoning
        news_sentiment = news_analysis.get("overall_sentiment", 0)
        sentiment_desc = "bullish" if news_sentiment > 0.2 else "bearish" if news_sentiment < -0.2 else "neutral"
        reasoning_parts.append(f"News sentiment is {sentiment_desc} ({news_sentiment:.2f})")

        if news_analysis.get("bullish_factors"):
            reasoning_parts.append(f"Bullish factors: {', '.join(news_analysis['bullish_factors'][:3])}")

        if news_analysis.get("bearish_factors"):
            reasoning_parts.append(f"Bearish factors: {', '.join(news_analysis['bearish_factors'][:3])}")

        # Generate recommendations
        recommendations = []
        final_signal = state.final_signal or "hold"

        if final_signal in ["strong_buy", "buy"]:
            recommendations.append(f"Consider entering a long position on {state.symbol}")
            if tech_analysis.get("support"):
                recommendations.append(f"Set stop loss near support at {tech_analysis['support']}")
            if tech_analysis.get("resistance"):
                recommendations.append(f"Consider taking profit near resistance at {tech_analysis['resistance']}")

        elif final_signal in ["strong_sell", "sell"]:
            recommendations.append(f"Consider closing long positions or entering short on {state.symbol}")
            recommendations.append("Reduce position size if holding")

        else:
            recommendations.append("Maintain current positions")
            recommendations.append("Wait for clearer signals before entering new positions")

        # Risk-based recommendations
        risk_level = tech_analysis.get("risk_level", "medium")
        if risk_level == "high":
            recommendations.append("Use smaller position sizes due to elevated risk")
        elif risk_level == "low":
            recommendations.append("Conditions favorable for standard position sizing")

        return {
            "reasoning": " | ".join(reasoning_parts),
            "recommendations": recommendations,
        }

    async def run(
        self,
        symbol: str,
        current_price: float,
        technical_indicators: dict[str, Any] | None = None,
        news_articles: list[dict[str, Any]] | None = None,
    ) -> TradingDecision:
        """Run the complete analysis workflow.

        Args:
            symbol: Trading symbol
            current_price: Current price
            technical_indicators: Technical indicator values
            news_articles: Recent news articles

        Returns:
            Trading decision
        """
        initial_state = AgentState(
            symbol=symbol,
            current_price=current_price,
            technical_indicators=technical_indicators or {},
            news_articles=news_articles or [],
        )

        # Run workflow
        result = await self.graph.ainvoke(initial_state)

        # Build decision
        tech_analysis = result.get("technical_analysis", {})
        news_analysis = result.get("news_analysis", {})

        signal_map = {
            "strong_buy": TradingSignal.STRONG_BUY,
            "buy": TradingSignal.BUY,
            "hold": TradingSignal.HOLD,
            "sell": TradingSignal.SELL,
            "strong_sell": TradingSignal.STRONG_SELL,
        }

        return TradingDecision(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal=signal_map.get(result.get("final_signal", "hold"), TradingSignal.HOLD),
            confidence=result.get("confidence", 0.5),
            technical_sentiment=tech_analysis.get("sentiment", "neutral"),
            news_sentiment=f"{news_analysis.get('overall_sentiment', 0):.2f}",
            combined_sentiment=result.get("confidence", 0.5),
            entry_price=current_price if result.get("final_signal") in ["buy", "strong_buy"] else None,
            stop_loss=tech_analysis.get("support"),
            take_profit=tech_analysis.get("resistance"),
            risk_level=tech_analysis.get("risk_level", "medium"),
            position_size_recommendation=self._calculate_position_size(
                result.get("confidence", 0.5),
                tech_analysis.get("risk_level", "medium"),
            ),
            reasoning=result.get("reasoning", ""),
            key_factors=result.get("recommendations", []),
        )

    def _calculate_position_size(
        self,
        confidence: float,
        risk_level: str,
    ) -> float:
        """Calculate recommended position size."""
        # Base position (2% of portfolio)
        base_position = 0.02

        # Adjust for confidence
        confidence_multiplier = 0.5 + confidence  # 0.5 to 1.5

        # Adjust for risk
        risk_multiplier = {
            "low": 1.2,
            "medium": 1.0,
            "high": 0.7,
            "critical": 0.3,
        }.get(risk_level, 1.0)

        return min(0.1, base_position * confidence_multiplier * risk_multiplier)


async def analyze_and_decide(
    symbol: str,
    current_price: float,
    technical_indicators: dict[str, Any] | None = None,
    news_articles: list[dict[str, Any]] | None = None,
) -> TradingDecision:
    """Convenience function for complete analysis.

    Args:
        symbol: Trading symbol
        current_price: Current price
        technical_indicators: Technical indicator values
        news_articles: Recent news articles

    Returns:
        Trading decision
    """
    workflow = TradingWorkflow()
    return await workflow.run(
        symbol=symbol,
        current_price=current_price,
        technical_indicators=technical_indicators,
        news_articles=news_articles,
    )
