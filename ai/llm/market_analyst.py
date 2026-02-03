"""LLM-based market analysis agent.

Uses LangChain/LangGraph for market analysis and trading insights.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from config import get_settings


class MarketSentiment(str, Enum):
    """Market sentiment classification."""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class TradingSignal(str, Enum):
    """Trading signal from analysis."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MarketAnalysis(BaseModel):
    """Structured market analysis output."""

    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sentiment: MarketSentiment
    signal: TradingSignal
    confidence: float = Field(ge=0.0, le=1.0)

    # Analysis components
    technical_summary: str
    fundamental_summary: str | None = None
    news_summary: str | None = None

    # Price targets
    support_level: float | None = None
    resistance_level: float | None = None
    price_target: float | None = None

    # Risk assessment
    risk_level: str  # low, medium, high
    key_risks: list[str] = Field(default_factory=list)

    # Reasoning
    reasoning: str
    key_factors: list[str] = Field(default_factory=list)


@dataclass
class MarketContext:
    """Context data for market analysis."""

    symbol: str
    current_price: Decimal
    price_change_24h: Decimal | None = None
    volume_24h: Decimal | None = None

    # Technical indicators
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    sma_20: Decimal | None = None
    sma_50: Decimal | None = None
    sma_200: Decimal | None = None
    bb_upper: Decimal | None = None
    bb_lower: Decimal | None = None
    atr: Decimal | None = None

    # Market data
    market_cap: Decimal | None = None
    fear_greed_index: int | None = None

    # News and sentiment
    recent_news: list[str] | None = None
    social_sentiment: float | None = None

    def to_prompt_context(self) -> str:
        """Convert to prompt-friendly string."""
        lines = [
            f"Symbol: {self.symbol}",
            f"Current Price: ${self.current_price}",
        ]

        if self.price_change_24h:
            lines.append(f"24h Change: {self.price_change_24h}%")
        if self.volume_24h:
            lines.append(f"24h Volume: ${self.volume_24h}")

        # Technical indicators
        tech_lines = ["", "Technical Indicators:"]
        if self.rsi is not None:
            tech_lines.append(f"  RSI(14): {self.rsi:.2f}")
        if self.macd is not None:
            tech_lines.append(f"  MACD: {self.macd:.4f}")
        if self.macd_signal is not None:
            tech_lines.append(f"  MACD Signal: {self.macd_signal:.4f}")
        if self.sma_20:
            tech_lines.append(f"  SMA(20): ${self.sma_20}")
        if self.sma_50:
            tech_lines.append(f"  SMA(50): ${self.sma_50}")
        if self.sma_200:
            tech_lines.append(f"  SMA(200): ${self.sma_200}")
        if self.bb_upper and self.bb_lower:
            tech_lines.append(f"  Bollinger Bands: ${self.bb_lower} - ${self.bb_upper}")
        if self.atr:
            tech_lines.append(f"  ATR(14): ${self.atr}")

        if len(tech_lines) > 2:
            lines.extend(tech_lines)

        # Market context
        if self.fear_greed_index is not None:
            lines.append(f"\nFear & Greed Index: {self.fear_greed_index}")

        # News
        if self.recent_news:
            lines.append("\nRecent News Headlines:")
            for news in self.recent_news[:5]:
                lines.append(f"  - {news}")

        return "\n".join(lines)


class MarketAnalyst:
    """LLM-based market analyst agent."""

    SYSTEM_PROMPT = """You are an expert financial market analyst with deep knowledge of technical analysis,
fundamental analysis, and market psychology. Your role is to analyze market data and provide actionable trading insights.

When analyzing markets, you should:
1. Consider technical indicators and price patterns
2. Assess market sentiment and psychology
3. Evaluate risk/reward ratios
4. Provide clear, actionable recommendations
5. Always explain your reasoning

Be objective and data-driven. Acknowledge uncertainty when appropriate.
Do not provide financial advice - only analysis and educational information."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
    ):
        """Initialize market analyst.

        Args:
            model_name: LLM model name
            temperature: Model temperature
        """
        settings = get_settings()

        self.model_name = model_name or settings.llm.primary_model
        self.temperature = temperature

        # Initialize LLM
        api_key = settings.llm.openai_api_key.get_secret_value()
        if not api_key:
            logger.warning("No OpenAI API key found. LLM features will not work.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
            )

    async def analyze(
        self,
        context: MarketContext,
    ) -> MarketAnalysis:
        """Analyze market conditions.

        Args:
            context: Market context data

        Returns:
            Structured market analysis
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized. Check API key.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", """Analyze the following market data and provide a comprehensive analysis.

{context}

Please provide your analysis in the following format:
1. Overall sentiment (very_bullish/bullish/neutral/bearish/very_bearish)
2. Trading signal (strong_buy/buy/hold/sell/strong_sell)
3. Confidence level (0.0-1.0)
4. Technical analysis summary
5. Support and resistance levels
6. Risk assessment (low/medium/high)
7. Key risks to consider
8. Detailed reasoning""")
        ])

        # Get LLM response
        chain = prompt | self.llm
        response = await chain.ainvoke({"context": context.to_prompt_context()})

        # Parse response
        return self._parse_analysis(context.symbol, response.content)

    def _parse_analysis(
        self,
        symbol: str,
        response: str,
    ) -> MarketAnalysis:
        """Parse LLM response into structured analysis."""
        # Default values
        sentiment = MarketSentiment.NEUTRAL
        signal = TradingSignal.HOLD
        confidence = 0.5
        risk_level = "medium"

        # Simple keyword parsing (in production, use structured output)
        response_lower = response.lower()

        # Parse sentiment
        if "very bullish" in response_lower or "extremely bullish" in response_lower:
            sentiment = MarketSentiment.VERY_BULLISH
        elif "bullish" in response_lower:
            sentiment = MarketSentiment.BULLISH
        elif "very bearish" in response_lower or "extremely bearish" in response_lower:
            sentiment = MarketSentiment.VERY_BEARISH
        elif "bearish" in response_lower:
            sentiment = MarketSentiment.BEARISH

        # Parse signal
        if "strong buy" in response_lower:
            signal = TradingSignal.STRONG_BUY
        elif "buy" in response_lower and "don't buy" not in response_lower:
            signal = TradingSignal.BUY
        elif "strong sell" in response_lower:
            signal = TradingSignal.STRONG_SELL
        elif "sell" in response_lower and "don't sell" not in response_lower:
            signal = TradingSignal.SELL

        # Parse confidence
        import re
        confidence_match = re.search(r"confidence[:\s]+(\d+\.?\d*)", response_lower)
        if confidence_match:
            conf_val = float(confidence_match.group(1))
            confidence = conf_val if conf_val <= 1.0 else conf_val / 100

        # Parse risk level
        if "high risk" in response_lower:
            risk_level = "high"
        elif "low risk" in response_lower:
            risk_level = "low"

        # Extract key risks (lines starting with risk indicators)
        key_risks = []
        for line in response.split("\n"):
            line_lower = line.lower().strip()
            if line_lower.startswith("-") or line_lower.startswith("•"):
                if any(word in line_lower for word in ["risk", "concern", "warning", "caution"]):
                    key_risks.append(line.strip("- •"))

        return MarketAnalysis(
            symbol=symbol,
            sentiment=sentiment,
            signal=signal,
            confidence=confidence,
            technical_summary=self._extract_section(response, "technical"),
            reasoning=response,
            risk_level=risk_level,
            key_risks=key_risks[:5],
            key_factors=self._extract_key_factors(response),
        )

    def _extract_section(self, response: str, section: str) -> str:
        """Extract a section from the response."""
        lines = response.split("\n")
        in_section = False
        section_lines = []

        for line in lines:
            if section.lower() in line.lower() and ":" in line:
                in_section = True
                continue
            elif in_section:
                if line.strip() and not any(
                    keyword in line.lower()
                    for keyword in ["summary", "analysis", "assessment", "signal", "recommendation"]
                ):
                    section_lines.append(line)
                elif line.strip() == "" and section_lines:
                    break

        return " ".join(section_lines).strip() or "Not available"

    def _extract_key_factors(self, response: str) -> list[str]:
        """Extract key factors from response."""
        factors = []
        lines = response.split("\n")

        for line in lines:
            # Look for numbered or bulleted items
            stripped = line.strip()
            if stripped and (
                stripped[0].isdigit() or
                stripped.startswith("-") or
                stripped.startswith("•") or
                stripped.startswith("*")
            ):
                # Clean and add
                clean = stripped.lstrip("0123456789.-•* ")
                if len(clean) > 10 and len(clean) < 200:
                    factors.append(clean)

        return factors[:10]

    async def get_quick_sentiment(
        self,
        symbol: str,
        current_price: Decimal,
        rsi: float | None = None,
        macd: float | None = None,
    ) -> dict[str, Any]:
        """Get quick sentiment analysis.

        Args:
            symbol: Trading symbol
            current_price: Current price
            rsi: RSI value
            macd: MACD value

        Returns:
            Quick sentiment dict
        """
        if not self.llm:
            # Return neutral if no LLM
            return {
                "sentiment": MarketSentiment.NEUTRAL.value,
                "signal": TradingSignal.HOLD.value,
                "confidence": 0.5,
            }

        context = MarketContext(
            symbol=symbol,
            current_price=current_price,
            rsi=rsi,
            macd=macd,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a market analyst. Respond only with: SENTIMENT: [value] SIGNAL: [value] CONFIDENCE: [0-1]"),
            ("human", "Quick analysis for {symbol} at ${price}. RSI: {rsi}, MACD: {macd}")
        ])

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "symbol": symbol,
                "price": current_price,
                "rsi": rsi or "N/A",
                "macd": macd or "N/A",
            })

            # Parse quick response
            text = response.content.lower()

            sentiment = MarketSentiment.NEUTRAL
            if "bullish" in text:
                sentiment = MarketSentiment.BULLISH
            elif "bearish" in text:
                sentiment = MarketSentiment.BEARISH

            signal = TradingSignal.HOLD
            if "buy" in text:
                signal = TradingSignal.BUY
            elif "sell" in text:
                signal = TradingSignal.SELL

            import re
            conf_match = re.search(r"confidence[:\s]+(\d+\.?\d*)", text)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            if confidence > 1:
                confidence /= 100

            return {
                "sentiment": sentiment.value,
                "signal": signal.value,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Quick sentiment analysis failed: {e}")
            return {
                "sentiment": MarketSentiment.NEUTRAL.value,
                "signal": TradingSignal.HOLD.value,
                "confidence": 0.5,
            }


async def analyze_market(
    symbol: str,
    current_price: Decimal,
    **kwargs: Any,
) -> MarketAnalysis:
    """Convenience function for market analysis.

    Args:
        symbol: Trading symbol
        current_price: Current price
        **kwargs: Additional context fields

    Returns:
        Market analysis
    """
    analyst = MarketAnalyst()
    context = MarketContext(symbol=symbol, current_price=current_price, **kwargs)
    return await analyst.analyze(context)
