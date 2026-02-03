"""News and sentiment analysis using LLM."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from config import get_settings


class NewsImpact(BaseModel):
    """Impact assessment of a news article."""

    headline: str
    source: str | None = None
    published_at: datetime | None = None

    # Sentiment
    sentiment_score: float = Field(ge=-1.0, le=1.0)  # -1 to 1
    sentiment_label: str  # positive, negative, neutral

    # Impact
    impact_level: str  # low, medium, high, critical
    impact_duration: str  # short-term, medium-term, long-term
    affected_assets: list[str] = Field(default_factory=list)

    # Trading implications
    price_impact_direction: str  # up, down, neutral
    confidence: float = Field(ge=0.0, le=1.0)

    # Summary
    key_points: list[str] = Field(default_factory=list)
    trading_implication: str


class NewsSummary(BaseModel):
    """Aggregated news summary."""

    period: str  # e.g., "last 24 hours"
    total_articles: int

    # Overall sentiment
    overall_sentiment: float  # -1 to 1
    sentiment_distribution: dict[str, int]  # positive/negative/neutral counts

    # Key themes
    key_themes: list[str] = Field(default_factory=list)
    trending_topics: list[str] = Field(default_factory=list)

    # Impact assessment
    high_impact_news: list[NewsImpact] = Field(default_factory=list)
    overall_market_impact: str

    # Trading implications
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)


@dataclass
class NewsArticle:
    """News article data."""

    headline: str
    body: str | None = None
    source: str | None = None
    published_at: datetime | None = None
    url: str | None = None
    categories: list[str] | None = None


class NewsAnalyzer:
    """LLM-based news and sentiment analyzer."""

    SYSTEM_PROMPT = """You are an expert financial news analyst. Your role is to analyze news articles and
assess their potential impact on financial markets and specific assets.

When analyzing news:
1. Assess the sentiment (positive/negative/neutral)
2. Evaluate the potential market impact (price direction, magnitude)
3. Consider the time horizon of the impact
4. Identify affected assets and sectors
5. Extract key actionable insights

Be objective and data-driven. Focus on facts and their implications."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
    ):
        """Initialize news analyzer.

        Args:
            model_name: LLM model name
            temperature: Model temperature
        """
        settings = get_settings()

        self.model_name = model_name or settings.llm.primary_model
        self.temperature = temperature

        api_key = settings.llm.openai_api_key.get_secret_value()
        if not api_key:
            logger.warning("No OpenAI API key found. News analysis will not work.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
            )

    async def analyze_article(
        self,
        article: NewsArticle,
        target_asset: str | None = None,
    ) -> NewsImpact:
        """Analyze a single news article.

        Args:
            article: News article to analyze
            target_asset: Optional specific asset to focus on

        Returns:
            News impact assessment
        """
        if not self.llm:
            return self._default_impact(article)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", """Analyze the following news article:

Headline: {headline}
Source: {source}
Content: {body}

{target_context}

Please provide:
1. Sentiment score (-1.0 to 1.0) and label (positive/negative/neutral)
2. Impact level (low/medium/high/critical)
3. Impact duration (short-term/medium-term/long-term)
4. Price impact direction (up/down/neutral)
5. Confidence level (0.0-1.0)
6. Key points (3-5 bullet points)
7. Trading implication (one sentence)""")
        ])

        target_context = ""
        if target_asset:
            target_context = f"Focus specifically on implications for {target_asset}."

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "headline": article.headline,
                "source": article.source or "Unknown",
                "body": article.body or article.headline,
                "target_context": target_context,
            })

            return self._parse_impact(article, response.content)

        except Exception as e:
            logger.error(f"Article analysis failed: {e}")
            return self._default_impact(article)

    def _default_impact(self, article: NewsArticle) -> NewsImpact:
        """Return default neutral impact."""
        return NewsImpact(
            headline=article.headline,
            source=article.source,
            published_at=article.published_at,
            sentiment_score=0.0,
            sentiment_label="neutral",
            impact_level="low",
            impact_duration="short-term",
            price_impact_direction="neutral",
            confidence=0.5,
            key_points=[],
            trading_implication="Unable to analyze - insufficient data",
        )

    def _parse_impact(
        self,
        article: NewsArticle,
        response: str,
    ) -> NewsImpact:
        """Parse LLM response into NewsImpact."""
        import re

        response_lower = response.lower()

        # Parse sentiment score
        sentiment_score = 0.0
        score_match = re.search(r"sentiment[:\s]+(-?\d+\.?\d*)", response_lower)
        if score_match:
            sentiment_score = max(-1.0, min(1.0, float(score_match.group(1))))

        # Parse sentiment label
        sentiment_label = "neutral"
        if "positive" in response_lower:
            sentiment_label = "positive"
            if sentiment_score == 0:
                sentiment_score = 0.3
        elif "negative" in response_lower:
            sentiment_label = "negative"
            if sentiment_score == 0:
                sentiment_score = -0.3

        # Parse impact level
        impact_level = "medium"
        for level in ["critical", "high", "medium", "low"]:
            if level in response_lower:
                impact_level = level
                break

        # Parse duration
        impact_duration = "short-term"
        if "long-term" in response_lower or "long term" in response_lower:
            impact_duration = "long-term"
        elif "medium-term" in response_lower or "medium term" in response_lower:
            impact_duration = "medium-term"

        # Parse price direction
        price_direction = "neutral"
        if any(word in response_lower for word in ["bullish", "upward", "rise", "increase"]):
            price_direction = "up"
        elif any(word in response_lower for word in ["bearish", "downward", "fall", "decrease", "drop"]):
            price_direction = "down"

        # Parse confidence
        confidence = 0.5
        conf_match = re.search(r"confidence[:\s]+(\d+\.?\d*)", response_lower)
        if conf_match:
            conf_val = float(conf_match.group(1))
            confidence = conf_val if conf_val <= 1.0 else conf_val / 100

        # Extract key points
        key_points = []
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.startswith("-") or stripped.startswith("•") or (
                stripped and stripped[0].isdigit() and "." in stripped[:3]
            ):
                clean = stripped.lstrip("0123456789.-•* ")
                if 10 < len(clean) < 200:
                    key_points.append(clean)

        # Extract trading implication
        trading_implication = "Review required"
        for line in response.split("\n"):
            if "trading" in line.lower() and "implication" in line.lower():
                parts = line.split(":")
                if len(parts) > 1:
                    trading_implication = parts[1].strip()
                    break

        return NewsImpact(
            headline=article.headline,
            source=article.source,
            published_at=article.published_at,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            impact_level=impact_level,
            impact_duration=impact_duration,
            price_impact_direction=price_direction,
            confidence=confidence,
            key_points=key_points[:5],
            trading_implication=trading_implication,
        )

    async def analyze_batch(
        self,
        articles: list[NewsArticle],
        target_asset: str | None = None,
    ) -> NewsSummary:
        """Analyze multiple news articles and summarize.

        Args:
            articles: List of news articles
            target_asset: Optional specific asset to focus on

        Returns:
            Aggregated news summary
        """
        if not articles:
            return self._empty_summary()

        # Analyze individual articles
        impacts = []
        for article in articles[:20]:  # Limit to 20 articles
            impact = await self.analyze_article(article, target_asset)
            impacts.append(impact)

        # Aggregate sentiment
        sentiment_scores = [i.sentiment_score for i in impacts]
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        sentiment_dist = {
            "positive": sum(1 for i in impacts if i.sentiment_label == "positive"),
            "negative": sum(1 for i in impacts if i.sentiment_label == "negative"),
            "neutral": sum(1 for i in impacts if i.sentiment_label == "neutral"),
        }

        # Identify high impact news
        high_impact = [i for i in impacts if i.impact_level in ["high", "critical"]]

        # Extract factors
        bullish = []
        bearish = []
        for impact in impacts:
            if impact.price_impact_direction == "up" and impact.confidence > 0.5:
                bullish.extend(impact.key_points[:2])
            elif impact.price_impact_direction == "down" and impact.confidence > 0.5:
                bearish.extend(impact.key_points[:2])

        # Determine overall impact
        if len(high_impact) > 3:
            overall_impact = "High market impact expected"
        elif len(high_impact) > 0:
            overall_impact = "Moderate market impact expected"
        else:
            overall_impact = "Low market impact expected"

        return NewsSummary(
            period="analyzed batch",
            total_articles=len(articles),
            overall_sentiment=overall_sentiment,
            sentiment_distribution=sentiment_dist,
            key_themes=self._extract_themes(impacts),
            trending_topics=[],
            high_impact_news=high_impact[:5],
            overall_market_impact=overall_impact,
            bullish_factors=list(set(bullish))[:5],
            bearish_factors=list(set(bearish))[:5],
        )

    def _empty_summary(self) -> NewsSummary:
        """Return empty summary."""
        return NewsSummary(
            period="N/A",
            total_articles=0,
            overall_sentiment=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0},
            key_themes=[],
            trending_topics=[],
            high_impact_news=[],
            overall_market_impact="No news to analyze",
            bullish_factors=[],
            bearish_factors=[],
        )

    def _extract_themes(self, impacts: list[NewsImpact]) -> list[str]:
        """Extract common themes from impacts."""
        # Simple keyword extraction
        all_points = []
        for impact in impacts:
            all_points.extend(impact.key_points)

        # Count word frequency (simplified)
        from collections import Counter
        words = []
        for point in all_points:
            words.extend(point.lower().split())

        # Filter common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                       "have", "has", "had", "do", "does", "did", "will", "would",
                       "could", "should", "may", "might", "to", "of", "in", "for",
                       "on", "with", "at", "by", "from", "as", "into", "through",
                       "and", "or", "but", "if", "then", "that", "this", "these",
                       "those", "it", "its", "their", "them", "they"}

        filtered = [w for w in words if w not in common_words and len(w) > 3]
        word_counts = Counter(filtered)

        return [word for word, _ in word_counts.most_common(5)]

    async def get_sentiment_score(
        self,
        text: str,
    ) -> float:
        """Get simple sentiment score for text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-1 to 1)
        """
        if not self.llm:
            return 0.0

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a sentiment analyzer. Respond only with a number between -1.0 (very negative) and 1.0 (very positive)."),
            ("human", "Sentiment of: {text}")
        ])

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"text": text[:500]})

            import re
            match = re.search(r"-?\d+\.?\d*", response.content)
            if match:
                return max(-1.0, min(1.0, float(match.group())))
            return 0.0

        except Exception as e:
            logger.error(f"Sentiment scoring failed: {e}")
            return 0.0


async def analyze_news(
    articles: list[dict[str, Any]],
    target_asset: str | None = None,
) -> NewsSummary:
    """Convenience function to analyze news articles.

    Args:
        articles: List of article dictionaries with 'headline' and optional 'body', 'source'
        target_asset: Optional asset to focus on

    Returns:
        News summary
    """
    analyzer = NewsAnalyzer()

    parsed_articles = [
        NewsArticle(
            headline=a.get("headline", a.get("title", "")),
            body=a.get("body", a.get("content", "")),
            source=a.get("source"),
            published_at=a.get("published_at"),
        )
        for a in articles
    ]

    return await analyzer.analyze_batch(parsed_articles, target_asset)
