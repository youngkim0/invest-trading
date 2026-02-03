"""Sentiment data collector for news and social media."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import httpx
from loguru import logger


class SentimentSource(str, Enum):
    """Sources for sentiment data."""

    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FEAR_GREED = "fear_greed"


class SentimentLevel(str, Enum):
    """Sentiment classification levels."""

    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentData:
    """Container for sentiment data."""

    source: SentimentSource
    symbol: str | None
    timestamp: datetime
    score: float  # -1.0 to 1.0
    level: SentimentLevel
    raw_data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SentimentCollector:
    """Collects sentiment data from various sources."""

    # Fear & Greed Index API (free, no auth required)
    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    # CryptoCompare News API
    CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"

    def __init__(
        self,
        cryptocompare_api_key: str | None = None,
        news_api_key: str | None = None,
    ):
        """Initialize sentiment collector.

        Args:
            cryptocompare_api_key: API key for CryptoCompare
            news_api_key: API key for NewsAPI
        """
        self.cryptocompare_api_key = cryptocompare_api_key
        self.news_api_key = news_api_key
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "AI Trading System/1.0"},
        )
        logger.info("Sentiment collector initialized")

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Sentiment collector closed")

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if not self._client:
            raise RuntimeError("Sentiment collector not initialized")
        return self._client

    # ==========================================================================
    # Fear & Greed Index
    # ==========================================================================

    async def fetch_fear_greed_index(
        self,
        limit: int = 1,
    ) -> list[SentimentData]:
        """Fetch Bitcoin Fear & Greed Index.

        Args:
            limit: Number of historical data points

        Returns:
            List of sentiment data
        """
        try:
            response = await self.client.get(
                self.FEAR_GREED_URL,
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("data", []):
                value = int(item["value"])
                classification = item["value_classification"].lower()

                # Map classification to SentimentLevel
                level_map = {
                    "extreme fear": SentimentLevel.EXTREME_FEAR,
                    "fear": SentimentLevel.FEAR,
                    "neutral": SentimentLevel.NEUTRAL,
                    "greed": SentimentLevel.GREED,
                    "extreme greed": SentimentLevel.EXTREME_GREED,
                }
                level = level_map.get(classification, SentimentLevel.NEUTRAL)

                # Convert 0-100 to -1.0 to 1.0
                score = (value - 50) / 50

                results.append(
                    SentimentData(
                        source=SentimentSource.FEAR_GREED,
                        symbol="BTC",
                        timestamp=datetime.fromtimestamp(int(item["timestamp"])),
                        score=score,
                        level=level,
                        raw_data=item,
                        metadata={"index_value": value},
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return []

    # ==========================================================================
    # News Sentiment
    # ==========================================================================

    async def fetch_crypto_news(
        self,
        categories: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch crypto news from CryptoCompare.

        Args:
            categories: News categories to filter
            limit: Number of articles

        Returns:
            List of news articles
        """
        try:
            params = {}
            if categories:
                params["categories"] = ",".join(categories)
            if self.cryptocompare_api_key:
                params["api_key"] = self.cryptocompare_api_key

            response = await self.client.get(
                self.CRYPTOCOMPARE_NEWS_URL,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            articles = data.get("Data", [])[:limit]

            return [
                {
                    "id": article["id"],
                    "title": article["title"],
                    "body": article["body"],
                    "url": article["url"],
                    "source": article["source"],
                    "published_at": datetime.fromtimestamp(article["published_on"]),
                    "categories": article.get("categories", "").split("|"),
                    "tags": article.get("tags", "").split("|"),
                }
                for article in articles
            ]

        except Exception as e:
            logger.error(f"Error fetching crypto news: {e}")
            return []

    async def analyze_news_sentiment(
        self,
        articles: list[dict[str, Any]],
    ) -> list[SentimentData]:
        """Analyze sentiment of news articles.

        This is a placeholder for actual sentiment analysis.
        In production, you'd use an LLM or sentiment model.

        Args:
            articles: List of news articles

        Returns:
            List of sentiment data
        """
        results = []

        # Simple keyword-based sentiment (placeholder)
        positive_keywords = [
            "bullish", "rally", "surge", "gain", "profit", "growth",
            "adoption", "breakthrough", "milestone", "record",
        ]
        negative_keywords = [
            "bearish", "crash", "plunge", "loss", "decline", "fear",
            "hack", "scam", "regulation", "ban", "warning",
        ]

        for article in articles:
            text = f"{article['title']} {article.get('body', '')}".lower()

            positive_count = sum(1 for kw in positive_keywords if kw in text)
            negative_count = sum(1 for kw in negative_keywords if kw in text)

            total = positive_count + negative_count
            if total == 0:
                score = 0.0
                level = SentimentLevel.NEUTRAL
            else:
                score = (positive_count - negative_count) / total
                if score >= 0.5:
                    level = SentimentLevel.GREED
                elif score >= 0.2:
                    level = SentimentLevel.NEUTRAL
                elif score >= -0.2:
                    level = SentimentLevel.NEUTRAL
                elif score >= -0.5:
                    level = SentimentLevel.FEAR
                else:
                    level = SentimentLevel.EXTREME_FEAR

            results.append(
                SentimentData(
                    source=SentimentSource.NEWS,
                    symbol=None,  # General crypto news
                    timestamp=article.get("published_at", datetime.utcnow()),
                    score=score,
                    level=level,
                    raw_data={"title": article["title"], "source": article["source"]},
                )
            )

        return results

    # ==========================================================================
    # Aggregate Sentiment
    # ==========================================================================

    async def get_market_sentiment(
        self,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated market sentiment.

        Args:
            symbol: Optional symbol to focus on

        Returns:
            Aggregated sentiment data
        """
        # Fetch from multiple sources
        fear_greed = await self.fetch_fear_greed_index(limit=1)
        news = await self.fetch_crypto_news(limit=20)
        news_sentiment = await self.analyze_news_sentiment(news)

        # Calculate aggregate scores
        all_sentiments = fear_greed + news_sentiment

        if not all_sentiments:
            return {
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "aggregate_score": 0.0,
                "level": SentimentLevel.NEUTRAL,
                "sources": {},
                "confidence": 0.0,
            }

        # Weight scores (Fear & Greed gets more weight)
        weighted_sum = 0.0
        total_weight = 0.0

        source_scores: dict[str, list[float]] = {}

        for sentiment in all_sentiments:
            weight = 2.0 if sentiment.source == SentimentSource.FEAR_GREED else 1.0
            weighted_sum += sentiment.score * weight
            total_weight += weight

            if sentiment.source.value not in source_scores:
                source_scores[sentiment.source.value] = []
            source_scores[sentiment.source.value].append(sentiment.score)

        aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine level
        if aggregate_score >= 0.5:
            level = SentimentLevel.EXTREME_GREED
        elif aggregate_score >= 0.2:
            level = SentimentLevel.GREED
        elif aggregate_score >= -0.2:
            level = SentimentLevel.NEUTRAL
        elif aggregate_score >= -0.5:
            level = SentimentLevel.FEAR
        else:
            level = SentimentLevel.EXTREME_FEAR

        # Calculate source averages
        source_averages = {
            source: sum(scores) / len(scores)
            for source, scores in source_scores.items()
        }

        # Confidence based on number of data points
        confidence = min(1.0, len(all_sentiments) / 10.0)

        return {
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "aggregate_score": round(aggregate_score, 4),
            "level": level,
            "sources": source_averages,
            "data_points": len(all_sentiments),
            "confidence": round(confidence, 2),
        }

    # ==========================================================================
    # Historical Sentiment
    # ==========================================================================

    async def get_historical_fear_greed(
        self,
        days: int = 30,
    ) -> list[SentimentData]:
        """Get historical Fear & Greed Index.

        Args:
            days: Number of days of history

        Returns:
            List of historical sentiment data
        """
        return await self.fetch_fear_greed_index(limit=days)

    async def get_sentiment_trend(
        self,
        days: int = 7,
    ) -> dict[str, Any]:
        """Calculate sentiment trend over time.

        Args:
            days: Number of days for trend analysis

        Returns:
            Trend analysis dictionary
        """
        historical = await self.get_historical_fear_greed(days=days)

        if len(historical) < 2:
            return {
                "trend": "unknown",
                "change": 0.0,
                "current": None,
                "previous": None,
            }

        current = historical[0]
        previous = historical[-1]

        change = current.score - previous.score

        if change > 0.1:
            trend = "improving"
        elif change < -0.1:
            trend = "deteriorating"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": round(change, 4),
            "current": {
                "score": current.score,
                "level": current.level.value,
                "timestamp": current.timestamp,
            },
            "previous": {
                "score": previous.score,
                "level": previous.level.value,
                "timestamp": previous.timestamp,
            },
        }


# Factory function
async def create_sentiment_collector(
    cryptocompare_api_key: str | None = None,
    news_api_key: str | None = None,
) -> SentimentCollector:
    """Create and initialize a sentiment collector.

    Args:
        cryptocompare_api_key: Optional CryptoCompare API key
        news_api_key: Optional NewsAPI key

    Returns:
        Initialized sentiment collector
    """
    collector = SentimentCollector(
        cryptocompare_api_key=cryptocompare_api_key,
        news_api_key=news_api_key,
    )
    await collector.initialize()
    return collector
