"""AI-powered feedback loop for continuous improvement."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from config import get_settings
from data.storage.repository import (
    DatabaseManager,
    LLMAnalysisRepository,
    TradeLogRepository,
)
from journal.performance import PerformanceAnalyzer, PerformanceMetrics


@dataclass
class ImprovementSuggestion:
    """Suggestion for strategy improvement."""

    category: str  # "entry", "exit", "risk_management", "position_sizing"
    priority: str  # "high", "medium", "low"
    suggestion: str
    rationale: str
    expected_impact: str
    implementation_steps: list[str]


@dataclass
class FeedbackReport:
    """Comprehensive feedback report."""

    timestamp: datetime
    analysis_period: str

    # Performance summary
    performance_grade: str  # A, B, C, D, F
    key_strengths: list[str]
    key_weaknesses: list[str]

    # Pattern analysis
    winning_patterns: list[dict[str, Any]]
    losing_patterns: list[dict[str, Any]]

    # Suggestions
    suggestions: list[ImprovementSuggestion]

    # Model recommendations
    retrain_recommended: bool
    parameter_adjustments: dict[str, Any]


class FeedbackLoop:
    """Automated feedback and improvement system."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        llm_model: str | None = None,
    ):
        """Initialize feedback loop.

        Args:
            db_manager: Database manager
            llm_model: LLM model for analysis
        """
        self.db_manager = db_manager or DatabaseManager()
        self.performance_analyzer = PerformanceAnalyzer(db_manager)

        settings = get_settings()
        api_key = settings.llm.openai_api_key.get_secret_value()

        if api_key:
            self.llm = ChatOpenAI(
                model=llm_model or settings.llm.primary_model,
                temperature=0.2,
                api_key=api_key,
            )
        else:
            self.llm = None

    async def analyze_performance(
        self,
        strategy_name: str | None = None,
        lookback_days: int = 30,
    ) -> FeedbackReport:
        """Analyze recent performance and generate feedback.

        Args:
            strategy_name: Strategy to analyze
            lookback_days: Number of days to analyze

        Returns:
            Feedback report
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        # Get performance metrics
        metrics = await self.performance_analyzer.calculate_metrics(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
        )

        # Analyze trades
        trade_patterns = await self._analyze_trade_patterns(
            strategy_name, start_date, end_date
        )

        # Generate AI insights
        ai_insights = await self._generate_ai_insights(metrics, trade_patterns)

        # Determine if retraining is needed
        retrain_recommended = self._should_retrain(metrics)

        # Get parameter adjustment suggestions
        param_adjustments = self._suggest_parameters(metrics, trade_patterns)

        return FeedbackReport(
            timestamp=datetime.utcnow(),
            analysis_period=f"{start_date.date()} to {end_date.date()}",
            performance_grade=self._calculate_grade(metrics),
            key_strengths=ai_insights.get("strengths", []),
            key_weaknesses=ai_insights.get("weaknesses", []),
            winning_patterns=trade_patterns.get("winning", []),
            losing_patterns=trade_patterns.get("losing", []),
            suggestions=ai_insights.get("suggestions", []),
            retrain_recommended=retrain_recommended,
            parameter_adjustments=param_adjustments,
        )

    async def _analyze_trade_patterns(
        self,
        strategy_name: str | None,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze patterns in winning and losing trades."""
        async with self.db_manager.session() as session:
            trade_repo = TradeLogRepository(session)

            trades = await trade_repo.get_trades(
                strategy_name=strategy_name,
                start_time=start_date,
                end_time=end_date,
                limit=500,
            )

        winning_trades = [t for t in trades if t.net_pnl and t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl and t.net_pnl < 0]

        # Analyze winning patterns
        winning_patterns = self._extract_patterns(winning_trades)
        losing_patterns = self._extract_patterns(losing_trades)

        return {
            "winning": winning_patterns,
            "losing": losing_patterns,
        }

    def _extract_patterns(self, trades: list) -> list[dict[str, Any]]:
        """Extract common patterns from trades."""
        if not trades:
            return []

        patterns = []

        # Time-based patterns
        hours = [t.entry_time.hour for t in trades if t.entry_time]
        if hours:
            most_common_hour = max(set(hours), key=hours.count)
            hour_count = hours.count(most_common_hour)
            if hour_count / len(hours) > 0.2:
                patterns.append({
                    "type": "time",
                    "description": f"Most trades at hour {most_common_hour}",
                    "frequency": hour_count / len(hours),
                })

        # Duration patterns
        durations = [
            t.duration_seconds / 3600  # Convert to hours
            for t in trades
            if t.duration_seconds
        ]
        if durations:
            avg_duration = sum(durations) / len(durations)
            patterns.append({
                "type": "duration",
                "description": f"Average trade duration: {avg_duration:.1f} hours",
                "value": avg_duration,
            })

        # Symbol patterns
        symbols = [t.symbol for t in trades]
        if symbols:
            most_common_symbol = max(set(symbols), key=symbols.count)
            symbol_count = symbols.count(most_common_symbol)
            patterns.append({
                "type": "symbol",
                "description": f"Most traded: {most_common_symbol}",
                "frequency": symbol_count / len(symbols),
            })

        # Signal source patterns
        sources = [t.signal_source.value for t in trades if t.signal_source]
        if sources:
            most_common_source = max(set(sources), key=sources.count)
            patterns.append({
                "type": "signal_source",
                "description": f"Primary signal source: {most_common_source}",
                "frequency": sources.count(most_common_source) / len(sources),
            })

        return patterns

    async def _generate_ai_insights(
        self,
        metrics: PerformanceMetrics,
        patterns: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Generate AI-powered insights."""
        if not self.llm:
            return self._generate_rule_based_insights(metrics, patterns)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert trading performance analyst.
Analyze the following trading performance data and provide actionable insights.
Focus on specific, implementable improvements."""),
            ("human", """Performance Metrics:
- Total Return: {total_return:.2%}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Win Rate: {win_rate:.2%}
- Profit Factor: {profit_factor:.2f}
- Max Drawdown: {max_drawdown:.2%}
- Total Trades: {total_trades}
- Average Win: ${avg_win:.2f}
- Average Loss: ${avg_loss:.2f}

Winning Trade Patterns:
{winning_patterns}

Losing Trade Patterns:
{losing_patterns}

Provide:
1. 3 key strengths
2. 3 key weaknesses
3. 3 specific improvement suggestions with priority (high/medium/low) and implementation steps""")
        ])

        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "max_drawdown": metrics.max_drawdown,
                "total_trades": metrics.total_trades,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "winning_patterns": str(patterns.get("winning", [])),
                "losing_patterns": str(patterns.get("losing", [])),
            })

            return self._parse_ai_response(response.content)

        except Exception as e:
            logger.error(f"AI insight generation failed: {e}")
            return self._generate_rule_based_insights(metrics, patterns)

    def _generate_rule_based_insights(
        self,
        metrics: PerformanceMetrics,
        patterns: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Generate rule-based insights when LLM is unavailable."""
        strengths = []
        weaknesses = []
        suggestions = []

        # Analyze win rate
        if metrics.win_rate > 0.55:
            strengths.append(f"Strong win rate of {metrics.win_rate:.1%}")
        elif metrics.win_rate < 0.45:
            weaknesses.append(f"Low win rate of {metrics.win_rate:.1%}")
            suggestions.append(ImprovementSuggestion(
                category="entry",
                priority="high",
                suggestion="Improve entry signal quality",
                rationale="Win rate below 45% indicates entry timing issues",
                expected_impact="Increase win rate by 5-10%",
                implementation_steps=[
                    "Add confirmation indicators",
                    "Increase minimum signal confidence threshold",
                    "Filter trades by market regime",
                ],
            ))

        # Analyze profit factor
        if metrics.profit_factor > 1.5:
            strengths.append(f"Excellent profit factor of {metrics.profit_factor:.2f}")
        elif metrics.profit_factor < 1.0:
            weaknesses.append("Unprofitable: profit factor below 1.0")
            suggestions.append(ImprovementSuggestion(
                category="risk_management",
                priority="high",
                suggestion="Improve risk/reward ratio",
                rationale="Average loss exceeds average win",
                expected_impact="Turn unprofitable strategy profitable",
                implementation_steps=[
                    "Tighten stop losses",
                    "Widen take profit targets",
                    "Cut losing trades earlier",
                ],
            ))

        # Analyze drawdown
        if metrics.max_drawdown < 0.1:
            strengths.append("Low maximum drawdown")
        elif metrics.max_drawdown > 0.2:
            weaknesses.append(f"High drawdown of {metrics.max_drawdown:.1%}")
            suggestions.append(ImprovementSuggestion(
                category="risk_management",
                priority="high",
                suggestion="Reduce position sizes during drawdown",
                rationale="Drawdown exceeds 20% threshold",
                expected_impact="Reduce max drawdown by 30-50%",
                implementation_steps=[
                    "Implement drawdown-based position scaling",
                    "Pause trading after 10% drawdown",
                    "Add correlation-based position limits",
                ],
            ))

        # Analyze Sharpe ratio
        if metrics.sharpe_ratio > 1.5:
            strengths.append(f"Strong risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
        elif metrics.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
            suggestions.append(ImprovementSuggestion(
                category="position_sizing",
                priority="medium",
                suggestion="Optimize position sizing for volatility",
                rationale="Sharpe ratio below 0.5 indicates poor risk-adjusted returns",
                expected_impact="Improve Sharpe ratio by 0.3-0.5",
                implementation_steps=[
                    "Implement volatility-adjusted position sizing",
                    "Reduce positions in high volatility periods",
                    "Add Kelly criterion for sizing",
                ],
            ))

        return {
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "suggestions": suggestions[:3],
        }

    def _parse_ai_response(self, response: str) -> dict[str, Any]:
        """Parse AI response into structured format."""
        lines = response.split("\n")

        strengths = []
        weaknesses = []
        suggestions = []

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if "strength" in lower:
                current_section = "strengths"
            elif "weakness" in lower:
                current_section = "weaknesses"
            elif "suggestion" in lower or "improvement" in lower:
                current_section = "suggestions"
            elif line.startswith("-") or line.startswith("•") or (line[0].isdigit() if line else False):
                clean = line.lstrip("0123456789.-•* ")
                if current_section == "strengths":
                    strengths.append(clean)
                elif current_section == "weaknesses":
                    weaknesses.append(clean)
                elif current_section == "suggestions":
                    suggestions.append(ImprovementSuggestion(
                        category="general",
                        priority="medium",
                        suggestion=clean,
                        rationale="AI-generated suggestion",
                        expected_impact="Potential improvement",
                        implementation_steps=[],
                    ))

        return {
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "suggestions": suggestions[:5],
        }

    def _calculate_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance grade."""
        score = 0

        # Sharpe ratio (up to 30 points)
        score += min(30, metrics.sharpe_ratio * 15)

        # Win rate (up to 20 points)
        score += min(20, metrics.win_rate * 40)

        # Profit factor (up to 20 points)
        score += min(20, (metrics.profit_factor - 1) * 10) if metrics.profit_factor > 1 else 0

        # Drawdown penalty (up to -20 points)
        score -= min(20, metrics.max_drawdown * 100)

        # Return bonus (up to 30 points)
        if metrics.total_return > 0:
            score += min(30, metrics.annualized_return * 50)

        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        elif score >= 20:
            return "D"
        else:
            return "F"

    def _should_retrain(self, metrics: PerformanceMetrics) -> bool:
        """Determine if model retraining is recommended."""
        # Retrain if:
        # 1. Performance significantly degraded
        # 2. Enough new data
        # 3. Win rate dropped

        if metrics.sharpe_ratio < 0:
            return True

        if metrics.max_drawdown > 0.25:
            return True

        if metrics.win_rate < 0.4 and metrics.total_trades > 50:
            return True

        if metrics.profit_factor < 0.8:
            return True

        return False

    def _suggest_parameters(
        self,
        metrics: PerformanceMetrics,
        patterns: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Suggest parameter adjustments."""
        adjustments = {}

        # Position sizing
        if metrics.max_drawdown > 0.15:
            adjustments["max_position_size"] = {
                "current": 0.1,
                "suggested": 0.07,
                "reason": "Reduce position size to manage drawdown",
            }

        # Stop loss
        if metrics.avg_loss > metrics.avg_win:
            adjustments["stop_loss_multiplier"] = {
                "current": 2.0,
                "suggested": 1.5,
                "reason": "Tighter stops to reduce average loss",
            }

        # Signal confidence threshold
        if metrics.win_rate < 0.5:
            adjustments["min_confidence"] = {
                "current": 0.6,
                "suggested": 0.7,
                "reason": "Higher confidence threshold to filter weak signals",
            }

        return adjustments

    async def run_periodic_analysis(
        self,
        strategy_name: str | None = None,
        interval_days: int = 7,
    ) -> FeedbackReport:
        """Run periodic analysis and store results.

        Args:
            strategy_name: Strategy to analyze
            interval_days: Analysis interval

        Returns:
            Feedback report
        """
        report = await self.analyze_performance(
            strategy_name=strategy_name,
            lookback_days=interval_days,
        )

        logger.info(
            f"Feedback analysis complete: Grade={report.performance_grade}, "
            f"Retrain={report.retrain_recommended}"
        )

        # Log to database for tracking
        async with self.db_manager.session() as session:
            llm_repo = LLMAnalysisRepository(session)

            from data.storage.models import LLMAnalysisLog

            log = LLMAnalysisLog(
                analysis_type="feedback_loop",
                prompt=f"Performance analysis for {strategy_name or 'all'}",
                response=str(report),
                model_name="feedback_loop",
            )
            await llm_repo.create(log)

        return report
