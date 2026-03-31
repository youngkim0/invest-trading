"""Prompt templates for Claude AI trading analysis."""

POST_TRADE_ANALYSIS_PROMPT = """You are an expert crypto futures trading analyst. Analyze this closed trade and provide actionable insights.

## Trade Details
- **Strategy**: {strategy_name}
- **Symbol**: {symbol}
- **Side**: {side}
- **Entry**: ${entry_price:,.4f} at {entry_time}
- **Exit**: ${exit_price:,.4f} at {exit_time}
- **PnL**: ${pnl:,.2f} ({pnl_pct:+.2f}% ROE)
- **Duration**: {duration}
- **Exit Reason**: {exit_reason}

## Market Context at Entry
{entry_indicators}

## Strategy Recent Performance (last 20 trades)
- Win rate: {win_rate:.0f}%
- Avg win: ${avg_win:,.2f} | Avg loss: ${avg_loss:,.2f}
- Consecutive {streak_type}: {streak_count}

## Active Positions at Time of Trade
{active_positions}

Provide a concise analysis (max 200 words):
1. **What happened**: Why did this trade {outcome}? Was the entry timing good?
2. **Pattern match**: Does this resemble any common failure/success pattern?
3. **One suggestion**: What single parameter or condition change would improve this strategy's edge?

Respond in plain text, no markdown headers."""


SIGNAL_EVALUATION_PROMPT = """You are a risk-aware crypto futures signal evaluator. Evaluate whether this signal should be taken.

## Proposed Signal
- **Strategy**: {strategy_name}
- **Symbol**: {symbol}
- **Direction**: {direction}
- **Confidence**: {confidence:.2f}
- **Reasoning**: {signal_reasoning}

## Key Indicators
{indicators}

## Strategy Performance (last 7 days)
- Trades: {recent_trade_count}
- Win rate: {win_rate:.0f}%
- Net PnL: ${net_pnl:,.2f}
- Last 5 results: {last_5_results}

## Current Portfolio State
- Open positions: {open_positions_count}
- Same-direction positions: {same_direction_count}
- Total exposure: ${total_exposure:,.2f}
- Unrealized PnL: ${unrealized_pnl:,.2f}

## Recent Losses in Similar Conditions
{similar_losses}

Evaluate this signal. Respond with EXACTLY this JSON format (no other text):
{{"adjusted_confidence": <float 0.0-1.0>, "reasoning": "<1-2 sentences>", "risk_flags": ["<flag1>", ...]}}

Rules:
- You may only REDUCE confidence, never increase above {confidence:.2f}
- Keep original confidence if the signal looks solid
- Flag correlation risk if multiple positions in same direction
- Flag regime mismatch if indicators conflict with strategy thesis"""


DAILY_REVIEW_PROMPT = """You are a senior crypto trading portfolio manager reviewing the last 24 hours of automated trading.

## Trading Summary ({review_date})
- Total trades: {total_trades}
- Winners: {winners} | Losers: {losers}
- Net PnL: ${net_pnl:,.2f}
- Best trade: ${best_trade:,.2f} ({best_strategy})
- Worst trade: ${worst_trade:,.2f} ({worst_strategy})

## Per-Strategy Breakdown
{strategy_breakdown}

## AI Trade Analyses (accumulated insights)
{trade_analyses_summary}

## 7-Day Trend
- This week PnL: ${week_pnl:,.2f}
- This week win rate: {week_win_rate:.0f}%
- Prior week PnL: ${prior_week_pnl:,.2f}

## Market Regime
{market_regime}

Provide a structured daily review (max 400 words):

1. **Performance Summary**: One paragraph assessing the day.
2. **Strategy Health**: Rate each strategy (strong/ok/weak/critical). Explain why.
3. **Top 3 Suggestions**: Ranked by expected impact. Each should be specific and actionable (e.g., "Reduce crash_momentum TP from 3.0x to 2.5x ATR because X" not "Consider adjusting parameters").
4. **Risk Alert**: Any concerning patterns that need immediate attention?

Respond in plain text."""


WEEKLY_REVIEW_PROMPT = """You are a senior crypto trading strategist conducting a weekly portfolio review.

## Weekly Summary ({start_date} to {end_date})
{weekly_stats}

## Per-Strategy Performance
{strategy_breakdown}

## Daily PnL Progression
{daily_pnl}

## Accumulated AI Insights This Week
{weekly_insights}

## Previous Week's Suggestions & Outcomes
{previous_suggestions}

## Market Conditions This Week
{market_conditions}

Provide a comprehensive weekly review (max 600 words):

1. **Week Assessment**: Overall performance trend and key takeaways.
2. **Strategy Rankings**: Rank all strategies by risk-adjusted performance. Recommend any to disable/enable.
3. **Parameter Recommendations**: Specific parameter changes with reasoning (e.g., ATR multipliers, confidence thresholds, capital allocation).
4. **Capital Reallocation**: Should capital be moved between strategies? How much?
5. **Next Week Outlook**: Based on market regime, which strategies should be prioritized?

Respond in plain text."""
