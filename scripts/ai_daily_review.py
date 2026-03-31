#!/usr/bin/env python
"""AI-powered daily trading performance review.

Runs once per day (via systemd timer or cron). Pulls recent trades,
accumulated AI analyses, and generates a comprehensive daily review
using Claude Sonnet.

Usage:
    python scripts/ai_daily_review.py              # Today's review
    python scripts/ai_daily_review.py --date 2026-03-30  # Specific date
    python scripts/ai_daily_review.py --weekly      # Weekly review
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from data.storage.supabase_client import (
    TradeLogRepository,
    TradeAnalysisRepository,
    AIReviewRepository,
)
from core.ai.claude_client import ClaudeAnalyzer


async def run_daily_review(review_date: str | None = None):
    """Generate and store a daily AI review."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return

    analyzer = ClaudeAnalyzer(api_key=api_key)
    trade_repo = TradeLogRepository()
    analysis_repo = TradeAnalysisRepository()
    review_repo = AIReviewRepository()

    if review_date is None:
        review_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Date ranges
    day_start = f"{review_date}T00:00:00+00:00"
    day_end = f"{review_date}T23:59:59+00:00"

    # Last 7 days for context
    review_dt = datetime.strptime(review_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    week_start = (review_dt - timedelta(days=7)).isoformat()
    prior_week_start = (review_dt - timedelta(days=14)).isoformat()

    logger.info(f"[AI Review] Generating daily review for {review_date}")

    # Fetch data
    try:
        today_trades = await asyncio.to_thread(
            lambda: trade_repo.table.select("*")
            .gte("exit_time", day_start)
            .lte("exit_time", day_end)
            .order("exit_time", desc=True)
            .execute()
        )
        today_trades = today_trades.data or []

        week_trades = await asyncio.to_thread(
            lambda: trade_repo.table.select("*")
            .gte("exit_time", week_start)
            .lte("exit_time", day_end)
            .order("exit_time", desc=True)
            .execute()
        )
        week_trades = week_trades.data or []

        prior_week_trades = await asyncio.to_thread(
            lambda: trade_repo.table.select("*")
            .gte("exit_time", prior_week_start)
            .lt("exit_time", week_start)
            .order("exit_time", desc=True)
            .execute()
        )
        prior_week_trades = prior_week_trades.data or []

        # Get AI trade analyses from today
        trade_analyses = await asyncio.to_thread(
            lambda: analysis_repo.table.select("*")
            .gte("created_at", day_start)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        trade_analyses = trade_analyses.data or []

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    if not today_trades:
        logger.info(f"[AI Review] No trades found for {review_date}, skipping review")
        return

    logger.info(f"[AI Review] Found {len(today_trades)} trades today, generating review...")

    # Generate review
    review = await analyzer.generate_daily_review(
        review_date=review_date,
        trades=today_trades,
        trade_analyses=trade_analyses,
        week_trades=week_trades,
        prior_week_trades=prior_week_trades,
    )

    if not review:
        logger.error("[AI Review] Failed to generate review")
        return

    # Save to Supabase
    try:
        import json
        await asyncio.to_thread(
            lambda: review_repo.table.insert({
                "review_date": review_date,
                "period": review.period,
                "summary": review.summary,
                "strategy_insights": json.dumps(review.strategy_insights),
                "suggestions": json.dumps(review.suggestions),
                "model_used": review.model_used,
                "tokens_used": review.tokens_used,
            }).execute()
        )
        logger.info(
            f"[AI Review] Daily review saved for {review_date} "
            f"({review.tokens_used} tokens, ${analyzer.daily_cost:.4f})"
        )
    except Exception as e:
        logger.error(f"Failed to save review: {e}")
        return

    # Print summary
    print(f"\n{'='*60}")
    print(f"  AI Daily Review — {review_date}")
    print(f"{'='*60}\n")
    print(review.summary)
    if review.suggestions:
        print(f"\n{'─'*60}")
        print("  Top Suggestions:")
        for s in review.suggestions:
            print(f"  {s.get('priority', '?')}. {s.get('suggestion', '')}")
    print(f"\n{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description="AI Daily Trading Review")
    parser.add_argument("--date", type=str, help="Review date (YYYY-MM-DD)")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly review")
    args = parser.parse_args()

    if args.weekly:
        # For weekly, review the last 7 days
        end_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info(f"[AI Review] Weekly review ending {end_date} (not yet implemented, running daily)")
        await run_daily_review(end_date)
    else:
        await run_daily_review(args.date)


if __name__ == "__main__":
    asyncio.run(main())
