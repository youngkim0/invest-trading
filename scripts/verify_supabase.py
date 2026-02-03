#!/usr/bin/env python
"""Verify Supabase tables and insert test data."""

import asyncio
from datetime import datetime, timedelta
import random
import uuid
import os

from loguru import logger
from dotenv import load_dotenv

# Change to project root
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
load_dotenv(project_root / ".env")

from data.storage.supabase_client import (
    get_supabase_client,
    OHLCVRepository,
    TradeLogRepository,
    PerformanceRepository,
    SignalRepository,
)


async def verify_tables():
    """Verify all tables exist."""
    client = get_supabase_client()

    tables = ["ohlcv", "trade_logs", "performance_snapshots", "signals", "model_checkpoints", "llm_analysis_logs"]
    results = {}

    for table in tables:
        try:
            result = client.table(table).select("*").limit(1).execute()
            results[table] = "✅ OK"
            logger.info(f"Table '{table}' exists and is accessible")
        except Exception as e:
            results[table] = f"❌ Error: {e}"
            logger.error(f"Table '{table}' error: {e}")

    return results


async def insert_sample_ohlcv():
    """Insert sample OHLCV data."""
    repo = OHLCVRepository()

    # Generate sample candles for the last 24 hours
    candles = []
    base_price = 95000.0  # BTC price
    now = datetime.utcnow()

    for i in range(24):
        timestamp = now - timedelta(hours=24-i)
        open_price = base_price + random.uniform(-500, 500)
        close_price = open_price + random.uniform(-300, 300)
        high_price = max(open_price, close_price) + random.uniform(0, 200)
        low_price = min(open_price, close_price) - random.uniform(0, 200)
        volume = random.uniform(100, 1000)

        candles.append({
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "timeframe": "1h",
            "timestamp": timestamp.isoformat(),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 4),
        })

        base_price = close_price

    count = await repo.save_candles(candles)
    logger.info(f"Inserted {count} OHLCV candles")
    return count


async def insert_sample_trades():
    """Insert sample trade logs."""
    repo = TradeLogRepository()

    trades = []
    now = datetime.utcnow()

    for i in range(10):
        entry_time = now - timedelta(days=10-i, hours=random.randint(0, 12))
        exit_time = entry_time + timedelta(hours=random.randint(1, 24))

        entry_price = 94000 + random.uniform(-2000, 2000)
        pnl_pct = random.uniform(-0.05, 0.08)
        exit_price = entry_price * (1 + pnl_pct)
        quantity = random.uniform(0.01, 0.1)

        gross_pnl = (exit_price - entry_price) * quantity

        trade = {
            "position_id": str(uuid.uuid4()),
            "symbol": random.choice(["BTCUSDT", "ETHUSDT"]),
            "exchange": "binance",
            "side": random.choice(["buy", "sell"]),
            "entry_price": round(entry_price, 2),
            "entry_time": entry_time.isoformat(),
            "exit_price": round(exit_price, 2),
            "exit_time": exit_time.isoformat(),
            "quantity": round(quantity, 6),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(gross_pnl * 0.999, 2),  # After fees
            "return_pct": round(pnl_pct * 100, 2),
            "duration_seconds": int((exit_time - entry_time).total_seconds()),
            "strategy_name": "hybrid_v1",
            "signal_source": random.choice(["rl", "llm", "hybrid"]),
            "signal_confidence": round(random.uniform(0.6, 0.95), 4),
        }

        result = await repo.log_trade(trade)
        if result:
            trades.append(result)

    logger.info(f"Inserted {len(trades)} trade logs")
    return len(trades)


async def insert_sample_performance():
    """Insert sample performance snapshots."""
    repo = PerformanceRepository()

    snapshots = []
    now = datetime.utcnow()
    base_equity = 10000.0

    for i in range(30):
        timestamp = now - timedelta(days=30-i)
        daily_change = random.uniform(-0.03, 0.05)
        base_equity *= (1 + daily_change)

        winning = random.randint(0, 5)
        losing = random.randint(0, 3)
        total = winning + losing

        snapshot = {
            "timestamp": timestamp.isoformat(),
            "strategy_name": "hybrid_v1",
            "total_equity": round(base_equity, 2),
            "cash_balance": round(base_equity * 0.3, 2),
            "positions_value": round(base_equity * 0.7, 2),
            "daily_pnl": round(base_equity * daily_change, 2),
            "total_pnl": round(base_equity - 10000, 2),
            "unrealized_pnl": round(random.uniform(-100, 200), 2),
            "total_trades": total,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": round(winning / total, 4) if total > 0 else 0,
            "max_drawdown": round(random.uniform(0.02, 0.15), 4),
            "sharpe_ratio": round(random.uniform(0.5, 2.5), 4),
            "sortino_ratio": round(random.uniform(0.8, 3.0), 4),
        }

        result = await repo.save_snapshot(snapshot)
        if result:
            snapshots.append(result)

    logger.info(f"Inserted {len(snapshots)} performance snapshots")
    return len(snapshots)


async def insert_sample_signals():
    """Insert sample trading signals."""
    repo = SignalRepository()

    signals = []
    now = datetime.utcnow()

    for i in range(20):
        timestamp = now - timedelta(hours=i*2)

        signal = {
            "timestamp": timestamp.isoformat(),
            "symbol": random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"]),
            "exchange": "binance",
            "timeframe": "1h",
            "signal_type": random.choice(["buy", "sell", "strong_buy", "strong_sell", "hold"]),
            "source": random.choice(["rl", "llm", "hybrid", "technical"]),
            "confidence": round(random.uniform(0.5, 0.98), 4),
            "entry_price": round(94000 + random.uniform(-2000, 2000), 2),
            "stop_loss": round(94000 - random.uniform(500, 1500), 2),
            "take_profit": round(94000 + random.uniform(1000, 3000), 2),
            "reasoning": f"Signal generated based on technical indicators and ML model prediction",
            "status": random.choice(["pending", "executed", "expired"]),
        }

        result = await repo.save_signal(signal)
        if result:
            signals.append(result)

    logger.info(f"Inserted {len(signals)} signals")
    return len(signals)


async def main():
    """Main function to verify and populate Supabase."""
    print("=" * 60)
    print("Supabase Verification and Test Data Insertion")
    print("=" * 60)

    # Step 1: Verify tables
    print("\n📋 Verifying tables...")
    results = await verify_tables()

    all_ok = all("OK" in v for v in results.values())

    for table, status in results.items():
        print(f"  {table}: {status}")

    if not all_ok:
        print("\n❌ Some tables are missing. Please run the SQL schema first.")
        return

    print("\n✅ All tables verified!")

    # Step 2: Insert sample data
    print("\n📊 Inserting sample data...")

    try:
        ohlcv_count = await insert_sample_ohlcv()
        print(f"  ✅ OHLCV: {ohlcv_count} candles")
    except Exception as e:
        print(f"  ❌ OHLCV: {e}")

    try:
        trades_count = await insert_sample_trades()
        print(f"  ✅ Trades: {trades_count} records")
    except Exception as e:
        print(f"  ❌ Trades: {e}")

    try:
        perf_count = await insert_sample_performance()
        print(f"  ✅ Performance: {perf_count} snapshots")
    except Exception as e:
        print(f"  ❌ Performance: {e}")

    try:
        signals_count = await insert_sample_signals()
        print(f"  ✅ Signals: {signals_count} records")
    except Exception as e:
        print(f"  ❌ Signals: {e}")

    print("\n" + "=" * 60)
    print("✅ Supabase setup complete!")
    print("You can now view the data in your Supabase dashboard")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
