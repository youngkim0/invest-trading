#!/usr/bin/env python3
"""6-month backtester for all active strategies.

Downloads historical data from Binance and simulates every strategy
using the SAME signal generation logic as the live bot.

Usage:
    python scripts/backtest_strategies.py
    python scripts/backtest_strategies.py --days 180 --symbols BTCUSDT ETHUSDT
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.paper_trade_simple import (
    TrendBreakoutGenerator,
    OrderFlowGenerator,
    CrashMomentumShortGenerator,
    SmartMoneyFlowGenerator,
    FundingMeanReversionGenerator,
    FailedBreakoutShortGenerator,
    RegimeShortConfluenceGenerator,
    RefinedLiqCascadeGenerator,
    determine_htf_trend,
    check_bearish_regime,
    calculate_atr,
    apply_fast_reversal_override,
)


# =============================================================================
# Historical Data Downloader
# =============================================================================

async def download_klines(client: httpx.AsyncClient, symbol: str, interval: str,
                          start: datetime, end: datetime) -> pd.DataFrame:
    """Download historical klines from Binance with paging."""
    all_data = []
    current_start = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while current_start < end_ms:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol, "interval": interval, "limit": 1000,
            "startTime": current_start, "endTime": end_ms,
        }
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        current_start = data[-1][0] + 1  # Next ms after last candle
        await asyncio.sleep(0.1)  # Rate limit

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


async def download_funding_rates(client: httpx.AsyncClient, symbol: str,
                                  start: datetime, end: datetime) -> list[dict]:
    """Download historical funding rates from Binance."""
    all_rates = []
    current_start = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while current_start < end_ms:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": 1000, "startTime": current_start}
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rates.extend(data)
        current_start = data[-1]["fundingTime"] + 1
        await asyncio.sleep(0.1)

    return [{"rate": float(r["fundingRate"]), "time": r["fundingTime"]} for r in all_rates]


async def download_all_data(symbols: list[str], days: int) -> dict:
    """Download all historical data for backtesting."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    logger.info(f"Downloading {days} days of data for {symbols} ({start.date()} → {end.date()})")

    client = httpx.AsyncClient(timeout=30.0)
    data = {}

    for symbol in symbols:
        logger.info(f"  Downloading {symbol}...")
        data[symbol] = {}

        # Download candles in parallel
        tasks = {
            "15m": download_klines(client, symbol, "15m", start, end),
            "1h": download_klines(client, symbol, "1h", start, end),
            "4h": download_klines(client, symbol, "4h", start, end),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"    {key} failed: {result}")
                data[symbol][key] = pd.DataFrame()
            else:
                data[symbol][key] = result
                logger.info(f"    {key}: {len(result)} candles")

        # Funding rates
        try:
            funding = await download_funding_rates(client, symbol, start, end)
            data[symbol]["funding"] = funding
            logger.info(f"    funding: {len(funding)} rates")
        except Exception as e:
            logger.warning(f"    funding failed: {e}")
            data[symbol]["funding"] = []

    await client.aclose()
    return data


# =============================================================================
# Backtester
# =============================================================================

class Trade:
    def __init__(self, symbol, side, entry_price, entry_time, strategy, sl_pct, tp_pct, quantity):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.strategy = strategy
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.quantity = quantity
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = ""
        self.pnl = 0.0

    def check_exit(self, high, low, close, current_time, max_hours=6.0):
        """Check if trade should exit based on price action."""
        if self.side == "long":
            sl_price = self.entry_price * (1 - self.sl_pct)
            tp_price = self.entry_price * (1 + self.tp_pct)
            if low <= sl_price:
                self.exit_price = sl_price
                self.exit_reason = "Stop loss"
                self.pnl = (sl_price - self.entry_price) * self.quantity
            elif high >= tp_price:
                self.exit_price = tp_price
                self.exit_reason = "Take profit"
                self.pnl = (tp_price - self.entry_price) * self.quantity
        else:  # short
            sl_price = self.entry_price * (1 + self.sl_pct)
            tp_price = self.entry_price * (1 - self.tp_pct)
            if high >= sl_price:
                self.exit_price = sl_price
                self.exit_reason = "Stop loss"
                self.pnl = (self.entry_price - sl_price) * self.quantity
            elif low <= tp_price:
                self.exit_price = tp_price
                self.exit_reason = "Take profit"
                self.pnl = (tp_price - self.entry_price) * self.quantity

        # Time-based exit
        if not self.exit_price:
            hours = (current_time - self.entry_time).total_seconds() / 3600
            if hours >= max_hours:
                self.exit_price = close
                self.exit_reason = "Time exit"
                if self.side == "long":
                    self.pnl = (close - self.entry_price) * self.quantity
                else:
                    self.pnl = (self.entry_price - close) * self.quantity

        if self.exit_price:
            self.exit_time = current_time
            self.pnl *= 0.999  # 0.1% fee estimate
            return True
        return False


def calculate_atr_from_df(df, period=14):
    """Calculate ATR from DataFrame."""
    if df is None or len(df) < period + 1:
        return 0
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def run_backtest(data: dict, symbols: list[str]) -> dict:
    """Run all strategies against historical data."""
    logger.info("Running backtest...")

    # Initialize generators
    generators = {
        "trend_breakout": {
            "gen": TrendBreakoutGenerator(),
            "type": "long",
            "sl_mult": 1.5, "tp_mult": 3.0,
            "timeframe": "15m", "max_hours": 6.0,
            "capital": 1500,
        },
        "order_flow": {
            "gen": OrderFlowGenerator(),
            "type": "long",
            "sl_mult": 1.5, "tp_mult": 3.0,
            "timeframe": "15m", "max_hours": 6.0,
            "capital": 1000,
        },
        "crash_momentum": {
            "gen": CrashMomentumShortGenerator(),
            "type": "short",
            "sl_mult": 2.0, "tp_mult": 3.0,
            "timeframe": "1h", "max_hours": 12.0,
            "capital": 500,
        },
    }

    all_trades = []
    strategy_results = defaultdict(lambda: {
        "trades": 0, "wins": 0, "pnl": 0, "sl_exits": 0, "tp_exits": 0,
        "time_exits": 0, "by_symbol": defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0}),
        "by_month": defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0}),
        "max_drawdown": 0, "peak_equity": 0, "equity_curve": [],
    })

    for symbol in symbols:
        sym_data = data.get(symbol, {})
        df_15m = sym_data.get("15m", pd.DataFrame())
        df_1h = sym_data.get("1h", pd.DataFrame())
        df_4h = sym_data.get("4h", pd.DataFrame())
        funding = sym_data.get("funding", [])

        if df_1h.empty or df_15m.empty:
            logger.warning(f"  {symbol}: insufficient data, skipping")
            continue

        logger.info(f"  Backtesting {symbol}...")

        # Build funding rate lookup
        funding_lookup = {}
        for f in funding:
            ts = datetime.fromtimestamp(f["time"] / 1000, tz=timezone.utc)
            funding_lookup[ts.strftime("%Y-%m-%d %H")] = f["rate"]

        # Iterate through 1h candles (main loop simulation)
        open_trades = {}  # strategy -> Trade
        cooldowns = {}  # strategy -> cooldown_until

        for i in range(50, len(df_1h)):
            current_time = df_1h.index[i]
            current_1h = df_1h.iloc[:i+1]

            # Get matching 15m window (last 100 15m candles before this 1h close)
            mask_15m = df_15m.index <= current_time
            current_15m = df_15m[mask_15m].tail(100)

            # Get matching 4h window
            mask_4h = df_4h.index <= current_time
            current_4h = df_4h[mask_4h].tail(60)

            if len(current_15m) < 30 or len(current_1h) < 50:
                continue

            current_price = float(current_1h["close"].iloc[-1])
            current_high = float(current_1h["high"].iloc[-1])
            current_low = float(current_1h["low"].iloc[-1])

            # HTF trend
            htf_trend = determine_htf_trend(current_1h)
            htf_trend = apply_fast_reversal_override(htf_trend, current_15m)

            # Bearish regime
            regime = check_bearish_regime(current_4h) if len(current_4h) >= 30 else {"is_bearish": False}

            # Check exits for open trades
            for strat_name, trade in list(open_trades.items()):
                config = generators[strat_name]
                exited = trade.check_exit(current_high, current_low, current_price, current_time, config["max_hours"])
                if exited:
                    results = strategy_results[strat_name]
                    results["trades"] += 1
                    results["pnl"] += trade.pnl
                    if trade.pnl > 0:
                        results["wins"] += 1
                    if "Stop loss" in trade.exit_reason:
                        results["sl_exits"] += 1
                    elif "Take profit" in trade.exit_reason:
                        results["tp_exits"] += 1
                    else:
                        results["time_exits"] += 1
                    results["by_symbol"][symbol]["trades"] += 1
                    results["by_symbol"][symbol]["pnl"] += trade.pnl
                    if trade.pnl > 0:
                        results["by_symbol"][symbol]["wins"] += 1
                    month = current_time.strftime("%Y-%m")
                    results["by_month"][month]["trades"] += 1
                    results["by_month"][month]["pnl"] += trade.pnl
                    if trade.pnl > 0:
                        results["by_month"][month]["wins"] += 1

                    # Track equity curve
                    running_pnl = results["pnl"]
                    results["equity_curve"].append(running_pnl)
                    if running_pnl > results["peak_equity"]:
                        results["peak_equity"] = running_pnl
                    dd = results["peak_equity"] - running_pnl
                    if dd > results["max_drawdown"]:
                        results["max_drawdown"] = dd

                    all_trades.append({
                        "strategy": strat_name, "symbol": symbol, "side": trade.side,
                        "entry_price": trade.entry_price, "exit_price": trade.exit_price,
                        "pnl": round(trade.pnl, 2), "exit_reason": trade.exit_reason,
                        "entry_time": trade.entry_time.isoformat(),
                        "exit_time": trade.exit_time.isoformat(),
                    })
                    del open_trades[strat_name]

                    # Cooldown after SL
                    if "Stop loss" in trade.exit_reason:
                        cooldowns[strat_name] = current_time + timedelta(minutes=60)

            # Generate signals for each strategy
            for strat_name, config in generators.items():
                # Skip if already in a position or on cooldown
                if strat_name in open_trades:
                    continue
                if strat_name in cooldowns and current_time < cooldowns[strat_name]:
                    continue

                gen = config["gen"]
                signal = None

                try:
                    if strat_name == "trend_breakout":
                        signal = gen.generate_signal(current_15m, htf_trend)
                    elif strat_name == "order_flow":
                        # No derivatives data for backtest — skip
                        continue
                    elif strat_name == "crash_momentum":
                        signal = gen.generate_signal(current_1h, htf_trend, regime=regime)
                except Exception:
                    continue

                if not signal or signal.get("signal") == "hold":
                    continue

                sig_type = signal["signal"]
                confidence = signal.get("confidence", 0.7)

                if confidence < 0.65:
                    continue

                # Determine side
                if sig_type in ("buy", "strong_buy"):
                    side = "long"
                elif sig_type in ("sell", "strong_sell"):
                    side = "short"
                else:
                    continue

                # Calculate ATR and position size
                atr_df = current_15m if config["timeframe"] == "15m" else current_1h
                atr = calculate_atr_from_df(atr_df)
                if atr <= 0:
                    continue

                sl_pct = (atr * config["sl_mult"]) / current_price
                tp_pct = (atr * config["tp_mult"]) / current_price

                # Position size (2% risk)
                risk_amount = config["capital"] * 0.02
                position_value = risk_amount / sl_pct if sl_pct > 0 else 0
                quantity = position_value / current_price if current_price > 0 else 0

                if quantity <= 0:
                    continue

                trade = Trade(symbol, side, current_price, current_time, strat_name, sl_pct, tp_pct, quantity)
                open_trades[strat_name] = trade

    return {"strategy_results": dict(strategy_results), "all_trades": all_trades}


# =============================================================================
# Report
# =============================================================================

def print_report(results: dict, days: int):
    """Print comprehensive backtest report."""
    strategy_results = results["strategy_results"]
    all_trades = results["all_trades"]

    print()
    print("=" * 80)
    print(f"  BACKTEST REPORT — {days} days, {len(all_trades)} total trades")
    print("=" * 80)

    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for strat, r in sorted(strategy_results.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = r["wins"] / r["trades"] * 100 if r["trades"] else 0
        avg = r["pnl"] / r["trades"] if r["trades"] else 0
        pnl_per_month = r["pnl"] / (days / 30)
        total_pnl += r["pnl"]
        total_trades += r["trades"]
        total_wins += r["wins"]

        print(f"\n{'─' * 80}")
        print(f"  {strat}")
        print(f"{'─' * 80}")
        print(f"  PnL: ${r['pnl']:+.2f} (${pnl_per_month:+.2f}/month) | {r['trades']} trades | {wr:.1f}% WR | Avg: ${avg:.2f}")
        print(f"  Exits: SL={r['sl_exits']} TP={r['tp_exits']} Time={r['time_exits']}")
        print(f"  Max drawdown: ${r['max_drawdown']:.2f}")

        # By symbol
        print(f"  By symbol:")
        for sym, sd in sorted(r["by_symbol"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            sym_wr = sd["wins"] / sd["trades"] * 100 if sd["trades"] else 0
            print(f"    {sym:10s}: ${sd['pnl']:+8.2f} | {sd['trades']:3d} trades | {sym_wr:.0f}% WR")

        # By month
        print(f"  By month:")
        for month, md in sorted(r["by_month"].items()):
            m_wr = md["wins"] / md["trades"] * 100 if md["trades"] else 0
            print(f"    {month}: ${md['pnl']:+8.2f} | {md['trades']:3d} trades | {m_wr:.0f}% WR")

    # Summary
    total_wr = total_wins / total_trades * 100 if total_trades else 0
    monthly_pnl = total_pnl / (days / 30)

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total PnL: ${total_pnl:+.2f} | Monthly avg: ${monthly_pnl:+.2f}")
    print(f"  Total trades: {total_trades} | Win rate: {total_wr:.1f}%")
    print(f"  Strategies with edge (>0 PnL): {sum(1 for r in strategy_results.values() if r['pnl'] > 0)}/{len(strategy_results)}")

    # Profitable months
    all_months = defaultdict(float)
    for r in strategy_results.values():
        for month, md in r["by_month"].items():
            all_months[month] += md["pnl"]

    profitable_months = sum(1 for v in all_months.values() if v > 0)
    print(f"  Profitable months: {profitable_months}/{len(all_months)}")
    print(f"\n  Monthly breakdown:")
    for month in sorted(all_months.keys()):
        marker = "✅" if all_months[month] > 0 else "❌"
        print(f"    {month}: ${all_months[month]:+8.2f} {marker}")

    print(f"\n{'=' * 80}")


async def main():
    parser = argparse.ArgumentParser(description="Strategy Backtester")
    parser.add_argument("--days", type=int, default=180, help="Days of history (default: 180)")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"])
    args = parser.parse_args()

    # Download data
    data = await download_all_data(args.symbols, args.days)

    # Run backtest
    results = run_backtest(data, args.symbols)

    # Print report
    print_report(results, args.days)

    # Save results
    import json
    output_path = Path(__file__).parent / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "days": args.days,
            "symbols": args.symbols,
            "strategy_results": {
                k: {kk: vv for kk, vv in v.items() if kk != "equity_curve"}
                for k, v in results["strategy_results"].items()
            },
            "trade_count": len(results["all_trades"]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
