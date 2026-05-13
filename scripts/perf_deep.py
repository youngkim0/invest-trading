"""Deeper performance dive — by (strategy × symbol), exit-category, and time trend."""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
c = create_client(os.environ["SUPABASE_URL"], os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"])

V84 = "2026-05-06T00:00:00Z"


def fetch_trades(since_iso: str) -> list[dict]:
    r = (
        c.table("trade_logs")
        .select("entry_time,exit_time,symbol,strategy_name,side,net_pnl,exit_reasoning,return_pct,duration_seconds")
        .gte("exit_time", since_iso)
        .not_.is_("exit_time", "null")
        .order("exit_time", desc=True)
        .limit(5000)
        .execute()
    )
    return r.data or []


def exit_category(reason: str | None) -> str:
    if not reason:
        return "(none)"
    r = reason.lower()
    if r.startswith("take profit"):
        return "take_profit"
    if r.startswith("stop loss"):
        return "stop_loss"
    if r.startswith("trailing"):
        return "trailing_stop"
    if r.startswith("reversal"):
        return "reversal_override"
    if r.startswith("stale"):
        return "stale_position"
    if r.startswith("session"):
        return "session_end"
    if r.startswith("max hold") or "max hold" in r:
        return "max_hold"
    return r.split("(")[0].strip()[:25]


def print_pair_table(trades: list[dict], title: str) -> None:
    """Print strategy × symbol matrix."""
    pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in trades:
        pairs[(t.get("strategy_name") or "?", t.get("symbol") or "?")].append(t)
    rows = []
    for (strat, sym), ts in pairs.items():
        n = len(ts)
        wins = sum(1 for t in ts if (t.get("net_pnl") or 0) > 0)
        pnl = sum((t.get("net_pnl") or 0) for t in ts)
        rows.append((strat, sym, n, wins, pnl, wins / n * 100 if n else 0))
    rows.sort(key=lambda r: r[4])  # worst first
    print(f"\n{title}")
    print(f"  {'strategy':<20} {'symbol':<10} {'n':>3} {'W':>3} {'PnL':>9} {'WR%':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*3} {'-'*3} {'-'*9} {'-'*6}")
    for strat, sym, n, w, pnl, wr in rows:
        mark = "✗" if pnl < -10 else ("✓" if pnl > 10 else "·")
        print(f"  {strat:<20} {sym:<10} {n:>3} {w:>3} {pnl:>+9.2f} {wr:>5.1f}% {mark}")


def print_exit_summary(trades: list[dict]) -> None:
    cats: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        cats[exit_category(t.get("exit_reasoning"))].append(t)
    print(f"\nExit category summary")
    print(f"  {'category':<22} {'n':>4} {'W':>4} {'PnL':>9} {'WR%':>6} {'avg':>7}")
    print(f"  {'-'*22} {'-'*4} {'-'*4} {'-'*9} {'-'*6} {'-'*7}")
    rows = []
    for cat, ts in cats.items():
        n = len(ts)
        wins = sum(1 for t in ts if (t.get("net_pnl") or 0) > 0)
        pnl = sum((t.get("net_pnl") or 0) for t in ts)
        rows.append((cat, n, wins, pnl, wins / n * 100 if n else 0, pnl / n if n else 0))
    rows.sort(key=lambda r: r[3], reverse=True)
    for cat, n, w, pnl, wr, avg in rows:
        print(f"  {cat:<22} {n:>4} {w:>4} {pnl:>+9.2f} {wr:>5.1f}% {avg:>+7.2f}")


def print_daily_trend(trades: list[dict]) -> None:
    """Daily PnL since v8.4."""
    days: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        d = t.get("exit_time", "")[:10]
        days[d].append(t)
    print(f"\nDaily PnL (since v8.4)")
    print(f"  {'date':<12} {'n':>4} {'W':>4} {'PnL':>9} {'WR%':>6}  cum")
    print(f"  {'-'*12} {'-'*4} {'-'*4} {'-'*9} {'-'*6}  ---")
    cum = 0.0
    for d in sorted(days):
        ts = days[d]
        n = len(ts)
        wins = sum(1 for t in ts if (t.get("net_pnl") or 0) > 0)
        pnl = sum((t.get("net_pnl") or 0) for t in ts)
        cum += pnl
        wr = wins / n * 100 if n else 0
        bar = "█" * min(20, abs(int(pnl / 5)))
        sign = "+" if pnl >= 0 else "-"
        print(f"  {d:<12} {n:>4} {wins:>4} {pnl:>+9.2f} {wr:>5.1f}%  {cum:>+8.2f}  {sign}{bar}")


def main() -> None:
    print("=" * 78)
    print(f"DEEP DIVE — since v8.4 ({V84})")
    print("=" * 78)
    trades = fetch_trades(V84)
    print(f"closed trades: {len(trades)}, total net_pnl: ${sum((t.get('net_pnl') or 0) for t in trades):+.2f}")

    print_pair_table(trades, "Strategy × Symbol (sorted worst → best)")
    print_exit_summary(trades)
    print_daily_trend(trades)


if __name__ == "__main__":
    main()
