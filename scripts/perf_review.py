"""Performance review across recent windows.

Pulls trade_logs from Supabase and summarizes by strategy/symbol/window.
Windows: all-time, since v8.3 (2026-04-26), since v8.4 (2026-05-06), last 7d, last 24h.
"""
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
URL = os.environ["SUPABASE_URL"]
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"]
c = create_client(URL, KEY)

V83 = "2026-04-26T12:00:00Z"  # v8.3 launch (NEW_SYSTEM_DATE)
V84 = "2026-05-06T00:00:00Z"  # v8.4 rsi_momentum gating + dashboard fix


def fetch_trades(since_iso: str | None) -> list[dict]:
    """Closed trades only (have exit_time)."""
    q = (
        c.table("trade_logs")
        .select("entry_time,exit_time,symbol,strategy_name,side,net_pnl,exit_reasoning,return_pct,duration_seconds,quantity,entry_price,exit_price")
        .order("exit_time", desc=True)
        .limit(5000)
    )
    if since_iso:
        q = q.gte("exit_time", since_iso)
    # Only closed trades
    q = q.not_.is_("exit_time", "null")
    r = q.execute()
    return r.data or []


def summarize(trades: list[dict], group_by: str) -> list[tuple]:
    """Return [(key, n, wins, pnl, win_rate, avg_pnl, avg_win, avg_loss), ...] sorted by pnl desc."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        buckets[t.get(group_by) or "(none)"].append(t)
    rows = []
    for k, ts in buckets.items():
        n = len(ts)
        wins = [t for t in ts if (t.get("net_pnl") or 0) > 0]
        losses = [t for t in ts if (t.get("net_pnl") or 0) <= 0]
        total_pnl = sum((t.get("net_pnl") or 0) for t in ts)
        wr = len(wins) / n * 100 if n else 0
        avg_pnl = total_pnl / n if n else 0
        avg_win = sum((t.get("net_pnl") or 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum((t.get("net_pnl") or 0) for t in losses) / len(losses) if losses else 0
        rows.append((k, n, len(wins), total_pnl, wr, avg_pnl, avg_win, avg_loss))
    return sorted(rows, key=lambda r: r[3], reverse=True)


def print_table(rows: list[tuple], title: str, key_label: str) -> None:
    print(f"\n{title}")
    print(f"  {key_label:<22} {'n':>4} {'W':>4} {'PnL':>9} {'WR%':>6} {'avg':>7} {'avg+':>7} {'avg-':>7}")
    print(f"  {'-'*22} {'-'*4} {'-'*4} {'-'*9} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    total_n = total_w = 0
    total_pnl = 0.0
    for k, n, w, pnl, wr, avg, avgw, avgl in rows:
        marker = "✓" if pnl > 0 else ("·" if pnl == 0 else "✗")
        print(f"  {str(k):<22} {n:>4} {w:>4} {pnl:>+9.2f} {wr:>5.1f}% {avg:>+7.2f} {avgw:>+7.2f} {avgl:>+7.2f} {marker}")
        total_n += n
        total_w += w
        total_pnl += pnl
    if rows:
        wr_total = total_w / total_n * 100 if total_n else 0
        print(f"  {'TOTAL':<22} {total_n:>4} {total_w:>4} {total_pnl:>+9.2f} {wr_total:>5.1f}%")


def window(label: str, since_iso: str | None) -> None:
    print(f"\n{'=' * 78}")
    print(f"WINDOW: {label}  (since {since_iso or 'beginning'})")
    print(f"{'=' * 78}")
    trades = fetch_trades(since_iso)
    if not trades:
        print("  (no closed trades in window)")
        return
    print(f"  closed trades: {len(trades)}")
    if since_iso:
        days = (datetime.now(timezone.utc) - datetime.fromisoformat(since_iso.replace("Z", "+00:00"))).days
        print(f"  span: ~{days} days")
    print_table(summarize(trades, "strategy_name"), "by strategy", "strategy")
    print_table(summarize(trades, "symbol"), "by symbol", "symbol")
    # Side breakdown only if it matters
    sides = summarize(trades, "side")
    if len(sides) > 1:
        print_table(sides, "by side", "side")


def exit_reason_breakdown(since_iso: str) -> None:
    trades = fetch_trades(since_iso)
    if not trades:
        return
    print(f"\n{'=' * 78}")
    print(f"EXIT REASONS (since {since_iso})")
    print(f"{'=' * 78}")
    print_table(summarize(trades, "exit_reasoning"), "by exit_reason", "exit_reasoning")


def main() -> None:
    now = datetime.now(timezone.utc)
    last_7d = (now - timedelta(days=7)).isoformat()
    last_24h = (now - timedelta(hours=24)).isoformat()

    window("since v8.3 launch (Apr 26)", V83)
    window("since v8.4 (May 6)", V84)
    window("last 7d", last_7d)
    window("last 24h", last_24h)
    exit_reason_breakdown(V84)


if __name__ == "__main__":
    main()
