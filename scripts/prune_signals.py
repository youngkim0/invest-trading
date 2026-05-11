"""Prune old `hold` signals from the Supabase signals table.

Deletes rows matching `signal_type = 'hold' AND timestamp < now() - retention_days`
in ID batches small enough to stay under Supabase's statement timeout.

Actionable signals (buy/sell/strong_buy/strong_sell) are never touched.

Usage:
    python scripts/prune_signals.py --dry-run            # show counts only
    python scripts/prune_signals.py --retention-days 7   # actually delete
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

URL = os.environ["SUPABASE_URL"]
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"]
client = create_client(URL, KEY)

WINDOW_HOURS = 6   # delete in 6-hour chunks (≈2K hold rows each at current rate)
LOOKBACK_DAYS = 120  # don't walk back further than this from the cutoff


def count_targets(cutoff_iso: str) -> int | None:
    """Best-effort count. Uses planner estimate so it works on large tables."""
    try:
        r = (
            client.table("signals")
            .select("id", count="planned", head=True)
            .eq("signal_type", "hold")
            .lt("timestamp", cutoff_iso)
            .execute()
        )
        return r.count
    except Exception as e:
        print(f"  (count unavailable: {str(e)[:80]})")
        return None


def delete_window(start_iso: str, end_iso: str) -> None:
    """DELETE WHERE signal_type='hold' AND start <= timestamp < end. Sends
    Prefer: return=minimal so the response body stays small."""
    (
        client.table("signals")
        .delete(returning="minimal")
        .eq("signal_type", "hold")
        .gte("timestamp", start_iso)
        .lt("timestamp", end_iso)
        .execute()
    )


def count_window(start_iso: str, end_iso: str) -> int:
    """Exact count for a small window (fast because both ends bounded)."""
    r = (
        client.table("signals")
        .select("id", count="exact", head=True)
        .eq("signal_type", "hold")
        .gte("timestamp", start_iso)
        .lt("timestamp", end_iso)
        .execute()
    )
    return r.count or 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retention-days", type=int, default=7,
                    help="Keep hold signals newer than this many days (default 7)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print the count; don't delete")
    ap.add_argument("--max-batches", type=int, default=1000,
                    help="Safety cap on iterations (default 1000)")
    args = ap.parse_args()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.retention_days)).isoformat()
    print(f"Cutoff: keep hold signals with timestamp >= {cutoff}")
    target_count = count_targets(cutoff)
    if target_count is not None:
        print(f"Planner estimate of rows to delete: {target_count:,}")

    if args.dry_run:
        print("(dry run — exiting)")
        return

    # Walk backward from cutoff in WINDOW_HOURS chunks. The newest "old hold"
    # row is at most `cutoff`; the oldest is bounded by LOOKBACK_DAYS.
    now = datetime.now(timezone.utc)
    cutoff_dt = now - timedelta(days=args.retention_days)
    horizon_dt = now - timedelta(days=LOOKBACK_DAYS)

    end_dt = cutoff_dt
    deleted = 0
    empty_streak = 0
    t_start = time.perf_counter()
    i = 0
    while end_dt > horizon_dt and i < args.max_batches:
        i += 1
        start_dt = end_dt - timedelta(hours=WINDOW_HOURS)
        start_iso, end_iso = start_dt.isoformat(), end_dt.isoformat()
        t0 = time.perf_counter()
        try:
            n_before = count_window(start_iso, end_iso)
            if n_before > 0:
                delete_window(start_iso, end_iso)
                deleted += n_before
                empty_streak = 0
                dt = (time.perf_counter() - t0) * 1000
                rate = deleted / max(0.001, time.perf_counter() - t_start)
                print(f"  [{i:>3}] {start_dt:%Y-%m-%d %H:%M}..{end_dt:%H:%M}  "
                      f"deleted {n_before:>5} (total {deleted:>7,}) in {dt:>5.0f} ms — {rate:.0f} rows/s")
            else:
                empty_streak += 1
                if empty_streak >= 8:  # 48h of empty windows => assume done
                    print(f"  [{i:>3}] {start_dt:%Y-%m-%d %H:%M}  no rows for 8 windows — stopping")
                    break
        except Exception as e:
            print(f"  [{i:>3}] {start_dt:%Y-%m-%d %H:%M}  ERROR: {str(e)[:120]}")
            # Retry once with a half-sized window
            mid_dt = start_dt + timedelta(hours=WINDOW_HOURS // 2)
            try:
                delete_window(start_dt.isoformat(), mid_dt.isoformat())
                delete_window(mid_dt.isoformat(), end_iso)
                print("       (recovered with two half-windows)")
            except Exception as e2:
                print(f"       still failing: {str(e2)[:80]}; skipping window")
        end_dt = start_dt

    print(f"\nDone: deleted {deleted:,} rows in {time.perf_counter()-t_start:.1f}s "
          f"across {i} windows")


if __name__ == "__main__":
    main()
