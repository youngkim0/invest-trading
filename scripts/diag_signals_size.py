"""Diagnose what's filling the Supabase signals table.

Counts total rows, date range, and breakdowns by signal_type / source / date
so we can pick a precise retention rule.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
URL = os.environ["SUPABASE_URL"]
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"]
c = create_client(URL, KEY)


def count(query_desc: str, **filters) -> int:
    q = c.table("signals").select("id", count="exact", head=True)
    for k, v in filters.items():
        op, val = v
        q = getattr(q, op)(k, val)
    t0 = time.perf_counter()
    r = q.execute()
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {query_desc:60s} {r.count:>10,} rows  ({dt:.0f} ms)")
    return r.count


def sample_oldest_newest() -> None:
    newest = (
        c.table("signals").select("timestamp").order("timestamp", desc=True).limit(1).execute()
    )
    print(f"  newest timestamp: {newest.data[0]['timestamp'] if newest.data else 'n/a'}")
    # Find oldest via binary-search on count to avoid ASC scan timeout.
    now = datetime.now(timezone.utc)
    # find smallest D such that count(timestamp >= now - D days) ~= total
    for days in (365, 180, 90, 60, 45, 30, 21, 14, 10, 7, 5, 3, 2, 1):
        cutoff = (now - timedelta(days=days)).isoformat()
        r = c.table("signals").select("id", count="exact", head=True).gte("timestamp", cutoff).execute()
        print(f"  rows newer than {days:>3}d: {r.count:>10,}")


def breakdown_recent_sources(days: int = 1) -> None:
    """Sample recent rows to estimate source / signal_type distribution."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    r = (
        c.table("signals")
        .select("source,signal_type")
        .gte("timestamp", cutoff)
        .limit(50000)
        .execute()
    )
    rows = r.data or []
    print(f"  sample size (last {days}d): {len(rows):,}")
    src = Counter(x.get("source") for x in rows)
    sig = Counter(x.get("signal_type") for x in rows)
    print(f"  by source (top 10):")
    for k, v in src.most_common(10):
        print(f"     {str(k):30s} {v:>7,}  ({v/max(1,len(rows))*100:.1f}%)")
    print(f"  by signal_type:")
    for k, v in sig.most_common():
        print(f"     {str(k):30s} {v:>7,}  ({v/max(1,len(rows))*100:.1f}%)")


def main() -> None:
    print("=== signals table diagnosis ===\n")

    print("Totals:")
    total = count("total rows")
    sample_oldest_newest()
    print()

    print("Last-1-day breakdown by source/signal_type:")
    breakdown_recent_sources(days=1)


if __name__ == "__main__":
    main()
