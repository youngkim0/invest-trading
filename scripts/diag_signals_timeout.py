"""Diagnose dashboard signals fetch timeout.

Runs the exact query the dashboard runs, plus a few variants, and times each.
Goal: confirm whether idx_signals_timestamp is doing its job.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

URL = os.environ["SUPABASE_URL"]
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"]
NEW_SYSTEM_DATE = "2026-04-26T12:00:00Z"

client = create_client(URL, KEY)


def time_query(label: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        rows = fn()
        dt = time.perf_counter() - t0
        n = len(rows.data) if hasattr(rows, "data") else len(rows)
        print(f"  ok    {label:55s}  {dt*1000:7.0f} ms  ({n} rows)")
    except Exception as e:
        dt = time.perf_counter() - t0
        msg = str(e)[:120]
        print(f"  FAIL  {label:55s}  {dt*1000:7.0f} ms  -> {msg}")


def main() -> None:
    print("Diagnosing signals fetch timeout")
    print(f"  URL    = {URL}")
    print(f"  filter = timestamp >= {NEW_SYSTEM_DATE}")
    print()

    print("Reproducing dashboard query (limit 200, with timestamp filter):")
    time_query(
        "select 7 cols, gte filter, order desc, limit 200",
        lambda: client.table("signals")
        .select("id,timestamp,symbol,source,signal_type,confidence,reasoning")
        .gte("timestamp", NEW_SYSTEM_DATE)
        .order("timestamp", desc=True)
        .limit(200)
        .execute(),
    )

    print()
    print("Variants to localize the cost:")
    time_query(
        "limit 200 WITHOUT timestamp filter",
        lambda: client.table("signals")
        .select("id,timestamp,symbol,source,signal_type,confidence,reasoning")
        .order("timestamp", desc=True)
        .limit(200)
        .execute(),
    )
    time_query(
        "limit 200 WITHOUT reasoning column",
        lambda: client.table("signals")
        .select("id,timestamp,symbol,source,signal_type,confidence")
        .gte("timestamp", NEW_SYSTEM_DATE)
        .order("timestamp", desc=True)
        .limit(200)
        .execute(),
    )
    time_query(
        "limit 50 with full select",
        lambda: client.table("signals")
        .select("id,timestamp,symbol,source,signal_type,confidence,reasoning")
        .gte("timestamp", NEW_SYSTEM_DATE)
        .order("timestamp", desc=True)
        .limit(50)
        .execute(),
    )
    time_query(
        "count(*) of signals",
        lambda: client.table("signals").select("id", count="exact").limit(1).execute(),
    )


if __name__ == "__main__":
    main()
