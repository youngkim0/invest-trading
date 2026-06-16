# Project Retrospective — Crypto Paper-Trading Bot

**Status: ARCHIVED / SHUT DOWN — 2026-06-16**
**Final version: v9.0 (`4edf537`, branch `feature/real-trading`)**

This project is being retired. This document is the honest postmortem: what it
was, what it achieved, why it's stopping, and what was learned. It is the
canonical "what happened here" reference for anyone (including future me)
who opens this repo again.

---

## What it was

A crypto paper-trading bot for 6 coins (BTC, ETH, SOL, AVAX, DOGE, XRP) on
Binance USDT-M Futures, 10x leverage, simulating an evidence-based,
multi-strategy book against live market data.

**Infrastructure built (and it works):**
- Live market-data collection (spot + futures) — `data/collectors/market_data.py`
- Paper trading engine with signal generation, cooldowns, circuit breaker —
  `scripts/paper_trade_simple.py`
- ATR-based SL/TP/trailing, Kelly + vol-adjusted position sizing
- Risk controls: daily loss limit, leverage cap, correlation awareness
- Supabase storage (`trade_logs`, `signals`) with retention cleanup
- Streamlit Cloud dashboard — `dashboard/app.py`
- Daily AI performance review via Claude — `scripts/ai_daily_review.py`
- GCP VM deploy pipeline (systemd service, `paper-trader`, asia-northeast3-a)
- A live-trading path (`scripts/live_trade.py` + `core/engine/order_manager.py`,
  CCXT → Binance) built but **never deployed with real money** — correctly so.

---

## The bottom line (why it's shutting down)

**The system never demonstrated a durable edge. It underperformed doing nothing.**

The decisive evidence — the post-v8.0 window, our most "evidence-based" build:

| metric | 5-strategy book (54 days) | BTC buy-and-hold |
|---|---|---|
| return | **−6.4% (−$400)** | **−1.9%** |
| trades | 693 (~12.8/day) | 0 |

We traded heavily, paid the activity cost, and lost more than passively
holding the benchmark. Per-strategy, only **one** of five had a live edge:

| strategy | n | WR | live PnL | verdict |
|---|---|---|---|---|
| **uptrend_pullback** | 102 | 54% | **+$340** | only live winner |
| smart_money | 241 | 39% | −$47 | no edge |
| bb_squeeze | 102 | 29% | −$92 | backtest died live |
| rsi_momentum | 54 | 35% | −$388 | backtest died live |
| funding_reversion | 0 | — | $0 | never triggered |

v9.0 concentrated all capital onto uptrend_pullback (2x) and disabled the rest —
the correct *final experiment*. But even that winner's edge (+$340 over 54 days,
54% WR) is too small a sample to bet real money on, and live trading would be
*worse* than paper once real fees, slippage, and funding drag are paid.

---

## The core lesson (the real takeaway)

**The backtest repeatedly failed to predict live results.**

This is the single most important finding, and it recurred across the entire
history — not once, but as a *pattern*:

| strategy | backtest | live |
|---|---|---|
| rsi_momentum | +$1,302 (64% WR) | −$388 (35% WR) |
| bb_squeeze | +$1,358 (82% WR) | −$92 (29% WR) |
| crash_momentum | (looked great) | −$5,871 — killed v8.0 |
| trend_breakout | (looked great) | −$1,451 — killed v8.0 |

Every strategy was added on a beautiful backtest and removed after live losses.
When the measurement tool you use to make decisions is consistently wrong, every
subsequent "improvement" is a coin flip dressed up as analysis. **Markets are
efficient enough that naive backtested edges do not survive contact with
reality** — especially in liquid crypto majors, especially net of costs,
especially on the short side (shorts were structurally broken and abandoned in
v8.0).

Secondary lessons worth keeping:
- **Over-optimization destroys edges** (v6.9.3): stacking filters onto a working
  strategy degrades it. Be conservative adding gates.
- **Exit R:R matters as much as entry** (v8.5): tight trailing stops bank
  winners early while full-width stops run to loss — silently inverting R:R.
- **Sizing pins are invisible bugs**: a vol-adjust multiplier reading 1h ATR
  pinned nominal 2% risk at a flat 3% on every trade.
- **Concentration risk is real** once correlation-group limits are removed
  (one day stacked 9 SOL trades into a single loss).

---

## Version history (condensed)

- **v9.0** (2026-06-03): Concentrated to uptrend_pullback (2x) + funding; disabled the 3 bleeders.
- **v8.5** (2026-05-27): rsi_momentum exit fix (trailing off, wider stop).
- **v8.4.x** (2026-05): Symbol gating; Supabase retention cleanup; dashboard perf.
- **v8.3** (2026-04-26): Profitability overhaul — tighter SL/TP, trailing on longs.
- **v8.0** (2026-04-09): Evidence-based rebuild from 6mo backtest. Killed
  crash_momentum, trend_breakout, and all shorts. uptrend_pullback the lone keeper.
- **v6.x–v7.x**: Portfolio risk controls, Kelly sizing, correlation groups (later removed). See `CHANGELOG.md`.

---

## What was genuinely worth it

Judged as a **learning and engineering project, it succeeded.** It produced a
real, end-to-end live-data → strategy → execution → storage → dashboard → review
pipeline, a clean GCP deploy, and — most valuably — a hard-won, first-hand
understanding of why retail algorithmic trading is hard: the backtest lies, and
beating buy-and-hold net of costs is genuinely difficult. That lesson is worth
more than the −$400 of paper losses it cost to learn.

Judged as a **profit engine, it did not work**, and after ~9 major iterations
the honest move is to stop rather than iterate further hoping the next tweak is
the one. The history says that path leads back to here.

---

## Shutdown actions taken (2026-06-16)

1. GCP systemd service `paper-trader` **stopped and disabled** (no restart on reboot).
2. No more writes to Supabase → DB overflow halted at the source.
3. This retrospective committed; `CHANGELOG.md` closed with a final entry.
4. Project memory updated to mark the project ARCHIVED.

The code remains intact and runnable. Nothing is deleted. If revived, the only
honest framing is option #2 from the shutdown discussion: run uptrend_pullback
alone, hands-off, for a fixed window, and judge it against BTC buy-and-hold with
zero mid-flight tinkering. Anything else repeats the pattern documented above.
