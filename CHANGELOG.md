# Paper Trader Changelog

## v8.3 — Profitability Overhaul: Execution Parameters Fix (2026-04-26)

**Root cause analysis**: Signals are profitable but execution parameters destroyed the edge.
883 trades, -$291 total. Active strategies alone: +$100 on 137 trades. The SL/TP ratio
demanded 60-73% win rates but strategies only deliver 35-50%. Trailing stops were disabled
on all long strategies, so winners reversed to losers.

Research: Freqtrade/Jesse communities, AdaptiveTrend paper (Sharpe 2.41), Phemex Q1 2026
bot results, overfitting studies. 73% of automated crypto traders lose money in 6 months.
Realistic target on $6,250: $50-150/month (10-30% annually).

### Changes

| Change | Before | After | Why |
|--------|--------|-------|-----|
| uptrend_pullback SL/TP | 1.5x / 2.5x ATR | 1.0x / 1.5x | BE WR 60%->40% (actual: 50%) |
| rsi_momentum SL/TP | 2.0x / 2.5x ATR | 1.5x / 2.0x | BE WR 44%->43% (actual: 64%) |
| smart_money SL/TP | 2.0x / 3.0x ATR | 1.5x / 2.0x | Tighter R:R for 40% WR |
| funding SL/TP | 2.0x / 4.0x ATR | 1.5x / 2.5x | Was unreachable |
| bb_squeeze SL/TP | 2.0x / 3.0x ATR | kept | 82% WR supports it |
| Trailing stops | DISABLED on all longs | ENABLED, 0.75x ATR activation | Winners no longer reverse to SL |
| Trailing distance | 1.0-1.5x ATR | 0.5x ATR | Tight trail locks in more profit |
| Daily loss limit | 3% | 4% | Was too tight, missed recoveries |
| Stale exit | Only at max hold time | Also at 50% hold if <0.3% profit | Frees capital from dead trades |
| Regime detection | None | ADX + EMA stacking (observe-only) | Logs bull/bear/sideways per symbol |
| 4h data fetch | Conditional | Always | Required for regime detection |

### v8.2 — Remove Joovier Scalp (2026-04-26)
- Removed Joovier Superscalp: -$192 on 137 trades in 6 days (overfitted backtest)
- Reverted 15s loop back to 60s, removed smart data fetching machinery
- Reallocated $2K capital to proven strategies

---

## v8.0 — Evidence-Based Rebuild: Backtest-Proven Strategies Only (2026-04-09)

**6-month backtest (Oct 2025 - Apr 2026, 1,199 trades) revealed the truth:**
- trend_breakout: -$1,451/6mo, 30.7% WR — proven loser across ALL parameter combos
- crash_momentum: -$5,871/6mo, 22.8% WR — structurally broken (shorting crypto long bias)
- All short strategies: massive losers over 6 months
- Only ONE strategy was profitable: **Pullback to SMA20 + volume confirmation**

### New strategy: UptrendPullbackGenerator (PROVEN)
- **Backtest: +$439/6mo, 44.7% WR, 215 trades, $243 max drawdown**
- Buys pullbacks to 1h SMA20 during confirmed uptrends (SMA20 > SMA50)
- Requires volume > 1.5x average (institutional buying the dip)
- Best coins: ETH (+$239), AVAX (+$169), XRP (+$118)
- SL 1.5x ATR (tight, below SMA20), TP 2.5x ATR

### Active strategies (4 total)
| Strategy | Capital | Edge | Evidence |
|----------|---------|------|----------|
| uptrend_pullback | $2,000 | PROVEN | 6mo backtest +$439, 44.7% WR |
| order_flow | $1,500 | Promising | +$61/30d live (can't backtest - needs derivatives) |
| smart_money | $1,250 | Promising | +$6/30d live (can't backtest - needs derivatives) |
| funding_reversion | $1,000 | Rare event | Idle capital shared via pool |

### Disabled strategies (5 removed)
- trend_breakout: -$1,451/6mo backtest
- crash_momentum: -$5,871/6mo backtest
- failed_breakout_short, regime_short, refined_liq_cascade: 0 trades or net losers

### Infrastructure retained
- Kelly quarter-sizing from rolling 30 trades
- Volatility-adjusted position sizing
- 1 position per strategy, daily loss limit 3%, portfolio heat 12%
- AI regime detection (observe-only), market data snapshots, post-trade analysis

---

## v7.0.2 — Full AI Trading Intelligence Suite (2026-04-06)

Six AI features that make the system learn and adapt from historical data:

| # | Feature | Frequency | Model | What it does |
|---|---------|-----------|-------|-------------|
| 1 | Pattern Learning | Daily | Sonnet | Analyzes 30d trades → per-strategy rules (avoid symbols, min thresholds) |
| 2+5 | Regime Detection | Hourly | Haiku | BTC multi-signal analysis → regime/bias/risk classification |
| 3 | Exit Optimization | Per-position | Haiku | AI decides: hold / tighten stop / close now |
| 4 | Parameter Tuning | Daily | Sonnet | Recommends SL/TP ATR multiplier changes from exit patterns |
| 6 | Symbol Selection | 12h | Haiku | Identifies which symbols to trade/avoid based on 7d WR |

- Learned rules from #1 applied as dynamic filters in entry logic
- Regime bias from #2 blocks counter-regime trades when confidence > 70%
- Exit optimization #3 added as step 7 in exit priority chain
- Parameter tuning #4 auto-adjusts SL/TP within safe ranges (SL 1-3x, TP 1.5-5x)
- Symbol avoidance from #6 blocks entry on poorly-performing coins
- Total cost: ~$4.20/month (1.5% of target monthly profit)

---

## v7.0.1 — Capital efficiency overhaul + relaxed portfolio limits (2026-04-06)

**Problem**: 79% of capital was idle. Three strategies (funding_reversion, regime_short, refined_liq_cascade) had ZERO trades in 30 days = $1,300 dead capital. Meanwhile trend_breakout (#1 earner at $0.83/hr) was capped at $1,250.

### Capital reallocation (follow the PnL/hr)
- **trend_breakout $1,250 → $2,500**: #1 earner, $0.83/hr in market, 38% utilization. Gets 45% of capital.
- **order_flow $500 → $1,000**: #2 earner, $0.54/hr in market.
- **smart_money $500 → $800**: Decent edge, was undercapitalized at 2.5% utilization.
- **Dead strategies $1,300 → $600**: funding_reversion/regime_short/refined_liq_cascade — 0 trades each, reduced to $200 minimum.
- **Losers cut**: failed_breakout_short $400→$300, crash_momentum $400→$300.
- Top 2 earners now get 64% of capital (was 40%).

### Portfolio limits relaxed (3-long cap killed best week)
- **Max longs 3 → 6**: Best week had 8 concurrent longs. Cap at 3 blocked $248 of profit.
- **Max shorts 3 → 4**: Shorts fire less frequently.
- **Correlation group 2 → 3**: Allow full group in strong trends.
- **Portfolio heat 8% → 12%**: 6 positions × 2% Kelly risk each = 12% needed.

### Projected impact
- trend_breakout at $2,500 × $0.83/hr × 38% utilization × 720h = ~$227-$454/month
- Plus order_flow and smart_money contributions
- Target: 5-10% monthly ($275-$550 on $5,500 total)

---

## v7.0 — Portfolio Risk Controls + Kelly Sizing (2026-04-06)

**Problem**: Bot had zero portfolio-level risk management. 7 strategies × 6 correlated coins = up to 42 concurrent positions with no aggregate limit. Position sizing boosted risk on win streaks and strong regimes (exactly when drawdowns compound). 30-day result: +$64 on 494 trades — profitable strategies undermined by uncontrolled risk.

### Portfolio Risk Controls (new)
- **Daily loss limit**: Stop opening new trades after 5% daily drawdown. Prevents catastrophic loss days.
- **Max directional exposure**: Cap at 3 longs and 3 shorts across all strategies. A single market move can no longer trigger 6+ simultaneous losses.
- **Correlation group limits**: BTC/ETH/SOL = large_cap group, XRP/DOGE/AVAX = alt_cap group. Max 2 same-direction positions per group. 6 coins at 80-95% correlation = 2 real bets, not 6.
- **Portfolio heat check**: Total open risk (all positions' SL distances × values) capped at 8% of capital. Prevents over-leveraged portfolio even when individual positions are within limits.

### Position Sizing Overhaul
- **Kelly-based sizing**: Replaced `calculate_adaptive_risk_pct` with `calculate_kelly_risk_pct`. Uses quarter-Kelly from actual rolling 30-trade win rate and avg win/loss ratio. No more arbitrary confidence scaling, win-streak boosts, regime boosts, or multi-strategy boosts.
- **Volatility-adjusted sizing**: Position size scales inversely with ATR. High vol → smaller positions, low vol → larger positions. Target: 1.5% ATR.
- **Max risk ceiling**: Reduced from 3% to 2.5% per trade.
- **Recent results history**: Increased from 20 to 30 trades for more stable Kelly estimates.

### What was removed
- Win streak risk boost (gambler's fallacy)
- Regime alignment risk boost (increases risk when drawdowns compound)
- Multi-strategy agreement risk boost (correlated signals aren't independent)
- Confidence-based linear risk scaling (replaced by Kelly from actual results)

### Expected impact
| Control | Prevention | Estimated benefit |
|---------|-----------|-------------------|
| Daily loss limit | -20% drawdown days | Caps worst day at -5% |
| Max directional exposure | 6 correlated losses | ~50% worst-case reduction |
| Correlation groups | BTC+ETH+SOL same-direction | Further ~30% risk reduction |
| Portfolio heat | Over-leveraged portfolio | Caps total open risk at 8% |
| Kelly sizing | Oversizing no-edge strategies | crash_momentum: same PnL, 75% less risk |
| Vol-adjusted sizing | Big losses in volatile markets | 15-25% smaller avg loss |

---

## v6.9.4 — Simplification: restore long strategies, reduce short overtrading (2026-04-06)

**Problem**: Since v6.9.2-v6.9.3 filters, ZERO long trades fired in 3 days. System became shorts-only. Missed the Apr 5 BTC rally ($67.3K→$69.1K) and actively shorted into it. Meanwhile crash_momentum churned 34 trades for -$7.

**Root cause**: Over-optimization. We kept adding filters to fix bad weeks, but 30-day data shows trend_breakout is the #1 earner (+$230) and order_flow is #2 (+$61). The bad weeks are paid for by good weeks — that's how trend following works.

### Rolled back (long strategy filters)
- **Removed 4h uptrend confirmation** (v6.9.3): Was blocking ALL longs even during real rallies
- **Restored HTF strength threshold 0.1** (was 0.3 in v6.9.2): Prevented entries on early trend moves
- **Kept neutral-mode disabled**: That filter has solid evidence (35% false break rate)

### Fixed (smart_money silent)
- **Score threshold 0.55 → 0.45**: v6.9.1 raised from 0.40→0.55 based on 5 AVAX losses. But 0.55 is unreachable — scores top out ~0.47. Zero trades in 6 days. 0.45 filters the weakest entries while letting the strategy fire.

### Tightened (short overtrading)
- **crash_momentum entry: 0.3% → 0.5% below SMA20**: 131 trades for +$9 in 30 days. 48 were "no momentum" exits. Tighter entry = fewer marginal setups.

### Capital rebalanced (follow the data)
- **trend_breakout $1000 → $1250**: #1 earner, restored to full allocation
- **failed_breakout_short $850 → $400**: Net loser (-$13/30d), cut in half
- **crash_momentum $500 → $400**: Overtrading for minimal return
- Total: $4,290 (was $4,650)

### 30-day performance context
| Strategy | 30d PnL | Trades | Decision |
|----------|---------|--------|----------|
| trend_breakout | +$230 | 122 | **Restore** |
| order_flow | +$61 | 44 | **Restore** |
| crash_momentum | +$9 | 131 | Tighten entry |
| failed_breakout_short | -$13 | 46 | Cut capital |

---

## v6.9.3 — 4h uptrend confirmation for long strategies (2026-04-03)

**Problem**: trend_breakout lost -$64 in 7 days (27% WR, 30 trades) despite all signals being "correct" on 1h timeframe. Root cause: 1h SMA20/SMA50 reads "bullish" on 2-3 day bear market rallies, and breakout buys into those bounces. Even htf_str=0.93+ trades were stopping out.

### Fix: `check_4h_uptrend()` — higher timeframe confirmation
- Long signals from trend_breakout and order_flow now require **4h structural uptrend** (2 of 3 confirmations):
  1. Price above 4h SMA50 (sustained uptrend ~8 days)
  2. 4h SMA20 slope positive (trend direction)
  3. Higher lows on 4h (last 10 vs previous 10 candles — trend structure)
- This is NOT a regime gate or inverse of bearish check — it's structural trend confirmation
- Mirrors how short strategies already use `check_bearish_regime(df_4h)` for validation
- When market turns bullish again, 4h structure naturally confirms and longs fire normally

### Current 4h status (at deploy)
- BTC: REJECTED (1/3) — below SMA50, lower lows
- ETH: CONFIRMED (3/3) — above SMA50, rising, higher lows
- SOL: REJECTED (0/3) — below SMA50, falling, lower lows

### Estimated impact
- Would have blocked most of the -$64 in breakout losses (bear market rally entries)
- Strategies stay fully intact — fire automatically when 4h confirms uptrend

---

## v6.9.2 — Capital rebalance + trend_breakout filter (2026-04-01)

Based on AI daily review analysis (Mar 30-31):

### Capital reallocation
- **crash_momentum $750 → $500**: 52% WR but SL losses (-$6 each) outweigh early-exit wins ($0.30-$1). Reduce exposure.
- **failed_breakout_short $600 → $850**: Best strategy (60% WR, +$17/day) was under-capitalized. Total stays $4,650.

### trend_breakout: HTF strength 0.1 → 0.3
- 17% WR on 6 trades — entering breakouts when trend barely exists. Requiring 0.3 strength blocks noise entries while keeping real trends.

---

## v6.9.1 — AI-driven fixes: smart_money + crash_momentum (2026-03-31)

Two fixes based on Claude AI trade analysis patterns:

### smart_money: tighten entry filters (29% WR → expect ~50%+)
- **Score threshold 0.40 → 0.55**: All 5 consecutive AVAX losses entered at score ~0.41. Raising to 0.55 blocks weak signals.
- **Require whale OR funding confirmation**: Every loss had `hl_whales=0.0` AND `funding=0.0`. Now requires at least one to be active (>0.1).

### crash_momentum: early momentum exit
- **30-min momentum check**: If price hasn't moved ±0.3% after 30 minutes, close early. Frees capital from stale trades (11/30 were stale/session-end exits with <$1 PnL).

---

## v6.9 — AI-Powered Trading Intelligence (2026-03-31)

Integrated Claude API for automated trade analysis, signal evaluation, and daily performance reviews.

### New: Post-Trade AI Analysis
- After every trade closes, Claude Haiku analyzes why it won/lost using entry indicators and strategy context
- Results stored in `trade_analysis` table, displayed on dashboard under "Claude AI Reviews"
- Fire-and-forget pattern — never blocks the trading loop
- Works in both paper and live trading modes

### New: Pre-Trade Signal Gate (opt-in)
- Before executing a signal, Claude evaluates it against recent performance and portfolio state
- Returns adjusted confidence (can only reduce, never inflate)
- Falls back to original confidence on timeout/error (3s max)
- Enable with `AI_SIGNAL_GATE_ENABLED=true` in .env (default: off)

### New: Daily AI Review
- Standalone script `scripts/ai_daily_review.py` generates comprehensive daily reviews via Claude Sonnet
- Per-strategy breakdown, cross-strategy correlations, 3 ranked suggestions
- Systemd timer runs at midnight UTC (`deploy/ai-review.timer`)
- Reviews stored in `ai_reviews` table, displayed on dashboard

### Infrastructure
- `core/ai/` module: `claude_client.py` (API wrapper), `prompts.py` (templates), `models.py` (dataclasses)
- New Supabase tables: `trade_analysis`, `ai_reviews`, `signals.ai_gate_result` column
- Config: `AI_POST_TRADE_ENABLED`, `AI_SIGNAL_GATE_ENABLED`, `AI_DAILY_REVIEW_ENABLED`
- Cost tracking: per-call token usage, daily cost estimate logged
- Estimated cost: ~$13-54/month depending on trade volume

### Dashboard
- New "Claude AI Reviews" section showing latest daily review and per-trade analyses
- Updated footer to reflect Claude AI integration

---

## v6.8 — Smart Money Flow strategy (2026-03-30)

New strategy tracking whale/smart money positions across multiple free data sources.

**New data**: Binance top trader POSITION ratio (capital-weighted), global L/S ratio (retail crowd), Hyperliquid whale positions (20 wallets, $4M-$81M accounts), Fear & Greed Index.

**SmartMoneyFlowGenerator**: Composite score from 6 components (top position 25%, pro-retail divergence 25%, HL whales 20%, taker flow 15%, F&G 10%, funding 5%). Entry: score > ±0.4, 3+ components agree, HTF not opposing. SL 2.0x ATR, TP 3.0x, trailing, max 2 concurrent.

**Capital**: breakout $1250→$1000, flow $750→$500, smart_money $500 (new). Total $4,650.

---

## v6.7.2 — Data-driven performance fix (2026-03-30)

**7-day assessment (85 trades, Mar 23-30): $4,650 → $4,650.60 (+$0.60, flat)**

One strategy bled the system: crash_momentum +$40, trend_breakout +$40, but trend_pullback -$81 wiped it all out.

### Changes:

1. **Disabled trend_pullback** — 23% WR, -$80.82, 10pts below breakeven. $750 redistributed to winners.
2. **Tightened trend_breakout** — disabled neutral-mode breakouts, vol threshold 1.2x→1.5x
3. **Per-strategy circuit breaker** — was global (pullback SLs froze crash_momentum). Now scoped per strategy.
4. **Crash momentum TP 4.0→3.0 ATR** — converts stale 12h timeouts into TP hits
5. **Funding reversion OI 0.5→0.2%** — 0 trades/week because threshold too strict

Capital: funding $500, breakout $1250, flow $750, regime_short $400, failed_bkout $600, cascade $400, crash $750 = $4650

---

## v6.7.1 — Correlation guard + directional SL guard + adaptive sizing ON (2026-03-24)

**Problem**: First 19h of v6.7 showed -$60.83 (-1.3%). Two issues:

1. **crash_momentum opened 6 correlated shorts simultaneously** (-$76.93): All 6 coins moved together, all 6 hit SL together. This was one bet, not six independent bets.
2. **trend_pullback kept buying into a decline** (5/5 SL, -$42.44): Strategy re-entered long after every SL hit, each time catching the next leg down.
3. **HTF conflict log spam**: 5,008 lines in 19h — logged once per strategy per symbol per cycle.

**Fixes**:

### 1. Max concurrent positions (crash_momentum: 2)
- New `max_concurrent_positions` field on StrategyConfig (0 = unlimited)
- crash_momentum capped at 2 open positions across all symbols
- Prevents correlated exposure from turning one bad call into 6x loss

### 2. Directional SL guard
- Tracks SL hits per strategy+direction (e.g., "trend_pullback:long")
- If 2+ SLs in same direction within 2h window, blocks further entries in that direction
- Prevents buying into sustained declines or shorting into sustained rallies
- Applies to ALL strategies (not just pullback)

### 3. HTF conflict log throttled
- Removed per-call logging from `apply_fast_reversal_override()`
- Now logs once per symbol per cycle (6 lines/cycle max, not 48)
- Conflict detail preserved in htf_trend dict for debugging

### 4. Adaptive sizing enabled
- Added `--adaptive-sizing` to GCP service
- Scales risk 0.5%-3% based on confidence, performance streak, regime alignment
- After v6.7's crash_momentum 6-loss streak, sizing would auto-reduce to 0.5x risk

---

## v6.7 — More assets + adaptive position sizing (2026-03-22)

**Two improvements targeting better monthly returns:**

### 1. Expanded to 6 symbols (was 3)
- Added SOL, DOGE, AVAX to default trading list (BTC/ETH/XRP unchanged)
- More assets = more signal opportunities with same strategy logic
- OrderManager now supports any USDT-M pair dynamically (no hardcoded map)
- Dashboard auto-detects traded symbols from DB (no more hardcoded lists)
- Rate limit safe: 6 symbols × ~14 API calls = 84/cycle, well under Binance 1200/min

### 2. Adaptive position sizing (opt-in: `--adaptive-sizing`)
Replaces fixed 2%/1.5% risk with dynamic sizing based on 4 factors:

| Factor | Logic | Range |
|--------|-------|-------|
| **Confidence** | Signal confidence 0.45→0.5x, 0.90→1.0x | 0.5x – 1.0x |
| **Performance** | 2+ consecutive losses → reduce 25%/loss. 3+ wins → boost 10%/win | 0.5x – 1.3x |
| **Regime alignment** | HTF aligned + strong → up to +15%. Misaligned → 0.75x | 0.75x – 1.15x |
| **Multi-strategy agreement** | 2+ strategies same direction+symbol → 1.1x | 1.0x – 1.1x |

- Hard floor: 0.5% risk minimum. Hard cap: 3% risk maximum.
- Worst case compound: ~0.19x base risk. Best case: ~1.65x, clamped by 3% cap.
- Default OFF (`--adaptive-sizing` flag to enable). Zero behavior change without flag.
- Works for both paper and live trading.
- Effective risk logged in trade indicators for analysis.

### Files changed
- `core/engine/order_manager.py` — dynamic symbol conversion
- `scripts/paper_trade_simple.py` — adaptive sizing, 6 symbols, performance tracking
- `scripts/live_trade.py` — adaptive sizing integration
- `dashboard/app.py` — dynamic symbols from DB, CoinGecko maps for 6 coins

---

## v6.6.1 — Switch crash_momentum to 1h candles (2026-03-18)

**Problem**: crash_momentum on 15m was too noisy — 89 signals in 24h, constant SL hits from small bounces. Backtest showed -$59 without cooldowns, -$15 with cooldowns.

**Fix**: Switch to 1h candles. Less noise, fewer false entries, bigger moves per candle.
- 1h SMA20 and RSI instead of 15m
- Lower low check vs 3 candles ago (3h) instead of 5 (75min on 15m)
- RSI floor raised to 25 (from 20) — 1h RSI rarely goes as low
- SL widened to 2.0x ATR(1h) (from 1.5x ATR(15m)) — crash bounces are violent
- TP widened to 4.0x ATR(1h) (from 3.0x) — bigger moves on 1h, 2:1 R:R
- Time stop 12h (from 6h) — 1h candles need more time
- Daily SL limit raised to 3 (from 2) — crashes are fast, need more chances

**Backtest result** (last 24h of crash):
| Version | Trades | PnL |
|---|---|---|
| 15m no cooldowns | 89 | -$59.72 |
| 15m with cooldowns | 10 | -$15.29 |
| **1h with cooldowns** | **10** | **+$22.63** |

---

## v6.6 — Fast regime gate + crash momentum strategy (2026-03-18)

**Problem**: v6.5 short strategies never fired despite market crashing (BTC -5%, ETH -6.5%, XRP -9%). Root cause: **two-layer failure**.

1. **Regime gate too slow**: Used 4h SMA200 (33 days of data) + SMA50 slope. During a selloff, price remains above SMA200 and SMA50 slope stays positive from prior uptrend. Gate blocked ALL shorts.
2. **Strategies designed for pre-crash only**: Even with gate open, existing strategies require positive funding, rising OI, and selling pressure (taker <0.95). During actual crashes, funding goes negative (longs liquidated), OI drops (positions unwound), and taker ratio >1.0 (dip buyers stepping in). The conditions flip.

**Fix 1: Faster regime gate** (3 triggers, any one = bearish):
- Price below 4h SMA50 (~8 days responsive vs ~33 days for SMA200)
- 4h SMA20 slope negative over last 3 candles (~3 days responsive)
- Price dropped >3% from 5-day high (30 4h-candles) — catches sudden selloffs immediately

**Fix 2: New `crash_momentum` strategy** — pure price action, no derivatives:
- Designed for the "during crash" phase that existing strategies miss
- Uses ONLY price action (no funding/OI/taker that flip during crashes)
- Entry: price < 15m SMA20 AND RSI 20-45 (not oversold) AND 2/3 red candles AND lower low vs 5 bars ago
- Fires after initial panic subsides and bounce fails — catches second leg down
- Config: SL 1.5x ATR, TP 3.0x ATR (2:1 R:R), max hold 6h, trailing ON (1.5x/0.8x), 1.5% risk
- Research: Dead cat bounces fail ~70% in first 48h of trend break (Jegadeesh & Titman)

**Capital reallocation** ($4,650 total, was $4,350):
| Strategy | v6.5 | v6.6 | Change |
|---|---|---|---|
| funding_reversion | $500 | $500 | — |
| trend_breakout | $1,000 | $1,000 | — |
| trend_pullback | $750 | $750 | — |
| order_flow | $750 | $750 | — |
| regime_short | $500 | $400 | -$100 (pre-crash, reduced) |
| failed_breakout_short | $400 | $350 | -$50 (pre-crash, reduced) |
| refined_liq_cascade | $450 | $400 | -$50 (pre-crash, reduced) |
| crash_momentum | — | $500 | NEW |

**Short strategy coverage by market phase**:
- **Pre-crash** (topping, crowded longs): regime_short, failed_breakout, refined_cascade
- **During crash** (active selloff, bounce failures): crash_momentum
- **Bull market**: all shorts disabled by regime gate (by design)

---

## v6.5 — Research-based short strategies with regime filter (2026-03-18)

**Problem**: v6.4 short strategies didn't work — liquidation_cascade had 0 trades ever (0.01% funding threshold never reached), panic_momentum had 12 trades/-$22 (taker 5m<0.90 too extreme), breakdown_reversal had 1 trade/-$7 (2.0x volume never met). $1,350 capital sitting idle while market dropped.

**Research findings** (BIS Working Paper 1087, QuantJourney, PocketOption studies):
1. Crypto funding rates are positive 92% of the time — structural long bias means mirroring long strategies for shorts has negative EV
2. No single derivatives indicator works alone — need moderate thresholds on 4+ conditions simultaneously
3. Shorts only work in bearish regimes — must be disabled during bull markets
4. Shorts need wider SL (2.0x ATR vs 1.0-1.3x), smaller size (1.5% risk vs 2%), and time stops

**3 new regime-filtered short strategies** (replace v6.4 shorts):

### Regime gate (shared by all shorts)
- 4h SMA50 slope negative OR price below 4h SMA200 → bearish regime → shorts enabled
- Bull regime → all shorts disabled, capital idle (by design)
- Fetches 4h candles (210 bars = ~35 days for SMA200)

### `regime_short` ($500 capital) — replaces liquidation_cascade
- **Concept**: Multi-condition derivatives confluence with moderate thresholds
- **Entry**: Funding > 0.03% AND taker 15m < 0.95 AND top traders > 55% long AND price < SMA20
- **Config**: SL 2.0x ATR(15m), TP 3.0x ATR, R:R 1.5:1, max hold 12h, trailing ON (1.5x/1.0x), 1.5% risk
- **Frequency**: 2-5/day (in bearish regime)
- **Why different**: 4 moderate conditions vs 3 extreme ones. Funding 0.03% vs 0.01%. Adds crowd positioning filter.

### `failed_breakout_short` ($400 capital) — replaces panic_momentum
- **Concept**: Price action exhaustion — breakout above resistance fails with rejection wick
- **Entry**: Prev candle made new 20-bar high AND volume > 1.5x avg AND close in lower 40% of range (rejection) AND current candle below prev midpoint (failure)
- **Config**: SL 2.0x ATR(15m), TP 2.5x ATR, R:R 1.25:1, max hold 6h, trailing ON (1.5x/0.8x), 1.5% risk
- **Frequency**: 1-3/day (in bearish regime)
- **Why different**: Pure price action (no derivatives dependency). Catches liquidity sweep reversals.

### `refined_liq_cascade` ($450 capital) — replaces breakdown_reversal
- **Concept**: Derivatives-based with realistic thresholds (v6.4 cascade redesigned)
- **Entry**: Funding > 0.05% AND OI rising > 1% (full lookback) + price flat/falling AND taker 15m < 0.93 AND RSI < 45
- **Config**: SL 2.0x ATR(15m), TP 4.0x ATR, R:R 2:1, max hold 8h, trailing ON (2.0x/1.0x), 1.5% risk
- **Frequency**: 0-2/day (rare but high conviction)
- **Why different**: Funding 0.05% (vs 0.01%), longer OI lookback (full history vs 30min), realistic taker threshold.

**Key changes from v6.4**:
| Aspect | v6.4 shorts | v6.5 shorts |
|---|---|---|
| Regime filter | None (always on) | 4h bearish regime required |
| Stop loss | 1.0-1.3x ATR (too tight) | 2.0x ATR (research: violent bounces) |
| Risk per trade | 2% | 1.5% (smaller for shorts) |
| Time stops | 3-5h | 6-12h |
| Conditions | 3 extreme thresholds | 4+ moderate thresholds |
| Funding threshold | 0.01% (never reached) | 0.03%/0.05% (actionable) |
| Price action | None | Failed breakout wick pattern |
| Capital | $1,350 | $1,350 (same total) |

---

## v6.4 — Short-only strategies (2026-03-17)

**Rationale**: Previous shorts failed (38 trades, 18.4% WR, -$262) because they mirrored long logic — same HTF trend dependency, same R:R, same thresholds. The 1h SMA20/SMA50 HTF trend is the proven long edge but the biggest short liability: crypto crashes happen in minutes-hours; by the time 1h SMAs cross bearish, the move is 60-80% done.

**3 new short-only strategies** (derivatives-based, no HTF dependency):

### `liquidation_cascade` ($500 capital)
- **Concept**: Over-leveraged longs + weakening price + rising OI = liquidation fuel
- **Entry**: Funding rate > 0.01% AND OI rising + price falling (30min divergence) AND 15m price < SMA20 + RSI < 45 + RSI falling
- **Config**: SL 1.2x ATR(15m), TP 4.0x ATR, R:R 3.33:1, max hold 4h, trailing ON (activate 2.0x, trail 0.8x)
- **Frequency**: 1-3/day

### `panic_momentum` ($350 capital)
- **Concept**: Multi-TF taker selling + negative futures premium = self-reinforcing panic
- **Entry**: 5m taker ratio < 0.90 AND 15m taker ratio < 0.95 AND futures premium <= 0 AND top traders < 55% short
- **Config**: SL 1.0x ATR(15m), TP 3.5x ATR, R:R 3.5:1, max hold 3h, trailing ON (activate 1.5x, trail 0.6x)
- **Frequency**: 0-2/day

### `breakdown_reversal` ($500 capital)
- **Concept**: Price breaks below 20-bar low with volume + rising OI confirms real breakdown
- **Entry**: 15m close < 20-bar low AND volume > 2.0x avg AND OI rising > 0.1% over 15min
- **Config**: SL 1.3x ATR(15m), TP 3.5x ATR, R:R 2.69:1, max hold 5h, trailing ON (activate 2.0x, trail 0.8x)
- **Frequency**: 1-3/day

**Key design differences from failed shorts**:
| Aspect | Old shorts | New short strategies |
|---|---|---|
| HTF dependency | Required 1h SMA bearish (too slow) | Zero HTF — derivatives-based |
| R:R | 2:1 (same as longs) | 2.7-3.5:1 (wider TP for cascades) |
| SL | 1.5x ATR | 1.0-1.3x ATR (tighter — cut fast) |
| Trailing stop | Disabled | Enabled (lock profits in fast moves) |
| Max hold | 6-12h | 3-5h (short moves resolve quickly) |

**Safety**: Circuit breaker, cross-strategy cooldown, pileup block all shared with longs. Reversal close won't affect shorts (checks HTF suppression which shorts don't use).

**Capital**: $3,000 (4 long) + $1,350 (3 short) = $4,350 total.

## v6.3.5 — Long-only mode (2026-03-17)

**Data basis**: All 156 closed trades. Sell side: 38 trades, 18.4% WR, -$262. Buy side: 118 trades, 55.9% WR, +$698.

**Sell performance by strategy**:
- `trend_breakout` shorts: 10 trades, 10% WR, -$68
- `taker_flow` shorts: 13 trades, 23.1% WR, -$138
- `order_flow` shorts: 7 trades, 14.3% WR, -$17
- `trend_pullback` shorts: 3 trades, 0% WR, -$16
- `oi_momentum` shorts: 5 trades, 40% WR, -$22

**Root causes identified**:
- HTF strength thresholds 2-3x higher for sells (0.3 vs 0.1-0.15 for buys), forcing entries only at marginal conditions
- Sell avg HTF strength at entry: 0.389 (barely above threshold) vs buy: 0.523 (well above)
- Counter-trend confidence penalty (-10-25%) caps sell confidence at 0.45
- Strategies designed and tuned for long bias; shorts bolted on as afterthought

**Change**: Disabled all sell/short signals across all 4 strategies. Each generator now returns hold_signal when sell conditions are met. Long-only mode eliminates $262 drag from losing shorts.

## v6.3.4 — SL bug fix + reversal close (2026-03-17)

**Data basis**: Mar 17 session. 5 long positions entered during bullish trend (01:10-02:31 UTC), market reversed at ~03:30. Positions lost $233 instead of ~$50-80 from proper SL exits.

**Bug fix — _check_exit undefined `symbol` variable**:
- `_check_exit()` used `symbol` without extracting it from pos_key
- NameError was silently caught by outer try/except in `_process_strategy_symbol`
- Result: SL exits NEVER executed — positions bled until liquidation or session end
- Fix: extract `_, symbol = self._parse_pos_key(pos_key)` at method start

**Feature — Reversal close for losing positions**:
- Previously, fast reversal override only blocked new entries
- Now: when reversal persists 2+ consecutive cycles (~2min), closes positions that are:
  - In loss (profitable positions keep running)
  - NOT pullback strategy (pullback expects counter-trend dips by design)
- Today's scenario: 5 longs sat underwater for 70+ min through reversal. With this fix, they would have been closed within ~2min of reversal detection, saving ~$150+.
- Tracking: `reversal_override_counts` per symbol, incremented once per main-loop cycle (not per strategy). `_reversal_counted_this_cycle` set prevents multi-counting.

## v6.3.3 — Hold reason logging + smarter pileup block (2026-03-16)

**Data basis**: v6.3.2 overnight (Mar 15-16). 7/7 trades won (+$228), but analysis showed:
- Flow generated 574 buy signals, only took 1 trade — pileup block killed the rest
- No post-hoc analysis possible because hold signals weren't logged to DB

**Fix 1 — Hold reason logging to DB**:
- Hold signals now saved to signals table (throttled: once per 15min per strategy:symbol, or on reason change)
- Enables post-hoc analysis of *why* strategies didn't trade during specific periods
- ~96 samples/day per strategy:symbol (vs 0 before), minimal DB load

**Fix 2 — Time-aware pileup block**:
- Old: if ANY strategy has same direction on symbol → block (killed flow all night)
- New: only block if existing position was opened within last 30 minutes
- Prevents simultaneous pileup (the Mar 13 problem: breakout+flow both entering at 13:19)
- Allows sequential entries (flow can enter BTC after pullback has held for 2+ hours)
- Flow gets independent entries when its taker ratio + top trader signals fire independently

## v6.3.2 — Three loss-reduction fixes: sell floor + symbol cooldown + pileup block (2026-03-16)

**Data basis**: 59 active-strategy trades (Mar 13-15, v6.3/v6.3.1).

**Fix 1 — Higher HTF strength floor for sells**:
- Sell signals now require HTF strength ≥ 0.3 for pullback (was 0.15) and flow (was 0.1)
- Buy thresholds unchanged (0.15 / 0.1). Breakout unchanged (already has volume gate)
- Evidence: Weak-HTF sells (strength < 0.3) had 0-12% WR, -$77 total. Strategies were shorting on barely-bearish noise.

**Fix 2 — Cross-strategy symbol cooldown after SL**:
- 30-min cooldown on the *symbol* (not strategy:symbol) after any SL hit
- Prevents different strategies from immediately re-entering the same losing symbol
- Evidence: 4 cross-strategy re-entries after SL on same symbol → -$43. E.g. breakout SL'd on ETHUSDT, flow re-entered immediately.

**Fix 3 — One strategy per symbol per direction**:
- If any strategy already has a long on ETHUSDT, other strategies blocked from also going long on ETHUSDT
- Different directions still allowed (unlikely with current logic but safe)
- Evidence: breakout + flow both bought ETH at 13:19 on Mar 13 → double exposure to same reversal, -$61 across 3 clusters.

## v6.3.1 — Fix sell losses: suppress instead of flip on HTF conflict (2026-03-15)

**Problem**: v6.3's fast reversal override (`apply_fast_reversal_override`) was flipping HTF direction (bullish→bearish) on every normal 15m pullback (RSI<40 + price<SMA20). This forced ALL strategies into shorts simultaneously. Result: sell trades 12% WR, -$61.24 vs buy trades 50% WR, +$95.65. Market was ranging — shorts into sideways = stop losses.

**Root cause**: Single-candle RSI/price condition fires on noise, not real reversals. Hard direction flip with 0.5x strength still passes every strategy threshold (breakout>0.1, pullback>0.15, flow>0.1). Creates whipsaw: rapid long/short reversals generating multiple SLs.

**Fix — Conflict suppression instead of direction flip**:
- When 15m conflicts with 1h, set direction to **neutral** instead of flipping to opposite
- Strength reduced to 0.3x (was 0.5x) — ensures suppression is meaningful
- `trend_pullback`: neutral direction → returns hold (blocked). No forced shorts.
- `order_flow`: neutral direction → returns hold (blocked). No forced shorts.
- `trend_breakout`: neutral = ranging mode → requires 1.5x volume for breakouts (existing logic). High-volume breakouts in either direction still allowed.
- Circuit breaker unchanged — still active as secondary protection
- Net effect: strategies **stop trading** when signals disagree, rather than **reversing**

## v6.3 — Fast reversal detection + circuit breaker (2026-03-14)

**Problem**: On Mar 13, all 3 active strategies kept buying into a bearish reversal because they all depend on the 1h HTF trend (SMA20 vs SMA50), which takes hours to flip. Result: 5 winners (+$174) followed by 9 consecutive stop losses (-$179). The daily SL limit (2 per strategy) eventually stopped trading, but by then the damage was done.

**Root cause**: `determine_htf_trend()` uses 1h SMA20/SMA50 cross. When market reverses on 15m, the 1h SMAs don't cross for hours. All strategies (breakout, pullback, order_flow) are gated on HTF direction, so they're structurally blind to reversals.

**Fix 1 — Fast 15m reversal override** (`apply_fast_reversal_override`):
- After computing 1h HTF trend, check 15m data for reversal signals
- Override to bearish when: HTF says bullish BUT 15m price < SMA20 AND 15m RSI < 40
- Override to bullish when: HTF says bearish BUT 15m price > SMA20 AND 15m RSI > 60
- Override strength is halved (0.5x) since it's an early/unconfirmed signal
- Only triggers on conflict (HTF vs 15m disagree) — no effect when they agree

**Fix 2 — Global circuit breaker**:
- Track SL timestamps across ALL strategies globally
- After 3 SLs within 2 hours → pause ALL entries for 1 hour
- Prevents the "9 consecutive SL" scenario — would have triggered after the 3rd SL at 13:31 UTC and saved 6 subsequent losses (~$112)
- Per-strategy daily SL limit (2/day) still active as secondary protection
- Circuit breaker resets after the pause expires

**Bug fix**: Added `"flow"` to `needs_derivatives` check for robustness (was working via `"funding"` but would break if funding_reversion disabled).

## v6.2 — Add order_flow strategy using taker ratio + top trader data (2026-03-13)

**Rationale**: Two derivatives data sources fetched every cycle but never used: taker buy/sell ratio (15m) and top trader long/short account ratio. Academic research shows order flow has permanent price impact (Sharpe 3.63 in Anastasopoulos 2025). Backtesting at 1.05/0.95 thresholds returned 142% vs 101% B&H (CryptoCoffeeShop 2021-2024). Top trader positioning works as contrarian filter — skip when crowd is too one-sided (>58%).

**New Strategy — Order Flow** (`order_flow`):
- Entry requires ALL 3 conditions:
  1. HTF trend established: strength > 0.1 (any directional trend)
  2. 15m taker buy/sell ratio confirms direction: >1.05 for buys, <0.95 for sells
  3. Top trader positioning NOT crowded in our direction: long_account < 0.58 (buy) or short_account < 0.58 (sell)
- Confidence boosts: +5% for strong flow imbalance (>1.10/<0.90), +5% for contrarian positioning alignment (<48%)
- SL 1.5x ATR(15m), TP 3.0x ATR(15m), R:R 2:1
- Max hold 6h, risk 2%/trade, capital $750
- MEDIUM frequency expected (2-5/day)

**Why this is different from breakout/pullback**:
- Breakout enters on **price structure** (break above highs)
- Pullback enters on **RSI reversion** (dip to SMA)
- Order flow enters on **aggressive taker volume** + **non-crowded positioning**
- All three use HTF trend as common filter but capture completely different edges

**Capital rebalance (4 strategies, $3,000 total)**:
- funding_reversion: $500 (0.5x, rare-event)
- trend_breakout: $1,000 (1.0x, proven)
- trend_pullback: $750 (0.75x, new)
- order_flow: $750 (0.75x, new)

**Dashboard**: Added order_flow to strategy selector, badges, comparison table, signal source labels. Updated version to v6.2.

## v6.1 — Replace oi_momentum with trend_pullback (2026-03-13)

**Problem**: oi_momentum lost money across EVERY version that generated trades. v6.0.2: 33 trades, 39% WR, -$85. v6.0.3 widened TP but the core thesis (RSI momentum zone + OI rising) is too weak — RSI 55-75 is where RSI *normally sits* in a trending market, producing 89% buy bias and 622 signals in 7 days. The signal is too common to have edge.

**Solution**: Replace oi_momentum with **trend_pullback** — a complementary strategy to trend_breakout. While breakout catches the START of trend moves, pullback catches CONTINUATION by buying dips in uptrends and selling rallies in downtrends.

**New Strategy — Trend Pullback** (`trend_pullback`):
- Entry requires ALL 3 conditions:
  1. HTF trend established: strength > 0.15 (moderate trend, lower bar than breakout's 0.3)
  2. 15m RSI shows pullback: 30-45 in uptrend (dip) or 55-70 in downtrend (rally)
  3. Price within 1.0x ATR(15m) of SMA20(15m) — pulled back to the mean
- Direction follows HTF trend (no directional bias like oi_momentum's 89% buy)
- SL 1.5x ATR(15m), TP 3.0x ATR(15m), R:R 2:1
- Max hold 8h, risk 2%/trade, capital $1,000
- Uses 15m timeframe (same as trend_breakout, less noisy than oi_momentum's 5m)
- MEDIUM frequency expected (2-5/day)

**Why this should work**:
- Uses our proven edge: 1h HTF trend filter (the ONLY consistently profitable edge across all versions)
- Entry at better prices than breakout (buying the dip, not the breakout)
- Naturally limited: only fires in trending markets with actual pullbacks to SMA
- Complementary: breakout enters on strength, pullback enters on weakness within same trend
- No derivatives dependency (OI, funding) — pure price action with trend confirmation

**Other changes**:
- Removed `OIMomentumGenerator` class and "oi" strategy type
- Capital unchanged: funding $500, trend_breakout $1,500, trend_pullback $1,000 (total $3,000)
- Dashboard: updated strategy selector, badge mappings, comparison table for v6.1

## v6.0.3 — Improve oi_momentum R:R, Rebalance Capital (2026-03-13)

**Problem**: v6.0.2 ran 2 days (65 trades, 36.9% WR, -$218). Strategy breakdown:
- **trend_breakout**: 19 trades, 42% WR, +$5 — only profitable strategy. Winners larger than losers (+1.2-1.6% vs -0.7-0.9%). Structurally sound.
- **oi_momentum**: 33 trades, 39% WR, -$85 — poor R:R. TP (2.5x ATR) and SL (1.5x ATR) produce similar-magnitude outcomes (~0.6-0.9% both sides). At 39% WR with 1:1.67 R:R, expected PnL is barely positive but stale exits and min_sl_pct floor erode the edge. Also 89% buy bias (RSI momentum zone 55-75 fires much more than 25-45 in trending markets). All 3 open positions often correlated (all long BTC/ETH/XRP simultaneously).
- **funding_reversion**: 0 trades — funding normal (-0.011%) entire period. $1,000 sitting idle.
- **taker_flow**: 13 legacy trades from Mar 9 pre-deployment (23% WR, -$138). Not in current code.

**Fixes**:
- **Widen oi_momentum TP 2.5x → 3.5x ATR**: R:R improves from 1:1.67 to 1:2.33. Breakeven WR drops from 38% to 30%. At current 39% WR, expected PnL per trade = 0.39*3.5 - 0.61*1.5 = +0.45 (was +0.06). Trades need more room to reach TP, so also extended max hours.
- **Extend oi_momentum max_position_hours 3h → 4h**: With wider TP (3.5x ATR on 5m), trades need more time. 4h gives better chance of hitting TP before stale exit.
- **Rebalance capital allocation**: Instead of equal $1,000 per strategy, now weighted by performance evidence:
  - funding_reversion: $1,000 → $500 (rare-event, 0 trades in 7 days, mostly idle)
  - trend_breakout: $1,000 → $1,500 (only profitable strategy, proven edge)
  - oi_momentum: $1,000 → $1,000 (unchanged, awaiting R:R fix results)
  - Total unchanged at $3,000.
- **Dashboard**: Updated version string to v6.0.3.

## v6.0.2 — Fix Trailing Stops, Signal Noise, ATR Floor, Loosen Thresholds (2026-03-11)

**Problem**: v6.0.1 ran 12h (16 trades, 8W/8L, -$18.57, 50% WR). Despite 50% WR, system lost money because trailing stops clipped all winners: 6/7 wins exited via trailing at 36-60% of designed TP. Actual R:R was 0.83:1 instead of designed 2:1. Only 1/16 trades hit full TP. Signal DB flooded with ~6500 hold signals/12h. oi_momentum's 5m ATR produced SL as tight as 0.33%, clipped by noise. After initial fixes, funding_reversion and trend_breakout had 0 trades in 2 days — thresholds too restrictive for ranging markets.

**Fixes**:
- **Disable trailing stops**: Added `trailing_enabled` flag to StrategyConfig (default False). Trailing update + trailing exit gated on flag. Backwards-compatible (existing positions default True). Pure SL/TP exits let winners run to designed 2:1 R:R. Can re-enable later with better tuning.
- **Only save actionable signals**: Skip `_save_signal()` for hold signals. Holds still logged to debug for journalctl. Eliminates ~6500 hold signals/12h noise in DB. Dashboard accuracy stats become meaningful.
- **ATR floor for oi_momentum**: Added `min_sl_pct` to StrategyConfig. When ATR-computed SL < floor, all stops scale proportionally (preserves R:R). Set 0.5% floor for oi_momentum. Prevents ultra-tight 0.33% stops from noise-clipping.
- **Strategy-aware stale exit**: `_is_stale_position` now uses `tp_pct * 0.25` instead of fixed 0.005. Adapts per strategy (funding_reversion needs different threshold than oi_momentum).
- **Remove HTF filter from oi_momentum**: The 0.2 strength threshold (from v6.0.1) made oi_momentum impossible to fire in ranging markets — `determine_htf_trend` halves strength when SMA spread < 0.3%, capping it at ~0.04. Since oi_momentum uses RSI for direction and OI for confirmation, it doesn't need HTF strength. Down to 3 conditions (from 4).
- **Lower OI threshold 1.0% → 0.5%**: In ranging markets, 30min OI changes peak at 0.5-0.7%. The 1% threshold only triggered once in 4 hours of data. At 0.5%, would have caught 6 more valid opportunities. Still meaningful (confirms new money entering).
- **Trend breakout ranging mode**: HTF strength 0.3 was impossible in ranging markets (max ~0.07). Now allows breakouts even when HTF is neutral, but requires higher volume (1.5x avg vs 1.2x). When trending, still uses original 1.2x volume threshold. Range breakouts get -5% confidence penalty.
- **Lower funding thresholds**: Funding >0.05%/<-0.03% too rare in normal markets (0 trades in 2 days). Lowered to >0.03%/<-0.02%. Still contrarian, but catches moderately elevated funding. Strong signal threshold stays at 0.1%.

## v6.0.1 — Fix Inverted R:R and OI Momentum Filtering (2026-03-10)

**Problem**: After 24h of v6.0, results were 4W/6L (-$40.88, 40% WR) with inverted R:R (0.83:1 instead of 2:1). Root causes: (1) trailing_atr_mult set equal to sl_atr_mult for oi_momentum (both 1.5x ATR) — trailing activates at SL distance, so winners exit for tiny gains; (2) oi_momentum entered on near-zero HTF strength (0.021), catching noise in ranging markets; (3) dashboard signal table flooded with hold signals, making accuracy stats 0/0.

**Fixes**:
- **Raise trailing activation**: oi_momentum 1.5x→2.0x ATR (dist 0.8x→1.0x); trend_breakout 2.0x→2.5x ATR. Now trailing activates well above SL, letting winners run to proper R:R.
- **Add HTF strength filter to oi_momentum**: Require strength >= 0.2 before entering. Prevents entries in directionless markets.
- **Dashboard: filter to actionable signals only**: Signal table now shows only buy/sell signals (not holds). Accuracy stats become meaningful.
- **Dashboard: add source labels**: Added "breakout" and "oi" labels for v6.0 strategies.
- **Data reset**: NEW_SYSTEM_DATE updated to clear v6.0 data and start fresh with fixes.

## v6.0 — Evidence-Based Strategy Redesign (2026-03-09)

**Problem**: v5.0-5.1 strategies all failed: funding_sentiment (0 trades — 8h funding data too slow), volatility_squeeze (0 trades — BB/KC squeeze too rare in crypto), taker_flow (5 trades, 25% WR, -$58 — 1m data too noisy, 0.8% SL too tight). Historical analysis shows the 1h HTF trend filter was the ONLY consistently profitable edge (v3.6 achieved 53% WR with it). Research confirms: funding rate mean reversion has strongest evidence (Sharpe 1.4-2.3), OI works as confirmer not standalone, ATR-based stops >> fixed %.

**Solution**: Replace all 3 strategy generators with simpler, evidence-based designs using boolean conditions (no additive scoring), ATR-based dynamic stops, and risk-based position sizing.

**3 New Strategies**:

1. **Funding Reversion** (`funding_reversion`) — Contrarian funding rate mean reversion. Entry: funding > 0.05% (short) or < -0.03% (long) + OI rising > 0.5% (30min) + price hasn't reversed > 0.5% (1h). SL 2.0x ATR(1h), TP 4.0x ATR(1h), trailing 3.0x/1.5x ATR, max 12h, R:R 2:1, 2% risk/trade. LOW frequency (0-2/day).

2. **Trend Breakout** (`trend_breakout`) — Trade 15m breakouts WITH the 1h trend. Entry: HTF trend strength > 0.3 + 15m close above 10-bar high/low + volume > 1.2x avg. SL 1.5x ATR(15m), TP 3.0x ATR(15m), trailing 2.0x/1.0x ATR, max 6h, R:R 2:1, 2% risk/trade. MEDIUM frequency (2-5/day).

3. **OI Momentum** (`oi_momentum`) — OI rising + RSI in momentum zone. Entry: 5m RSI 55-75 (long) or 25-45 (short) + OI rising > 1% (30min) + price within 2x ATR of SMA20. SL 1.5x ATR(5m), TP 2.5x ATR(5m), trailing 1.5x/0.8x ATR, max 3h, R:R 1.67:1, 1.5% risk/trade. HIGH frequency (3-8/day).

**Key architectural changes**:

- **ATR-based dynamic stops**: `calculate_atr()` standalone function. SL/TP computed at entry as ATR multiples, stored on position as pct for exit checks. Adapts per-asset and per-market-volatility automatically.
- **Risk-based position sizing**: `calculate_position_size()` — risk 1.5-2% capital per trade, capped at 30% margin. Replaces flat 20% margin allocation.
- **HTF trend returns dict**: `determine_htf_trend()` now returns `{direction, strength, slope}`. `apply_htf_adjustment()` modulates confidence by trend strength instead of hard-blocking.
- **Simple boolean conditions**: Each strategy checks 3 binary conditions. ALL must be true. No additive scoring = fewer parameters = less overfitting.
- **New candle fetches**: Added 5m + 15m klines. Removed 4h.
- **Position stores ATR params**: `sl_pct`, `tp_pct`, `trailing_act_pct`, `trailing_dist_pct` computed at entry from ATR and stored on position dict + DB indicators_at_entry. Exit logic reads from position, not strategy config.

**Removed**: `FundingSentimentGenerator`, `VolatilitySqueezeGenerator`, `TakerFlowGenerator`, 4h candle fetch, `max_position_pct` (replaced by risk-based sizing), fixed SL/TP percentages on StrategyConfig.

**Dashboard**: Updated strategy selector, badge mappings, comparison table, version string for v6.0 strategy names.

---

## v5.1 — Tuned Thresholds + Soft HTF Filter + DB Fix (2026-03-09)

**Problem**: After 6 hours of v5.0 running, only taker_flow generated trades (5 trades, all SHORT, 25% WR, -$54). funding_sentiment and volatility_squeeze had zero trades due to:
1. Signal thresholds too high (needed composite score >= 2.5 for entry)
2. HTF trend filter was a **hard block** — bearish trend blocked ALL buy signals, bullish blocked ALL sell signals, creating a short-only bias in ranging markets
3. Critical DB bug: `await` on synchronous Supabase `APIResponse` meant **trade exits never saved to database** (broken since v4.0)

**Changes**:

1. **Fixed trade exit DB bug**: Wrapped `trade_repo.table.update()` in `asyncio.to_thread()` (same pattern as other DB calls). Trade exits now persist correctly.

2. **Lowered signal thresholds** for funding_sentiment and volatility_squeeze:
   - Strong signal: 4.0 → 3.0
   - Normal signal: 2.5 → 1.8
   - taker_flow unchanged (already at 3.5/2.0)

3. **Soft HTF trend filter** (all 3 strategies): Instead of blocking counter-trend trades entirely, reduce confidence by 15%. This allows high-conviction counter-trend entries while still preferring trend-aligned trades. The confidence gate (0.65 min) naturally filters out weak counter-trend signals.

4. **Ranging market detection**: `determine_htf_trend()` now returns "neutral" when SMA20 and SMA50 are within 0.3% of each other, preventing false directional bias in choppy markets.

**Expected impact**: All 3 strategies should now generate trades. Counter-trend trades allowed but penalized. Ranging markets treated as neutral instead of forcing a directional bias.

---

## v5.0 — Derivatives-Data Signal Strategies (2026-03-09)

**Problem**: v4.0's 3 strategies (agreement_classic, agreement_mtf, momentum) all used lagging technical indicators (MACD/SMA/RSI) on 1m candles. After 3 days and 143 trades, signal accuracy was **43%** — worse than a coinflip. Testing across all timeframes (1m through 1d) showed MACD+SMA had ~47-51% accuracy on every timeframe. The indicators had zero predictive power. The system only profited because the 1h HTF filter correctly identified macro trend direction. Entry signals were decorative.

**Solution**: Replace all 3 signal generators with strategies using **derivatives market data** (funding rates, open interest, taker flow, order book) — forward-looking data that measures what participants are *doing*, not what price already *did*. All data is freely available from Binance public Futures API (no API key needed).

**3 New Strategies**:

1. **Funding Sentiment** (`funding_sentiment`) — Funding rate + OI divergence + futures basis scoring. Extreme funding precedes reversals. OI divergence (OI rising while price falls) is a leading indicator. Basis measures futures premium/discount. Params: SL 1.5%, TP 4.0%, trailing 2.0%/1.0%, max hold 8h, R:R 2.67:1.

2. **Volatility Squeeze** (`volatility_squeeze`) — Bollinger Band squeeze inside Keltner Channel on **4h candles**. Volatility is mean-reverting; breakout from squeeze produces asymmetric payoff. Only acts on completed 4h candles. Volume confirmation from 1m data. Params: SL 1.0%, TP 3.0%, trailing 1.5%/0.7%, max hold 12h, R:R 3.0:1.

3. **Taker Flow** (`taker_flow`) — Taker buy/sell volume ratio + order book imbalance + top trader L/S ratio (contrarian). Real-time aggression data. Short-term (1m) strategy since flow data is inherently short-lived. Params: SL 0.8%, TP 1.6%, trailing 0.8%/0.4%, max hold 2h, R:R 2.0:1.

**Data sources** (all Binance Futures public API):
- `fapi/v1/fundingRate` — funding rate history
- `fapi/v1/openInterest` — current OI snapshot
- `futures/data/openInterestHist` — OI trend (5m intervals)
- `fapi/v1/premiumIndex` — mark price, index price, predicted funding
- `futures/data/takerlongshortRatio` — taker buy/sell volume (5m/15m/1h)
- `futures/data/topLongShortAccountRatio` — top trader positioning
- `api/v3/depth` — order book (existing)

**Architecture changes**:
- Removed: `TechnicalSignalGenerator`, `MTFSignalGenerator`, `MomentumBreakoutGenerator`, all SMC imports
- Kept: `StrategyConfig`, `SimplePaperTrader` (position management, exit logic, DB logging), `determine_htf_trend()` (standalone), `calculate_rsi()` (standalone, for RSI exit), SL cooldown/daily max SL
- Added: `FundingSentimentGenerator`, `VolatilitySqueezeGenerator`, `TakerFlowGenerator`
- `_fetch_candles()` → `_fetch_market_data()`: parallel fetch candles + derivatives + orderbook
- 4h candle fetch added for squeeze strategy
- `get_derivatives_data()` in MarketDataCollector: parallel fetch all 8 futures endpoints via asyncio.gather

**Rate limits**: 8 derivatives API calls per symbol per cycle x 3 symbols x 1/min = 24 calls/min (Binance allows 1200/min).

**Dashboard**: Updated strategy selector, badge mappings, and comparison table for new strategy names. Version bumped to v5.0.

---

## v4.0 — Multi-Strategy Architecture (2026-03-06)

**Goal**: Compare multiple signal generation strategies side-by-side with independent capital allocation and tracking. Determines which approach works best in different market conditions.

**3 Strategies**:
1. **Agreement Classic** (1m+1h) — existing strategy, unchanged logic. MACD+SMC+SMA must agree on 1m, filtered by 1h HTF trend.
2. **Agreement MTF** (1m+5m+30m+1h) — same agreement logic on 1m, plus 5m/30m confluence scoring. 5m confirms: +1.0 pts, disagrees: -0.5 pts. 30m confirms: +1.5 pts, disagrees: -1.0 pts. MTF disagreement (bonus ≤ -1.5) downgrades to hold. Strong confirmation (bonus ≥ 2.0) upgrades signal strength.
3. **Momentum/Breakout** (1m+1h) — new strategy. Breakout above 20-bar high (2.0 pts) + volume spike ≥2x avg (2.0 pts) + RSI 40-70 (1.0 pt) + MACD rising (1.5 pts) + SMA20>SMA50 (1.5 pts). Threshold ≥5.0 for entry, ≥7.0 for strong. Tighter exits: SL 0.8%, TP 1.8%, max hold 2h.

**Architecture changes**:
- `StrategyConfig` dataclass holds per-strategy parameters (SL, TP, trailing, max hold, capital)
- `MTFSignalGenerator` inherits from `TechnicalSignalGenerator`, adds 5m/30m confluence
- `MomentumBreakoutGenerator` inherits from `TechnicalSignalGenerator`, overrides signal logic
- `SimplePaperTrader` accepts list of strategies, tracks positions/PnL/capital independently per strategy
- Positions keyed by `strategy_name:symbol` (same symbol can have positions from different strategies)
- Candles fetched once per symbol, shared across all strategies (efficient)
- CLI `--strategies` flag to select which strategies to run

**Capital allocation**: Total capital split equally across selected strategies ($333.33 each with $1000).

**Dashboard changes**:
- Strategy selector dropdown: view all or filter by individual strategy
- Strategy comparison table when viewing "All Strategies"
- All existing views (trades, signals, P&L) filtered by selected strategy

**DB compatibility**: No schema changes needed. `strategy_name` and `source` fields already exist. Legacy "paper_technical" trades map to "agreement_classic".

---

## v3.6 — Remove RSI reversal entries (2026-03-06)

**Problem**: RSI reversal entries (buy when RSI <25 + rising) had **25% win rate** and lost **$65.24** over 8 trades. Despite 3 rounds of tightening (v3.4, v3.5), the strategy kept catching falling knives. RSI extremes on 1m candles indicate strong trend momentum, not imminent reversal.

Meanwhile, agreement-based entries (SMC+Tech aligned) had **53% win rate** and made **+$138.57** over 17 trades.

**Changes**:
- **Removed RSI reversal as entry trigger** entirely (both oversold buy and overbought sell)
- **Kept RSI as VETO** — oversold still blocks sells, overbought still blocks buys (this part works)
- Only entry path is now agreement-check: MACD + SMC + SMA must all align

**History**: RSI reversal was introduced in v3.0 (commit 93d4560, Mar 1) alongside the RSI VETO fix. The VETO was the right fix; the reversal entry was a bonus that never worked on 1m timeframe.

---

## v3.5 — Tighten entry filters across all buy paths (2026-03-06)

**Problem**: Worst day since launch — 5L/2W, -$53.12. Market dropped (BTC -2.3%, ETH -1.8%) and the system kept entering longs through two leaky paths:

1. **RSI reversal still catching knives**: Trade #5 (ETH -$23.75) — RSI=24 oversold reversal fired because SMA20>SMA50=True (v3.4 check passed). But price was BELOW SMA20 and MACD was deeply negative (-1.72). The reversal bought into active selling.
2. **Agreement-check ignoring SMA cross**: Trade #6 (BTC -$24.94) — MACD+SMC agreed bullish, but SMA20 had already crossed below SMA50 (short-term trend turned bearish). Agreement path had no SMA cross check.

**Changes**:
- **RSI reversal buy now requires ALL THREE**: SMA20>SMA50, price above SMA20, and positive MACD histogram. Previously only checked SMA20>SMA50. Blocks entries where price is falling through the moving average.
- **Agreement-check buy now requires SMA20>SMA50**: Prevents entering longs when the 1m short-term trend has crossed bearish, even if MACD+SMC agree.
- **Better diagnostics**: Falling knife block now reports which specific conditions failed (SMA20<SMA50, price<SMA20, MACD negative).

**Analysis**: Of the 16 RSI oversold signals that passed v3.4 yesterday, most had price below SMA20 and/or negative MACD. These new filters would have blocked all the losing RSI entries while still allowing legitimate reversals where price is above the short-term average with positive momentum.

---

## v3.4 — Signal quality + stop loss protection (2026-03-05)

**Problem**: 3-trade losing streak at end of day (-$88). Root cause analysis:
- 2 of 3 losses were RSI oversold reversal buys during a pullback. RSI=16 and RSI=8 triggered buy signals even though MACD, SMA20, SMA50 all said bearish. The RSI override bypassed the agreement check.
- 1 loss had MACD histogram at 0.02 (near-zero) — no real momentum behind the signal.
- System immediately re-entered after each stop loss, compounding losses.

**Signal quality fixes**:
- RSI oversold reversal buy now **requires 1m SMA20 > SMA50**. Won't catch falling knives when short-term trend is bearish.
- MACD histogram minimum threshold: buy signals require meaningfully positive MACD (>0.001% of price), sell signals require meaningfully negative. Filters out zero-momentum entries.

**Risk management fixes**:
- **Stop loss cooldown**: 60-minute block on re-entry after a stop loss on the same symbol.
- **Max 2 stop losses per symbol per day**: After 2 SL hits, stop trading that symbol for the rest of the day.

**Analysis**: RSI reversal wins (RSI=27, RSI=20) happened when 1m SMA20>SMA50 (short-term trend was up). RSI reversal losses (RSI=16, RSI=8) happened when SMA20<SMA50 (short-term trend was down). Same signal type, opposite context — the SMA filter distinguishes them.

---

## v3.3 — Fix dashboard accuracy evaluation (2026-03-04)

**Problem**: Dashboard evaluated signal accuracy by comparing signal entry price to **current price** at render time. This meant all signals appeared wrong during a dip, and all appeared correct during a rally — regardless of actual signal quality.

**Changes**:
- Signal accuracy now evaluated against price **10 minutes after** signal was generated
- Fetches historical 1m candles from Binance to look up the actual price at evaluation time
- Signals younger than 10 minutes shown as "Pending"
- Same fix applied to AI insights accuracy calculation fed to Gemini
- Added `fetch_price_at_time()` and `lookup_price_after()` helper functions

**Impact**: Accuracy metric now reflects actual short-term predictive quality of signals, independent of current market direction.

---

## v3.2 — Sell signals require bearish HTF (2026-03-04)

**Problem**: Sell signals had 0% accuracy (0/54) because RSI extreme overbought reversal sells fired constantly on 1m candles during a 5-6% rally. 49 of 54 sells had HTF=neutral, meaning no trend confirmation.

**Changes**:
- Sells now **only allowed when HTF (1h) is bearish**. Previously allowed in neutral too.
- Buys still allowed in bullish and neutral (with -10% confidence penalty in neutral).

**Rationale**: The system is intentionally long-biased. In crypto, strong uptrends produce frequent RSI overbought readings on 1m that don't indicate reversals. Requiring bearish 1h confirmation prevents counter-trend sells.

**Expected impact**: Far fewer sell signals, but higher quality. Buy signals unaffected (were already 100% accurate in this session).

---

## v3.1 — Fix RSI exit cutting winners short (2026-03-04)

**Problem**: RSI profit-taking exit fired at 0.5% profit with RSI > 68. On 1m candles, RSI crosses 68 constantly. Both winning trades exited at +0.50% and +0.67%, never reaching the designed 2.5% take profit. Realized R:R was ~1:2 inverted (losses 2x wins).

**Changes**:
- RSI exit min profit: 0.5% -> 1.5% (trailing stop activation level)
- RSI exit threshold: 68/32 -> 80/20

**Result**: Next trade hit the **actual take profit** at +2.88% (+$57.47). Single trade covered all previous losses. R:R now working as designed (2.08:1).

---

## v3.0 — Multi-timeframe signal generation (2026-03-04)

**Problem**: Signal accuracy was 0% (0/20 in v2.x). Root causes:
- System used 1h candles but polled every 60s -> same stale signal repeated 60x/hour
- `smc_bullish` fallback was too loose (market_trend alone triggered buys)
- Combined score threshold of 3.5 was too easy to meet
- BUY trades were 0W/4L because lagging 1h indicators couldn't detect reversals

**Changes**:
- **1h candles** -> hard trend filter only (HTF: bullish/bearish/neutral via SMA20/SMA50)
- **1m candles** -> entry signals, SMC analysis, technical scoring (matches 60s poll)
- `smc_bullish`/`smc_bearish`: removed loose `market_trend` fallback, require actual confluence direction
- Combined score thresholds raised: strong 5.0->6.0, normal 3.5->4.5
- Fixed `smc_result` NameError (was undefined when SMC analysis failed)
- Signal timeframe saved as "1m" instead of "1h"
- Dashboard version bumped to v3.0, accuracy label updated

**Result (18h)**:
- Hold rate: 90.2% (system is selective)
- Buy accuracy: 44/44 (100%) — all buys correct in rallying market
- 6 trades: 3W/3L (50%), PnL: +$30.97 (+3.10%)
- Market (buy-and-hold): BTC +5.75%, ETH +6.24%
- System underperformed buy-and-hold on return, but used only 20% capital per trade

---

## v2.5 and earlier

Pre-multi-timeframe. Signal accuracy ~0%. See git history for details.
