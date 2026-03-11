# Paper Trader Changelog

## v6.0.2 — Fix Trailing Stops, Signal Noise, ATR Floor (2026-03-11)

**Problem**: v6.0.1 ran 12h (16 trades, 8W/8L, -$18.57, 50% WR). Despite 50% WR, system lost money because trailing stops clipped all winners: 6/7 wins exited via trailing at 36-60% of designed TP. Actual R:R was 0.83:1 instead of designed 2:1. Only 1/16 trades hit full TP. Signal DB flooded with ~6500 hold signals/12h. oi_momentum's 5m ATR produced SL as tight as 0.33%, clipped by noise.

**Fixes**:
- **Disable trailing stops**: Added `trailing_enabled` flag to StrategyConfig (default False). Trailing update + trailing exit gated on flag. Backwards-compatible (existing positions default True). Pure SL/TP exits let winners run to designed 2:1 R:R. Can re-enable later with better tuning.
- **Only save actionable signals**: Skip `_save_signal()` for hold signals. Holds still logged to debug for journalctl. Eliminates ~6500 hold signals/12h noise in DB. Dashboard accuracy stats become meaningful.
- **ATR floor for oi_momentum**: Added `min_sl_pct` to StrategyConfig. When ATR-computed SL < floor, all stops scale proportionally (preserves R:R). Set 0.5% floor for oi_momentum. Prevents ultra-tight 0.33% stops from noise-clipping.
- **Strategy-aware stale exit**: `_is_stale_position` now uses `tp_pct * 0.25` instead of fixed 0.005. Adapts per strategy (funding_reversion needs different threshold than oi_momentum).

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
