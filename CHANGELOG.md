# Paper Trader Changelog

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
