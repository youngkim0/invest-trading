# Paper Trader Changelog

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
