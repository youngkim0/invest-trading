# Paper Trader Architecture

## Quick Reference

| What | Where |
|------|-------|
| **Paper trader (main bot)** | `scripts/paper_trade_simple.py` |
| **Dashboard** | `dashboard/app.py` (deployed on Streamlit Cloud) |
| **Signal generation** | `TechnicalSignalGenerator` class in `paper_trade_simple.py` |
| **SMC analysis** | `data/features/smc/detector.py`, `confluence.py`, `zones.py` |
| **Market data fetcher** | `data/collectors/market_data.py` |
| **Database client** | `data/storage/supabase_client.py` |
| **GCP deployment** | `deploy/setup-vm.sh`, `deploy/paper-trader.service` |
| **Config / settings** | `config/settings.py`, `.env` |
| **Changelog** | `CHANGELOG.md` |

## System Architecture

```
Binance Public API
    │
    ├── 1h candles ──► determine_htf_trend() ──► bullish / bearish / neutral
    │
    └── 1m candles ──► generate_signal()
                          ├── RSI, MACD, SMA (technical scoring)
                          ├── SMC detector (order blocks, FVGs, liquidity)
                          ├── Confluence engine (scoring)
                          └── Agreement check + HTF hard filter
                                │
                                ▼
                          Signal: buy / sell / hold
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              _save_signal  _check_entry  _check_exit
              (Supabase)    (open pos)    (SL/TP/trailing)
                                │
                                ▼
                          Supabase DB
                    ┌───────────┼───────────┐
                    │           │           │
                 signals    trade_logs   performance_snapshots
                    │
                    ▼
              Streamlit Dashboard (app.py)
              + Gemini AI insights
```

## Deployment

### GCP VM (paper trader bot)
- **Instance**: `paper-trader` in `asia-northeast3-a` (e2-micro, 1GB RAM)
- **User**: `paper` (system user, no login shell)
- **App dir**: `/opt/paper-trader`
- **Service**: `paper-trader.service` (systemd, auto-restart)
- **Logs**: `sudo journalctl -u paper-trader -f`

**Deploy new code:**
```bash
gcloud compute ssh paper-trader --zone=asia-northeast3-a \
  --command="cd /opt/paper-trader && sudo -u paper git pull && sudo systemctl restart paper-trader"
```

### Streamlit Cloud (dashboard)
- Auto-deploys from `main` branch on push
- Entry point: `dashboard/app.py`
- Config: `.streamlit/config.toml`

### Supabase (database)
- Cloud-hosted PostgreSQL
- Tables: `signals`, `trade_logs`, `performance_snapshots`, `ohlcv`
- Auth: anon key (dashboard read), service role key (bot write)

## Signal Generation Pipeline

### Entry signal flow (`generate_signal()`)
1. Calculate technical indicators on **1m candles**: RSI(14), MACD(12,26,9), SMA(20,50), momentum
2. Run **SMC analysis**: order blocks, FVGs, liquidity sweeps, market structure
3. Score confluence (min 0.55 to count)
4. **Technical direction**: buy score vs sell score (threshold: 1.5)
5. **Agreement check**: SMC direction must match technical direction
6. RSI extremes can override (< 25 rising = buy, > 75 falling = sell)
7. **Combined score thresholds**: buy >= 4.5, strong_buy >= 6.0
8. **HTF hard filter** (1h candles):
   - Bearish HTF → block buys
   - Non-bearish HTF → block sells (v3.2)
   - Neutral HTF + buy → confidence -10%

### Exit logic (`_check_exit()`, priority order)
1. **Liquidation**: forced exit at max loss
2. **Take profit**: 2.5% price move
3. **Trailing stop**: activates at 1.5%, trails 0.8% behind peak
4. **Stop loss**: 1.2% against
5. **Stale position**: > 4h with < 0.5% profit
6. **RSI profit taking**: RSI > 80 (long) or < 20 (short), requires 1.5% profit (v3.1)
7. **Signal exit**: opposite signal with high confidence

### Position sizing
- 20% of capital per position (margin)
- 10x leverage
- Max 1 position per symbol
- R:R ratio: 2.08:1 (1.2% SL / 2.5% TP)
- Break-even win rate: 32%

## Key Classes

### `TechnicalSignalGenerator` (paper_trade_simple.py)
- `calculate_rsi(prices)` → RSI with momentum
- `calculate_macd(prices)` → MACD with histogram
- `calculate_sma_crossover(prices)` → SMA20/50 with price position
- `calculate_price_momentum(prices)` → 5/10 bar momentum
- `determine_htf_trend(df_1h)` → bullish/bearish/neutral from 1h SMA20/50
- `generate_signal(df, htf_trend)` → complete signal with reasoning

### `SimplePaperTrader` (paper_trade_simple.py)
- `run()` → main loop, polls every 60s
- `_process_symbol(symbol, collector)` → fetch data, generate signal, trade
- `_check_entry()` → open position if signal + confidence >= 0.65
- `_check_exit()` → 7-level exit priority chain
- `_save_signal()` → log to Supabase signals table
- `_save_performance_snapshot()` → log equity/metrics

### `SMCDetector` (data/features/smc/detector.py)
- `analyze(df)` → returns order blocks, FVGs, liquidity sweeps, market structure

### `ConfluenceEngine` (data/features/smc/confluence.py)
- `analyze(price, order_blocks, fvgs, sweeps, channels, structure, atr)` → confluence score + direction

### Supabase Repositories (data/storage/supabase_client.py)
- `SignalRepository` → table: `signals`
- `TradeLogRepository` → table: `trade_logs`
- `PerformanceRepository` → table: `performance_snapshots`
- `OHLCVRepository` → table: `ohlcv`

## Dashboard (dashboard/app.py)

### Key functions
- `fetch_signals(limit)` → latest signals from Supabase
- `fetch_trades(limit)` → latest trades from Supabase
- `fetch_klines(symbol, interval, limit)` → live candles from Binance
- `generate_ai_insights(signals, trades, prices, api_key)` → Gemini analysis
- `main()` → Streamlit layout with tabs

### Signal accuracy evaluation
- Compares each signal's entry price to **current price** at dashboard render time
- Not a fixed time-window evaluation (known limitation)
- Buy correct if price went up > 0.1%, sell correct if down > 0.1%
- Signals < 10 min old shown as "Pending"

## File Tree (key files only)

```
invest/
├── ARCHITECTURE.md          ← this file
├── CHANGELOG.md             ← version history with results
├── .env                     ← secrets (not in git)
├── scripts/
│   └── paper_trade_simple.py  ← MAIN BOT (TechnicalSignalGenerator + SimplePaperTrader)
├── dashboard/
│   └── app.py               ← Streamlit dashboard
├── data/
│   ├── collectors/
│   │   └── market_data.py   ← Binance API wrapper (klines, ticker)
│   ├── features/
│   │   └── smc/
│   │       ├── detector.py  ← SMC pattern detection
│   │       ├── confluence.py ← confluence scoring
│   │       └── zones.py     ← data structures (OrderBlock, FVG, etc.)
│   └── storage/
│       ├── supabase_client.py ← DB repositories
│       └── models.py        ← SQLAlchemy models
├── config/
│   ├── settings.py          ← Pydantic settings from .env
│   └── strategies.py        ← strategy enums, timeframes
├── deploy/
│   ├── setup-vm.sh          ← GCP VM provisioning
│   ├── paper-trader.service ← systemd unit file
│   └── requirements-paper.txt
└── supabase/
    ├── schema.sql           ← table definitions
    └── config.toml          ← local dev settings
```
