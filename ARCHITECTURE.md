# Paper Trader Architecture

## Quick Reference

| What | Where |
|------|-------|
| **Paper trader (main bot)** | `scripts/paper_trade_simple.py` |
| **Dashboard** | `dashboard/app.py` (deployed on Streamlit Cloud) |
| **Market data fetcher** | `data/collectors/market_data.py` |
| **Database client** | `data/storage/supabase_client.py` |
| **GCP deployment** | `deploy/setup-vm.sh`, `deploy/paper-trader.service` |
| **Config / settings** | `config/settings.py`, `.env` |
| **Changelog** | `CHANGELOG.md` |

## System Architecture

```
Binance Public API
    │
    ├── 1h candles ──► determine_htf_trend() ──► {direction, strength, slope}
    │
    ├── 15m candles ──► TrendBreakoutGenerator (10-bar breakout + volume)
    │
    ├── 5m candles ──► OIMomentumGenerator (RSI zone + OI + ATR distance)
    │                  calculate_atr(5m) ──► dynamic SL/TP
    │
    ├── 1m candles ──► FundingMeanReversionGenerator (funding + OI + price check)
    │
    └── Futures API ──► get_derivatives_data()
         (funding_rate, oi_history, premium_index, taker ratios, L/S ratio)
                              │
                              ▼
                    3 boolean conditions per strategy
                    ALL must be true → buy/sell signal
                              │
                    ┌─────────┼───────────┐
                    ▼         ▼           ▼
              _save_signal  _check_entry  _check_exit
              (Supabase)    (risk sizing) (ATR-based SL/TP)
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

## Strategy Overview (v6.0)

### Funding Reversion (LOW freq, 0-2/day)
- **Thesis**: Extreme funding rates precede reversals (strongest academic evidence)
- **Conditions**: Extreme funding + OI rising + price not already reversed
- **ATR timeframe**: 1h | SL: 2.0x ATR | TP: 4.0x ATR | Risk: 2%/trade | Max hold: 12h

### Trend Breakout (MEDIUM freq, 2-5/day)
- **Thesis**: The 1h trend was our only proven edge — trade breakouts in its direction
- **Conditions**: HTF trend strength > 0.3 + 15m breakout + volume confirmation
- **ATR timeframe**: 15m | SL: 1.5x ATR | TP: 3.0x ATR | Risk: 2%/trade | Max hold: 6h

### OI Momentum (HIGH freq, 3-8/day)
- **Thesis**: OI rising + price momentum = new money entering in one direction
- **Conditions**: RSI momentum zone + OI rising > 1% + price near SMA20
- **ATR timeframe**: 5m | SL: 1.5x ATR | TP: 2.5x ATR | Risk: 1.5%/trade | Max hold: 3h

## Key Design Decisions (v6.0)

- **ATR-based stops**: SL/TP computed at entry as multiples of ATR, stored on position as percentages. Automatically adapts per asset (XRP gets wider stops than BTC) and per market condition.
- **Risk-based sizing**: `calculate_position_size()` risks 1.5-2% of capital per trade. Position size = risk_amount / sl_distance. Capped at 30% margin.
- **Boolean conditions**: Each strategy checks exactly 3 binary conditions. ALL must be true. No additive scoring = fewer parameters = less overfitting.
- **HTF returns strength**: `determine_htf_trend()` returns `{direction, strength, slope}`. Strength modulates confidence via `apply_htf_adjustment()` instead of hard-blocking counter-trend trades.

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

## Exit Logic (`_check_exit()`, priority order)
1. **Liquidation**: forced exit at max loss
2. **Take profit**: ATR-based TP (stored as pct on position at entry)
3. **Trailing stop**: activates at trailing_act_pct, trails trailing_dist_pct behind peak
4. **Stop loss**: ATR-based SL (stored as pct on position at entry)
5. **Stale position**: > max_position_hours with < 0.5% profit
6. **RSI profit taking**: RSI > 80 (long) or < 20 (short), requires trailing_act_pct profit
7. **Signal exit**: opposite signal with high confidence

## Key Functions

### Standalone Functions (paper_trade_simple.py)
- `calculate_rsi(prices, period)` → RSI with momentum detection
- `calculate_atr(df, period)` → Average True Range value
- `determine_htf_trend(df_1h)` → `{direction, strength, slope}` from 1h SMA20/50
- `hold_signal(reasoning, htf_trend)` → standard hold signal dict
- `apply_htf_adjustment(signal_type, confidence, htf_trend, reasons)` → adjusted confidence
- `calculate_position_size(capital, risk_pct, sl_distance_pct, leverage, price)` → (quantity, margin)

### Strategy Generators
- `FundingMeanReversionGenerator.generate_signal(df_1m, htf_trend, derivatives)` → signal dict
- `TrendBreakoutGenerator.generate_signal(df_15m, htf_trend)` → signal dict
- `OIMomentumGenerator.generate_signal(df_5m, htf_trend, derivatives, atr_5m)` → signal dict

### `SimplePaperTrader` (paper_trade_simple.py)
- `start()` → main loop, polls every 60s
- `_fetch_market_data(symbol, collector)` → 1m, 5m, 15m, 1h candles + ticker + derivatives
- `_process_strategy_symbol(strategy, symbol, market_data)` → compute ATR, generate signal, trade
- `_check_entry()` → open position if signal + confidence >= 0.65, risk-based sizing
- `_check_exit()` → 7-level exit priority chain using position-stored ATR params
- `_open_position()` → compute ATR-based SL/TP, risk-based sizing, store params on position
- `_save_signal()` → log to Supabase signals table
- `_save_performance_snapshots()` → log equity/metrics per strategy

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

## File Tree (key files only)

```
invest/
├── ARCHITECTURE.md          ← this file
├── CHANGELOG.md             ← version history with results
├── .env                     ← secrets (not in git)
├── scripts/
│   └── paper_trade_simple.py  ← MAIN BOT (3 strategy generators + SimplePaperTrader)
├── dashboard/
│   └── app.py               ← Streamlit dashboard
├── data/
│   ├── collectors/
│   │   └── market_data.py   ← Binance API wrapper (klines, ticker, derivatives)
│   ├── features/
│   │   └── smc/             ← legacy SMC analysis (not used in v6.0)
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
