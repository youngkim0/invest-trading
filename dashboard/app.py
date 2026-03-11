"""Simplified AI Trading Dashboard - Focus on Signals, Trades, Performance

Updated: 2026-02-27 - Added portfolio value banner at top
"""

import os
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Gemini API via REST (no package needed)
GEMINI_AVAILABLE = True  # Always available via REST API

# Korea Standard Time (UTC+9) - Define fallbacks first, then try to upgrade
KST = timezone(timedelta(hours=9))
UTC = timezone.utc
try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
    UTC = ZoneInfo("UTC")
except Exception:
    pass  # Keep the fallback values defined above
NEW_SYSTEM_DATE = "2026-03-11T07:27:00Z"


def to_kst(timestamp_str: str) -> str:
    """Convert UTC timestamp string to KST formatted string."""
    if not timestamp_str:
        return ""
    try:
        # Parse the timestamp (handles both with and without timezone)
        if '+' in timestamp_str or 'Z' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=UTC)
        # Convert to KST
        dt_kst = dt.astimezone(KST)
        return dt_kst.strftime('%m/%d %H:%M')
    except:
        return timestamp_str[:16].replace('T', ' ')

# Load environment variables
project_root = Path(__file__).parent.parent
try:
    if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
        os.environ['SUPABASE_URL'] = st.secrets['SUPABASE_URL']
        os.environ['SUPABASE_ANON_KEY'] = st.secrets['SUPABASE_ANON_KEY']
    else:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")
except Exception:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

# Page config
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .big-metric { font-size: 28px; font-weight: bold; }
    .positive { color: #00ff88; }
    .negative { color: #ff4444; }
    .neutral { color: #888; }
    .signal-buy { background: #1a4d1a; padding: 5px 10px; border-radius: 5px; }
    .signal-sell { background: #4d1a1a; padding: 5px 10px; border-radius: 5px; }
    .signal-hold { background: #4d4d1a; padding: 5px 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)


def get_supabase():
    """Get Supabase client."""
    from supabase import create_client
    if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
        url = st.secrets['SUPABASE_URL']
        key = st.secrets['SUPABASE_ANON_KEY']
    else:
        url = os.environ.get('SUPABASE_URL', '')
        key = os.environ.get('SUPABASE_ANON_KEY', '')
    if not url or not key:
        return None
    return create_client(url, key)


@st.cache_data(ttl=30)
def fetch_signals(limit: int = 100):
    """Fetch recent signals."""
    try:
        client = get_supabase()
        if not client:
            return []
        result = client.table('signals').select('*').gte('timestamp', NEW_SYSTEM_DATE).order('timestamp', desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        st.error(f"Error fetching signals: {e}")
        return []


@st.cache_data(ttl=30)
def fetch_trades(limit: int = 100):
    """Fetch trades."""
    try:
        client = get_supabase()
        if not client:
            return []
        result = client.table('trade_logs').select('*').gte('entry_time', NEW_SYSTEM_DATE).order('entry_time', desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        st.error(f"Error fetching trades: {e}")
        return []


# Strategy name → signal source mapping
STRATEGY_SOURCE_MAP = {
    "funding_reversion": "funding",
    "trend_breakout": "breakout",
    "oi_momentum": "oi",
    # Legacy strategies
    "funding_sentiment": "funding",
    "volatility_squeeze": "squeeze",
    "taker_flow": "taker",
    "agreement_classic": "agreement",
    "agreement_mtf": "agreement_mtf",
    "momentum": "momentum",
    "paper_technical": "technical",
}

# All known strategy names (for filtering)
ALL_STRATEGY_NAMES = [
    "funding_reversion", "trend_breakout", "oi_momentum",
    "funding_sentiment", "volatility_squeeze", "taker_flow",
    "agreement_classic", "agreement_mtf", "momentum", "paper_technical",
]


def filter_trades_by_strategy(trades, strategy_name):
    """Filter trades by strategy_name. None = all."""
    if strategy_name is None:
        return trades
    # Include legacy "paper_technical" when viewing "agreement_classic"
    names = [strategy_name]
    if strategy_name == "agreement_classic":
        names.append("paper_technical")
    return [t for t in trades if t.get("strategy_name", "paper_technical") in names]


def filter_signals_by_strategy(signals, strategy_name):
    """Filter signals by source matching strategy. None = all."""
    if strategy_name is None:
        return signals
    source = STRATEGY_SOURCE_MAP.get(strategy_name, strategy_name)
    # Include legacy "technical" when viewing "agreement_classic"
    sources = [source]
    if strategy_name == "agreement_classic":
        sources.append("technical")
    return [s for s in signals if s.get("source", "technical") in sources]


@st.cache_data(ttl=10)
def fetch_current_prices():
    """Fetch current BTC, ETH, and XRP prices from multiple sources."""
    prices = {}
    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]

    # Try multiple Binance endpoints
    binance_urls = [
        "https://api.binance.com/api/v3/ticker/price",
        "https://api.binance.us/api/v3/ticker/price",
        "https://api1.binance.com/api/v3/ticker/price",
    ]

    for base_url in binance_urls:
        if len(prices) == len(symbols):
            break
        for symbol in symbols:
            if symbol in prices:
                continue
            try:
                resp = requests.get(f"{base_url}?symbol={symbol}", timeout=5)
                if resp.status_code == 200:
                    prices[symbol] = float(resp.json()["price"])
            except:
                continue

    # Fallback to CoinGecko if any symbol missing
    if len(prices) < len(symbols):
        try:
            coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "XRPUSDT": "ripple"}
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,ripple&vs_currencies=usd",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                for sym, coin_id in coin_map.items():
                    if sym not in prices and coin_id in data:
                        prices[sym] = float(data[coin_id]["usd"])
        except:
            pass

    return prices


@st.cache_data(ttl=60)
def fetch_klines(symbol: str, interval: str = "1h", limit: int = 48):
    """Fetch candlestick data from multiple sources."""
    # Try multiple Binance endpoints
    urls = [
        f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                # Convert timestamp to datetime, add 9 hours for KST
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=9)
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except:
            continue

    # Fallback to CoinGecko OHLC
    try:
        coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "XRPUSDT": "ripple"}
        coin_id = coin_map.get(symbol, "bitcoin")
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=2"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=9)
            df['volume'] = 0.0
            return df
    except:
        pass

    return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_price_at_time(symbol: str, limit: int = 500):
    """Fetch recent 1m candles and return a dict of minute_timestamp -> close price.

    Used to evaluate signal accuracy by looking up price N minutes after signal.
    """
    price_map = {}
    urls = [
        f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                for candle in resp.json():
                    # candle[0] = open time in ms, candle[4] = close price
                    ts = int(candle[0]) // 60000  # minute-level key
                    price_map[ts] = float(candle[4])
                return price_map
        except:
            continue
    return price_map


def lookup_price_after(price_map: dict, signal_dt: datetime, minutes_after: int = 10):
    """Look up price N minutes after a signal timestamp.

    Returns (price, found) tuple. If exact minute not found, tries nearby minutes.
    """
    sig_epoch_min = int(signal_dt.timestamp()) // 60
    target_min = sig_epoch_min + minutes_after

    # Try exact minute, then +/- 1 minute
    for offset in [0, 1, -1, 2, -2]:
        price = price_map.get(target_min + offset)
        if price is not None:
            return price, True
    return 0, False


def get_gemini_api_key():
    """Get Gemini API key from secrets or environment."""
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        return st.secrets['GEMINI_API_KEY']
    return os.environ.get('GEMINI_API_KEY', '')


def call_gemini_api(prompt: str, api_key: str) -> str:
    """Call Gemini API via REST (no package needed)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
        }
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)

    if response.status_code == 200:
        result = response.json()
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    else:
        raise Exception(f"Gemini API error: {response.status_code} - {response.text[:200]}")


def get_analysis_cache_key(trades, signals):
    """Generate cache key based on data hash."""
    data_str = json.dumps({
        'trades': [t.get('position_id', '') for t in trades[-20:]],
        'signals': len(signals),
        'hour_bucket': datetime.now(timezone.utc).hour // 4  # Changes every 4 hours
    }, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


@st.cache_data(ttl=14400)  # Cache for 4 hours (14400 seconds)
def generate_ai_insights(trades_json: str, signals_json: str, prices_json: str, cache_key: str):
    """Generate AI insights about trading performance using Gemini REST API."""
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            return None

        trades = json.loads(trades_json)
        signals = json.loads(signals_json)
        prices = json.loads(prices_json)

        # Prepare analysis data
        closed_trades = [t for t in trades if t.get('exit_time')]
        open_trades = [t for t in trades if not t.get('exit_time')]

        # Calculate signal accuracy using price 10 min after signal
        signal_analysis = []
        eval_minutes = 10
        ai_price_maps = {}
        for sym in ["BTCUSDT", "ETHUSDT", "XRPUSDT"]:
            ai_price_maps[sym] = fetch_price_at_time(sym, 500)

        for sig in signals[-50:]:  # Last 50 signals
            symbol = sig.get('symbol', '')
            signal_type = sig.get('signal_type', '')
            sig_price = float(sig.get('entry_price', 0) or 0)

            if sig_price > 0 and signal_type and signal_type != 'hold':
                # Parse signal timestamp and look up price 10 min later
                sig_time = sig.get('timestamp', '')
                eval_price = 0
                sig_dt = None
                if sig_time:
                    try:
                        if '+' in sig_time or 'Z' in sig_time:
                            sig_dt = datetime.fromisoformat(sig_time.replace('Z', '+00:00'))
                        else:
                            sig_dt = datetime.fromisoformat(sig_time).replace(tzinfo=timezone.utc)
                    except:
                        pass

                if sig_dt:
                    age_minutes = (datetime.now(timezone.utc) - sig_dt).total_seconds() / 60
                    if age_minutes >= eval_minutes:
                        eval_price, found = lookup_price_after(
                            ai_price_maps.get(symbol, {}), sig_dt, eval_minutes
                        )
                        if not found:
                            eval_price = 0

                if eval_price > 0:
                    price_change = (eval_price - sig_price) / sig_price * 100

                    if 'buy' in signal_type.lower():
                        correct = price_change > 0
                    elif 'sell' in signal_type.lower():
                        correct = price_change < 0
                    else:
                        correct = None

                    signal_analysis.append({
                        'signal': signal_type,
                        'price_at_signal': sig_price,
                        'price_10min_later': eval_price,
                        'price_change_pct': price_change,
                        'correct': correct
                    })

        # Count correct signals
        correct_signals = sum(1 for s in signal_analysis if s.get('correct') is True)
        total_actionable = sum(1 for s in signal_analysis if s.get('correct') is not None)
        signal_accuracy = (correct_signals / total_actionable * 100) if total_actionable > 0 else 0

        # Build prompt
        prompt = f"""You are a trading analyst reviewing an AI trading system's recent performance.
Analyze the following data and provide:
1. **Signal Accuracy Assessment**: Were the signals correct? (Signal accuracy: {signal_accuracy:.1f}% based on {total_actionable} actionable signals)
2. **Trade Performance Review**: Analysis of recent closed trades
3. **Concerns**: Any red flags or issues you see
4. **Suggestions**: Specific improvements for better performance

## Recent Trades (Last 10)
{json.dumps(closed_trades[-10:], indent=2, default=str)}

## Open Positions
{json.dumps(open_trades, indent=2, default=str)}

## Signal Accuracy Details (sample)
{json.dumps(signal_analysis[-10:], indent=2, default=str)}

## Current Prices
BTC: ${prices.get('BTCUSDT', 0):,.2f}
ETH: ${prices.get('ETHUSDT', 0):,.2f}
XRP: ${prices.get('XRPUSDT', 0):,.4f}

Provide your analysis in a clear, concise format using markdown. Be specific and actionable.
Focus on patterns you see in the data, not generic advice.
Keep response under 500 words."""

        analysis_text = call_gemini_api(prompt, api_key)

        return {
            'analysis': analysis_text,
            'signal_accuracy': signal_accuracy,
            'total_signals_analyzed': total_actionable,
            'correct_signals': correct_signals,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {'error': str(e), 'generated_at': datetime.now(timezone.utc).isoformat()}


def main():
    st.title("📈 AI Trading Dashboard")
    kst_now = datetime.now(timezone.utc) + timedelta(hours=9)
    st.caption(f"Last updated: {kst_now.strftime('%Y-%m-%d %H:%M:%S KST')} | v6.0.2 started: Mar 11, 2026 16:27 KST | v6.0.2 (disable trailing, signal filter, ATR floor)")

    # Auto refresh + strategy selector
    col1, col2, col3 = st.columns([2.5, 1.5, 1])
    with col2:
        strategy_options = {
            "All Strategies": None,
            "🔄 Funding Reversion": "funding_reversion",
            "📈 Trend Breakout": "trend_breakout",
            "📊 OI Momentum": "oi_momentum",
        }
        selected_label = st.selectbox("Strategy", list(strategy_options.keys()), key="strategy_select")
        strategy_filter = strategy_options[selected_label]
    with col3:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # Fetch all data, then filter by strategy
    all_signals = fetch_signals(500)
    all_trades = fetch_trades(200)
    prices = fetch_current_prices()

    signals = filter_signals_by_strategy(all_signals, strategy_filter)
    trades = filter_trades_by_strategy(all_trades, strategy_filter)

    # ============================================
    # STRATEGY COMPARISON (only when "All Strategies" selected)
    # ============================================
    if strategy_filter is None:
        st.markdown("---")
        st.header("⚖️ Strategy Comparison")

        strat_names = ["funding_reversion", "trend_breakout", "oi_momentum"]
        strat_labels = ["Funding Reversion", "Trend Breakout", "OI Momentum"]
        comp_cols = st.columns(len(strat_names))

        for col, sname, slabel in zip(comp_cols, strat_names, strat_labels):
            with col:
                s_trades = filter_trades_by_strategy(all_trades, sname)
                s_closed = [t for t in s_trades if t.get('exit_time')]
                s_open = [t for t in s_trades if not t.get('exit_time')]
                s_wins = [t for t in s_closed if (t.get('net_pnl') or 0) > 0]
                s_pnl = sum(t.get('net_pnl') or 0 for t in s_closed)
                s_wr = len(s_wins) / len(s_closed) * 100 if s_closed else 0

                # Unrealized P&L for open positions
                s_unrealized = 0.0
                for t in s_open:
                    sym = t.get('symbol', '')
                    side = t.get('side', '')
                    ep = float(t.get('entry_price') or 0)
                    qty = float(t.get('quantity') or 0)
                    cp = prices.get(sym, ep)
                    if ep > 0 and qty > 0:
                        if side == 'buy':
                            s_unrealized += (cp - ep) * qty
                        else:
                            s_unrealized += (ep - cp) * qty

                s_total = s_pnl + s_unrealized
                pnl_color = "#00ff88" if s_total >= 0 else "#ff4444"
                st.markdown(f"**{slabel}**")
                st.markdown(f"<span style='color:{pnl_color}; font-size:22px;'>${s_total:+.2f}</span>", unsafe_allow_html=True)
                details = f"{len(s_closed)} closed | {s_wr:.0f}% WR | {len(s_wins)}W/{len(s_closed)-len(s_wins)}L"
                if s_open:
                    ur_color = "#00ff88" if s_unrealized >= 0 else "#ff4444"
                    details += f" | {len(s_open)} open (<span style='color:{ur_color}'>${s_unrealized:+.2f}</span>)"
                st.markdown(details, unsafe_allow_html=True)

    # ============================================
    # PORTFOLIO VALUE BANNER (at the top)
    # ============================================
    # Each strategy gets $1000, total $3000 when viewing all; $1000 per strategy
    if strategy_filter is None:
        STARTING_CAPITAL = 3000.0
    else:
        STARTING_CAPITAL = 1000.0

    # Calculate realized P&L from closed trades
    closed_trades = [t for t in trades if t.get('exit_time')]
    realized_pnl = sum(t.get('net_pnl') or 0 for t in closed_trades)

    # Calculate unrealized P&L from open positions
    open_trades = [t for t in trades if not t.get('exit_time')]
    unrealized_pnl = 0.0
    for t in open_trades:
        symbol = t.get('symbol', '')
        side = t.get('side', '')
        entry_price = float(t.get('entry_price') or 0)
        quantity = float(t.get('quantity') or 0)
        current_price = prices.get(symbol, entry_price)

        if entry_price > 0 and quantity > 0:
            if side == 'buy':
                unrealized_pnl += (current_price - entry_price) * quantity
            else:
                unrealized_pnl += (entry_price - current_price) * quantity

    # Total portfolio value
    portfolio_value = STARTING_CAPITAL + realized_pnl + unrealized_pnl
    total_return = ((portfolio_value - STARTING_CAPITAL) / STARTING_CAPITAL) * 100

    # Display prominent portfolio banner
    st.markdown("---")
    port_col1, port_col2, port_col3, port_col4 = st.columns([2, 1.5, 1.5, 1.5])

    with port_col1:
        if portfolio_value >= STARTING_CAPITAL:
            st.markdown(f"### 💰 Portfolio: <span style='color: #00ff88; font-size: 32px;'>${portfolio_value:,.2f}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"### 💰 Portfolio: <span style='color: #ff4444; font-size: 32px;'>${portfolio_value:,.2f}</span>", unsafe_allow_html=True)

    with port_col2:
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric("Total Return", f"{total_return:+.2f}%", delta=f"${portfolio_value - STARTING_CAPITAL:+,.2f}")

    with port_col3:
        st.metric("Realized", f"${realized_pnl:+,.2f}", delta=f"{len(closed_trades)} trades")

    with port_col4:
        st.metric("Unrealized", f"${unrealized_pnl:+,.2f}", delta=f"{len(open_trades)} open")

    # ============================================
    # SECTION 1: TRADE STATISTICS
    # ============================================
    st.markdown("---")
    st.header("📊 Trade Statistics")

    # Calculate stats (reuse closed_trades and open_trades from above)
    total_trades = len(closed_trades)
    winners = [t for t in closed_trades if (t.get('net_pnl') or 0) > 0]
    losers = [t for t in closed_trades if (t.get('net_pnl') or 0) < 0]

    gross_profit = sum(t.get('net_pnl') or 0 for t in winners)
    gross_loss = abs(sum(t.get('net_pnl') or 0 for t in losers))

    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    avg_win = (gross_profit / len(winners)) if winners else 0
    avg_loss = (gross_loss / len(losers)) if losers else 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{len(winners)}W / {len(losers)}L")

    with col2:
        st.metric("Profit Factor", f"{profit_factor:.2f}")

    with col3:
        st.metric("Avg Win", f"${avg_win:.2f}")

    with col4:
        st.metric("Avg Loss", f"${avg_loss:.2f}")

    # ============================================
    # SECTION 2: OPEN POSITIONS
    # ============================================
    st.markdown("---")
    st.header("🔥 Open Positions")

    if open_trades:
        # Strategy label mapping
        strat_badge = {
            "funding_reversion": "🔄 Funding",
            "trend_breakout": "📈 Breakout",
            "oi_momentum": "📊 OI Mom",
            "funding_sentiment": "💰 Funding(v5)",
            "volatility_squeeze": "📊 Squeeze(v5)",
            "taker_flow": "🌊 Flow(v5)",
            "agreement_classic": "🔵 Classic",
            "agreement_mtf": "🟣 MTF",
            "momentum": "🟠 Momentum",
            "paper_technical": "🔵 Classic",
        }
        for t in open_trades:
            symbol = t.get('symbol', 'N/A')
            side = t.get('side', 'N/A').upper()
            entry_price = float(t.get('entry_price') or 0)
            entry_time_kst = to_kst(t.get('entry_time', ''))
            current_price = prices.get(symbol, entry_price)
            strategy = t.get('strategy_name', 'paper_technical')
            badge = strat_badge.get(strategy, strategy)

            # Calculate P&L
            if side == 'BUY':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            pnl_color = "🟢" if pnl_pct >= 0 else "🔴"
            side_emoji = "📈" if side == "BUY" else "📉"

            col1, col2, col3, col4, col5, col6 = st.columns([1.5, 2, 2, 2, 1.5, 1.5])
            with col1:
                st.markdown(f"**{badge}**")
            with col2:
                st.markdown(f"**{side_emoji} {symbol}** {side}")
            with col3:
                st.markdown(f"Entry: **${entry_price:,.2f}**")
            with col4:
                st.markdown(f"Current: **${current_price:,.2f}**")
            with col5:
                st.markdown(f"{pnl_color} **{pnl_pct:+.2f}%**")
            with col6:
                st.markdown(f"📅 {entry_time_kst}")
    else:
        st.info("No open positions")

    # ============================================
    # SECTION 3: RECENT SIGNALS (Every Minute Decisions)
    # ============================================
    st.markdown("---")
    st.header("⚡ Recent Signals")

    if signals:
        # Pagination controls
        signals_per_page = 30
        total_signals = len(signals)
        total_pages = max(1, (total_signals + signals_per_page - 1) // signals_per_page)

        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 2, 1])
        with nav_col1:
            st.write(f"**{total_signals}** signals")
        with nav_col2:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="signal_page")
        with nav_col3:
            st.write(f"of {total_pages} pages")
        with nav_col4:
            show_holds = st.checkbox("Show holds", value=False, key="show_holds")

        # Filter signals if needed
        if not show_holds:
            filtered_signals = [s for s in signals if s.get('signal_type', 'hold') != 'hold']
        else:
            filtered_signals = signals

        # Recalculate pagination after filter
        total_filtered = len(filtered_signals)
        total_pages = max(1, (total_filtered + signals_per_page - 1) // signals_per_page)
        page = min(page, total_pages)

        # Get signals for current page
        start_idx = (page - 1) * signals_per_page
        end_idx = start_idx + signals_per_page
        page_signals = filtered_signals[start_idx:end_idx]

        # Format display with KST timezone
        display_data = []
        signal_results = {'correct': 0, 'incorrect': 0, 'pending': 0}

        # Fetch historical 1m candles for price-at-time lookups
        eval_minutes = 10  # Evaluate signal accuracy 10 minutes after
        price_maps = {}
        for sym in ["BTCUSDT", "ETHUSDT", "XRPUSDT"]:
            price_maps[sym] = fetch_price_at_time(sym, 500)

        # Filter to actionable signals only (buy/sell) — hold signals are noise
        actionable_signals = [s for s in page_signals if (s.get('signal_type') or 'hold') != 'hold']

        for i, sig in enumerate(actionable_signals):
            signal_type = sig.get('signal_type', 'hold') or 'hold'
            confidence = float(sig.get('confidence') or 0) * 100
            sig_price = float(sig.get('entry_price') or 0)
            symbol = sig.get('symbol', 'N/A')
            timestamp_kst = to_kst(sig.get('timestamp', ''))

            # Signal emoji
            if 'strong_buy' in signal_type:
                signal_emoji = "🟢🟢"
            elif 'buy' in signal_type:
                signal_emoji = "🟢"
            elif 'strong_sell' in signal_type:
                signal_emoji = "🔴🔴"
            elif 'sell' in signal_type:
                signal_emoji = "🔴"
            else:
                signal_emoji = "⚪"

            # Evaluate signal accuracy using price 10 min after signal
            # Recent signals (< 10 min) are "Pending"
            result = "⏳"
            result_text = "Pending"

            if sig_price > 0:
                # Parse signal timestamp
                sig_time = sig.get('timestamp', '')
                sig_dt = None
                is_old_enough = False
                if sig_time:
                    try:
                        if '+' in sig_time or 'Z' in sig_time:
                            sig_dt = datetime.fromisoformat(sig_time.replace('Z', '+00:00'))
                        else:
                            sig_dt = datetime.fromisoformat(sig_time).replace(tzinfo=timezone.utc)
                        age_minutes = (datetime.now(timezone.utc) - sig_dt).total_seconds() / 60
                        is_old_enough = age_minutes >= eval_minutes
                    except:
                        is_old_enough = i >= 10

                if is_old_enough and sig_dt:
                    # Look up price 10 min after signal
                    eval_price, found = lookup_price_after(
                        price_maps.get(symbol, {}), sig_dt, eval_minutes
                    )

                    if found and eval_price > 0:
                        price_change = (eval_price - sig_price) / sig_price * 100

                        if 'buy' in signal_type.lower():
                            is_correct = price_change > 0.1
                        elif 'sell' in signal_type.lower():
                            is_correct = price_change < -0.1
                        else:
                            is_correct = None

                        if is_correct is True:
                            result = "✅"
                            result_text = f"+{abs(price_change):.2f}%"
                            signal_results['correct'] += 1
                        elif is_correct is False:
                            result = "❌"
                            result_text = f"{price_change:+.2f}%"
                            signal_results['incorrect'] += 1
                        else:
                            signal_results['pending'] += 1
                    else:
                        # Price data not available for this time window
                        signal_results['pending'] += 1
                else:
                    signal_results['pending'] += 1
            # Strategy source label
            sig_source = sig.get('source', 'technical')
            source_labels = {
                'funding': '💰 Funding',
                'breakout': '📈 Breakout',
                'oi': '⚡ OI Momentum',
                'squeeze': '📊 Squeeze',
                'taker': '🌊 Flow',
                'agreement': '🔵 Classic',
                'agreement_mtf': '🟣 MTF',
                'momentum': '🟠 Momentum',
                'technical': '🔵 Classic',
            }
            source_label = source_labels.get(sig_source, sig_source)

            display_data.append({
                'Time (KST)': timestamp_kst,
                'Strategy': source_label,
                'Symbol': symbol,
                'Signal': f"{signal_emoji} {signal_type}",
                'Confidence': f"{confidence:.0f}%",
                'Price': f"${sig_price:,.2f}" if sig_price > 0 else "—",
                'Result': f"{result} {result_text}",
            })

        # Show signal accuracy summary
        total_evaluated = signal_results['correct'] + signal_results['incorrect']
        if total_evaluated > 0:
            accuracy = signal_results['correct'] / total_evaluated * 100
            acc_color = "🟢" if accuracy >= 55 else "🔴" if accuracy < 45 else "🟡"
            st.markdown(f"**Signal Accuracy (10min):** {acc_color} {accuracy:.1f}% ({signal_results['correct']}/{total_evaluated} correct) | ⏳ {signal_results['pending']} pending")

        st.dataframe(
            pd.DataFrame(display_data),
            use_container_width=True,
            hide_index=True,
        )

        # Signal diagnostics expander
        with st.expander("🔍 Signal Quality Diagnostics"):
            # Count signal flips (buy→sell or sell→buy)
            flips = 0
            last_direction = {}
            signal_by_symbol = {'BTCUSDT': [], 'ETHUSDT': [], 'XRPUSDT': []}

            for sig in signals:
                symbol = sig.get('symbol', '')
                sig_type = sig.get('signal_type', 'hold')

                if symbol in signal_by_symbol:
                    signal_by_symbol[symbol].append(sig_type)

                # Determine direction
                if 'buy' in sig_type:
                    direction = 'buy'
                elif 'sell' in sig_type:
                    direction = 'sell'
                else:
                    continue

                if symbol in last_direction and last_direction[symbol] != direction:
                    flips += 1
                last_direction[symbol] = direction

            # Calculate metrics
            total_actionable = sum(1 for s in signals if s.get('signal_type', 'hold') != 'hold')
            flip_rate = (flips / total_actionable * 100) if total_actionable > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                flip_color = "🔴" if flip_rate > 40 else "🟡" if flip_rate > 25 else "🟢"
                st.metric("Signal Flip Rate", f"{flip_color} {flip_rate:.1f}%", help="How often signal changes direction (lower is better)")
            with col2:
                st.metric("Direction Changes", f"{flips}", help="Total buy↔sell flips in recent signals")
            with col3:
                hold_count = sum(1 for s in signals if s.get('signal_type') == 'hold')
                hold_pct = hold_count / len(signals) * 100 if signals else 0
                st.metric("Hold Signals", f"{hold_pct:.1f}%", help="Percentage of hold (no action) signals")

            st.markdown("""
            **Why signals may be inaccurate:**
            - 🔴 **High flip rate (>40%)**: System is indecisive, constantly changing direction
            - 🔴 **Low hold rate (<20%)**: System is overtrading, not waiting for clear setups
            - 🔴 **~50% accuracy**: Signals are essentially random

            **Potential fixes:**
            - Increase signal thresholds to reduce noise
            - Add trend confirmation (only trade with trend)
            - Require multiple timeframe agreement
            - Add cooldown between opposite signals
            """)
    else:
        st.info("No signals found")

    # ============================================
    # SECTION 4: TRADE HISTORY (What Actually Happened)
    # ============================================
    st.markdown("---")
    st.header("📋 Trade History (Since Feb 20)")

    if trades:
        # Strategy badge mapping for trades
        trade_strat_badge = {
            "funding_reversion": "🔄 Funding",
            "trend_breakout": "📈 Breakout",
            "oi_momentum": "📊 OI Mom",
            "funding_sentiment": "💰 Funding(v5)",
            "volatility_squeeze": "📊 Squeeze(v5)",
            "taker_flow": "🌊 Flow(v5)",
            "agreement_classic": "🔵 Classic",
            "agreement_mtf": "🟣 MTF",
            "momentum": "🟠 Momentum",
            "paper_technical": "🔵 Classic",
        }
        trade_data = []
        for t in trades:
            symbol = t.get('symbol', 'N/A')
            side = t.get('side', 'N/A').upper()
            entry_price = float(t.get('entry_price') or 0)
            exit_price = float(t.get('exit_price') or 0)
            entry_time_kst = to_kst(t.get('entry_time', ''))
            exit_time = t.get('exit_time', '')
            net_pnl = float(t.get('net_pnl') or 0)
            return_pct = float(t.get('return_pct') or 0)
            strategy = t.get('strategy_name', 'paper_technical')
            badge = trade_strat_badge.get(strategy, strategy)

            # Status
            if exit_time:
                exit_time_kst = to_kst(exit_time)
                status = "✅" if net_pnl > 0 else "❌"
                pnl_display = f"${net_pnl:+.2f} ({return_pct:+.2f}%)"
            else:
                exit_time_kst = "OPEN"
                status = "🔄"
                pnl_display = "—"

            side_emoji = "📈" if side == "BUY" else "📉"

            trade_data.append({
                'Status': status,
                'Strategy': badge,
                'Symbol': f"{side_emoji} {symbol}",
                'Side': side,
                'Entry': f"${entry_price:,.2f}",
                'Exit': f"${exit_price:,.2f}" if exit_price else "—",
                'P&L': pnl_display,
                'Entry (KST)': entry_time_kst,
                'Exit (KST)': exit_time_kst,
            })

        st.dataframe(
            pd.DataFrame(trade_data),
            use_container_width=True,
            hide_index=True,
        )

        # Summary by symbol/side
        st.subheader("📊 Breakdown by Symbol & Side")

        breakdown = {}
        for t in closed_trades:
            key = f"{t.get('symbol', 'N/A')}_{t.get('side', 'N/A').upper()}"
            if key not in breakdown:
                breakdown[key] = {'wins': 0, 'losses': 0, 'pnl': 0}

            pnl = float(t.get('net_pnl') or 0)
            breakdown[key]['pnl'] += pnl
            if pnl > 0:
                breakdown[key]['wins'] += 1
            else:
                breakdown[key]['losses'] += 1

        cols = st.columns(len(breakdown) if breakdown else 1)
        for i, (key, data) in enumerate(breakdown.items()):
            total = data['wins'] + data['losses']
            wr = (data['wins'] / total * 100) if total > 0 else 0
            with cols[i % len(cols)]:
                pnl_color = "🟢" if data['pnl'] >= 0 else "🔴"
                st.markdown(f"""
                **{key}**
                {pnl_color} P&L: ${data['pnl']:+.2f}
                Win Rate: {wr:.0f}% ({data['wins']}W/{data['losses']}L)
                """)
    else:
        st.info("No trades found since Feb 20")

    # ============================================
    # SECTION 5: PRICE CHARTS
    # ============================================
    st.markdown("---")
    st.header("📈 Price Charts")

    # Timeframe selector
    timeframe_options = {
        "1분": ("1m", 60),
        "5분": ("5m", 60),
        "15분": ("15m", 48),
        "1시간": ("1h", 48),
        "4시간": ("4h", 48),
        "1일": ("1d", 30),
    }

    selected_tf = st.selectbox(
        "차트 시간대 선택",
        options=list(timeframe_options.keys()),
        index=3,  # Default to 1시간
        key="timeframe_select"
    )

    interval, limit = timeframe_options[selected_tf]

    chart_symbols = [
        ("BTCUSDT", "Bitcoin (BTC)"),
        ("ETHUSDT", "Ethereum (ETH)"),
        ("XRPUSDT", "Ripple (XRP)"),
    ]
    chart_cols = st.columns(len(chart_symbols))

    for col, (sym, label) in zip(chart_cols, chart_symbols):
        with col:
            st.subheader(label)
            df_chart = fetch_klines(sym, interval, limit)
            if not df_chart.empty:
                fig = go.Figure(data=[go.Candlestick(
                    x=df_chart['timestamp'],
                    open=df_chart['open'],
                    high=df_chart['high'],
                    low=df_chart['low'],
                    close=df_chart['close'],
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                )])
                fig.update_layout(
                    height=400,
                    template="plotly_dark",
                    xaxis_title="Time (KST)",
                    yaxis_title="Price (USDT)",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                current_p = df_chart['close'].iloc[-1]
                prev_p = df_chart['close'].iloc[-2]
                change_p = ((current_p - prev_p) / prev_p) * 100
                color = "🟢" if change_p >= 0 else "🔴"
                fmt = f"${current_p:,.4f}" if current_p < 10 else f"${current_p:,.2f}"
                st.markdown(f"**Current: {fmt}** {color} ({change_p:+.2f}%)")
            else:
                st.warning(f"Could not load {label} data")

    # ============================================
    # SECTION 6: P&L BREAKDOWN (Daily / Weekly / Monthly)
    # ============================================
    st.markdown("---")
    st.header("📊 P&L Breakdown")

    now_utc = datetime.now(timezone.utc)

    def parse_time(ts):
        if not ts:
            return None
        try:
            if '+' in ts or 'Z' in ts:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        except:
            return None

    def calc_stats(trade_list):
        if not trade_list:
            return {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'win_rate': 0, 'avg_pnl': 0}
        wins = [t for t in trade_list if (t.get('net_pnl') or 0) > 0]
        losses = [t for t in trade_list if (t.get('net_pnl') or 0) < 0]
        pnl = sum(t.get('net_pnl') or 0 for t in trade_list)
        return {
            'trades': len(trade_list),
            'wins': len(wins),
            'losses': len(losses),
            'pnl': pnl,
            'win_rate': (len(wins) / len(trade_list) * 100) if trade_list else 0,
            'avg_pnl': pnl / len(trade_list) if trade_list else 0
        }

    # Parse exit times for all closed trades
    trades_with_time = []
    for t in closed_trades:
        exit_time = parse_time(t.get('exit_time'))
        if exit_time:
            trades_with_time.append({**t, '_exit_dt': exit_time})

    # --- Group trades by day ---
    daily_groups = {}
    for t in trades_with_time:
        day_key = t['_exit_dt'].strftime('%Y-%m-%d')
        daily_groups.setdefault(day_key, []).append(t)

    # --- Group trades by week (Monday start) ---
    weekly_groups = {}
    for t in trades_with_time:
        week_start = t['_exit_dt'] - timedelta(days=t['_exit_dt'].weekday())
        week_key = week_start.strftime('%Y-%m-%d')
        weekly_groups.setdefault(week_key, []).append(t)

    # --- Group trades by month ---
    monthly_groups = {}
    for t in trades_with_time:
        month_key = t['_exit_dt'].strftime('%Y-%m')
        monthly_groups.setdefault(month_key, []).append(t)

    # --- Tab view: Daily | Weekly | Monthly ---
    pnl_tab1, pnl_tab2, pnl_tab3 = st.tabs(["📅 Daily", "📆 Weekly", "📈 Monthly"])

    def render_pnl_table(groups, date_label="Date"):
        """Render a P&L summary table and bar chart for grouped trades."""
        if not groups:
            st.info("No closed trades yet")
            return

        rows = []
        for key in sorted(groups.keys()):
            stats = calc_stats(groups[key])
            rows.append({
                date_label: key,
                'Trades': stats['trades'],
                'W/L': f"{stats['wins']}W/{stats['losses']}L",
                'Win Rate': f"{stats['win_rate']:.0f}%",
                'P&L': stats['pnl'],
                'Avg P&L': stats['avg_pnl'],
            })

        df = pd.DataFrame(rows)

        # P&L bar chart
        colors = ['#00ff88' if v >= 0 else '#ff4444' for v in df['P&L']]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df[date_label],
            y=df['P&L'],
            marker_color=colors,
            text=[f"${v:+.2f}" for v in df['P&L']],
            textposition='outside',
            textfont=dict(size=11),
        ))

        # Cumulative P&L line
        cumulative = df['P&L'].cumsum()
        fig.add_trace(go.Scatter(
            x=df[date_label],
            y=cumulative,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='#00bfff', width=2),
            marker=dict(size=6),
            yaxis='y2',
        ))

        fig.update_layout(
            title=f"P&L by {date_label}",
            yaxis=dict(title="Period P&L ($)", zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
            yaxis2=dict(title="Cumulative P&L ($)", overlaying='y', side='right', zeroline=True, zerolinecolor='rgba(0,191,255,0.3)'),
            template="plotly_dark",
            height=350,
            margin=dict(t=40, b=40, l=50, r=50),
            showlegend=False,
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        display_df = df.copy()
        display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:+.2f}")
        display_df['Avg P&L'] = display_df['Avg P&L'].apply(lambda x: f"${x:+.2f}")
        # Show most recent first
        display_df = display_df.iloc[::-1].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with pnl_tab1:
        render_pnl_table(daily_groups, "Date")

    with pnl_tab2:
        render_pnl_table(weekly_groups, "Week")

    with pnl_tab3:
        render_pnl_table(monthly_groups, "Month")

    # --- Summary row: Today / This Week / This Month / All Time ---
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    this_week_start = today_start - timedelta(days=now_utc.weekday())
    this_month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    today_trades = [t for t in trades_with_time if t['_exit_dt'] >= today_start]
    week_trades = [t for t in trades_with_time if t['_exit_dt'] >= this_week_start]
    month_trades = [t for t in trades_with_time if t['_exit_dt'] >= this_month_start]

    today_stats = calc_stats(today_trades)
    week_stats = calc_stats(week_trades)
    month_stats = calc_stats(month_trades)
    all_stats = calc_stats(closed_trades)

    st.markdown("#### Summary")
    s1, s2, s3, s4 = st.columns(4)
    for col, label, stats in [
        (s1, "Today", today_stats),
        (s2, "This Week", week_stats),
        (s3, "This Month", month_stats),
        (s4, "All Time", all_stats),
    ]:
        with col:
            pnl = stats['pnl']
            color = "#00ff88" if pnl >= 0 else "#ff4444"
            st.markdown(f"**{label}**")
            st.markdown(f"<span style='color:{color}; font-size:20px;'>${pnl:+.2f}</span>", unsafe_allow_html=True)
            if stats['trades'] > 0:
                st.caption(f"{stats['trades']} trades | {stats['win_rate']:.0f}% WR | {stats['wins']}W/{stats['losses']}L")

    # Progress tracker towards 100 trades
    st.markdown("---")
    st.subheader("🎯 Progress to Real Money")

    target_trades = 100
    current_trades = all_stats['trades']
    progress_pct = min(current_trades / target_trades * 100, 100)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(progress_pct / 100)
    with col2:
        st.markdown(f"**{current_trades} / {target_trades} trades**")

    # Status checklist
    days_trading = (now_utc - datetime.fromisoformat(NEW_SYSTEM_DATE.replace('Z', '+00:00'))).days
    target_days = 30

    checks = {
        "100+ closed trades": current_trades >= 100,
        "30+ days of data": days_trading >= 30,
        "Win rate > 50%": all_stats['win_rate'] > 50,
        "Profit factor > 1.5": profit_factor > 1.5,
        "Positive total P&L": all_stats['pnl'] > 0,
    }

    st.markdown("**Readiness Checklist:**")
    cols = st.columns(len(checks))
    for i, (check, passed) in enumerate(checks.items()):
        with cols[i]:
            icon = "✅" if passed else "⬜"
            st.markdown(f"{icon} {check}")

    passed_checks = sum(checks.values())
    if passed_checks == len(checks):
        st.success("🎉 All criteria met! System may be ready for real money testing with small amounts.")
    elif passed_checks >= 3:
        st.warning(f"⏳ {passed_checks}/{len(checks)} criteria met. Keep paper trading.")
    else:
        st.info(f"📊 {passed_checks}/{len(checks)} criteria met. More data needed.")

    # ============================================
    # SECTION 7: AI INSIGHTS (On-Demand)
    # ============================================
    st.markdown("---")
    st.header("🤖 AI Trading Analyst")

    st.caption("Click the button below to generate an AI analysis (uses Gemini API)")

    # Show previous analysis from session state if available
    if 'ai_insights' in st.session_state:
        insights = st.session_state['ai_insights']
        if 'error' in insights:
            st.error(f"AI Analysis Error: {insights['error']}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                accuracy = insights.get('signal_accuracy', 0)
                st.metric(
                    "Signal Accuracy",
                    f"{accuracy:.1f}%",
                    delta=f"{insights.get('correct_signals', 0)}/{insights.get('total_signals_analyzed', 0)} correct"
                )
            with col2:
                generated_at = insights.get('generated_at', '')
                if generated_at:
                    try:
                        gen_time = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                        gen_time_kst = gen_time + timedelta(hours=9)
                        st.metric("Generated", gen_time_kst.strftime('%m/%d %H:%M KST'))
                    except:
                        st.metric("Generated", "Just now")

            st.markdown("### 📋 Analysis & Recommendations")
            st.markdown(insights.get('analysis', 'No analysis available'))

    # Button to trigger analysis
    if st.button("🧠 Run AI Analysis", key="run_ai"):
        if not get_gemini_api_key():
            st.warning("GEMINI_API_KEY not found in secrets.")
        else:
            try:
                trades_json = json.dumps(trades, default=str)
                signals_json = json.dumps(signals[-100:], default=str)
                prices_json = json.dumps(prices)
                cache_key = get_analysis_cache_key(trades, signals)

                with st.spinner("🧠 AI is analyzing your trading performance..."):
                    insights = generate_ai_insights(trades_json, signals_json, prices_json, cache_key)

                if insights:
                    st.session_state['ai_insights'] = insights
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating AI insights: {e}")

    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    st.caption("🤖 AI Trading System v4.0 | Multi-Strategy | Signals every minute | Gemini 2.5 Flash AI | 🇰🇷 KST (UTC+9)")


if __name__ == "__main__":
    main()
