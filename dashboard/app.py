"""Simplified AI Trading Dashboard - Focus on Signals, Trades, Performance"""

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

# Korea Standard Time (UTC+9)
try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except ImportError:
    # Fallback for older Python versions
    from datetime import timezone as tz
    KST = tz(timedelta(hours=9))
NEW_SYSTEM_DATE = "2026-02-20T00:00:00Z"


def to_kst(timestamp_str: str) -> str:
    """Convert UTC timestamp string to KST formatted string."""
    if not timestamp_str:
        return ""
    try:
        # Parse the timestamp (handles both with and without timezone)
        if '+' in timestamp_str or 'Z' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=ZoneInfo("UTC"))
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
        result = client.table('signals').select('*').order('timestamp', desc=True).limit(limit).execute()
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


@st.cache_data(ttl=10)
def fetch_current_prices():
    """Fetch current BTC and ETH prices from multiple sources."""
    prices = {}
    symbols = ["BTCUSDT", "ETHUSDT"]

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
            coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum"}
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if "BTCUSDT" not in prices and "bitcoin" in data:
                    prices["BTCUSDT"] = float(data["bitcoin"]["usd"])
                if "ETHUSDT" not in prices and "ethereum" in data:
                    prices["ETHUSDT"] = float(data["ethereum"]["usd"])
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
        coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum"}
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
            "maxOutputTokens": 1024,
        }
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)

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
        'hour_bucket': datetime.now(timezone.utc).hour // 6  # Changes every 6 hours
    }, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


@st.cache_data(ttl=21600)  # Cache for 6 hours (21600 seconds)
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

        # Calculate signal accuracy
        signal_analysis = []
        for sig in signals[-50:]:  # Last 50 signals
            symbol = sig.get('symbol', '')
            signal_type = sig.get('signal_type', '')
            sig_price = float(sig.get('entry_price', 0) or 0)
            current_price = prices.get(symbol, sig_price)

            if sig_price > 0 and current_price > 0:
                price_change = (current_price - sig_price) / sig_price * 100

                # Was the signal correct?
                if 'buy' in signal_type.lower():
                    correct = price_change > 0
                elif 'sell' in signal_type.lower():
                    correct = price_change < 0
                else:
                    correct = None

                signal_analysis.append({
                    'signal': signal_type,
                    'price_at_signal': sig_price,
                    'current_price': current_price,
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
    st.caption(f"Last updated: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')} | System started: Feb 20, 2026")

    # Auto refresh
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # Fetch data
    signals = fetch_signals(100)
    trades = fetch_trades(50)
    prices = fetch_current_prices()

    # ============================================
    # SECTION 1: OVERALL PERFORMANCE (Since Feb 20)
    # ============================================
    st.markdown("---")
    st.header("📊 Performance Since Update (Feb 20)")

    # Calculate stats
    closed_trades = [t for t in trades if t.get('exit_time')]
    open_trades = [t for t in trades if not t.get('exit_time')]

    total_trades = len(closed_trades)
    winners = [t for t in closed_trades if (t.get('net_pnl') or 0) > 0]
    losers = [t for t in closed_trades if (t.get('net_pnl') or 0) < 0]

    total_pnl = sum(t.get('net_pnl') or 0 for t in closed_trades)
    gross_profit = sum(t.get('net_pnl') or 0 for t in winners)
    gross_loss = abs(sum(t.get('net_pnl') or 0 for t in losers))

    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    avg_win = (gross_profit / len(winners)) if winners else 0
    avg_loss = (gross_loss / len(losers)) if losers else 0

    # Calculate unrealized P&L for open positions
    unrealized_pnl = 0
    for t in open_trades:
        symbol = t.get('symbol', '')
        side = t.get('side', '')
        entry_price = float(t.get('entry_price') or 0)
        current_price = prices.get(symbol, entry_price)

        if side == 'buy':
            unrealized_pnl += (current_price - entry_price) / entry_price * 1000  # Approx position value
        else:
            unrealized_pnl += (entry_price - current_price) / entry_price * 1000

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        color = "positive" if total_pnl >= 0 else "negative"
        st.metric("Realized P&L", f"${total_pnl:+,.2f}", delta=f"{total_trades} trades")

    with col2:
        color = "positive" if unrealized_pnl >= 0 else "negative"
        st.metric("Unrealized P&L", f"${unrealized_pnl:+,.2f}", delta=f"{len(open_trades)} open")

    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{len(winners)}W / {len(losers)}L")

    with col4:
        st.metric("Profit Factor", f"{profit_factor:.2f}")

    with col5:
        st.metric("Avg Win/Loss", f"${avg_win:.2f} / ${avg_loss:.2f}")

    # ============================================
    # SECTION 2: OPEN POSITIONS
    # ============================================
    st.markdown("---")
    st.header("🔥 Open Positions")

    if open_trades:
        for t in open_trades:
            symbol = t.get('symbol', 'N/A')
            side = t.get('side', 'N/A').upper()
            entry_price = float(t.get('entry_price') or 0)
            entry_time_kst = to_kst(t.get('entry_time', ''))
            current_price = prices.get(symbol, entry_price)

            # Calculate P&L
            if side == 'BUY':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            pnl_color = "🟢" if pnl_pct >= 0 else "🔴"
            side_emoji = "📈" if side == "BUY" else "📉"

            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
            with col1:
                st.markdown(f"**{side_emoji} {symbol}** {side}")
            with col2:
                st.markdown(f"Entry: **${entry_price:,.2f}**")
            with col3:
                st.markdown(f"Current: **${current_price:,.2f}**")
            with col4:
                st.markdown(f"{pnl_color} **{pnl_pct:+.2f}%**")
            with col5:
                st.markdown(f"📅 {entry_time_kst}")
    else:
        st.info("No open positions")

    # ============================================
    # SECTION 3: RECENT SIGNALS (Every Minute Decisions)
    # ============================================
    st.markdown("---")
    st.header("⚡ Recent Signals (Every Minute Decisions)")

    if signals:
        # Format display with KST timezone
        display_data = []
        for sig in signals[:30]:
            signal_type = sig.get('signal_type', 'hold') or 'hold'
            confidence = float(sig.get('confidence') or 0) * 100
            risk_score = float(sig.get('risk_score') or 0)
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

            display_data.append({
                'Time (KST)': timestamp_kst,
                'Symbol': sig.get('symbol', 'N/A'),
                'Signal': f"{signal_emoji} {signal_type}",
                'Confidence': f"{confidence:.0f}%",
                'Risk': f"{risk_score:.2f}",
            })

        st.dataframe(
            pd.DataFrame(display_data),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No signals found")

    # ============================================
    # SECTION 4: TRADE HISTORY (What Actually Happened)
    # ============================================
    st.markdown("---")
    st.header("📋 Trade History (Since Feb 20)")

    if trades:
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

    col1, col2 = st.columns(2)

    # BTC Chart
    with col1:
        st.subheader("Bitcoin (BTC)")
        btc_df = fetch_klines("BTCUSDT", interval, limit)
        if not btc_df.empty:
            fig_btc = go.Figure(data=[go.Candlestick(
                x=btc_df['timestamp'],
                open=btc_df['open'],
                high=btc_df['high'],
                low=btc_df['low'],
                close=btc_df['close'],
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            )])
            fig_btc.update_layout(
                height=400,
                template="plotly_dark",
                xaxis_title="Time (KST)",
                yaxis_title="Price (USDT)",
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_btc, use_container_width=True)

            # Show current price
            current_btc = btc_df['close'].iloc[-1]
            prev_btc = btc_df['close'].iloc[-2]
            change_btc = ((current_btc - prev_btc) / prev_btc) * 100
            color = "🟢" if change_btc >= 0 else "🔴"
            st.markdown(f"**Current: ${current_btc:,.2f}** {color} ({change_btc:+.2f}%)")
        else:
            st.warning("Could not load BTC data")

    # ETH Chart
    with col2:
        st.subheader("Ethereum (ETH)")
        eth_df = fetch_klines("ETHUSDT", interval, limit)
        if not eth_df.empty:
            fig_eth = go.Figure(data=[go.Candlestick(
                x=eth_df['timestamp'],
                open=eth_df['open'],
                high=eth_df['high'],
                low=eth_df['low'],
                close=eth_df['close'],
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            )])
            fig_eth.update_layout(
                height=400,
                template="plotly_dark",
                xaxis_title="Time (KST)",
                yaxis_title="Price (USDT)",
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_eth, use_container_width=True)

            # Show current price
            current_eth = eth_df['close'].iloc[-1]
            prev_eth = eth_df['close'].iloc[-2]
            change_eth = ((current_eth - prev_eth) / prev_eth) * 100
            color = "🟢" if change_eth >= 0 else "🔴"
            st.markdown(f"**Current: ${current_eth:,.2f}** {color} ({change_eth:+.2f}%)")
        else:
            st.warning("Could not load ETH data")

    # ============================================
    # SECTION 6: PERFORMANCE REPORTS
    # ============================================
    st.markdown("---")
    st.header("📊 Performance Reports")

    # Calculate daily and weekly stats
    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    def parse_time(ts):
        if not ts:
            return None
        try:
            if '+' in ts or 'Z' in ts:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        except:
            return None

    # Filter trades by period
    today_trades = []
    week_trades = []
    for t in closed_trades:
        exit_time = parse_time(t.get('exit_time'))
        if exit_time:
            if exit_time >= today_start:
                today_trades.append(t)
            if exit_time >= week_start:
                week_trades.append(t)

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

    today_stats = calc_stats(today_trades)
    week_stats = calc_stats(week_trades)
    all_stats = calc_stats(closed_trades)

    # Display reports in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📅 Today")
        if today_stats['trades'] > 0:
            pnl_color = "🟢" if today_stats['pnl'] >= 0 else "🔴"
            st.markdown(f"""
            - **Trades:** {today_stats['trades']}
            - **Win Rate:** {today_stats['win_rate']:.1f}% ({today_stats['wins']}W/{today_stats['losses']}L)
            - **P&L:** {pnl_color} ${today_stats['pnl']:+.2f}
            - **Avg P&L:** ${today_stats['avg_pnl']:+.2f}/trade
            """)
        else:
            st.info("No closed trades today")

    with col2:
        st.subheader("📆 Last 7 Days")
        if week_stats['trades'] > 0:
            pnl_color = "🟢" if week_stats['pnl'] >= 0 else "🔴"
            st.markdown(f"""
            - **Trades:** {week_stats['trades']}
            - **Win Rate:** {week_stats['win_rate']:.1f}% ({week_stats['wins']}W/{week_stats['losses']}L)
            - **P&L:** {pnl_color} ${week_stats['pnl']:+.2f}
            - **Avg P&L:** ${week_stats['avg_pnl']:+.2f}/trade
            """)
        else:
            st.info("No closed trades this week")

    with col3:
        st.subheader("📈 All Time")
        if all_stats['trades'] > 0:
            pnl_color = "🟢" if all_stats['pnl'] >= 0 else "🔴"
            st.markdown(f"""
            - **Trades:** {all_stats['trades']}
            - **Win Rate:** {all_stats['win_rate']:.1f}% ({all_stats['wins']}W/{all_stats['losses']}L)
            - **P&L:** {pnl_color} ${all_stats['pnl']:+.2f}
            - **Avg P&L:** ${all_stats['avg_pnl']:+.2f}/trade
            """)
        else:
            st.info("No closed trades yet")

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
    # SECTION 7: AI INSIGHTS (Every 6 Hours)
    # ============================================
    st.markdown("---")
    st.header("🤖 AI Trading Analyst")

    # Calculate when next analysis will be generated
    current_hour = datetime.now(timezone.utc).hour
    next_analysis_hour = ((current_hour // 6) + 1) * 6 % 24
    hours_until_next = (next_analysis_hour - current_hour) % 6
    if hours_until_next == 0:
        hours_until_next = 6

    st.caption(f"Analysis updates every 6 hours | Next update in ~{hours_until_next}h")

    # Generate cache key and get AI insights
    try:
        cache_key = get_analysis_cache_key(trades, signals)

        # Convert data to JSON for caching (Streamlit cache requires hashable args)
        trades_json = json.dumps(trades, default=str)
        signals_json = json.dumps(signals[-100:], default=str)  # Last 100 signals
        prices_json = json.dumps(prices)

        with st.spinner("🧠 AI is analyzing your trading performance..."):
            insights = generate_ai_insights(trades_json, signals_json, prices_json, cache_key)

        if insights:
            if 'error' in insights:
                st.error(f"AI Analysis Error: {insights['error']}")
            else:
                # Signal Accuracy Metric
                col1, col2, col3 = st.columns(3)
                with col1:
                    accuracy = insights.get('signal_accuracy', 0)
                    accuracy_color = "🟢" if accuracy >= 50 else "🔴"
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
                            gen_time_kst = gen_time.astimezone(KST)
                            st.metric("Last Analysis", gen_time_kst.strftime('%m/%d %H:%M KST'))
                        except:
                            st.metric("Last Analysis", "Just now")
                with col3:
                    st.metric("Analysis Period", "Last 6 hours")

                # AI Analysis Content
                st.markdown("### 📋 Analysis & Recommendations")
                st.markdown(insights.get('analysis', 'No analysis available'))

        else:
            # Show details about why AI is not available
            if not get_gemini_api_key():
                st.warning("AI analysis not available. GEMINI_API_KEY not found in secrets.")
            else:
                st.warning("AI analysis not available. Check API key configuration.")

    except Exception as e:
        st.error(f"Error generating AI insights: {e}")

    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    st.caption("🤖 AI Trading System | Signals every minute | Gemini 2.5 Flash AI | 🇰🇷 All times in KST (UTC+9)")


if __name__ == "__main__":
    main()
