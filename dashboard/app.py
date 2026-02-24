"""Simplified AI Trading Dashboard - Focus on Signals, Trades, Performance"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

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


@st.cache_data(ttl=30)
def fetch_current_prices():
    """Fetch current BTC and ETH prices."""
    prices = {}
    try:
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5)
            if resp.status_code == 200:
                prices[symbol] = float(resp.json()["price"])
    except:
        pass
    return prices


@st.cache_data(ttl=60)
def fetch_klines(symbol: str, interval: str = "1h", limit: int = 48):
    """Fetch candlestick data from Binance."""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        resp = requests.get(url, timeout=10)
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
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
    return pd.DataFrame()


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
            signal_type = sig.get('signal_type', 'hold')
            confidence = float(sig.get('confidence', 0)) * 100
            risk_score = float(sig.get('risk_score', 0))
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
    # FOOTER
    # ============================================
    st.markdown("---")
    st.caption("🤖 AI Trading System | Signals every minute | Gemini 2.5 Flash AI | 🇰🇷 All times in KST (UTC+9)")


if __name__ == "__main__":
    main()
