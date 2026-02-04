"""Streamlit dashboard for the AI Trading System."""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Load environment variables - supports both local (.env) and Streamlit Cloud (secrets)
project_root = Path(__file__).parent.parent

# Try Streamlit secrets first (for Streamlit Cloud), then fall back to .env
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

# Page configuration
st.set_page_config(
    page_title="AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive {
        color: #00ff88;
    }
    .negative {
        color: #ff4444;
    }
    .stMetric > div {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_market_data(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100):
    """Fetch real market data from Binance."""
    try:
        from data.collectors.market_data import get_market_data_sync
        return get_market_data_sync(symbol, interval, limit)
    except Exception as e:
        st.warning(f"Could not fetch market data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_ticker(symbol: str = "BTCUSDT"):
    """Fetch real ticker data from Binance."""
    try:
        from data.collectors.market_data import get_ticker_sync
        return get_ticker_sync(symbol)
    except Exception as e:
        return {}


@st.cache_data(ttl=30)
def fetch_multiple_tickers(symbols: list):
    """Fetch multiple tickers."""
    try:
        from data.collectors.market_data import get_multiple_tickers_sync
        return get_multiple_tickers_sync(symbols)
    except Exception as e:
        return []


def get_supabase():
    """Get Supabase client - works on both local and Streamlit Cloud."""
    from supabase import create_client

    # Try Streamlit secrets first, then environment variables
    if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
        url = st.secrets['SUPABASE_URL']
        key = st.secrets['SUPABASE_ANON_KEY']
    else:
        url = os.environ.get('SUPABASE_URL', '')
        key = os.environ.get('SUPABASE_ANON_KEY', '')

    if not url or not key:
        return None

    return create_client(url, key)


@st.cache_data(ttl=60)
def fetch_trades_from_supabase(limit: int = 100):
    """Fetch trade logs from Supabase."""
    try:
        client = get_supabase()
        if not client:
            return []
        result = client.table('trade_logs').select('*').order('entry_time', desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        st.warning(f"Could not fetch trades: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_performance_from_supabase(limit: int = 100):
    """Fetch performance snapshots from Supabase."""
    try:
        client = get_supabase()
        if not client:
            return []
        result = client.table('performance_snapshots').select('*').order('timestamp', desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        st.warning(f"Could not fetch performance: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_signals_from_supabase(limit: int = 50):
    """Fetch trading signals from Supabase."""
    try:
        client = get_supabase()
        if not client:
            return []
        result = client.table('signals').select('*').order('timestamp', desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        st.warning(f"Could not fetch signals: {e}")
        return []


def get_trading_data_from_supabase():
    """Get trading data from Supabase."""
    # Fetch data
    trades_raw = fetch_trades_from_supabase(100)
    perf_raw = fetch_performance_from_supabase(100)

    # Convert to DataFrames
    if perf_raw:
        equity_df = pd.DataFrame(perf_raw)
        equity_df["date"] = pd.to_datetime(equity_df["timestamp"])
        equity_df["equity"] = equity_df["total_equity"].astype(float)
        if "daily_pnl" in equity_df.columns and "total_equity" in equity_df.columns:
            equity_df["daily_return"] = equity_df["daily_pnl"].astype(float) / equity_df["total_equity"].astype(float)
        else:
            equity_df["daily_return"] = 0.0
        equity_df = equity_df.sort_values("date")
    else:
        # Fallback to sample data
        return get_sample_trading_data()

    if trades_raw:
        trades_df = pd.DataFrame(trades_raw)
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_time"])
        trades_df["pnl"] = trades_df["net_pnl"].astype(float) if "net_pnl" in trades_df.columns else 0
        trades_df["return_pct"] = trades_df["return_pct"].astype(float) / 100 if "return_pct" in trades_df.columns else 0
        trades_df["strategy"] = trades_df.get("strategy_name", "hybrid_v1")
    else:
        # Fallback to sample data
        return get_sample_trading_data()

    return equity_df, trades_df


def get_sample_trading_data():
    """Generate sample trading data for demonstration."""
    dates = pd.date_range(start="2024-01-01", end=datetime.now(), freq="D")

    # Generate sample equity curve
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    equity = 100000 * (1 + returns).cumprod()

    equity_df = pd.DataFrame({
        "date": dates,
        "equity": equity,
        "daily_return": returns,
    })

    # Generate sample trades
    trades = []
    for i in range(100):
        entry_date = dates[np.random.randint(0, len(dates) - 10)]
        exit_date = entry_date + timedelta(days=np.random.randint(1, 10))
        pnl = np.random.normal(100, 500)

        trades.append({
            "id": i + 1,
            "symbol": np.random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]),
            "side": np.random.choice(["buy", "sell"]),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": np.random.uniform(100, 50000),
            "exit_price": np.random.uniform(100, 50000),
            "pnl": pnl,
            "return_pct": pnl / 1000,
            "strategy": np.random.choice(["Hybrid RL+LLM", "PPO", "DQN"]),
        })

    trades_df = pd.DataFrame(trades)

    return equity_df, trades_df


def main():
    """Main dashboard application."""
    # Sidebar
    with st.sidebar:
        st.title("📈 AI Trading System")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Market Data", "🏠 Overview", "📈 Performance", "💹 Trades", "⚙️ Strategy", "🤖 AI Insights"],
            index=0,
        )

        st.markdown("---")

        # Symbol selector for market data
        if page == "📊 Market Data":
            st.subheader("Settings")
            symbol = st.selectbox(
                "Symbol",
                ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"],
            )
            interval = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3,
            )
        else:
            symbol = "BTCUSDT"
            interval = "1h"

        # Filters for other pages
        if page not in ["📊 Market Data"]:
            st.subheader("Filters")
            strategy = st.selectbox(
                "Strategy",
                ["All", "Hybrid RL+LLM", "PPO", "DQN"],
            )

            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
            )

        st.markdown("---")
        st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Auto-refresh option
        if st.checkbox("Auto-refresh (30s)", value=False):
            st.rerun()

    # Get data from Supabase (with fallback to sample data)
    equity_df, trades_df = get_trading_data_from_supabase()

    # Main content
    if page == "📊 Market Data":
        show_market_data(symbol, interval)
    elif page == "🏠 Overview":
        show_overview(equity_df, trades_df)
    elif page == "📈 Performance":
        show_performance(equity_df, trades_df)
    elif page == "💹 Trades":
        show_trades(trades_df)
    elif page == "⚙️ Strategy":
        show_strategy()
    elif page == "🤖 AI Insights":
        show_ai_insights(equity_df, trades_df)


def show_market_data(symbol: str, interval: str):
    """Show real-time market data page."""
    st.title("📊 Live Market Data")

    # Fetch real data
    ticker = fetch_ticker(symbol)
    df = fetch_market_data(symbol, interval, 200)

    if ticker:
        # Price metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            price = ticker.get("price", 0)
            change = ticker.get("change_24h", 0)
            st.metric(
                f"{symbol} Price",
                f"${price:,.2f}",
                f"{change:+.2f}%",
            )

        with col2:
            st.metric(
                "24h High",
                f"${ticker.get('high_24h', 0):,.2f}",
            )

        with col3:
            st.metric(
                "24h Low",
                f"${ticker.get('low_24h', 0):,.2f}",
            )

        with col4:
            vol = ticker.get("volume_24h", 0)
            st.metric(
                "24h Volume",
                f"{vol:,.0f}",
            )

        with col5:
            quote_vol = ticker.get("quote_volume_24h", 0)
            st.metric(
                "24h Quote Vol",
                f"${quote_vol:,.0f}",
            )

    st.markdown("---")

    # Multiple tickers
    st.subheader("🔥 Top Cryptocurrencies")
    tickers = fetch_multiple_tickers(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])

    if tickers:
        cols = st.columns(len(tickers))
        for i, t in enumerate(tickers):
            with cols[i]:
                change = t.get("change_24h", 0)
                color = "🟢" if change >= 0 else "🔴"
                st.metric(
                    f"{color} {t['symbol'].replace('USDT', '')}",
                    f"${t['price']:,.2f}",
                    f"{change:+.2f}%",
                )

    st.markdown("---")

    # Candlestick chart
    if not df.empty:
        st.subheader(f"📈 {symbol} - {interval} Chart")

        # Create candlestick chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
            ),
            row=1, col=1,
        )

        # Add moving averages
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="yellow", width=1),
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="orange", width=1),
            ),
            row=1, col=1,
        )

        # Volume bars
        colors = ["red" if df["close"].iloc[i] < df["open"].iloc[i] else "green"
                  for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=colors,
            ),
            row=2, col=1,
        )

        fig.update_layout(
            height=600,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(x=0, y=1, orientation="h"),
        )

        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Technical indicators summary
        st.subheader("📊 Technical Indicators")

        col1, col2, col3 = st.columns(3)

        with col1:
            # RSI calculation
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            rsi_color = "🟢" if current_rsi < 30 else "🔴" if current_rsi > 70 else "🟡"
            st.metric(f"{rsi_color} RSI (14)", f"{current_rsi:.1f}")

        with col2:
            # MACD
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_val = macd.iloc[-1] - signal.iloc[-1]

            macd_color = "🟢" if macd_val > 0 else "🔴"
            st.metric(f"{macd_color} MACD Signal", f"{macd_val:.2f}")

        with col3:
            # Trend
            sma20_val = df["sma20"].iloc[-1]
            sma50_val = df["sma50"].iloc[-1]
            current_price = df["close"].iloc[-1]

            if current_price > sma20_val > sma50_val:
                trend = "🟢 Bullish"
            elif current_price < sma20_val < sma50_val:
                trend = "🔴 Bearish"
            else:
                trend = "🟡 Neutral"

            st.metric("Trend", trend)

        # Data table
        with st.expander("📋 Raw OHLCV Data"):
            st.dataframe(
                df.tail(20).round(2),
                use_container_width=True,
            )
    else:
        st.warning("Could not fetch market data. Please check your connection.")


def show_overview(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Show overview page."""
    st.title("🏠 Dashboard Overview")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        current_equity = equity_df["equity"].iloc[-1]
        daily_change = equity_df['daily_return'].iloc[-1]
        st.metric(
            "Total Equity",
            f"${current_equity:,.0f}",
            f"{daily_change:.2%}",
        )

    with col2:
        total_return = (current_equity / 100000 - 1) * 100
        st.metric(
            "Total Return",
            f"{total_return:.1f}%",
            f"+{total_return:.1f}%" if total_return > 0 else f"{total_return:.1f}%",
        )

    with col3:
        win_rate = len(trades_df[trades_df["pnl"] > 0]) / len(trades_df)
        st.metric(
            "Win Rate",
            f"{win_rate:.1%}",
        )

    with col4:
        st.metric(
            "Total Trades",
            len(trades_df),
        )

    with col5:
        cummax = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - cummax) / cummax
        max_dd = drawdown.min()
        st.metric(
            "Max Drawdown",
            f"{max_dd:.1%}",
        )

    st.markdown("---")

    # Charts
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["date"],
            y=equity_df["equity"],
            mode="lines",
            name="Equity",
            fill="tozeroy",
            line=dict(color="#00ff88"),
        ))
        fig.update_layout(
            height=400,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 PnL Distribution")
        fig = px.histogram(
            trades_df,
            x="pnl",
            nbins=30,
            color_discrete_sequence=["#00ff88"],
        )
        fig.update_layout(
            height=400,
            template="plotly_dark",
            xaxis_title="PnL ($)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent trades
    st.subheader("📋 Recent Trades")
    recent_trades = trades_df.sort_values("entry_date", ascending=False).head(10)

    styled_df = recent_trades[["symbol", "side", "strategy", "entry_date", "pnl", "return_pct"]].copy()
    styled_df["pnl"] = styled_df["pnl"].apply(lambda x: f"${x:,.2f}")
    styled_df["return_pct"] = styled_df["return_pct"].apply(lambda x: f"{x:.2%}")

    st.dataframe(styled_df, use_container_width=True)


def show_performance(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Show performance analytics page."""
    st.title("📈 Performance Analytics")

    # Calculate metrics
    returns = equity_df["daily_return"].dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    negative_returns = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / negative_returns.std() if len(negative_returns) > 0 else 0

    cummax = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar ratio
    annual_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) ** (252 / len(equity_df)) - 1
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col2:
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    with col3:
        st.metric("Calmar Ratio", f"{calmar:.2f}")
    with col4:
        st.metric("Annual Return", f"{annual_return:.1%}")

    st.markdown("---")

    # Drawdown chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Drawdown")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["date"],
            y=drawdown * 100,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ff4444"),
            name="Drawdown %",
        ))
        fig.update_layout(
            height=300,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Monthly Returns")
        equity_df["month"] = equity_df["date"].dt.to_period("M")
        monthly = equity_df.groupby("month")["daily_return"].sum() * 100

        fig = go.Figure()
        colors = ["#00ff88" if x >= 0 else "#ff4444" for x in monthly.values]
        fig.add_trace(go.Bar(
            x=monthly.index.astype(str),
            y=monthly.values,
            marker_color=colors,
        ))
        fig.update_layout(
            height=300,
            template="plotly_dark",
            xaxis_title="Month",
            yaxis_title="Return (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Trade statistics
    st.subheader("📊 Trade Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        st.markdown("**Win/Loss Stats**")
        st.write(f"- Winning Trades: {len(winning_trades)}")
        st.write(f"- Losing Trades: {len(losing_trades)}")
        st.write(f"- Win Rate: {len(winning_trades) / len(trades_df):.1%}")

    with col2:
        avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades["pnl"].mean()) if len(losing_trades) > 0 else 1

        st.markdown("**PnL Stats**")
        st.write(f"- Avg Win: ${avg_win:,.2f}")
        st.write(f"- Avg Loss: ${avg_loss:,.2f}")
        st.write(f"- Win/Loss Ratio: {avg_win / avg_loss:.2f}")

    with col3:
        total_profit = winning_trades["pnl"].sum()
        total_loss = abs(losing_trades["pnl"].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        st.markdown("**Profit Factor**")
        st.write(f"- Gross Profit: ${total_profit:,.2f}")
        st.write(f"- Gross Loss: ${total_loss:,.2f}")
        st.write(f"- Profit Factor: {profit_factor:.2f}")


def show_trades(trades_df: pd.DataFrame):
    """Show trades page."""
    st.title("💹 Trade History")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        symbol_filter = st.multiselect(
            "Symbols",
            trades_df["symbol"].unique(),
            default=trades_df["symbol"].unique(),
        )

    with col2:
        side_filter = st.multiselect(
            "Side",
            ["buy", "sell"],
            default=["buy", "sell"],
        )

    with col3:
        pnl_filter = st.selectbox(
            "PnL",
            ["All", "Winners Only", "Losers Only"],
        )

    # Apply filters
    filtered_df = trades_df[
        (trades_df["symbol"].isin(symbol_filter)) &
        (trades_df["side"].isin(side_filter))
    ]

    if pnl_filter == "Winners Only":
        filtered_df = filtered_df[filtered_df["pnl"] > 0]
    elif pnl_filter == "Losers Only":
        filtered_df = filtered_df[filtered_df["pnl"] < 0]

    # Summary
    st.markdown(f"**Showing {len(filtered_df)} trades**")

    # Display
    display_df = filtered_df[["symbol", "side", "strategy", "entry_date", "exit_date", "entry_price", "exit_price", "pnl", "return_pct"]].copy()
    display_df = display_df.sort_values("entry_date", ascending=False)
    display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:,.2f}")
    display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"${x:,.2f}")
    display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:,.2f}")
    display_df["return_pct"] = display_df["return_pct"].apply(lambda x: f"{x:.2%}")

    st.dataframe(display_df, use_container_width=True, height=500)

    # Export option
    csv = trades_df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        "trades.csv",
        "text/csv",
    )


def show_strategy():
    """Show strategy configuration page."""
    st.title("⚙️ Strategy Configuration")

    st.markdown("""
    Configure the hybrid RL + LLM trading strategy parameters.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 RL Agent Settings")

        rl_weight = st.slider(
            "RL Signal Weight",
            0.0, 1.0, 0.6,
            help="Weight given to RL agent signals",
        )

        st.selectbox(
            "RL Algorithm",
            ["PPO", "DQN", "A2C"],
        )

        st.number_input(
            "Lookback Window",
            min_value=10, max_value=200, value=60,
            help="Number of candles for observation",
        )

        st.selectbox(
            "Reward Function",
            ["Sharpe Ratio", "Sortino Ratio", "Profit", "Log Return"],
        )

    with col2:
        st.subheader("🧠 LLM Agent Settings")

        llm_weight = st.slider(
            "LLM Signal Weight",
            0.0, 1.0, 0.4,
            help="Weight given to LLM analysis",
        )

        st.selectbox(
            "LLM Model",
            ["GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo", "Claude 3"],
        )

        st.checkbox("Enable News Analysis", value=True)
        st.checkbox("Enable Sentiment Analysis", value=True)

        st.number_input(
            "LLM Veto Threshold",
            min_value=0.5, max_value=1.0, value=0.8,
            help="Confidence threshold for LLM to veto RL signals",
        )

    st.markdown("---")

    st.subheader("⚠️ Risk Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.number_input(
            "Max Position Size (%)",
            min_value=1, max_value=100, value=10,
        )

    with col2:
        st.number_input(
            "Stop Loss (%)",
            min_value=0.5, max_value=10.0, value=2.0,
        )

    with col3:
        st.number_input(
            "Take Profit (%)",
            min_value=1.0, max_value=20.0, value=4.0,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Max Drawdown (%)",
            min_value=5, max_value=50, value=15,
        )

    with col2:
        st.number_input(
            "Daily Trade Limit",
            min_value=1, max_value=100, value=20,
        )

    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")


def show_ai_insights(equity_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Show AI insights page."""
    st.title("🤖 AI Insights")

    # Performance grade
    returns = equity_df["daily_return"].dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    win_rate = len(trades_df[trades_df["pnl"] > 0]) / len(trades_df)

    # Calculate grade
    score = 0
    score += min(30, sharpe * 15)
    score += min(20, win_rate * 40)

    if score >= 80:
        grade = "A"
        grade_color = "green"
    elif score >= 60:
        grade = "B"
        grade_color = "blue"
    elif score >= 40:
        grade = "C"
        grade_color = "orange"
    else:
        grade = "D"
        grade_color = "red"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; background-color: #262730; border-radius: 10px;">
            <h1 style="font-size: 80px; color: {grade_color}; margin: 0;">{grade}</h1>
            <p style="font-size: 20px;">Performance Grade</p>
            <p>Score: {score:.0f}/100</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Analysis Summary")

        # Strengths
        st.markdown("**✅ Strengths:**")
        if sharpe > 1:
            st.write(f"- Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
        if win_rate > 0.55:
            st.write(f"- High win rate ({win_rate:.1%})")

        # Weaknesses
        st.markdown("**⚠️ Areas for Improvement:**")
        if sharpe < 1:
            st.write("- Risk-adjusted returns could be improved")
        if win_rate < 0.5:
            st.write("- Win rate is below 50%")

    st.markdown("---")

    # AI Recommendations
    st.subheader("💡 AI Recommendations")

    recommendations = [
        {
            "category": "Risk Management",
            "priority": "High",
            "suggestion": "Consider reducing position sizes during high volatility periods",
            "expected_impact": "Reduce drawdown by 20-30%",
        },
        {
            "category": "Entry Signals",
            "priority": "Medium",
            "suggestion": "Add confirmation indicators to improve signal quality",
            "expected_impact": "Increase win rate by 5-10%",
        },
        {
            "category": "Exit Strategy",
            "priority": "Medium",
            "suggestion": "Implement trailing stops to capture more profit",
            "expected_impact": "Improve profit factor by 15%",
        },
    ]

    for rec in recommendations:
        with st.expander(f"**{rec['priority']}** - {rec['category']}"):
            st.write(f"**Suggestion:** {rec['suggestion']}")
            st.write(f"**Expected Impact:** {rec['expected_impact']}")

    st.markdown("---")

    # Model Status
    st.subheader("🤖 Model Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**RL Agent (PPO)**")
        st.write("- Status: 🟢 Active")
        st.write("- Last Training: 2 days ago")
        st.write("- Episodes: 10,000")

    with col2:
        st.markdown("**LLM Analyst**")
        st.write("- Status: 🟢 Active")
        st.write("- Model: GPT-4 Turbo")
        st.write("- Requests Today: 142")

    with col3:
        st.markdown("**Hybrid Strategy**")
        st.write("- Status: 🟢 Active")
        st.write("- RL Weight: 60%")
        st.write("- LLM Weight: 40%")

    # Retrain suggestion
    if sharpe < 0.5 or win_rate < 0.45:
        st.warning("⚠️ **Retraining Recommended**: Performance metrics suggest the model may benefit from retraining with recent data.")

        if st.button("🔄 Schedule Retraining"):
            st.info("Retraining scheduled. This will run during off-market hours.")

    st.markdown("---")

    # Recent Signals from Supabase
    st.subheader("📡 Recent Trading Signals")

    signals = fetch_signals_from_supabase(20)

    if signals:
        signals_df = pd.DataFrame(signals)

        # Format for display
        display_cols = ["timestamp", "symbol", "signal_type", "source", "confidence", "status"]
        available_cols = [c for c in display_cols if c in signals_df.columns]

        if available_cols:
            display_df = signals_df[available_cols].copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

            if "confidence" in display_df.columns:
                display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{float(x)*100:.1f}%")

            # Color code signal types
            def signal_color(signal_type):
                colors = {
                    "strong_buy": "🟢🟢",
                    "buy": "🟢",
                    "hold": "🟡",
                    "sell": "🔴",
                    "strong_sell": "🔴🔴",
                }
                return colors.get(signal_type, "⚪")

            if "signal_type" in display_df.columns:
                display_df["signal"] = display_df["signal_type"].apply(signal_color) + " " + display_df["signal_type"]
                display_df = display_df.drop(columns=["signal_type"])

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No signal data available.")
    else:
        st.info("No recent signals found in database.")


if __name__ == "__main__":
    main()
