-- Supabase Schema for AI Trading System
-- Run this in your Supabase SQL editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- OHLCV (Candlestick) Data
-- ============================================
CREATE TABLE IF NOT EXISTS ohlcv (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(symbol, exchange, timeframe, timestamp)
);

-- Index for fast querying
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, exchange, timeframe, timestamp DESC);

-- ============================================
-- Trade Logs
-- ============================================
CREATE TABLE IF NOT EXISTS trade_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    position_id VARCHAR(100) UNIQUE,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'

    -- Entry
    entry_price DECIMAL(20, 8) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    entry_order_id VARCHAR(100),
    entry_reasoning TEXT,

    -- Exit
    exit_price DECIMAL(20, 8),
    exit_time TIMESTAMPTZ,
    exit_order_id VARCHAR(100),
    exit_reasoning TEXT,

    -- Position
    quantity DECIMAL(20, 8) NOT NULL,

    -- PnL
    gross_pnl DECIMAL(20, 8),
    net_pnl DECIMAL(20, 8),
    total_commission DECIMAL(20, 8) DEFAULT 0,
    return_pct DECIMAL(10, 4),

    -- Duration
    duration_seconds INTEGER,

    -- Strategy & Signal Info
    strategy_name VARCHAR(100),
    signal_source VARCHAR(50), -- 'rl', 'llm', 'hybrid', 'manual'
    signal_confidence DECIMAL(5, 4),

    -- Context (JSON)
    market_context JSONB,
    indicators_at_entry JSONB,
    indicators_at_exit JSONB,

    -- Tags
    tags TEXT[],

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol ON trade_logs(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_logs_strategy ON trade_logs(strategy_name);
CREATE INDEX IF NOT EXISTS idx_trade_logs_entry_time ON trade_logs(entry_time DESC);

-- ============================================
-- Performance Snapshots
-- ============================================
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_name VARCHAR(100),

    -- Portfolio
    total_equity DECIMAL(20, 8) NOT NULL,
    cash_balance DECIMAL(20, 8) NOT NULL,
    positions_value DECIMAL(20, 8) NOT NULL,

    -- PnL
    daily_pnl DECIMAL(20, 8),
    total_pnl DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),

    -- Trade Stats
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),

    -- Risk Metrics
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),

    -- Positions (JSON)
    open_positions JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index
CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_strategy ON performance_snapshots(strategy_name, timestamp DESC);

-- ============================================
-- Trading Signals
-- ============================================
CREATE TABLE IF NOT EXISTS signals (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Signal Info
    signal_type VARCHAR(20) NOT NULL, -- 'buy', 'sell', 'strong_buy', 'strong_sell', 'hold'
    source VARCHAR(50) NOT NULL, -- 'rl', 'llm', 'hybrid', 'technical'
    confidence DECIMAL(5, 4) NOT NULL,

    -- Price Targets
    entry_price DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),

    -- Analysis
    reasoning TEXT,
    indicators JSONB,
    metadata JSONB,

    -- Status
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'executed', 'expired', 'cancelled'
    executed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

-- ============================================
-- Model Checkpoints
-- ============================================
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'ppo', 'dqn', 'llm'
    version VARCHAR(50) NOT NULL,

    -- Storage
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,

    -- Training Info
    training_episodes INTEGER,
    total_timesteps BIGINT,
    final_reward DECIMAL(20, 8),

    -- Performance
    sharpe_ratio DECIMAL(10, 4),
    win_rate DECIMAL(5, 4),
    total_return DECIMAL(10, 4),

    -- Hyperparameters (JSON)
    hyperparameters JSONB,

    -- Status
    is_active BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index
CREATE INDEX IF NOT EXISTS idx_model_checkpoints_name ON model_checkpoints(model_name, created_at DESC);

-- ============================================
-- LLM Analysis Logs
-- ============================================
CREATE TABLE IF NOT EXISTS llm_analysis_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    analysis_type VARCHAR(50) NOT NULL, -- 'market', 'news', 'sentiment', 'feedback'
    model_name VARCHAR(100) NOT NULL,

    -- Request/Response
    prompt TEXT NOT NULL,
    response TEXT,

    -- Metrics
    tokens_used INTEGER,
    latency_ms INTEGER,

    -- Cost
    cost_usd DECIMAL(10, 6),

    -- Metadata
    metadata JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index
CREATE INDEX IF NOT EXISTS idx_llm_logs_type ON llm_analysis_logs(analysis_type, timestamp DESC);

-- ============================================
-- Row Level Security (RLS) Policies
-- ============================================
-- Enable RLS on all tables (optional - uncomment if needed)
-- ALTER TABLE ohlcv ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trade_logs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE performance_snapshots ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE model_checkpoints ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE llm_analysis_logs ENABLE ROW LEVEL SECURITY;

-- ============================================
-- Functions
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for trade_logs
DROP TRIGGER IF EXISTS update_trade_logs_updated_at ON trade_logs;
CREATE TRIGGER update_trade_logs_updated_at
    BEFORE UPDATE ON trade_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Views
-- ============================================

-- Daily Performance View
CREATE OR REPLACE VIEW daily_performance AS
SELECT
    DATE(entry_time) as trade_date,
    strategy_name,
    COUNT(*) as total_trades,
    SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN net_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    ROUND(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    SUM(net_pnl) as total_pnl,
    AVG(net_pnl) as avg_pnl,
    MAX(net_pnl) as max_win,
    MIN(net_pnl) as max_loss
FROM trade_logs
WHERE exit_time IS NOT NULL
GROUP BY DATE(entry_time), strategy_name
ORDER BY trade_date DESC;

-- Latest Signals View
CREATE OR REPLACE VIEW latest_signals AS
SELECT
    s.*,
    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as signal_rank
FROM signals s
WHERE timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;
