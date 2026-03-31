-- AI-powered trading analysis tables
-- Run this migration in Supabase SQL Editor

-- Table for per-trade AI analysis
CREATE TABLE IF NOT EXISTS trade_analysis (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    position_id text NOT NULL,
    analysis_text text NOT NULL,
    patterns_identified jsonb DEFAULT '[]'::jsonb,
    suggestion text DEFAULT '',
    model_used text DEFAULT '',
    tokens_used integer DEFAULT 0,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trade_analysis_position_id ON trade_analysis(position_id);
CREATE INDEX IF NOT EXISTS idx_trade_analysis_created_at ON trade_analysis(created_at DESC);

-- Table for daily/weekly AI reviews
CREATE TABLE IF NOT EXISTS ai_reviews (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    review_date date NOT NULL,
    period text NOT NULL DEFAULT 'daily',  -- 'daily' or 'weekly'
    summary text NOT NULL,
    strategy_insights jsonb DEFAULT '{}'::jsonb,
    suggestions jsonb DEFAULT '[]'::jsonb,
    model_used text DEFAULT '',
    tokens_used integer DEFAULT 0,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_reviews_date ON ai_reviews(review_date DESC);
CREATE INDEX IF NOT EXISTS idx_ai_reviews_period ON ai_reviews(period);

-- Add AI gate result column to signals table (for tracking gate decisions)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'signals' AND column_name = 'ai_gate_result'
    ) THEN
        ALTER TABLE signals ADD COLUMN ai_gate_result jsonb DEFAULT NULL;
    END IF;
END $$;

-- Enable RLS (row level security) - allow all for anon key
ALTER TABLE trade_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_reviews ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all for trade_analysis" ON trade_analysis;
DROP POLICY IF EXISTS "Allow all for ai_reviews" ON ai_reviews;
CREATE POLICY "Allow all for trade_analysis" ON trade_analysis FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for ai_reviews" ON ai_reviews FOR ALL USING (true) WITH CHECK (true);
