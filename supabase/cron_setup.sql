-- Setup pg_cron to trigger paper trading Edge Function
-- Run this in Supabase SQL Editor after deploying the Edge Function

-- Enable pg_cron extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Enable pg_net extension for HTTP calls
CREATE EXTENSION IF NOT EXISTS pg_net;

-- Grant usage to postgres role
GRANT USAGE ON SCHEMA cron TO postgres;

-- Create the cron job to run every 5 minutes
-- Replace YOUR_PROJECT_REF with your actual project reference
-- Replace YOUR_ANON_KEY with your actual anon key

SELECT cron.schedule(
  'paper-trading-job',           -- Job name
  '*/5 * * * *',                  -- Every 5 minutes
  $$
  SELECT net.http_post(
    url := 'https://YOUR_PROJECT_REF.supabase.co/functions/v1/paper-trader',
    headers := '{"Content-Type": "application/json", "Authorization": "Bearer YOUR_ANON_KEY"}'::jsonb,
    body := '{}'::jsonb
  ) AS request_id;
  $$
);

-- To verify the job was created:
-- SELECT * FROM cron.job;

-- To see job run history:
-- SELECT * FROM cron.job_run_details ORDER BY start_time DESC LIMIT 10;

-- To delete the job if needed:
-- SELECT cron.unschedule('paper-trading-job');
