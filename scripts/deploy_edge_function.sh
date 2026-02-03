#!/bin/bash
# Deploy Paper Trading Edge Function to Supabase

set -e

echo "=========================================="
echo "  Deploying Paper Trading Edge Function"
echo "=========================================="

# Navigate to project directory
cd /Users/young/p/invest

# Check if logged in
echo ""
echo "Step 1: Checking Supabase login..."
if ! supabase projects list &>/dev/null; then
    echo "❌ Not logged in. Running login..."
    supabase login
fi
echo "✅ Logged in to Supabase"

# Link project
echo ""
echo "Step 2: Linking to project..."
supabase link --project-ref ylukecwiyfpkqxzxfzcf || true
echo "✅ Project linked"

# Deploy the function
echo ""
echo "Step 3: Deploying Edge Function..."
supabase functions deploy paper-trader --no-verify-jwt

echo ""
echo "✅ Edge Function deployed!"
echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "1. Go to Supabase Dashboard:"
echo "   https://supabase.com/dashboard/project/ylukecwiyfpkqxzxfzcf/functions"
echo ""
echo "2. Test the function by clicking 'paper-trader' and then 'Invoke'"
echo ""
echo "3. To set up automatic scheduling (every 5 minutes):"
echo "   - Go to SQL Editor in Supabase Dashboard"
echo "   - Run the SQL in: supabase/cron_setup.sql"
echo "   - Replace YOUR_ANON_KEY with your actual anon key"
echo ""
echo "4. Monitor trades in your dashboard:"
echo "   http://localhost:8501"
echo ""
