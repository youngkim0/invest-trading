#!/bin/bash
# Performance Tracking Script for AI Trading System
# Tracks performance since new filters deployed (Feb 20, 2026)

SUPABASE_URL="https://ylukecwiyfpkqxzxfzcf.supabase.co"
ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlsdWtlY3dpeWZwa3F4enhmemNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAwNjUxNzgsImV4cCI6MjA4NTY0MTE3OH0.AYzXc7EQZSKsIKLGeEuiX9mPjN9vUVSsiVO7qNJFUUk"

# New system start date
NEW_SYSTEM_DATE="2026-02-20T00:00:00"

echo "=============================================="
echo "  AI TRADING SYSTEM - PERFORMANCE TRACKER"
echo "=============================================="
echo ""
echo "📅 New System Start: Feb 20, 2026"
echo "📅 Current Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Get latest performance snapshot
echo "📊 OVERALL PERFORMANCE"
echo "----------------------------------------------"
curl -s "${SUPABASE_URL}/rest/v1/performance_snapshots?select=total_equity,total_pnl,total_trades,winning_trades,losing_trades,win_rate,sharpe_ratio,profit_factor,open_positions&strategy_name=eq.edge_paper&order=timestamp.desc&limit=1" \
  -H "apikey: ${ANON_KEY}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data:
    d = data[0]
    print(f\"💰 Total Equity:    \${d['total_equity']:,.2f}\")
    print(f\"📈 Total P&L:       \${d['total_pnl']:+,.2f} ({d['total_pnl']/100:+.2f}%)\")
    print(f\"🎯 Win Rate:        {d['win_rate']*100:.1f}% ({d['winning_trades']}/{d['winning_trades']+d['losing_trades']})\")
    print(f\"📊 Sharpe Ratio:    {d['sharpe_ratio']:.2f}\")
    print(f\"⚖️  Profit Factor:   {d['profit_factor']:.2f}\")
    print(f\"📍 Open Positions:  {d['open_positions']}\")
"

echo ""
echo "🆕 NEW SYSTEM PERFORMANCE (Since Feb 20)"
echo "----------------------------------------------"

# Get trades since new system
curl -s "${SUPABASE_URL}/rest/v1/trade_logs?select=return_pct,net_pnl,entry_time&strategy_name=eq.edge_paper&entry_time=gte.${NEW_SYSTEM_DATE}&exit_time=not.is.null" \
  -H "apikey: ${ANON_KEY}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data:
    print('No completed trades yet with new system.')
    print('Keep monitoring - need 100+ trades for validation.')
else:
    wins = sum(1 for t in data if t['return_pct'] > 0)
    losses = len(data) - wins
    total_pnl = sum(t['net_pnl'] or 0 for t in data)
    avg_return = sum(t['return_pct'] or 0 for t in data) / len(data) if data else 0
    win_rate = wins / len(data) * 100 if data else 0

    print(f'Total Trades:    {len(data)}')
    print(f'Win Rate:        {win_rate:.1f}% ({wins}W / {losses}L)')
    print(f'Total P&L:       \${total_pnl:+,.2f}')
    print(f'Avg Return:      {avg_return:+.2f}%')
    print()

    # Progress to go-live
    trades_needed = max(0, 100 - len(data))
    print(f'📋 GO-LIVE CHECKLIST:')
    print(f'   [{'✅' if len(data) >= 100 else '⬜'}] 100+ trades ({len(data)}/100)')
    print(f'   [{'✅' if win_rate >= 62 else '⬜'}] Win rate ≥62% ({win_rate:.1f}%)')
    print(f'   [{'✅' if avg_return > 0 else '⬜'}] Positive avg return ({avg_return:+.2f}%)')

    if trades_needed > 0:
        print(f'')
        print(f'⏳ {trades_needed} more trades needed for validation')
"

echo ""
echo "📈 SYMBOL BREAKDOWN (New System)"
echo "----------------------------------------------"

curl -s "${SUPABASE_URL}/rest/v1/trade_logs?select=symbol,side,return_pct&strategy_name=eq.edge_paper&entry_time=gte.${NEW_SYSTEM_DATE}&exit_time=not.is.null" \
  -H "apikey: ${ANON_KEY}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data:
    print('No data yet.')
else:
    stats = {}
    for t in data:
        key = f\"{t['symbol']} {t['side'].upper()}\"
        if key not in stats:
            stats[key] = {'wins': 0, 'losses': 0}
        if t['return_pct'] > 0:
            stats[key]['wins'] += 1
        else:
            stats[key]['losses'] += 1

    for key in sorted(stats.keys()):
        s = stats[key]
        total = s['wins'] + s['losses']
        wr = s['wins'] / total * 100 if total > 0 else 0
        print(f\"{key}: {wr:.0f}% ({s['wins']}W/{s['losses']}L)\")
"

echo ""
echo "🔄 OPEN POSITIONS"
echo "----------------------------------------------"

curl -s "${SUPABASE_URL}/rest/v1/trade_logs?select=symbol,side,entry_price,entry_time&strategy_name=eq.edge_paper&exit_time=is.null" \
  -H "apikey: ${ANON_KEY}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data:
    print('No open positions.')
else:
    for t in data:
        print(f\"{t['symbol']} {t['side'].upper()} @ \${float(t['entry_price']):,.2f}\")
"

echo ""
echo "=============================================="
echo "  Run this script daily to track progress"
echo "=============================================="
