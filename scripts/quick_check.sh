#!/bin/bash
# Quick performance check - just run: ./scripts/quick_check.sh

curl -s -X POST "https://ylukecwiyfpkqxzxfzcf.supabase.co/functions/v1/daily-report" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlsdWtlY3dpeWZwa3F4enhmemNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAwNjUxNzgsImV4cCI6MjA4NTY0MTE3OH0.AYzXc7EQZSKsIKLGeEuiX9mPjN9vUVSsiVO7qNJFUUk" \
  -H "Content-Type: application/json" 2>/dev/null | python3 -c "
import json, sys

data = json.load(sys.stdin)
o = data['overall']
n = data['newSystem']
g = data['goLiveChecklist']

print('=' * 50)
print('  AI TRADING SYSTEM - QUICK CHECK')
print('=' * 50)
print()
print('📊 OVERALL')
print(f\"   Equity: \${o['totalEquity']:,.2f} ({o['totalPnl']:+.2f})\")
print(f\"   Win Rate: {o['winRate']:.1f}% | Sharpe: {o['sharpeRatio']:.2f} | PF: {o['profitFactor']:.2f}\")
print()
print('🆕 NEW SYSTEM (since Feb 20)')
print(f\"   Trades: {n['totalTrades']} | Win Rate: {n['winRate']:.1f}%\")
print(f\"   P&L: \${n['totalPnl']:+.2f} | Days Active: {n['daysActive']}\")
print()
print('📋 GO-LIVE CHECKLIST')
print(f\"   [{'✅' if g['trades100'] else '⬜'}] 100+ trades\")
print(f\"   [{'✅' if g['winRate62'] else '⬜'}] Win rate ≥62%\")
print(f\"   [{'✅' if g['sharpe13'] else '⬜'}] Sharpe ≥1.3\")
print(f\"   [{'✅' if g['profitFactor14'] else '⬜'}] Profit Factor ≥1.4\")
print(f\"   [{'✅' if g['noMajorDrawdown'] else '⬜'}] No major drawdown\")
print()
if g['readyForLive']:
    print('🚀 READY FOR LIVE TRADING!')
else:
    print(f\"⏳ Estimated ready: {g['estimatedReadyDate']}\")
print()
print('📍 Open Positions:')
for p in data['openPositions']:
    print(f\"   {p['symbol']} {p['side'].upper()} @ \${p['entryPrice']:,.2f}\")
print('=' * 50)
"
