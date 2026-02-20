// Daily Performance Report Edge Function
// Tracks new system performance since Feb 20, 2026

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const NEW_SYSTEM_DATE = "2026-02-20T00:00:00Z";

interface PerformanceReport {
  timestamp: string;

  // Overall stats
  overall: {
    totalEquity: number;
    totalPnl: number;
    totalTrades: number;
    winRate: number;
    sharpeRatio: number;
    profitFactor: number;
    openPositions: number;
  };

  // New system stats (since Feb 20)
  newSystem: {
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnl: number;
    avgReturn: number;
    daysActive: number;
  };

  // By symbol/side
  breakdown: Record<string, { wins: number; losses: number; winRate: number }>;

  // Go-live readiness
  goLiveChecklist: {
    trades100: boolean;
    winRate62: boolean;
    sharpe13: boolean;
    profitFactor14: boolean;
    noMajorDrawdown: boolean;
    readyForLive: boolean;
    estimatedReadyDate: string;
  };

  // Open positions
  openPositions: Array<{ symbol: string; side: string; entryPrice: number; entryTime: string }>;
}

serve(async (req) => {
  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Get latest performance snapshot
    const { data: perfData } = await supabase
      .from("performance_snapshots")
      .select("*")
      .eq("strategy_name", "edge_paper")
      .order("timestamp", { ascending: false })
      .limit(1);

    const perf = perfData?.[0];

    // Get all trades since new system
    const { data: newTrades } = await supabase
      .from("trade_logs")
      .select("symbol, side, return_pct, net_pnl, entry_time, exit_time")
      .eq("strategy_name", "edge_paper")
      .gte("entry_time", NEW_SYSTEM_DATE)
      .not("exit_time", "is", null);

    // Get open positions
    const { data: openPos } = await supabase
      .from("trade_logs")
      .select("symbol, side, entry_price, entry_time")
      .eq("strategy_name", "edge_paper")
      .is("exit_time", null);

    // Calculate new system stats
    const trades = newTrades || [];
    const wins = trades.filter(t => (t.return_pct || 0) > 0).length;
    const losses = trades.length - wins;
    const totalPnl = trades.reduce((sum, t) => sum + (t.net_pnl || 0), 0);
    const avgReturn = trades.length > 0
      ? trades.reduce((sum, t) => sum + (t.return_pct || 0), 0) / trades.length
      : 0;
    const winRate = trades.length > 0 ? (wins / trades.length) * 100 : 0;

    // Calculate days active
    const startDate = new Date(NEW_SYSTEM_DATE);
    const now = new Date();
    const daysActive = Math.floor((now.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

    // Breakdown by symbol/side
    const breakdown: Record<string, { wins: number; losses: number; winRate: number }> = {};
    for (const t of trades) {
      const key = `${t.symbol}_${t.side}`;
      if (!breakdown[key]) {
        breakdown[key] = { wins: 0, losses: 0, winRate: 0 };
      }
      if ((t.return_pct || 0) > 0) {
        breakdown[key].wins++;
      } else {
        breakdown[key].losses++;
      }
    }
    for (const key in breakdown) {
      const b = breakdown[key];
      b.winRate = (b.wins / (b.wins + b.losses)) * 100;
    }

    // Go-live checklist
    const trades100 = trades.length >= 100;
    const winRate62 = winRate >= 62;
    const sharpe13 = (perf?.sharpe_ratio || 0) >= 1.3;
    const profitFactor14 = (perf?.profit_factor || 0) >= 1.4;
    const noMajorDrawdown = (perf?.total_pnl || 0) > -800; // Max 8% drawdown on 10k

    const criteriamet = [trades100, winRate62, sharpe13, profitFactor14, noMajorDrawdown].filter(Boolean).length;
    const readyForLive = criteriamet >= 4;

    // Estimate ready date based on current trade rate
    const tradesPerDay = daysActive > 0 ? trades.length / daysActive : 5;
    const tradesNeeded = Math.max(0, 100 - trades.length);
    const daysNeeded = tradesPerDay > 0 ? Math.ceil(tradesNeeded / tradesPerDay) : 30;
    const estimatedDate = new Date(now.getTime() + daysNeeded * 24 * 60 * 60 * 1000);
    const estimatedReadyDate = tradesNeeded > 0
      ? estimatedDate.toISOString().split('T')[0]
      : "Ready for validation!";

    const report: PerformanceReport = {
      timestamp: new Date().toISOString(),
      overall: {
        totalEquity: perf?.total_equity || 10000,
        totalPnl: perf?.total_pnl || 0,
        totalTrades: perf?.total_trades || 0,
        winRate: (perf?.win_rate || 0) * 100,
        sharpeRatio: perf?.sharpe_ratio || 0,
        profitFactor: perf?.profit_factor || 0,
        openPositions: perf?.open_positions || 0,
      },
      newSystem: {
        totalTrades: trades.length,
        wins,
        losses,
        winRate,
        totalPnl,
        avgReturn,
        daysActive,
      },
      breakdown,
      goLiveChecklist: {
        trades100,
        winRate62,
        sharpe13,
        profitFactor14,
        noMajorDrawdown,
        readyForLive,
        estimatedReadyDate,
      },
      openPositions: (openPos || []).map(p => ({
        symbol: p.symbol,
        side: p.side,
        entryPrice: parseFloat(p.entry_price),
        entryTime: p.entry_time,
      })),
    };

    // Store daily report (table might not exist yet, that's ok)
    try {
      await supabase.from("daily_reports").insert({
        timestamp: report.timestamp,
        report_data: report,
        strategy_name: "edge_paper",
      });
    } catch (e) {
      console.log("Could not store report:", e);
    }

    console.log("📊 Daily Report Generated");
    console.log(`   New System Trades: ${trades.length}`);
    console.log(`   Win Rate: ${winRate.toFixed(1)}%`);
    console.log(`   Ready for Live: ${readyForLive ? "YES" : "NO"}`);

    return new Response(JSON.stringify(report, null, 2), {
      headers: { "Content-Type": "application/json" },
    });

  } catch (error) {
    console.error("Error generating report:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
});
