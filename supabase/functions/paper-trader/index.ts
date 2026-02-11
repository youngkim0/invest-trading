// Supabase Edge Function for Paper Trading
// Runs on Supabase servers 24/7 via cron trigger

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SYMBOLS = ["BTCUSDT", "ETHUSDT"];
const BINANCE_APIS = [
  "https://api.binance.us/api/v3",
  "https://api.binance.com/api/v3",
];

// CoinGecko mapping for fallback
const COINGECKO_MAP: Record<string, string> = {
  "BTCUSDT": "bitcoin",
  "ETHUSDT": "ethereum",
};

// Trading configuration
const CONFIG = {
  initialCapital: 10000,
  maxPositionPct: 0.2,
  stopLossPct: 0.02,
  takeProfitPct: 0.04,
  minConfidence: 0.5,
  rsiOversold: 30,
  rsiOverbought: 70,
};

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Signal {
  signal: string;
  confidence: number;
  reasoning: string;
  indicators: Record<string, number>;
}

interface Position {
  symbol: string;
  side: string;
  entry_price: number;
  quantity: number;
  entry_time: string;
  position_id: string;
}

// Fetch OHLCV data with fallback sources
async function fetchKlines(symbol: string, interval = "1h", limit = 100): Promise<Candle[]> {
  // Try Binance APIs first
  for (const baseUrl of BINANCE_APIS) {
    try {
      const url = `${baseUrl}/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
      const response = await fetch(url, { signal: AbortSignal.timeout(10000) });
      if (response.ok) {
        const data = await response.json();
        return data.map((d: any[]) => ({
          timestamp: d[0],
          open: parseFloat(d[1]),
          high: parseFloat(d[2]),
          low: parseFloat(d[3]),
          close: parseFloat(d[4]),
          volume: parseFloat(d[5]),
        }));
      }
    } catch (e) {
      console.log(`Binance API failed: ${baseUrl}, trying next...`);
    }
  }

  // Fallback to CoinGecko
  try {
    const coinId = COINGECKO_MAP[symbol] || "bitcoin";
    const url = `https://api.coingecko.com/api/v3/coins/${coinId}/ohlc?vs_currency=usd&days=7`;
    const response = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (response.ok) {
      const data = await response.json();
      return data.map((d: number[]) => ({
        timestamp: d[0],
        open: d[1],
        high: d[2],
        low: d[3],
        close: d[4],
        volume: 0,
      }));
    }
  } catch (e) {
    console.log(`CoinGecko API failed: ${e}`);
  }

  throw new Error(`Failed to fetch klines for ${symbol}`);
}

// Fetch current ticker with fallback sources
async function fetchTicker(symbol: string): Promise<{ price: number; change24h: number }> {
  // Try Binance APIs first
  for (const baseUrl of BINANCE_APIS) {
    try {
      const url = `${baseUrl}/ticker/24hr?symbol=${symbol}`;
      const response = await fetch(url, { signal: AbortSignal.timeout(10000) });
      if (response.ok) {
        const data = await response.json();
        return {
          price: parseFloat(data.lastPrice),
          change24h: parseFloat(data.priceChangePercent),
        };
      }
    } catch (e) {
      console.log(`Binance ticker failed: ${baseUrl}, trying next...`);
    }
  }

  // Fallback to CoinGecko
  try {
    const coinId = COINGECKO_MAP[symbol] || "bitcoin";
    const url = `https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd&include_24hr_change=true`;
    const response = await fetch(url, { signal: AbortSignal.timeout(10000) });
    if (response.ok) {
      const data = await response.json();
      const coinData = data[coinId];
      return {
        price: coinData.usd,
        change24h: coinData.usd_24h_change || 0,
      };
    }
  } catch (e) {
    console.log(`CoinGecko ticker failed: ${e}`);
  }

  throw new Error(`Failed to fetch ticker for ${symbol}`);
}

// Calculate RSI
function calculateRSI(prices: number[], period = 14): number {
  if (prices.length < period + 1) return 50;

  const changes = prices.slice(1).map((p, i) => p - prices[i]);
  const gains = changes.map(c => c > 0 ? c : 0);
  const losses = changes.map(c => c < 0 ? -c : 0);

  const recentGains = gains.slice(-period);
  const recentLosses = losses.slice(-period);

  const avgGain = recentGains.reduce((a, b) => a + b, 0) / period;
  const avgLoss = recentLosses.reduce((a, b) => a + b, 0) / period;

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

// Calculate EMA
function calculateEMA(prices: number[], period: number): number[] {
  const multiplier = 2 / (period + 1);
  const ema: number[] = [prices[0]];

  for (let i = 1; i < prices.length; i++) {
    ema.push((prices[i] - ema[i - 1]) * multiplier + ema[i - 1]);
  }

  return ema;
}

// Calculate MACD
function calculateMACD(prices: number[]): { macd: number; signal: number; histogram: number } {
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);

  const macdLine = ema12.map((e, i) => e - ema26[i]);
  const signalLine = calculateEMA(macdLine, 9);

  const macd = macdLine[macdLine.length - 1];
  const signal = signalLine[signalLine.length - 1];

  return {
    macd,
    signal,
    histogram: macd - signal,
  };
}

// Calculate SMA
function calculateSMA(prices: number[], period: number): number {
  const slice = prices.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}

// Generate trading signal
function generateSignal(candles: Candle[]): Signal {
  if (candles.length < 60) {
    return { signal: "hold", confidence: 0.5, reasoning: "Insufficient data", indicators: {} };
  }

  const prices = candles.map(c => c.close);
  const currentPrice = prices[prices.length - 1];

  // Calculate indicators
  const rsi = calculateRSI(prices);
  const macd = calculateMACD(prices);
  const sma20 = calculateSMA(prices, 20);
  const sma50 = calculateSMA(prices, 50);

  // Previous SMAs for crossover detection
  const prevPrices = prices.slice(0, -1);
  const prevSma20 = calculateSMA(prevPrices, 20);
  const prevSma50 = calculateSMA(prevPrices, 50);

  const bullishCross = sma20 > sma50 && prevSma20 <= prevSma50;
  const bearishCross = sma20 < sma50 && prevSma20 >= prevSma50;

  // Scoring
  let buyScore = 0;
  let sellScore = 0;
  const reasons: string[] = [];

  // RSI signals
  if (rsi < CONFIG.rsiOversold) {
    buyScore += 2;
    reasons.push(`RSI oversold (${rsi.toFixed(1)})`);
  } else if (rsi > CONFIG.rsiOverbought) {
    sellScore += 2;
    reasons.push(`RSI overbought (${rsi.toFixed(1)})`);
  } else if (rsi < 45) {
    buyScore += 1;
  } else if (rsi > 55) {
    sellScore += 1;
  }

  // MACD signals
  if (macd.histogram > 0 && macd.macd > macd.signal) {
    buyScore += 1;
    reasons.push("MACD bullish");
  } else if (macd.histogram < 0 && macd.macd < macd.signal) {
    sellScore += 1;
    reasons.push("MACD bearish");
  }

  // SMA crossover
  if (bullishCross) {
    buyScore += 2;
    reasons.push("SMA bullish crossover");
  } else if (bearishCross) {
    sellScore += 2;
    reasons.push("SMA bearish crossover");
  } else if (sma20 > sma50) {
    buyScore += 0.5;
  } else {
    sellScore += 0.5;
  }

  // Determine signal
  const totalScore = buyScore - sellScore;
  const maxScore = 5.0;

  let signal: string;
  let confidence: number;

  // More aggressive thresholds for paper trading
  if (totalScore >= 0.5) {
    signal = totalScore >= 2 ? "strong_buy" : "buy";
    confidence = Math.min(0.9, 0.55 + (totalScore / maxScore) * 0.35);
  } else if (totalScore <= -0.5) {
    signal = totalScore <= -2 ? "strong_sell" : "sell";
    confidence = Math.min(0.9, 0.55 + (Math.abs(totalScore) / maxScore) * 0.35);
  } else {
    signal = "hold";
    confidence = 0.5;
  }

  return {
    signal,
    confidence,
    reasoning: reasons.length > 0 ? reasons.join(", ") : "No strong signals",
    indicators: {
      rsi,
      macd: macd.macd,
      macd_signal: macd.signal,
      macd_histogram: macd.histogram,
      sma20,
      sma50,
      price: currentPrice,
    },
  };
}

// Main handler
serve(async (req) => {
  try {
    // Initialize Supabase client
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    console.log("🚀 Paper Trading Edge Function triggered");

    const results: any[] = [];

    for (const symbol of SYMBOLS) {
      console.log(`Processing ${symbol}...`);

      // Fetch market data
      const candles = await fetchKlines(symbol);
      const ticker = await fetchTicker(symbol);

      // Generate signal
      const signal = generateSignal(candles);

      // Log signal to database
      await supabase.from("signals").insert({
        timestamp: new Date().toISOString(),
        symbol,
        exchange: "binance",
        timeframe: "1h",
        signal_type: signal.signal,
        source: "edge_function",
        confidence: signal.confidence,
        entry_price: ticker.price,
        reasoning: signal.reasoning,
        indicators: signal.indicators,
        status: "pending",
      });

      // Check for existing position
      const { data: existingPositions } = await supabase
        .from("trade_logs")
        .select("*")
        .eq("symbol", symbol)
        .eq("strategy_name", "edge_paper")
        .is("exit_time", null)
        .limit(1);

      const hasPosition = existingPositions && existingPositions.length > 0;
      const position = hasPosition ? existingPositions[0] : null;

      if (hasPosition && position) {
        // Check for exit conditions
        const entryPrice = parseFloat(position.entry_price);
        const pnlPct = position.side === "buy"
          ? (ticker.price - entryPrice) / entryPrice
          : (entryPrice - ticker.price) / entryPrice;

        let shouldExit = false;
        let exitReason = "";

        // Stop loss
        if (pnlPct < -CONFIG.stopLossPct) {
          shouldExit = true;
          exitReason = "Stop loss triggered";
        }

        // Take profit
        if (pnlPct > CONFIG.takeProfitPct) {
          shouldExit = true;
          exitReason = "Take profit triggered";
        }

        // Opposite signal
        if (position.side === "buy" && signal.signal.includes("sell") && signal.confidence >= CONFIG.minConfidence) {
          shouldExit = true;
          exitReason = "Sell signal received";
        } else if (position.side === "sell" && signal.signal.includes("buy") && signal.confidence >= CONFIG.minConfidence) {
          shouldExit = true;
          exitReason = "Buy signal received";
        }

        if (shouldExit) {
          const quantity = parseFloat(position.quantity);
          const pnl = position.side === "buy"
            ? (ticker.price - entryPrice) * quantity
            : (entryPrice - ticker.price) * quantity;

          await supabase
            .from("trade_logs")
            .update({
              exit_price: ticker.price,
              exit_time: new Date().toISOString(),
              gross_pnl: pnl,
              net_pnl: pnl * 0.999,
              return_pct: pnlPct * 100,
              exit_reasoning: exitReason,
            })
            .eq("id", position.id);

          console.log(`📉 CLOSED ${symbol} | PnL: $${pnl.toFixed(2)} | Reason: ${exitReason}`);
          results.push({ symbol, action: "close", pnl, reason: exitReason });
        }
      } else {
        // Check for entry
        if (signal.confidence >= CONFIG.minConfidence) {
          if (signal.signal.includes("buy") || signal.signal.includes("sell")) {
            const side = signal.signal.includes("buy") ? "buy" : "sell";
            const positionValue = CONFIG.initialCapital * CONFIG.maxPositionPct;
            const quantity = positionValue / ticker.price;

            const positionId = crypto.randomUUID();

            await supabase.from("trade_logs").insert({
              position_id: positionId,
              symbol,
              exchange: "binance",
              side,
              entry_price: ticker.price,
              entry_time: new Date().toISOString(),
              quantity,
              strategy_name: "edge_paper",
              signal_source: "technical",
              signal_confidence: signal.confidence,
              entry_reasoning: signal.reasoning,
              indicators_at_entry: signal.indicators,
            });

            console.log(`📈 OPENED ${side.toUpperCase()} ${symbol} @ $${ticker.price.toFixed(2)}`);
            results.push({ symbol, action: "open", side, price: ticker.price });
          }
        }
      }

      results.push({
        symbol,
        price: ticker.price,
        signal: signal.signal,
        confidence: signal.confidence,
        reasoning: signal.reasoning,
      });
    }

    // Save performance snapshot
    const { data: allTrades } = await supabase
      .from("trade_logs")
      .select("*")
      .eq("strategy_name", "edge_paper");

    const completedTrades = allTrades?.filter(t => t.exit_time) || [];
    const totalPnl = completedTrades.reduce((sum, t) => sum + parseFloat(t.net_pnl || 0), 0);
    const winningTrades = completedTrades.filter(t => parseFloat(t.net_pnl || 0) > 0).length;

    await supabase.from("performance_snapshots").insert({
      timestamp: new Date().toISOString(),
      strategy_name: "edge_paper",
      total_equity: CONFIG.initialCapital + totalPnl,
      cash_balance: CONFIG.initialCapital + totalPnl,
      positions_value: 0,
      total_pnl: totalPnl,
      total_trades: completedTrades.length,
      winning_trades: winningTrades,
      losing_trades: completedTrades.length - winningTrades,
      win_rate: completedTrades.length > 0 ? winningTrades / completedTrades.length : 0,
    });

    console.log("✅ Paper trading cycle complete");

    return new Response(JSON.stringify({ success: true, results }), {
      headers: { "Content-Type": "application/json" },
    });

  } catch (error) {
    console.error("Error:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
});
