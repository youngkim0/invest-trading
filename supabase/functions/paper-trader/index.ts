// Supabase Edge Function for Paper Trading
// Advanced Strategy with TradingView-style indicators + Claude AI Brain
// Runs on Supabase servers 24/7 via cron trigger

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SYMBOLS = ["BTCUSDT", "ETHUSDT"];
const BINANCE_APIS = [
  "https://api.binance.us/api/v3",
  "https://api.binance.com/api/v3",
];

const COINGECKO_MAP: Record<string, string> = {
  "BTCUSDT": "bitcoin",
  "ETHUSDT": "ethereum",
};

// Advanced Trading Configuration
const CONFIG = {
  initialCapital: 10000,
  maxPositionPct: 0.15,        // More conservative position sizing
  stopLossPct: 0.025,          // 2.5% stop loss
  takeProfitPct: 0.05,         // 5% take profit
  trailingStopPct: 0.015,      // 1.5% trailing stop
  minConfidence: 0.6,          // Base confidence threshold
  maxDrawdownPct: 0.10,        // Max 10% drawdown before pausing
  cooldownMinutes: 30,         // Cooldown after loss

  // NEW: Symbol-Side Win Rate Optimization
  // Based on historical analysis: ETH BUY=71%, BTC SELL=58%, ETH SELL=57%, BTC BUY=52%
  // Adjusted Feb 22: Lower thresholds for more trades
  symbolSideFilters: {
    "BTCUSDT_buy": {
      minConfidence: 0.60,      // Lowered from 0.65
      requireMTFAlignment: false, // Removed MTF requirement for more trades
      requireSqueeze: false,
      positionMultiplier: 0.7,   // Still smaller position size (worst performer)
    },
    "BTCUSDT_sell": {
      minConfidence: 0.55,      // Lowered from 0.60
      requireMTFAlignment: false,
      requireSqueeze: false,
      positionMultiplier: 1.0,
    },
    "ETHUSDT_buy": {
      minConfidence: 0.55,       // Lowered from 0.58
      requireMTFAlignment: false,
      requireSqueeze: false,
      positionMultiplier: 1.2,   // Larger position size (best performer)
    },
    "ETHUSDT_sell": {
      minConfidence: 0.55,      // Lowered from 0.60
      requireMTFAlignment: false,
      requireSqueeze: false,
      positionMultiplier: 1.0,
    },
  } as Record<string, { minConfidence: number; requireMTFAlignment: boolean; requireSqueeze: boolean; positionMultiplier: number }>,

  // Loss protection
  maxConsecutiveLosses: 3,     // Pause after 3 consecutive losses
  lossRecoveryTrades: 2,       // Need 2 wins to reset loss counter

  // Correlation filter - avoid same-direction bets on correlated assets
  useCorrelationFilter: true,  // Don't open BTC & ETH in same direction
  correlatedPairs: [["BTCUSDT", "ETHUSDT"]], // Define correlated pairs

  // Indicator thresholds
  rsiOversold: 30,
  rsiOverbought: 70,
  stochOversold: 20,
  stochOverbought: 80,
  adxTrendStrength: 25,        // ADX > 25 = strong trend
  bbWidthMin: 0.02,            // Minimum BB width for volatility

  // Gemini AI
  useGeminiAI: true,
  geminiModel: "gemini-2.5-flash",
};

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface AdvancedIndicators {
  // Trend
  rsi: number;
  stochRsi: { k: number; d: number };
  macd: { macd: number; signal: number; histogram: number };
  adx: number;
  plusDI: number;
  minusDI: number;

  // Moving Averages
  sma20: number;
  sma50: number;
  sma200: number;
  ema9: number;
  ema21: number;

  // Volatility
  atr: number;
  atrPercent: number;
  bollingerBands: { upper: number; middle: number; lower: number; width: number };
  parkinsonVolatility: number;  // NEW: More efficient volatility estimator
  garmanKlassVolatility: number; // NEW: OHLC-based volatility

  // Volume
  obv: number;
  obvTrend: string;
  volumeSma: number;
  volumeRatio: number;
  volumeMomentum: number;  // NEW: Volume rate of change
  buyPressure: number;     // NEW: Estimated buy vs sell pressure

  // Support/Resistance
  pivotPoints: { pivot: number; r1: number; r2: number; s1: number; s2: number };
  supportLevels: number[];  // NEW: Key support levels
  resistanceLevels: number[]; // NEW: Key resistance levels

  // Price Action
  currentPrice: number;
  priceChange24h: number;
  highLowRange: number;
  candlePattern: string;

  // Market Regime
  marketRegime: "trending_up" | "trending_down" | "ranging" | "volatile";
  trendStrength: number;

  // NEW: Multi-Timeframe Momentum
  momentum: {
    m1h: number;   // 1 hour momentum
    m4h: number;   // 4 hour momentum
    m12h: number;  // 12 hour momentum
    m24h: number;  // 24 hour momentum
    m72h: number;  // 3 day momentum
    alignment: number; // -1 to 1, all timeframes aligned
  };

  // NEW: Mean Reversion Indicators
  meanReversion: {
    zScore: number;         // Standard deviations from mean
    percentileRank: number; // Where current price sits in range (0-100)
    rsiDivergence: boolean; // Price/RSI divergence detected
    bbPosition: number;     // 0 = lower band, 1 = upper band
  };

  // NEW: Risk-Adjusted Returns
  riskMetrics: {
    sharpeRatio6h: number;
    sortinoRatio6h: number;
    volatilityRegime: "low" | "medium" | "high" | "extreme";
  };

  // NEW: TradingView-Style Advanced Indicators
  squeezeMomentum: {
    isSqueezing: boolean;      // BB inside Keltner = squeeze
    momentum: number;          // Momentum value
    momentumColor: "lime" | "green" | "red" | "maroon"; // Histogram color
    squeezeCount: number;      // How many bars in squeeze
  };

  superTrend: {
    value: number;             // SuperTrend line value
    direction: "up" | "down";  // Current trend direction
    flipRecent: boolean;       // Trend flipped in last 3 bars
  };

  vwap: {
    value: number;             // VWAP value
    upperBand1: number;        // +1 StdDev
    upperBand2: number;        // +2 StdDev
    lowerBand1: number;        // -1 StdDev
    lowerBand2: number;        // -2 StdDev
    pricePosition: number;     // -2 to +2 (which band zone)
  };

  waveTrend: {
    wt1: number;               // Main WaveTrend line
    wt2: number;               // Signal line
    crossUp: boolean;          // Bullish cross
    crossDown: boolean;        // Bearish cross
    overbought: boolean;       // WT1 > 60
    oversold: boolean;         // WT1 < -60
  };

  fairValueGaps: {
    bullishFVG: number[];      // Price levels of unfilled bullish FVGs
    bearishFVG: number[];      // Price levels of unfilled bearish FVGs
    nearestBullish: number | null;
    nearestBearish: number | null;
  };
}

interface Signal {
  signal: string;
  confidence: number;
  reasoning: string;
  indicators: AdvancedIndicators;
  aiAnalysis?: string;
  riskScore: number;
}

// ============================================
// TECHNICAL INDICATOR CALCULATIONS
// ============================================

function calculateRSI(prices: number[], period = 14): number {
  if (prices.length < period + 1) return 50;

  const changes = prices.slice(1).map((p, i) => p - prices[i]);
  let avgGain = 0;
  let avgLoss = 0;

  // Initial averages
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i];
    else avgLoss += Math.abs(changes[i]);
  }
  avgGain /= period;
  avgLoss /= period;

  // Smoothed RSI
  for (let i = period; i < changes.length; i++) {
    if (changes[i] > 0) {
      avgGain = (avgGain * (period - 1) + changes[i]) / period;
      avgLoss = (avgLoss * (period - 1)) / period;
    } else {
      avgGain = (avgGain * (period - 1)) / period;
      avgLoss = (avgLoss * (period - 1) + Math.abs(changes[i])) / period;
    }
  }

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

function calculateStochRSI(prices: number[], period = 14, smoothK = 3, smoothD = 3): { k: number; d: number } {
  const rsiValues: number[] = [];
  for (let i = period; i <= prices.length; i++) {
    rsiValues.push(calculateRSI(prices.slice(0, i), period));
  }

  if (rsiValues.length < period) return { k: 50, d: 50 };

  const recentRsi = rsiValues.slice(-period);
  const minRsi = Math.min(...recentRsi);
  const maxRsi = Math.max(...recentRsi);

  const stochRsi = maxRsi === minRsi ? 50 : ((rsiValues[rsiValues.length - 1] - minRsi) / (maxRsi - minRsi)) * 100;

  // Simplified K and D
  const k = stochRsi;
  const d = k; // Would need more history for proper smoothing

  return { k, d };
}

function calculateEMA(prices: number[], period: number): number[] {
  const multiplier = 2 / (period + 1);
  const ema: number[] = [prices[0]];

  for (let i = 1; i < prices.length; i++) {
    ema.push((prices[i] - ema[i - 1]) * multiplier + ema[i - 1]);
  }

  return ema;
}

function calculateSMA(prices: number[], period: number): number {
  if (prices.length < period) return prices[prices.length - 1];
  const slice = prices.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}

function calculateMACD(prices: number[]): { macd: number; signal: number; histogram: number } {
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);

  const macdLine = ema12.map((e, i) => e - ema26[i]);
  const signalLine = calculateEMA(macdLine, 9);

  const macd = macdLine[macdLine.length - 1];
  const signal = signalLine[signalLine.length - 1];

  return { macd, signal, histogram: macd - signal };
}

function calculateATR(candles: Candle[], period = 14): number {
  if (candles.length < period + 1) return 0;

  const trueRanges: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;

    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trueRanges.push(tr);
  }

  // EMA of true range
  const atr = calculateEMA(trueRanges, period);
  return atr[atr.length - 1];
}

function calculateADX(candles: Candle[], period = 14): { adx: number; plusDI: number; minusDI: number } {
  if (candles.length < period * 2) return { adx: 25, plusDI: 25, minusDI: 25 };

  const plusDM: number[] = [];
  const minusDM: number[] = [];
  const tr: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const highDiff = candles[i].high - candles[i - 1].high;
    const lowDiff = candles[i - 1].low - candles[i].low;

    plusDM.push(highDiff > lowDiff && highDiff > 0 ? highDiff : 0);
    minusDM.push(lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0);

    tr.push(Math.max(
      candles[i].high - candles[i].low,
      Math.abs(candles[i].high - candles[i - 1].close),
      Math.abs(candles[i].low - candles[i - 1].close)
    ));
  }

  const smoothedPlusDM = calculateEMA(plusDM, period);
  const smoothedMinusDM = calculateEMA(minusDM, period);
  const smoothedTR = calculateEMA(tr, period);

  const plusDI = (smoothedPlusDM[smoothedPlusDM.length - 1] / smoothedTR[smoothedTR.length - 1]) * 100;
  const minusDI = (smoothedMinusDM[smoothedMinusDM.length - 1] / smoothedTR[smoothedTR.length - 1]) * 100;

  const dx = Math.abs(plusDI - minusDI) / (plusDI + minusDI) * 100;

  return { adx: dx, plusDI, minusDI };
}

function calculateBollingerBands(prices: number[], period = 20, stdDev = 2): { upper: number; middle: number; lower: number; width: number } {
  const sma = calculateSMA(prices, period);
  const slice = prices.slice(-period);

  const variance = slice.reduce((sum, p) => sum + Math.pow(p - sma, 2), 0) / period;
  const std = Math.sqrt(variance);

  const upper = sma + stdDev * std;
  const lower = sma - stdDev * std;
  const width = (upper - lower) / sma;

  return { upper, middle: sma, lower, width };
}

function calculateOBV(candles: Candle[]): { obv: number; trend: string } {
  let obv = 0;
  const obvHistory: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    if (candles[i].close > candles[i - 1].close) {
      obv += candles[i].volume;
    } else if (candles[i].close < candles[i - 1].close) {
      obv -= candles[i].volume;
    }
    obvHistory.push(obv);
  }

  // OBV trend (compare current vs 10 periods ago)
  const lookback = Math.min(10, obvHistory.length - 1);
  const trend = obv > obvHistory[obvHistory.length - 1 - lookback] ? "bullish" : "bearish";

  return { obv, trend };
}

function calculatePivotPoints(candles: Candle[]): { pivot: number; r1: number; r2: number; s1: number; s2: number } {
  // Use previous day's high, low, close
  const recentCandles = candles.slice(-24); // Last 24 hours for 1h candles
  const high = Math.max(...recentCandles.map(c => c.high));
  const low = Math.min(...recentCandles.map(c => c.low));
  const close = recentCandles[recentCandles.length - 1].close;

  const pivot = (high + low + close) / 3;
  const r1 = 2 * pivot - low;
  const s1 = 2 * pivot - high;
  const r2 = pivot + (high - low);
  const s2 = pivot - (high - low);

  return { pivot, r1, r2, s1, s2 };
}

function detectCandlePattern(candles: Candle[]): string {
  if (candles.length < 3) return "none";

  const last = candles[candles.length - 1];
  const prev = candles[candles.length - 2];
  const prev2 = candles[candles.length - 3];

  const lastBody = Math.abs(last.close - last.open);
  const lastRange = last.high - last.low;
  const lastUpperWick = last.high - Math.max(last.close, last.open);
  const lastLowerWick = Math.min(last.close, last.open) - last.low;

  // Doji
  if (lastBody < lastRange * 0.1) return "doji";

  // Hammer (bullish reversal)
  if (lastLowerWick > lastBody * 2 && lastUpperWick < lastBody * 0.5 && last.close > last.open) {
    return "hammer";
  }

  // Shooting Star (bearish reversal)
  if (lastUpperWick > lastBody * 2 && lastLowerWick < lastBody * 0.5 && last.close < last.open) {
    return "shooting_star";
  }

  // Engulfing patterns
  if (last.close > last.open && prev.close < prev.open &&
      last.close > prev.open && last.open < prev.close) {
    return "bullish_engulfing";
  }
  if (last.close < last.open && prev.close > prev.open &&
      last.open > prev.close && last.close < prev.open) {
    return "bearish_engulfing";
  }

  // Morning/Evening Star
  const prev2Body = Math.abs(prev2.close - prev2.open);
  const prevBody = Math.abs(prev.close - prev.open);

  if (prev2.close < prev2.open && prevBody < prev2Body * 0.3 && last.close > last.open && last.close > (prev2.open + prev2.close) / 2) {
    return "morning_star";
  }
  if (prev2.close > prev2.open && prevBody < prev2Body * 0.3 && last.close < last.open && last.close < (prev2.open + prev2.close) / 2) {
    return "evening_star";
  }

  return "none";
}

function detectMarketRegime(indicators: Partial<AdvancedIndicators>): { regime: "trending_up" | "trending_down" | "ranging" | "volatile"; strength: number } {
  const { adx = 25, plusDI = 25, minusDI = 25, atrPercent = 2, bollingerBands } = indicators;

  // High volatility
  if (atrPercent > 4 || (bollingerBands && bollingerBands.width > 0.08)) {
    return { regime: "volatile", strength: Math.min(atrPercent / 5, 1) };
  }

  // Strong trend
  if (adx > CONFIG.adxTrendStrength) {
    if (plusDI > minusDI) {
      return { regime: "trending_up", strength: Math.min(adx / 50, 1) };
    } else {
      return { regime: "trending_down", strength: Math.min(adx / 50, 1) };
    }
  }

  // Ranging market
  return { regime: "ranging", strength: 1 - (adx / 50) };
}

// ============================================
// NEW: ADVANCED VOLATILITY ESTIMATORS
// ============================================

function calculateParkinsonVolatility(candles: Candle[], period = 24): number {
  // Parkinson volatility uses high/low - more efficient than close-to-close
  if (candles.length < period) return 0;

  const recentCandles = candles.slice(-period);
  const sum = recentCandles.reduce((acc, c) => {
    const logHL = Math.log(c.high / c.low);
    return acc + logHL * logHL;
  }, 0);

  const variance = sum / (4 * Math.log(2) * period);
  return Math.sqrt(variance) * Math.sqrt(365 * 24); // Annualized
}

function calculateGarmanKlassVolatility(candles: Candle[], period = 24): number {
  // Garman-Klass uses OHLC - even more efficient estimator
  if (candles.length < period) return 0;

  const recentCandles = candles.slice(-period);
  const sum = recentCandles.reduce((acc, c) => {
    const logHL = Math.log(c.high / c.low);
    const logCO = Math.log(c.close / c.open);
    return acc + 0.5 * logHL * logHL - (2 * Math.log(2) - 1) * logCO * logCO;
  }, 0);

  const variance = sum / period;
  return Math.sqrt(Math.max(0, variance)) * Math.sqrt(365 * 24); // Annualized
}

// ============================================
// NEW: MULTI-TIMEFRAME MOMENTUM
// ============================================

function calculateMomentum(prices: number[], lookback: number): number {
  if (prices.length < lookback + 1) return 0;
  const current = prices[prices.length - 1];
  const past = prices[prices.length - 1 - lookback];
  return (current / past - 1) * 100; // Percentage change
}

function calculateMultiTimeframeMomentum(candles: Candle[]): {
  m1h: number;
  m4h: number;
  m12h: number;
  m24h: number;
  m72h: number;
  alignment: number;
} {
  const prices = candles.map(c => c.close);

  const m1h = calculateMomentum(prices, 1);
  const m4h = calculateMomentum(prices, 4);
  const m12h = calculateMomentum(prices, 12);
  const m24h = calculateMomentum(prices, 24);
  const m72h = calculateMomentum(prices, Math.min(72, prices.length - 1));

  // Calculate alignment score (-1 to 1)
  // 1 = all timeframes positive, -1 = all negative, 0 = mixed
  const momentums = [m1h, m4h, m12h, m24h, m72h];
  const positives = momentums.filter(m => m > 0).length;
  const negatives = momentums.filter(m => m < 0).length;
  const alignment = (positives - negatives) / 5;

  return { m1h, m4h, m12h, m24h, m72h, alignment };
}

// ============================================
// NEW: MEAN REVERSION INDICATORS
// ============================================

function calculateMeanReversionIndicators(
  candles: Candle[],
  prices: number[],
  rsi: number,
  bollingerBands: { upper: number; middle: number; lower: number }
): {
  zScore: number;
  percentileRank: number;
  rsiDivergence: boolean;
  bbPosition: number;
} {
  const currentPrice = prices[prices.length - 1];

  // Z-Score: How many standard deviations from 50-period mean
  const period = 50;
  const recentPrices = prices.slice(-period);
  const mean = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
  const variance = recentPrices.reduce((acc, p) => acc + Math.pow(p - mean, 2), 0) / recentPrices.length;
  const stdDev = Math.sqrt(variance);
  const zScore = stdDev > 0 ? (currentPrice - mean) / stdDev : 0;

  // Percentile Rank: Where current price sits in recent range (0-100)
  const lookback = Math.min(100, prices.length);
  const lookbackPrices = prices.slice(-lookback);
  const sortedPrices = [...lookbackPrices].sort((a, b) => a - b);
  const rank = sortedPrices.findIndex(p => p >= currentPrice);
  const percentileRank = (rank / sortedPrices.length) * 100;

  // RSI Divergence: Price making new highs/lows but RSI not confirming
  let rsiDivergence = false;
  if (candles.length >= 20) {
    const recentHighs = candles.slice(-20).map(c => c.high);
    const currentHigh = Math.max(...recentHighs.slice(-5));
    const prevHigh = Math.max(...recentHighs.slice(0, 15));

    // Bearish divergence: price making higher highs but RSI falling
    if (currentHigh > prevHigh && rsi < 50) {
      rsiDivergence = true;
    }

    // Bullish divergence: price making lower lows but RSI rising
    const recentLows = candles.slice(-20).map(c => c.low);
    const currentLow = Math.min(...recentLows.slice(-5));
    const prevLow = Math.min(...recentLows.slice(0, 15));

    if (currentLow < prevLow && rsi > 50) {
      rsiDivergence = true;
    }
  }

  // BB Position: 0 = at lower band, 1 = at upper band
  const bbRange = bollingerBands.upper - bollingerBands.lower;
  const bbPosition = bbRange > 0
    ? (currentPrice - bollingerBands.lower) / bbRange
    : 0.5;

  return { zScore, percentileRank, rsiDivergence, bbPosition: Math.max(0, Math.min(1, bbPosition)) };
}

// ============================================
// NEW: SUPPORT & RESISTANCE DETECTION
// ============================================

function detectSupportResistance(candles: Candle[], numLevels = 3): { support: number[]; resistance: number[] } {
  if (candles.length < 20) return { support: [], resistance: [] };

  const prices = candles.map(c => ({ high: c.high, low: c.low, close: c.close }));
  const currentPrice = prices[prices.length - 1].close;

  // Find local minima (support) and maxima (resistance)
  const levels: { price: number; type: "support" | "resistance"; strength: number }[] = [];

  for (let i = 5; i < prices.length - 5; i++) {
    const localHigh = prices[i].high;
    const localLow = prices[i].low;

    // Check if this is a local maximum (resistance)
    let isResistance = true;
    let isSupport = true;

    for (let j = -5; j <= 5; j++) {
      if (j !== 0) {
        if (prices[i + j].high >= localHigh) isResistance = false;
        if (prices[i + j].low <= localLow) isSupport = false;
      }
    }

    if (isResistance) {
      // Count how many times this level was tested
      const tolerance = localHigh * 0.005; // 0.5% tolerance
      const touches = prices.filter(p => Math.abs(p.high - localHigh) < tolerance).length;
      levels.push({ price: localHigh, type: "resistance", strength: touches });
    }

    if (isSupport) {
      const tolerance = localLow * 0.005;
      const touches = prices.filter(p => Math.abs(p.low - localLow) < tolerance).length;
      levels.push({ price: localLow, type: "support", strength: touches });
    }
  }

  // Sort by strength and filter to relevant levels
  const supportLevels = levels
    .filter(l => l.type === "support" && l.price < currentPrice)
    .sort((a, b) => b.strength - a.strength)
    .slice(0, numLevels)
    .map(l => l.price)
    .sort((a, b) => b - a); // Nearest first

  const resistanceLevels = levels
    .filter(l => l.type === "resistance" && l.price > currentPrice)
    .sort((a, b) => b.strength - a.strength)
    .slice(0, numLevels)
    .map(l => l.price)
    .sort((a, b) => a - b); // Nearest first

  return { support: supportLevels, resistance: resistanceLevels };
}

// ============================================
// NEW: VOLUME ANALYSIS
// ============================================

function calculateVolumeMetrics(candles: Candle[]): { volumeMomentum: number; buyPressure: number } {
  if (candles.length < 20) return { volumeMomentum: 0, buyPressure: 0.5 };

  const volumes = candles.map(c => c.volume);

  // Volume momentum: ROC of volume
  const recentVolume = volumes.slice(-5).reduce((a, b) => a + b, 0) / 5;
  const prevVolume = volumes.slice(-15, -5).reduce((a, b) => a + b, 0) / 10;
  const volumeMomentum = prevVolume > 0 ? (recentVolume / prevVolume - 1) * 100 : 0;

  // Buy pressure estimation using candle body position
  // If close is near high, it's buying pressure; near low, selling pressure
  let totalBuyPressure = 0;
  const recent = candles.slice(-20);

  for (const c of recent) {
    const range = c.high - c.low;
    if (range > 0) {
      const bodyPosition = (c.close - c.low) / range; // 0 = closed at low, 1 = closed at high
      totalBuyPressure += bodyPosition * c.volume;
    }
  }

  const totalVolume = recent.reduce((sum, c) => sum + c.volume, 0);
  const buyPressure = totalVolume > 0 ? totalBuyPressure / totalVolume : 0.5;

  return { volumeMomentum, buyPressure };
}

// ============================================
// NEW: RISK-ADJUSTED METRICS
// ============================================

function calculateRiskMetrics(candles: Candle[]): {
  sharpeRatio6h: number;
  sortinoRatio6h: number;
  volatilityRegime: "low" | "medium" | "high" | "extreme";
} {
  if (candles.length < 24) {
    return { sharpeRatio6h: 0, sortinoRatio6h: 0, volatilityRegime: "medium" };
  }

  // Calculate 6h returns
  const returns: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    returns.push((candles[i].close / candles[i - 1].close) - 1);
  }

  const recent6h = returns.slice(-6);
  const avgReturn = recent6h.reduce((a, b) => a + b, 0) / recent6h.length;
  const variance = recent6h.reduce((acc, r) => acc + Math.pow(r - avgReturn, 2), 0) / recent6h.length;
  const stdDev = Math.sqrt(variance);

  // Sharpe ratio (simplified, assuming risk-free = 0)
  const sharpeRatio6h = stdDev > 0 ? avgReturn / stdDev : 0;

  // Sortino ratio (only downside volatility)
  const negativeReturns = recent6h.filter(r => r < 0);
  const downsideVariance = negativeReturns.length > 0
    ? negativeReturns.reduce((acc, r) => acc + r * r, 0) / negativeReturns.length
    : 0;
  const downsideStdDev = Math.sqrt(downsideVariance);
  const sortinoRatio6h = downsideStdDev > 0 ? avgReturn / downsideStdDev : 0;

  // Volatility regime based on 24h realized volatility
  const recent24h = returns.slice(-24);
  const vol24h = Math.sqrt(recent24h.reduce((acc, r) => acc + r * r, 0) / recent24h.length) * Math.sqrt(365 * 24);

  let volatilityRegime: "low" | "medium" | "high" | "extreme";
  if (vol24h < 0.3) volatilityRegime = "low";
  else if (vol24h < 0.6) volatilityRegime = "medium";
  else if (vol24h < 1.0) volatilityRegime = "high";
  else volatilityRegime = "extreme";

  return { sharpeRatio6h, sortinoRatio6h, volatilityRegime };
}

// ============================================
// NEW: KELTNER CHANNELS (for Squeeze Momentum)
// ============================================

function calculateKeltnerChannels(candles: Candle[], period = 20, multiplier = 1.5): {
  upper: number;
  middle: number;
  lower: number;
} {
  if (candles.length < period) {
    const price = candles[candles.length - 1].close;
    return { upper: price, middle: price, lower: price };
  }

  const prices = candles.map(c => c.close);
  const ema = calculateEMA(prices, period);
  const middle = ema[ema.length - 1];

  // Calculate ATR for the bands
  const atr = calculateATR(candles, period);
  const upper = middle + (multiplier * atr);
  const lower = middle - (multiplier * atr);

  return { upper, middle, lower };
}

// ============================================
// NEW: SQUEEZE MOMENTUM INDICATOR
// LazyBear's Squeeze Momentum from TradingView
// ============================================

function calculateSqueezeMomentum(candles: Candle[]): {
  isSqueezing: boolean;
  momentum: number;
  momentumColor: "lime" | "green" | "red" | "maroon";
  squeezeCount: number;
} {
  if (candles.length < 30) {
    return { isSqueezing: false, momentum: 0, momentumColor: "green", squeezeCount: 0 };
  }

  const prices = candles.map(c => c.close);

  // Bollinger Bands (20, 2)
  const bb = calculateBollingerBands(prices, 20, 2);

  // Keltner Channels (20, 1.5)
  const kc = calculateKeltnerChannels(candles, 20, 1.5);

  // Squeeze: BB inside KC
  const isSqueezing = bb.lower > kc.lower && bb.upper < kc.upper;

  // Count squeeze bars
  let squeezeCount = 0;
  for (let i = candles.length - 1; i >= Math.max(0, candles.length - 20); i--) {
    const slicedCandles = candles.slice(0, i + 1);
    const slicedPrices = slicedCandles.map(c => c.close);
    if (slicedPrices.length < 20) break;

    const tempBB = calculateBollingerBands(slicedPrices, 20, 2);
    const tempKC = calculateKeltnerChannels(slicedCandles, 20, 1.5);

    if (tempBB.lower > tempKC.lower && tempBB.upper < tempKC.upper) {
      squeezeCount++;
    } else {
      break;
    }
  }

  // Momentum calculation (Linear Regression of price - midline)
  // Simplified: Use price momentum relative to mean
  const period = 20;
  const recentPrices = prices.slice(-period);
  const highest = Math.max(...candles.slice(-period).map(c => c.high));
  const lowest = Math.min(...candles.slice(-period).map(c => c.low));
  const midline = (highest + lowest) / 2 + bb.middle;
  const avgMidline = midline / 2;

  const currentPrice = prices[prices.length - 1];
  const momentum = ((currentPrice - avgMidline) / avgMidline) * 100;

  // Previous momentum for color
  const prevPrice = prices[prices.length - 2];
  const prevMomentum = ((prevPrice - avgMidline) / avgMidline) * 100;

  // Color logic:
  // lime = positive and increasing, green = positive and decreasing
  // red = negative and decreasing, maroon = negative and increasing
  let momentumColor: "lime" | "green" | "red" | "maroon";
  if (momentum > 0) {
    momentumColor = momentum > prevMomentum ? "lime" : "green";
  } else {
    momentumColor = momentum < prevMomentum ? "red" : "maroon";
  }

  return { isSqueezing, momentum, momentumColor, squeezeCount };
}

// ============================================
// NEW: SUPERTREND INDICATOR
// ============================================

function calculateSuperTrend(candles: Candle[], period = 10, multiplier = 3): {
  value: number;
  direction: "up" | "down";
  flipRecent: boolean;
} {
  if (candles.length < period + 5) {
    return { value: candles[candles.length - 1].close, direction: "up", flipRecent: false };
  }

  const atr = calculateATR(candles, period);

  // Calculate SuperTrend
  const superTrendHistory: { upper: number; lower: number; trend: number }[] = [];

  for (let i = period; i < candles.length; i++) {
    const hl2 = (candles[i].high + candles[i].low) / 2;
    const sliceATR = calculateATR(candles.slice(0, i + 1), period);

    let upperBand = hl2 + (multiplier * sliceATR);
    let lowerBand = hl2 - (multiplier * sliceATR);

    // Adjust bands based on previous values
    if (superTrendHistory.length > 0) {
      const prev = superTrendHistory[superTrendHistory.length - 1];

      if (lowerBand > prev.lower || candles[i - 1].close < prev.lower) {
        lowerBand = lowerBand;
      } else {
        lowerBand = prev.lower;
      }

      if (upperBand < prev.upper || candles[i - 1].close > prev.upper) {
        upperBand = upperBand;
      } else {
        upperBand = prev.upper;
      }
    }

    // Determine trend
    let trend: number;
    if (superTrendHistory.length === 0) {
      trend = candles[i].close > upperBand ? 1 : -1;
    } else {
      const prev = superTrendHistory[superTrendHistory.length - 1];
      if (prev.trend === -1 && candles[i].close > prev.upper) {
        trend = 1;
      } else if (prev.trend === 1 && candles[i].close < prev.lower) {
        trend = -1;
      } else {
        trend = prev.trend;
      }
    }

    superTrendHistory.push({ upper: upperBand, lower: lowerBand, trend });
  }

  const current = superTrendHistory[superTrendHistory.length - 1];
  const direction: "up" | "down" = current.trend === 1 ? "up" : "down";
  const value = direction === "up" ? current.lower : current.upper;

  // Check for recent flip (last 3 bars)
  let flipRecent = false;
  for (let i = superTrendHistory.length - 3; i < superTrendHistory.length - 1; i++) {
    if (i >= 0 && superTrendHistory[i].trend !== superTrendHistory[i + 1].trend) {
      flipRecent = true;
      break;
    }
  }

  return { value, direction, flipRecent };
}

// ============================================
// NEW: VWAP (Volume Weighted Average Price)
// ============================================

function calculateVWAP(candles: Candle[]): {
  value: number;
  upperBand1: number;
  upperBand2: number;
  lowerBand1: number;
  lowerBand2: number;
  pricePosition: number;
} {
  if (candles.length < 10) {
    const price = candles[candles.length - 1].close;
    return {
      value: price,
      upperBand1: price,
      upperBand2: price,
      lowerBand1: price,
      lowerBand2: price,
      pricePosition: 0
    };
  }

  // Use last 24 candles for intraday VWAP (24h on 1h timeframe)
  const period = Math.min(24, candles.length);
  const recentCandles = candles.slice(-period);

  let cumulativeTPV = 0;  // Cumulative Typical Price * Volume
  let cumulativeVolume = 0;
  const tpvHistory: number[] = [];
  const vwapHistory: number[] = [];

  for (const c of recentCandles) {
    const typicalPrice = (c.high + c.low + c.close) / 3;
    cumulativeTPV += typicalPrice * c.volume;
    cumulativeVolume += c.volume;
    const vwap = cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : typicalPrice;
    vwapHistory.push(vwap);
    tpvHistory.push(typicalPrice);
  }

  const vwap = vwapHistory[vwapHistory.length - 1];

  // Calculate standard deviation for bands
  const deviations = tpvHistory.map((tp, i) => Math.pow(tp - vwapHistory[i], 2));
  const variance = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  const stdDev = Math.sqrt(variance);

  const upperBand1 = vwap + stdDev;
  const upperBand2 = vwap + (2 * stdDev);
  const lowerBand1 = vwap - stdDev;
  const lowerBand2 = vwap - (2 * stdDev);

  // Calculate price position (-2 to +2)
  const currentPrice = candles[candles.length - 1].close;
  let pricePosition: number;
  if (currentPrice >= upperBand2) {
    pricePosition = 2;
  } else if (currentPrice >= upperBand1) {
    pricePosition = 1 + (currentPrice - upperBand1) / (upperBand2 - upperBand1);
  } else if (currentPrice >= vwap) {
    pricePosition = (currentPrice - vwap) / (upperBand1 - vwap);
  } else if (currentPrice >= lowerBand1) {
    pricePosition = (currentPrice - vwap) / (vwap - lowerBand1);
  } else if (currentPrice >= lowerBand2) {
    pricePosition = -1 - (lowerBand1 - currentPrice) / (lowerBand1 - lowerBand2);
  } else {
    pricePosition = -2;
  }

  return { value: vwap, upperBand1, upperBand2, lowerBand1, lowerBand2, pricePosition };
}

// ============================================
// NEW: WAVETREND OSCILLATOR
// Better overbought/oversold than RSI
// ============================================

function calculateWaveTrend(candles: Candle[], channelLen = 10, avgLen = 21): {
  wt1: number;
  wt2: number;
  crossUp: boolean;
  crossDown: boolean;
  overbought: boolean;
  oversold: boolean;
} {
  if (candles.length < channelLen + avgLen + 5) {
    return { wt1: 0, wt2: 0, crossUp: false, crossDown: false, overbought: false, oversold: false };
  }

  // Calculate HLC3 (average price)
  const hlc3 = candles.map(c => (c.high + c.low + c.close) / 3);

  // EMA of HLC3
  const esa = calculateEMA(hlc3, channelLen);

  // Calculate absolute deviation
  const d: number[] = [];
  for (let i = 0; i < hlc3.length; i++) {
    d.push(Math.abs(hlc3[i] - esa[i]));
  }

  // EMA of deviation
  const de = calculateEMA(d, channelLen);

  // Calculate CI (momentum)
  const ci: number[] = [];
  for (let i = 0; i < hlc3.length; i++) {
    ci.push(de[i] !== 0 ? (hlc3[i] - esa[i]) / (0.015 * de[i]) : 0);
  }

  // WaveTrend lines (WT1 and WT2)
  const wt1Array = calculateEMA(ci, avgLen);
  const wt2Array: number[] = [];

  // WT2 is SMA of WT1 (4 period)
  for (let i = 0; i < wt1Array.length; i++) {
    if (i < 3) {
      wt2Array.push(wt1Array[i]);
    } else {
      const sma4 = (wt1Array[i] + wt1Array[i-1] + wt1Array[i-2] + wt1Array[i-3]) / 4;
      wt2Array.push(sma4);
    }
  }

  const wt1 = wt1Array[wt1Array.length - 1];
  const wt2 = wt2Array[wt2Array.length - 1];
  const prevWt1 = wt1Array[wt1Array.length - 2];
  const prevWt2 = wt2Array[wt2Array.length - 2];

  // Cross detection
  const crossUp = prevWt1 < prevWt2 && wt1 > wt2;
  const crossDown = prevWt1 > prevWt2 && wt1 < wt2;

  // Overbought/Oversold levels
  const overbought = wt1 > 60;
  const oversold = wt1 < -60;

  return { wt1, wt2, crossUp, crossDown, overbought, oversold };
}

// ============================================
// NEW: FAIR VALUE GAPS (Smart Money Concept)
// ============================================

function detectFairValueGaps(candles: Candle[], lookback = 50): {
  bullishFVG: number[];
  bearishFVG: number[];
  nearestBullish: number | null;
  nearestBearish: number | null;
} {
  if (candles.length < 5) {
    return { bullishFVG: [], bearishFVG: [], nearestBullish: null, nearestBearish: null };
  }

  const bullishFVG: number[] = [];
  const bearishFVG: number[] = [];
  const currentPrice = candles[candles.length - 1].close;

  // Look for FVGs in recent history
  const startIdx = Math.max(2, candles.length - lookback);

  for (let i = startIdx; i < candles.length - 1; i++) {
    const candle1 = candles[i - 2];
    const candle2 = candles[i - 1];
    const candle3 = candles[i];

    // Bullish FVG: Gap between candle1's high and candle3's low
    // (candle2 is the impulse candle)
    if (candle3.low > candle1.high) {
      const gapMid = (candle3.low + candle1.high) / 2;
      // Only track if not filled yet
      if (currentPrice > gapMid) {
        bullishFVG.push(gapMid);
      }
    }

    // Bearish FVG: Gap between candle1's low and candle3's high
    if (candle3.high < candle1.low) {
      const gapMid = (candle3.high + candle1.low) / 2;
      // Only track if not filled yet
      if (currentPrice < gapMid) {
        bearishFVG.push(gapMid);
      }
    }
  }

  // Find nearest FVGs to current price
  const nearestBullish = bullishFVG.length > 0
    ? bullishFVG.reduce((nearest, fvg) =>
        Math.abs(fvg - currentPrice) < Math.abs(nearest - currentPrice) ? fvg : nearest
      )
    : null;

  const nearestBearish = bearishFVG.length > 0
    ? bearishFVG.reduce((nearest, fvg) =>
        Math.abs(fvg - currentPrice) < Math.abs(nearest - currentPrice) ? fvg : nearest
      )
    : null;

  return {
    bullishFVG: bullishFVG.slice(-5),  // Keep last 5
    bearishFVG: bearishFVG.slice(-5),
    nearestBullish,
    nearestBearish
  };
}

// ============================================
// GEMINI AI ANALYSIS
// ============================================

async function getGeminiAnalysis(
  symbol: string,
  indicators: AdvancedIndicators,
  recentTrades: any[]
): Promise<{ analysis: string; recommendation: string; confidence: number }> {
  const geminiKey = Deno.env.get("GEMINI_API_KEY");

  if (!geminiKey || !CONFIG.useGeminiAI) {
    return { analysis: "AI analysis disabled", recommendation: "use_technical", confidence: 0.5 };
  }

  try {
    const prompt = `You are an expert crypto trading analyst using a multi-strategy ensemble approach. Analyze the following market data for ${symbol} and provide a trading recommendation.

## Current Market Data
- Price: $${indicators.currentPrice.toFixed(2)}
- 24h Change: ${indicators.priceChange24h.toFixed(2)}%
- Market Regime: ${indicators.marketRegime} (strength: ${(indicators.trendStrength * 100).toFixed(0)}%)
- Volatility Regime: ${indicators.riskMetrics.volatilityRegime}

## Multi-Timeframe Momentum (CRITICAL)
- 1H Momentum: ${indicators.momentum.m1h.toFixed(2)}%
- 4H Momentum: ${indicators.momentum.m4h.toFixed(2)}%
- 12H Momentum: ${indicators.momentum.m12h.toFixed(2)}%
- 24H Momentum: ${indicators.momentum.m24h.toFixed(2)}%
- MTF Alignment: ${(indicators.momentum.alignment * 100).toFixed(0)}% ${indicators.momentum.alignment > 0.5 ? '(BULLISH)' : indicators.momentum.alignment < -0.5 ? '(BEARISH)' : '(MIXED)'}

## Technical Indicators
- RSI(14): ${indicators.rsi.toFixed(1)} ${indicators.rsi < 30 ? '(OVERSOLD)' : indicators.rsi > 70 ? '(OVERBOUGHT)' : ''}
- Stoch RSI: K=${indicators.stochRsi.k.toFixed(1)}, D=${indicators.stochRsi.d.toFixed(1)}
- MACD: ${indicators.macd.histogram > 0 ? 'BULLISH' : 'BEARISH'} (histogram: ${indicators.macd.histogram.toFixed(4)})
- ADX: ${indicators.adx.toFixed(1)} ${indicators.adx > 25 ? '(STRONG TREND)' : '(WEAK TREND)'}

## Mean Reversion Signals
- Z-Score: ${indicators.meanReversion.zScore.toFixed(2)} ${Math.abs(indicators.meanReversion.zScore) > 2 ? '(EXTREME)' : ''}
- Percentile Rank: ${indicators.meanReversion.percentileRank.toFixed(0)}%
- BB Position: ${(indicators.meanReversion.bbPosition * 100).toFixed(0)}%
- RSI Divergence: ${indicators.meanReversion.rsiDivergence ? 'DETECTED ⚡' : 'None'}

## Volatility Analysis
- ATR: ${indicators.atrPercent.toFixed(2)}% of price
- Parkinson Vol: ${(indicators.parkinsonVolatility * 100).toFixed(1)}% annualized
- BB Width: ${(indicators.bollingerBands.width * 100).toFixed(2)}%

## Volume Analysis
- Volume Ratio: ${indicators.volumeRatio.toFixed(2)}x average
- Volume Momentum: ${indicators.volumeMomentum > 0 ? '+' : ''}${indicators.volumeMomentum.toFixed(0)}%
- Buy Pressure: ${(indicators.buyPressure * 100).toFixed(0)}%
- OBV Trend: ${indicators.obvTrend}

## Key Levels
- Support: ${indicators.supportLevels.slice(0, 2).map(s => '$' + s.toFixed(0)).join(', ') || 'None detected'}
- Resistance: ${indicators.resistanceLevels.slice(0, 2).map(r => '$' + r.toFixed(0)).join(', ') || 'None detected'}
- Candle Pattern: ${indicators.candlePattern}

## Risk Metrics
- 6H Sharpe: ${indicators.riskMetrics.sharpeRatio6h.toFixed(2)}
- 6H Sortino: ${indicators.riskMetrics.sortinoRatio6h.toFixed(2)}

## TradingView-Style Indicators (IMPORTANT)
### Squeeze Momentum (LazyBear)
- Squeezing: ${indicators.squeezeMomentum.isSqueezing ? `YES (${indicators.squeezeMomentum.squeezeCount} bars)` : 'NO - Active momentum'}
- Momentum: ${indicators.squeezeMomentum.momentum > 0 ? '+' : ''}${indicators.squeezeMomentum.momentum.toFixed(2)} (${indicators.squeezeMomentum.momentumColor})
${indicators.squeezeMomentum.squeezeCount >= 3 && !indicators.squeezeMomentum.isSqueezing ? '⚡ BREAKOUT IN PROGRESS!' : ''}

### SuperTrend
- Direction: ${indicators.superTrend.direction.toUpperCase()}
- Value: $${indicators.superTrend.value.toFixed(2)}
${indicators.superTrend.flipRecent ? '🔄 TREND JUST FLIPPED!' : ''}

### VWAP Analysis
- VWAP: $${indicators.vwap.value.toFixed(2)}
- Price Position: ${indicators.vwap.pricePosition.toFixed(2)} stddev ${indicators.vwap.pricePosition > 1.5 ? '(OVERBOUGHT)' : indicators.vwap.pricePosition < -1.5 ? '(OVERSOLD)' : ''}

### WaveTrend Oscillator
- WT1/WT2: ${indicators.waveTrend.wt1.toFixed(1)} / ${indicators.waveTrend.wt2.toFixed(1)}
- Signal: ${indicators.waveTrend.crossUp ? '⬆️ BULLISH CROSS' : indicators.waveTrend.crossDown ? '⬇️ BEARISH CROSS' : 'No cross'}
${indicators.waveTrend.overbought ? '⚠️ OVERBOUGHT (>60)' : indicators.waveTrend.oversold ? '✅ OVERSOLD (<-60)' : ''}

### Fair Value Gaps (Smart Money)
- Nearest Bullish FVG: ${indicators.fairValueGaps.nearestBullish ? '$' + indicators.fairValueGaps.nearestBullish.toFixed(0) : 'None'}
- Nearest Bearish FVG: ${indicators.fairValueGaps.nearestBearish ? '$' + indicators.fairValueGaps.nearestBearish.toFixed(0) : 'None'}

## Recent Trading Performance
${recentTrades.length > 0 ? recentTrades.slice(0, 5).map(t =>
  `- ${t.side.toUpperCase()} @ $${t.entry_price} → ${t.exit_price ? `$${t.exit_price} (${t.return_pct > 0 ? '+' : ''}${t.return_pct?.toFixed(2)}%)` : 'OPEN'}`
).join('\n') : '- No recent trades'}

STRATEGY SELECTION:
- In TRENDING markets (ADX > 25): Prioritize trend-following and momentum
- In RANGING markets: Prioritize mean reversion signals (Z-score, BB position)
- In VOLATILE markets: Be cautious, reduce position size recommendations

IMPORTANT: Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{"recommendation": "STRONG_BUY", "confidence": 0.75, "reasoning": "Brief explanation", "key_factors": ["factor1", "factor2"], "risk_level": "medium", "strategy_used": "trend_following|mean_reversion|momentum"}

Valid recommendations: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
Confidence must be between 0.0 and 1.0`;

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${CONFIG.geminiModel}:generateContent?key=${geminiKey}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.3,
            maxOutputTokens: 512,
          },
        }),
        signal: AbortSignal.timeout(30000),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Gemini API error: ${response.status} - ${errorText}`);
      return { analysis: "API error", recommendation: "use_technical", confidence: 0.5 };
    }

    const data = await response.json();
    const content = data.candidates?.[0]?.content?.parts?.[0]?.text || "";

    console.log(`🤖 Gemini raw response: ${content.substring(0, 200)}...`);

    // Parse JSON from response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          analysis: `${parsed.reasoning || "AI analysis complete"} Key factors: ${parsed.key_factors?.join(", ") || "N/A"}`,
          recommendation: parsed.recommendation || "HOLD",
          confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0.5,
        };
      } catch (parseError) {
        console.error("JSON parse error:", parseError);
      }
    }

    return { analysis: content, recommendation: "HOLD", confidence: 0.5 };
  } catch (error) {
    console.error("Gemini analysis error:", error);
    return { analysis: `Error: ${error.message}`, recommendation: "use_technical", confidence: 0.5 };
  }
}

// ============================================
// FETCH MARKET DATA
// ============================================

async function fetchKlines(symbol: string, interval = "1h", limit = 100): Promise<Candle[]> {
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

async function fetchTicker(symbol: string): Promise<{ price: number; change24h: number }> {
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

// ============================================
// SIGNAL GENERATION
// ============================================

function calculateAllIndicators(candles: Candle[], ticker: { price: number; change24h: number }): AdvancedIndicators {
  const prices = candles.map(c => c.close);
  const volumes = candles.map(c => c.volume);

  // Core indicators
  const rsi = calculateRSI(prices);
  const stochRsi = calculateStochRSI(prices);
  const macd = calculateMACD(prices);
  const { adx, plusDI, minusDI } = calculateADX(candles);

  // Moving averages
  const ema9 = calculateEMA(prices, 9);
  const ema21 = calculateEMA(prices, 21);
  const sma20 = calculateSMA(prices, 20);
  const sma50 = calculateSMA(prices, 50);
  const sma200 = calculateSMA(prices, 200);

  // Volatility (enhanced)
  const atr = calculateATR(candles);
  const atrPercent = (atr / ticker.price) * 100;
  const bollingerBands = calculateBollingerBands(prices);
  const parkinsonVolatility = calculateParkinsonVolatility(candles);
  const garmanKlassVolatility = calculateGarmanKlassVolatility(candles);

  // Volume (enhanced)
  const { obv, trend: obvTrend } = calculateOBV(candles);
  const volumeSma = calculateSMA(volumes, 20);
  const volumeRatio = volumes[volumes.length - 1] / volumeSma;
  const { volumeMomentum, buyPressure } = calculateVolumeMetrics(candles);

  // Support/Resistance (enhanced)
  const pivotPoints = calculatePivotPoints(candles);
  const { support: supportLevels, resistance: resistanceLevels } = detectSupportResistance(candles);

  // Price action
  const candlePattern = detectCandlePattern(candles);
  const highLowRange = (candles[candles.length - 1].high - candles[candles.length - 1].low) / ticker.price;

  const partialIndicators = {
    adx,
    plusDI,
    minusDI,
    atrPercent,
    bollingerBands,
  };

  const { regime, strength } = detectMarketRegime(partialIndicators);

  // NEW: Multi-timeframe momentum
  const momentum = calculateMultiTimeframeMomentum(candles);

  // NEW: Mean reversion indicators
  const meanReversion = calculateMeanReversionIndicators(candles, prices, rsi, bollingerBands);

  // NEW: Risk metrics
  const riskMetrics = calculateRiskMetrics(candles);

  // NEW: TradingView-style indicators
  const squeezeMomentum = calculateSqueezeMomentum(candles);
  const superTrend = calculateSuperTrend(candles);
  const vwap = calculateVWAP(candles);
  const waveTrend = calculateWaveTrend(candles);
  const fairValueGaps = detectFairValueGaps(candles);

  return {
    rsi,
    stochRsi,
    macd,
    adx,
    plusDI,
    minusDI,
    sma20,
    sma50,
    sma200,
    ema9: ema9[ema9.length - 1],
    ema21: ema21[ema21.length - 1],
    atr,
    atrPercent,
    bollingerBands,
    parkinsonVolatility,
    garmanKlassVolatility,
    obv,
    obvTrend,
    volumeSma,
    volumeRatio,
    volumeMomentum,
    buyPressure,
    pivotPoints,
    supportLevels,
    resistanceLevels,
    currentPrice: ticker.price,
    priceChange24h: ticker.change24h,
    highLowRange,
    candlePattern,
    marketRegime: regime,
    trendStrength: strength,
    momentum,
    meanReversion,
    riskMetrics,
    squeezeMomentum,
    superTrend,
    vwap,
    waveTrend,
    fairValueGaps,
  };
}

async function generateAdvancedSignal(
  symbol: string,
  candles: Candle[],
  ticker: { price: number; change24h: number },
  recentTrades: any[]
): Promise<Signal> {
  if (candles.length < 60) {
    return {
      signal: "hold",
      confidence: 0.5,
      reasoning: "Insufficient data",
      indicators: {} as AdvancedIndicators,
      riskScore: 0.5,
    };
  }

  const indicators = calculateAllIndicators(candles, ticker);

  // ============================================
  // ENHANCED MULTI-FACTOR ENSEMBLE SCORING
  // ============================================

  let buyScore = 0;
  let sellScore = 0;
  const reasons: string[] = [];
  let riskScore = 0.5;

  // Track individual strategy scores for ensemble
  const strategyScores = {
    trend: { buy: 0, sell: 0 },
    momentum: { buy: 0, sell: 0 },
    meanReversion: { buy: 0, sell: 0 },
    volume: { buy: 0, sell: 0 },
    multiTimeframe: { buy: 0, sell: 0 },
    // NEW: TradingView-style strategies
    squeezeMomentum: { buy: 0, sell: 0 },
    superTrend: { buy: 0, sell: 0 },
    waveTrend: { buy: 0, sell: 0 },
  };

  // 1. TREND FOLLOWING STRATEGY (Weight: 25%)
  // --------------------------------

  // ADX + DI
  if (indicators.adx > CONFIG.adxTrendStrength) {
    if (indicators.plusDI > indicators.minusDI) {
      strategyScores.trend.buy += 2;
      buyScore += 2;
      reasons.push(`Strong uptrend (ADX=${indicators.adx.toFixed(0)}, +DI>${indicators.minusDI.toFixed(0)})`);
    } else {
      strategyScores.trend.sell += 2;
      sellScore += 2;
      reasons.push(`Strong downtrend (ADX=${indicators.adx.toFixed(0)}, -DI>${indicators.plusDI.toFixed(0)})`);
    }
  }

  // EMA alignment
  if (indicators.ema9 > indicators.ema21 && indicators.ema21 > indicators.sma50) {
    strategyScores.trend.buy += 1.5;
    buyScore += 1.5;
    reasons.push("EMAs bullish aligned");
  } else if (indicators.ema9 < indicators.ema21 && indicators.ema21 < indicators.sma50) {
    strategyScores.trend.sell += 1.5;
    sellScore += 1.5;
    reasons.push("EMAs bearish aligned");
  }

  // Price vs SMA200 (long-term trend)
  if (indicators.currentPrice > indicators.sma200) {
    buyScore += 0.5;
  } else {
    sellScore += 0.5;
  }

  // 2. MOMENTUM STRATEGY (Weight: 20%)
  // --------------------------------

  // RSI
  if (indicators.rsi < CONFIG.rsiOversold) {
    strategyScores.momentum.buy += 2;
    buyScore += 2;
    reasons.push(`RSI oversold (${indicators.rsi.toFixed(1)})`);
  } else if (indicators.rsi > CONFIG.rsiOverbought) {
    strategyScores.momentum.sell += 2;
    sellScore += 2;
    reasons.push(`RSI overbought (${indicators.rsi.toFixed(1)})`);
  } else if (indicators.rsi < 40) {
    strategyScores.momentum.buy += 0.5;
    buyScore += 0.5;
  } else if (indicators.rsi > 60) {
    strategyScores.momentum.sell += 0.5;
    sellScore += 0.5;
  }

  // Stochastic RSI
  if (indicators.stochRsi.k < CONFIG.stochOversold && indicators.stochRsi.k > indicators.stochRsi.d) {
    strategyScores.momentum.buy += 1.5;
    buyScore += 1.5;
    reasons.push("StochRSI bullish crossover in oversold");
  } else if (indicators.stochRsi.k > CONFIG.stochOverbought && indicators.stochRsi.k < indicators.stochRsi.d) {
    strategyScores.momentum.sell += 1.5;
    sellScore += 1.5;
    reasons.push("StochRSI bearish crossover in overbought");
  }

  // MACD
  if (indicators.macd.histogram > 0 && indicators.macd.macd > indicators.macd.signal) {
    strategyScores.momentum.buy += 1;
    buyScore += 1;
    reasons.push("MACD bullish");
  } else if (indicators.macd.histogram < 0 && indicators.macd.macd < indicators.macd.signal) {
    strategyScores.momentum.sell += 1;
    sellScore += 1;
    reasons.push("MACD bearish");
  }

  // 3. NEW: MULTI-TIMEFRAME MOMENTUM (Weight: 20%)
  // --------------------------------

  const mtfAlignment = indicators.momentum.alignment;

  // Strong alignment across all timeframes
  if (mtfAlignment > 0.6) {
    strategyScores.multiTimeframe.buy += 2.5;
    buyScore += 2.5;
    reasons.push(`MTF bullish aligned (${(mtfAlignment * 100).toFixed(0)}%)`);
  } else if (mtfAlignment < -0.6) {
    strategyScores.multiTimeframe.sell += 2.5;
    sellScore += 2.5;
    reasons.push(`MTF bearish aligned (${(Math.abs(mtfAlignment) * 100).toFixed(0)}%)`);
  }

  // Short-term momentum acceleration
  if (indicators.momentum.m1h > 0.5 && indicators.momentum.m4h > 0) {
    strategyScores.multiTimeframe.buy += 1;
    buyScore += 1;
    reasons.push(`Short-term momentum +${indicators.momentum.m1h.toFixed(1)}%`);
  } else if (indicators.momentum.m1h < -0.5 && indicators.momentum.m4h < 0) {
    strategyScores.multiTimeframe.sell += 1;
    sellScore += 1;
    reasons.push(`Short-term momentum ${indicators.momentum.m1h.toFixed(1)}%`);
  }

  // 4. NEW: MEAN REVERSION STRATEGY (Weight: 20% - Higher in ranging markets)
  // --------------------------------

  const { zScore, percentileRank, rsiDivergence, bbPosition } = indicators.meanReversion;

  // Only apply mean reversion strongly in ranging/low volatility markets
  const meanReversionWeight = indicators.marketRegime === "ranging" ? 1.5 : 0.8;

  // Extreme oversold (Z-score < -2 or percentile < 10)
  if (zScore < -2 || percentileRank < 10) {
    const score = 2 * meanReversionWeight;
    strategyScores.meanReversion.buy += score;
    buyScore += score;
    reasons.push(`Mean reversion BUY (Z=${zScore.toFixed(1)}, P=${percentileRank.toFixed(0)}%)`);
  } else if (zScore > 2 || percentileRank > 90) {
    const score = 2 * meanReversionWeight;
    strategyScores.meanReversion.sell += score;
    sellScore += score;
    reasons.push(`Mean reversion SELL (Z=${zScore.toFixed(1)}, P=${percentileRank.toFixed(0)}%)`);
  }

  // Bollinger Band mean reversion
  if (bbPosition < 0.1 && indicators.marketRegime === "ranging") {
    strategyScores.meanReversion.buy += 1.5;
    buyScore += 1.5;
    reasons.push("BB oversold in range");
  } else if (bbPosition > 0.9 && indicators.marketRegime === "ranging") {
    strategyScores.meanReversion.sell += 1.5;
    sellScore += 1.5;
    reasons.push("BB overbought in range");
  }

  // RSI Divergence (strong signal)
  if (rsiDivergence) {
    if (indicators.rsi > 50) {
      strategyScores.meanReversion.sell += 1;
      sellScore += 1;
      reasons.push("⚡ RSI bearish divergence");
    } else {
      strategyScores.meanReversion.buy += 1;
      buyScore += 1;
      reasons.push("⚡ RSI bullish divergence");
    }
  }

  // 5. VOLUME CONFIRMATION (Weight: 15%)
  // --------------------------------

  if (indicators.volumeRatio > 1.5) {
    if (indicators.obvTrend === "bullish") {
      strategyScores.volume.buy += 1;
      buyScore += 1;
      reasons.push("High volume + bullish OBV");
    } else {
      strategyScores.volume.sell += 1;
      sellScore += 1;
      reasons.push("High volume + bearish OBV");
    }
  }

  // NEW: Volume momentum confirmation
  if (indicators.volumeMomentum > 50 && indicators.buyPressure > 0.6) {
    strategyScores.volume.buy += 1;
    buyScore += 1;
    reasons.push(`Volume surge +${indicators.volumeMomentum.toFixed(0)}% (buy pressure)`);
  } else if (indicators.volumeMomentum > 50 && indicators.buyPressure < 0.4) {
    strategyScores.volume.sell += 1;
    sellScore += 1;
    reasons.push(`Volume surge +${indicators.volumeMomentum.toFixed(0)}% (sell pressure)`);
  }

  // 6. NEW: SQUEEZE MOMENTUM STRATEGY (Weight: 15%)
  // --------------------------------
  // Squeeze Momentum is highly effective for detecting breakouts

  const { isSqueezing, momentum: sqzMomentum, momentumColor, squeezeCount } = indicators.squeezeMomentum;

  // Squeeze release is a high-probability setup
  if (squeezeCount >= 3 && !isSqueezing) {
    // Just released from squeeze - powerful breakout signal!
    if (sqzMomentum > 0 && (momentumColor === "lime" || momentumColor === "green")) {
      strategyScores.squeezeMomentum.buy += 2.5;
      buyScore += 2.5;
      reasons.push(`🔥 Squeeze BREAKOUT UP (${squeezeCount} bars squeezed)`);
    } else if (sqzMomentum < 0 && (momentumColor === "red" || momentumColor === "maroon")) {
      strategyScores.squeezeMomentum.sell += 2.5;
      sellScore += 2.5;
      reasons.push(`🔥 Squeeze BREAKOUT DOWN (${squeezeCount} bars squeezed)`);
    }
  } else if (isSqueezing && squeezeCount >= 6) {
    // Extended squeeze - expect big move coming
    reasons.push(`⏳ Extended squeeze (${squeezeCount} bars) - expect breakout`);
    riskScore -= 0.1; // Lower risk, good opportunity coming
  }

  // Momentum direction within squeeze
  if (!isSqueezing) {
    if (momentumColor === "lime") {
      strategyScores.squeezeMomentum.buy += 1;
      buyScore += 1;
      reasons.push("Squeeze momentum accelerating UP");
    } else if (momentumColor === "red") {
      strategyScores.squeezeMomentum.sell += 1;
      sellScore += 1;
      reasons.push("Squeeze momentum accelerating DOWN");
    }
  }

  // 7. NEW: SUPERTREND STRATEGY (Weight: 15%)
  // --------------------------------

  const { direction: stDirection, flipRecent } = indicators.superTrend;

  if (stDirection === "up") {
    strategyScores.superTrend.buy += 1.5;
    buyScore += 1.5;
    if (flipRecent) {
      strategyScores.superTrend.buy += 1;
      buyScore += 1;
      reasons.push("🔄 SuperTrend FLIPPED BULLISH");
    } else {
      reasons.push("SuperTrend bullish");
    }
  } else {
    strategyScores.superTrend.sell += 1.5;
    sellScore += 1.5;
    if (flipRecent) {
      strategyScores.superTrend.sell += 1;
      sellScore += 1;
      reasons.push("🔄 SuperTrend FLIPPED BEARISH");
    } else {
      reasons.push("SuperTrend bearish");
    }
  }

  // 8. NEW: VWAP STRATEGY (Weight: 10%)
  // --------------------------------

  const { pricePosition: vwapPosition } = indicators.vwap;

  // VWAP acts as institutional support/resistance
  if (vwapPosition <= -1.5) {
    strategyScores.meanReversion.buy += 1.5;
    buyScore += 1.5;
    reasons.push(`VWAP oversold (${vwapPosition.toFixed(1)} stddev)`);
  } else if (vwapPosition >= 1.5) {
    strategyScores.meanReversion.sell += 1.5;
    sellScore += 1.5;
    reasons.push(`VWAP overbought (${vwapPosition.toFixed(1)} stddev)`);
  } else if (vwapPosition > -0.3 && vwapPosition < 0.3) {
    // Price at VWAP - waiting for direction
    reasons.push("Price at VWAP equilibrium");
  }

  // 9. NEW: WAVETREND STRATEGY (Weight: 12%)
  // --------------------------------

  const { wt1, crossUp, crossDown, overbought: wtOB, oversold: wtOS } = indicators.waveTrend;

  // WaveTrend crossovers in extreme zones are high-probability
  if (crossUp && wtOS) {
    strategyScores.waveTrend.buy += 2.5;
    buyScore += 2.5;
    reasons.push("⚡ WaveTrend BULLISH cross in oversold!");
  } else if (crossDown && wtOB) {
    strategyScores.waveTrend.sell += 2.5;
    sellScore += 2.5;
    reasons.push("⚡ WaveTrend BEARISH cross in overbought!");
  } else if (crossUp) {
    strategyScores.waveTrend.buy += 1;
    buyScore += 1;
    reasons.push("WaveTrend bullish cross");
  } else if (crossDown) {
    strategyScores.waveTrend.sell += 1;
    sellScore += 1;
    reasons.push("WaveTrend bearish cross");
  }

  // Extreme WT levels
  if (wt1 < -80) {
    strategyScores.waveTrend.buy += 1.5;
    buyScore += 1.5;
    reasons.push(`WaveTrend extremely oversold (${wt1.toFixed(0)})`);
  } else if (wt1 > 80) {
    strategyScores.waveTrend.sell += 1.5;
    sellScore += 1.5;
    reasons.push(`WaveTrend extremely overbought (${wt1.toFixed(0)})`);
  }

  // 10. NEW: FAIR VALUE GAPS (Smart Money)
  // --------------------------------

  const { nearestBullish, nearestBearish } = indicators.fairValueGaps;
  const currentPrice = indicators.currentPrice;

  // FVGs act as magnets - price tends to fill them
  if (nearestBullish !== null) {
    const distToFVG = (currentPrice - nearestBullish) / currentPrice;
    if (distToFVG > 0 && distToFVG < 0.02) {
      // Price just above bullish FVG - could act as support
      buyScore += 0.5;
      reasons.push(`Bullish FVG support @ $${nearestBullish.toFixed(0)}`);
    }
  }

  if (nearestBearish !== null) {
    const distToFVG = (nearestBearish - currentPrice) / currentPrice;
    if (distToFVG > 0 && distToFVG < 0.02) {
      // Price just below bearish FVG - could act as resistance
      sellScore += 0.5;
      reasons.push(`Bearish FVG resistance @ $${nearestBearish.toFixed(0)}`);
    }
  }

  // 11. PRICE ACTION & PATTERNS
  // --------------------------------

  // Candle patterns
  if (["hammer", "bullish_engulfing", "morning_star"].includes(indicators.candlePattern)) {
    buyScore += 1.5;
    reasons.push(`Bullish pattern: ${indicators.candlePattern}`);
  } else if (["shooting_star", "bearish_engulfing", "evening_star"].includes(indicators.candlePattern)) {
    sellScore += 1.5;
    reasons.push(`Bearish pattern: ${indicators.candlePattern}`);
  }

  // NEW: Support/Resistance from detected levels
  if (indicators.supportLevels.length > 0) {
    const nearestSupport = indicators.supportLevels[0];
    const distToSupport = (indicators.currentPrice - nearestSupport) / indicators.currentPrice;
    if (distToSupport < 0.01 && distToSupport > 0) {
      buyScore += 1.5;
      reasons.push(`Near key support $${nearestSupport.toFixed(0)}`);
    }
  }

  if (indicators.resistanceLevels.length > 0) {
    const nearestResistance = indicators.resistanceLevels[0];
    const distToResistance = (nearestResistance - indicators.currentPrice) / indicators.currentPrice;
    if (distToResistance < 0.01 && distToResistance > 0) {
      sellScore += 1.5;
      reasons.push(`Near key resistance $${nearestResistance.toFixed(0)}`);
    }
  }

  // 13. VOLATILITY & RISK ADJUSTMENT
  // --------------------------------

  // Volatility regime adjustment
  if (indicators.riskMetrics.volatilityRegime === "extreme") {
    buyScore *= 0.5;
    sellScore *= 0.5;
    riskScore += 0.3;
    reasons.push("⚠️ EXTREME volatility - halved confidence");
  } else if (indicators.riskMetrics.volatilityRegime === "high") {
    buyScore *= 0.7;
    sellScore *= 0.7;
    riskScore += 0.2;
    reasons.push("⚠️ High volatility");
  } else if (indicators.riskMetrics.volatilityRegime === "low") {
    // Low volatility is good for mean reversion
    if (indicators.marketRegime === "ranging") {
      buyScore *= 1.1;
      sellScore *= 1.1;
      reasons.push("Low vol + ranging = mean reversion favorable");
    }
  }

  // Parkinson/GK volatility spike warning
  if (indicators.parkinsonVolatility > 1.0) {
    riskScore += 0.15;
  }

  // 12. ENSEMBLE STRATEGY AGREEMENT (8 strategies now)
  // --------------------------------

  // Count how many strategies agree
  const totalStrategies = Object.keys(strategyScores).length; // 8 strategies
  const bullishStrategies = Object.values(strategyScores).filter(s => s.buy > s.sell).length;
  const bearishStrategies = Object.values(strategyScores).filter(s => s.sell > s.buy).length;

  // Boost confidence when multiple strategies agree
  if (bullishStrategies >= 6) {
    buyScore *= 1.2;
    reasons.push(`📊 STRONG: ${bullishStrategies}/${totalStrategies} strategies bullish`);
  } else if (bullishStrategies >= 5) {
    buyScore *= 1.15;
    reasons.push(`📊 ${bullishStrategies}/${totalStrategies} strategies bullish`);
  } else if (bearishStrategies >= 6) {
    sellScore *= 1.2;
    reasons.push(`📊 STRONG: ${bearishStrategies}/${totalStrategies} strategies bearish`);
  } else if (bearishStrategies >= 5) {
    sellScore *= 1.15;
    reasons.push(`📊 ${bearishStrategies}/${totalStrategies} strategies bearish`);
  } else if (bullishStrategies >= 3 && bearishStrategies >= 3) {
    // Conflicting signals - reduce confidence
    buyScore *= 0.75;
    sellScore *= 0.75;
    reasons.push("⚠️ Mixed strategy signals - conflicting");
  }

  // NEW: TradingView indicator alignment bonus
  // When SuperTrend + SqueeZe + WaveTrend all agree = very strong signal
  const tvBullish = indicators.superTrend.direction === "up" &&
                    indicators.squeezeMomentum.momentum > 0 &&
                    indicators.waveTrend.wt1 > indicators.waveTrend.wt2;
  const tvBearish = indicators.superTrend.direction === "down" &&
                    indicators.squeezeMomentum.momentum < 0 &&
                    indicators.waveTrend.wt1 < indicators.waveTrend.wt2;

  if (tvBullish) {
    buyScore *= 1.1;
    reasons.push("🎯 TV indicators aligned bullish");
  } else if (tvBearish) {
    sellScore *= 1.1;
    reasons.push("🎯 TV indicators aligned bearish");
  }

  // ============================================
  // FINAL SIGNAL CALCULATION (Updated for 8 strategies)
  // Adjusted Feb 22: Lower thresholds for more trades
  // ============================================

  const totalScore = buyScore - sellScore;
  const maxScore = 18.0;  // Increased from 12 due to more strategies

  let signal: string;
  let confidence: number;

  // ADJUSTED: Lower thresholds to trade more frequently
  // Old: 6/3, New: 4/2 (more trades, still requires some agreement)
  if (totalScore >= 4) {
    signal = "strong_buy";
    confidence = Math.min(0.95, 0.65 + (totalScore / maxScore) * 0.30);
  } else if (totalScore >= 2) {
    signal = "buy";
    confidence = Math.min(0.85, 0.58 + (totalScore / maxScore) * 0.27);
  } else if (totalScore <= -4) {
    signal = "strong_sell";
    confidence = Math.min(0.95, 0.65 + (Math.abs(totalScore) / maxScore) * 0.30);
  } else if (totalScore <= -2) {
    signal = "sell";
    confidence = Math.min(0.85, 0.58 + (Math.abs(totalScore) / maxScore) * 0.27);
  } else {
    signal = "hold";
    confidence = 0.5;
  }

  // Normalize risk score
  riskScore = Math.min(1, Math.max(0, riskScore));

  // ============================================
  // GEMINI AI ENHANCEMENT
  // ============================================

  let aiAnalysis = "";

  if (CONFIG.useGeminiAI && (signal !== "hold" || Math.random() < 0.2)) {
    const geminiResult = await getGeminiAnalysis(symbol, indicators, recentTrades);
    aiAnalysis = geminiResult.analysis;

    // Adjust confidence based on AI agreement
    if (geminiResult.recommendation !== "use_technical") {
      const aiSignalMap: Record<string, number> = {
        "STRONG_BUY": 2,
        "BUY": 1,
        "HOLD": 0,
        "SELL": -1,
        "STRONG_SELL": -2,
      };

      const technicalDirection = totalScore > 0 ? 1 : totalScore < 0 ? -1 : 0;
      const aiDirection = Math.sign(aiSignalMap[geminiResult.recommendation] || 0);

      if (technicalDirection === aiDirection && technicalDirection !== 0) {
        // AI agrees - boost confidence
        confidence = Math.min(0.95, confidence + 0.1);
        reasons.push(`🤖 Gemini confirms: ${geminiResult.recommendation}`);
      } else if (technicalDirection !== aiDirection && aiDirection !== 0) {
        // AI disagrees - reduce confidence
        confidence = Math.max(0.5, confidence - 0.15);
        reasons.push(`🤖 Gemini caution: ${geminiResult.recommendation}`);

        // If AI strongly disagrees and has high confidence, consider overriding
        if (geminiResult.confidence > 0.8 && Math.abs(aiSignalMap[geminiResult.recommendation]) === 2) {
          signal = "hold";
          reasons.push("🤖 Gemini override - conflicting signals");
        }
      }
    }
  }

  return {
    signal,
    confidence,
    reasoning: reasons.join(" | "),
    indicators,
    aiAnalysis,
    riskScore,
  };
}

// ============================================
// MAIN HANDLER
// ============================================

serve(async (req) => {
  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    console.log("🚀 Advanced Paper Trading triggered");

    const results: any[] = [];

    // Check global drawdown
    const { data: perfData } = await supabase
      .from("performance_snapshots")
      .select("*")
      .eq("strategy_name", "edge_paper")
      .order("timestamp", { ascending: false })
      .limit(1);

    if (perfData && perfData.length > 0) {
      const currentEquity = perfData[0].total_equity;
      const drawdown = (CONFIG.initialCapital - currentEquity) / CONFIG.initialCapital;

      if (drawdown > CONFIG.maxDrawdownPct) {
        console.log(`⚠️ Max drawdown reached (${(drawdown * 100).toFixed(1)}%). Trading paused.`);
        return new Response(JSON.stringify({
          success: true,
          message: "Trading paused due to max drawdown",
          drawdown: drawdown * 100
        }), {
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    // NEW: Check for consecutive losses (loss streak protection)
    const { data: recentClosedTrades } = await supabase
      .from("trade_logs")
      .select("net_pnl")
      .eq("strategy_name", "edge_paper")
      .not("exit_time", "is", null)
      .order("exit_time", { ascending: false })
      .limit(CONFIG.maxConsecutiveLosses + CONFIG.lossRecoveryTrades);

    let consecutiveLosses = 0;
    if (recentClosedTrades && recentClosedTrades.length > 0) {
      for (const trade of recentClosedTrades) {
        if (parseFloat(trade.net_pnl || 0) < 0) {
          consecutiveLosses++;
        } else {
          break; // Stop counting at first win
        }
      }
    }

    const onLossStreak = consecutiveLosses >= CONFIG.maxConsecutiveLosses;
    if (onLossStreak) {
      console.log(`⚠️ Loss streak detected (${consecutiveLosses} losses). Trading more cautiously.`);
    }

    // NEW: Get all open positions for correlation filter
    const { data: allOpenPositions } = await supabase
      .from("trade_logs")
      .select("symbol, side")
      .eq("strategy_name", "edge_paper")
      .is("exit_time", null);

    const openPositionMap = new Map<string, string>();
    if (allOpenPositions) {
      for (const pos of allOpenPositions) {
        openPositionMap.set(pos.symbol, pos.side);
      }
    }
    console.log(`📍 Open positions: ${JSON.stringify(Object.fromEntries(openPositionMap))}`);

    for (const symbol of SYMBOLS) {
      console.log(`\n📊 Processing ${symbol}...`);

      // Fetch market data
      const candles = await fetchKlines(symbol, "1h", 200);
      const ticker = await fetchTicker(symbol);

      // Get recent trades for context
      const { data: recentTrades } = await supabase
        .from("trade_logs")
        .select("*")
        .eq("symbol", symbol)
        .eq("strategy_name", "edge_paper")
        .order("entry_time", { ascending: false })
        .limit(10);

      // Generate advanced signal
      const signal = await generateAdvancedSignal(symbol, candles, ticker, recentTrades || []);

      console.log(`Signal: ${signal.signal} (confidence: ${(signal.confidence * 100).toFixed(0)}%)`);
      console.log(`Reasoning: ${signal.reasoning}`);

      // Log signal to database
      await supabase.from("signals").insert({
        timestamp: new Date().toISOString(),
        symbol,
        exchange: "binance",
        timeframe: "1h",
        signal_type: signal.signal,
        source: "edge_function_v2",
        confidence: signal.confidence,
        entry_price: ticker.price,
        reasoning: signal.reasoning,
        indicators: signal.indicators,
        ai_analysis: signal.aiAnalysis,
        risk_score: signal.riskScore,
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
        // POSITION MANAGEMENT
        const entryPrice = parseFloat(position.entry_price);
        const pnlPct = position.side === "buy"
          ? (ticker.price - entryPrice) / entryPrice
          : (entryPrice - ticker.price) / entryPrice;

        let shouldExit = false;
        let exitReason = "";

        // Dynamic stop loss based on ATR
        const atrStopLoss = Math.max(CONFIG.stopLossPct, signal.indicators.atrPercent / 100 * 1.5);

        // Stop loss
        if (pnlPct < -atrStopLoss) {
          shouldExit = true;
          exitReason = `Stop loss (ATR-adjusted: ${(atrStopLoss * 100).toFixed(1)}%)`;
        }

        // Take profit
        if (pnlPct > CONFIG.takeProfitPct) {
          shouldExit = true;
          exitReason = "Take profit triggered";
        }

        // Trailing stop (after 2% profit)
        if (pnlPct > 0.02 && pnlPct < CONFIG.takeProfitPct) {
          const trailingStop = pnlPct - CONFIG.trailingStopPct;
          // Check if price has pulled back more than trailing stop allows
          if (pnlPct < trailingStop) {
            shouldExit = true;
            exitReason = "Trailing stop triggered";
          }
        }

        // Signal-based exit
        if (position.side === "buy" && signal.signal.includes("sell") && signal.confidence >= 0.7) {
          shouldExit = true;
          exitReason = `Strong sell signal (confidence: ${(signal.confidence * 100).toFixed(0)}%)`;
        } else if (position.side === "sell" && signal.signal.includes("buy") && signal.confidence >= 0.7) {
          shouldExit = true;
          exitReason = `Strong buy signal (confidence: ${(signal.confidence * 100).toFixed(0)}%)`;
        }

        // High risk exit
        if (signal.riskScore > 0.8 && pnlPct > 0) {
          shouldExit = true;
          exitReason = "Risk management - securing profits";
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
              exit_indicators: signal.indicators,
            })
            .eq("id", position.id);

          console.log(`📉 CLOSED ${symbol} | PnL: $${pnl.toFixed(2)} (${(pnlPct * 100).toFixed(2)}%) | ${exitReason}`);
          results.push({ symbol, action: "close", pnl, pnlPct: pnlPct * 100, reason: exitReason });
        }
      } else {
        // ENTRY LOGIC with Symbol-Side Optimization
        const side = signal.signal.includes("buy") ? "buy" : "sell";
        const symbolSideKey = `${symbol}_${side}`;
        const symbolFilter = CONFIG.symbolSideFilters[symbolSideKey] || {
          minConfidence: CONFIG.minConfidence,
          requireMTFAlignment: false,
          requireSqueeze: false,
          positionMultiplier: 1.0,
        };

        // Check symbol-specific minimum confidence
        const meetsConfidence = signal.confidence >= symbolFilter.minConfidence;

        // Check MTF alignment requirement (for weak setups like BTC BUY)
        const mtfAligned = Math.abs(signal.indicators.momentum.alignment) > 0.4;
        const meetsMTF = !symbolFilter.requireMTFAlignment || mtfAligned;

        // Check squeeze requirement if enabled
        const inSqueeze = signal.indicators.squeezeMomentum.isSqueezing;
        const meetsSqueeze = !symbolFilter.requireSqueeze || inSqueeze;

        // On loss streak: require higher confidence and strong signal only
        const lossStreakFilter = onLossStreak
          ? (signal.confidence >= 0.70 && signal.signal.includes("strong"))
          : true;

        // NEW: Correlation filter - don't open same direction on correlated assets
        let correlationFilter = true;
        if (CONFIG.useCorrelationFilter) {
          for (const pair of CONFIG.correlatedPairs) {
            if (pair.includes(symbol)) {
              // Find the correlated symbol
              const correlatedSymbol = pair.find(s => s !== symbol);
              if (correlatedSymbol && openPositionMap.has(correlatedSymbol)) {
                const correlatedSide = openPositionMap.get(correlatedSymbol);
                if (correlatedSide === side) {
                  // Already have same-direction position in correlated asset
                  correlationFilter = false;
                  console.log(`   🔗 Correlation filter: Already ${correlatedSide} ${correlatedSymbol}, skipping ${side} ${symbol}`);
                }
              }
            }
          }
        }

        const shouldEnter = meetsConfidence &&
                           meetsMTF &&
                           meetsSqueeze &&
                           lossStreakFilter &&
                           correlationFilter &&
                           signal.riskScore < 0.7 &&
                           (signal.signal.includes("buy") || signal.signal.includes("sell"));

        // Log filter results for debugging
        if (!shouldEnter && (signal.signal.includes("buy") || signal.signal.includes("sell"))) {
          console.log(`   ⏭️ Filtered out: ${symbolSideKey} | conf=${signal.confidence.toFixed(2)} (need ${symbolFilter.minConfidence}) | MTF=${mtfAligned} (need ${symbolFilter.requireMTFAlignment})`);
        }

        if (shouldEnter) {

          // ENHANCED: Volatility-adjusted position sizing
          // Base sizing from confidence
          const confidenceMultiplier = 0.5 + (signal.confidence * 0.5);

          // Risk-based adjustment
          const riskMultiplier = 1 - (signal.riskScore * 0.5);

          // NEW: Volatility regime adjustment
          let volatilityMultiplier = 1.0;
          const volRegime = signal.indicators.riskMetrics.volatilityRegime;
          if (volRegime === "extreme") {
            volatilityMultiplier = 0.3; // Very small positions in extreme vol
          } else if (volRegime === "high") {
            volatilityMultiplier = 0.5;
          } else if (volRegime === "low") {
            volatilityMultiplier = 1.2; // Can size up in low vol
          }

          // NEW: ATR-based position sizing (Kelly-inspired)
          // Smaller positions when ATR is high relative to expected gain
          const atrAdjustment = Math.min(1.0, 2.0 / Math.max(signal.indicators.atrPercent, 0.5));

          // NEW: Multi-timeframe alignment bonus
          const mtfBonus = Math.abs(signal.indicators.momentum.alignment) > 0.6 ? 1.1 : 1.0;

          // NEW: Symbol-side position multiplier based on historical win rates
          const symbolPositionMultiplier = symbolFilter.positionMultiplier;

          // Calculate final position size
          const positionSize = CONFIG.maxPositionPct *
            confidenceMultiplier *
            riskMultiplier *
            volatilityMultiplier *
            atrAdjustment *
            mtfBonus *
            symbolPositionMultiplier;  // Apply symbol-specific adjustment

          // Cap at max position and ensure minimum
          const finalPositionSize = Math.max(0.02, Math.min(CONFIG.maxPositionPct, positionSize));

          console.log(`   📊 ${symbolSideKey} multiplier: ${symbolPositionMultiplier}x`);

          const positionValue = CONFIG.initialCapital * finalPositionSize;
          const quantity = positionValue / ticker.price;

          // ENHANCED: ATR-based stop loss with volatility regime adjustment
          let atrMultiplier = 2.0;
          if (volRegime === "high" || volRegime === "extreme") {
            atrMultiplier = 2.5; // Wider stops in high vol
          } else if (volRegime === "low") {
            atrMultiplier = 1.5; // Tighter stops in low vol
          }

          const atrStop = signal.indicators.atr * atrMultiplier;
          const stopLossPrice = side === "buy"
            ? ticker.price - atrStop
            : ticker.price + atrStop;

          // Dynamic take profit based on risk:reward (minimum 2:1)
          const riskRewardRatio = signal.confidence > 0.8 ? 3.0 : 2.0;
          const takeProfitPrice = side === "buy"
            ? ticker.price + (atrStop * riskRewardRatio)
            : ticker.price - (atrStop * riskRewardRatio);

          console.log(`   Position sizing: conf=${(confidenceMultiplier*100).toFixed(0)}% vol=${volRegime} atr=${atrAdjustment.toFixed(2)} → ${(finalPositionSize*100).toFixed(1)}%`);

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
            signal_source: "technical_ai_hybrid",
            signal_confidence: signal.confidence,
            entry_reasoning: signal.reasoning,
            indicators_at_entry: signal.indicators,
            ai_analysis: signal.aiAnalysis,
            stop_loss_price: stopLossPrice,
            take_profit_price: takeProfitPrice,
            risk_score: signal.riskScore,
          });

          console.log(`📈 OPENED ${side.toUpperCase()} ${symbol} @ $${ticker.price.toFixed(2)} | Size: ${(finalPositionSize * 100).toFixed(1)}%`);
          console.log(`   SL: $${stopLossPrice.toFixed(2)} | TP: $${takeProfitPrice.toFixed(2)} | R:R=${riskRewardRatio}:1`);
          results.push({ symbol, action: "open", side, price: ticker.price, size: finalPositionSize });
        }
      }

      results.push({
        symbol,
        price: ticker.price,
        signal: signal.signal,
        confidence: signal.confidence,
        riskScore: signal.riskScore,
        reasoning: signal.reasoning,
        regime: signal.indicators.marketRegime,
      });
    }

    // Save performance snapshot
    const { data: allTrades } = await supabase
      .from("trade_logs")
      .select("*")
      .eq("strategy_name", "edge_paper");

    const completedTrades = allTrades?.filter(t => t.exit_time) || [];
    const openTrades = allTrades?.filter(t => !t.exit_time) || [];
    const totalPnl = completedTrades.reduce((sum, t) => sum + parseFloat(t.net_pnl || 0), 0);
    const winningTrades = completedTrades.filter(t => parseFloat(t.net_pnl || 0) > 0).length;

    // Calculate advanced metrics
    const returns = completedTrades.map(t => parseFloat(t.return_pct || 0) / 100);
    const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
    const stdReturn = returns.length > 1
      ? Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / (returns.length - 1))
      : 0;
    const sharpeRatio = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(252) : 0;

    // Calculate profit factor
    const grossProfit = completedTrades.filter(t => parseFloat(t.net_pnl || 0) > 0)
      .reduce((sum, t) => sum + parseFloat(t.net_pnl || 0), 0);
    const grossLoss = Math.abs(completedTrades.filter(t => parseFloat(t.net_pnl || 0) < 0)
      .reduce((sum, t) => sum + parseFloat(t.net_pnl || 0), 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 10 : 0;

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
      sharpe_ratio: sharpeRatio,
      profit_factor: profitFactor,
      open_positions: openTrades.length,
    });

    console.log("\n✅ Advanced paper trading cycle complete");
    console.log(`📊 Stats: ${completedTrades.length} trades | ${(winningTrades / Math.max(completedTrades.length, 1) * 100).toFixed(1)}% win rate | PF: ${profitFactor.toFixed(2)}`);

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
