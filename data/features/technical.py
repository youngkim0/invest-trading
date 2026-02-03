"""Technical indicators module.

Provides 100+ technical indicators for trading analysis.
Uses the 'ta' library (pure Python) for indicator calculations.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
    ROCIndicator,
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
    WMAIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)


@dataclass
class IndicatorResult:
    """Result container for technical indicators."""

    name: str
    value: float | None
    signal: str | None = None  # "buy", "sell", "neutral"
    metadata: dict[str, Any] | None = None


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data."""

    def __init__(
        self,
        df: pd.DataFrame,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ):
        """Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV data
            open_col: Name of open column
            high_col: Name of high column
            low_col: Name of low column
            close_col: Name of close column
            volume_col: Name of volume column
        """
        self.df = df.copy()
        self.open = self.df[open_col].astype(float)
        self.high = self.df[high_col].astype(float)
        self.low = self.df[low_col].astype(float)
        self.close = self.df[close_col].astype(float)
        self.volume = self.df[volume_col].astype(float)

    # ==========================================================================
    # Trend Indicators
    # ==========================================================================

    def sma(self, period: int = 20) -> pd.Series:
        """Simple Moving Average."""
        indicator = SMAIndicator(self.close, window=period)
        return indicator.sma_indicator()

    def ema(self, period: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        indicator = EMAIndicator(self.close, window=period)
        return indicator.ema_indicator()

    def wma(self, period: int = 20) -> pd.Series:
        """Weighted Moving Average."""
        indicator = WMAIndicator(self.close, window=period)
        return indicator.wma()

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        indicator = MACD(
            self.close,
            window_slow=slow,
            window_fast=fast,
            window_sign=signal,
        )
        return {
            "macd": indicator.macd(),
            "signal": indicator.macd_signal(),
            "histogram": indicator.macd_diff(),
        }

    def adx(self, period: int = 14) -> dict[str, pd.Series]:
        """ADX (Average Directional Index)."""
        indicator = ADXIndicator(
            self.high, self.low, self.close, window=period
        )
        return {
            "adx": indicator.adx(),
            "di_pos": indicator.adx_pos(),
            "di_neg": indicator.adx_neg(),
        }

    def aroon(self, period: int = 25) -> dict[str, pd.Series]:
        """Aroon Indicator."""
        indicator = AroonIndicator(self.close, window=period)
        return {
            "aroon_up": indicator.aroon_up(),
            "aroon_down": indicator.aroon_down(),
            "aroon_indicator": indicator.aroon_indicator(),
        }

    def cci(self, period: int = 20, constant: float = 0.015) -> pd.Series:
        """Commodity Channel Index."""
        indicator = CCIIndicator(
            self.high, self.low, self.close, window=period, constant=constant
        )
        return indicator.cci()

    def dpo(self, period: int = 20) -> pd.Series:
        """Detrended Price Oscillator."""
        indicator = DPOIndicator(self.close, window=period)
        return indicator.dpo()

    def ichimoku(
        self,
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52,
    ) -> dict[str, pd.Series]:
        """Ichimoku Cloud."""
        indicator = IchimokuIndicator(
            self.high, self.low,
            window1=tenkan, window2=kijun, window3=senkou
        )
        return {
            "tenkan_sen": indicator.ichimoku_conversion_line(),
            "kijun_sen": indicator.ichimoku_base_line(),
            "senkou_span_a": indicator.ichimoku_a(),
            "senkou_span_b": indicator.ichimoku_b(),
        }

    def kst(self) -> dict[str, pd.Series]:
        """KST Oscillator."""
        indicator = KSTIndicator(self.close)
        return {
            "kst": indicator.kst(),
            "signal": indicator.kst_sig(),
            "diff": indicator.kst_diff(),
        }

    def mass_index(self, fast: int = 9, slow: int = 25) -> pd.Series:
        """Mass Index."""
        indicator = MassIndex(self.high, self.low, window_fast=fast, window_slow=slow)
        return indicator.mass_index()

    def psar(self, step: float = 0.02, max_step: float = 0.2) -> dict[str, pd.Series]:
        """Parabolic SAR."""
        indicator = PSARIndicator(
            self.high, self.low, self.close, step=step, max_step=max_step
        )
        return {
            "psar": indicator.psar(),
            "psar_up": indicator.psar_up(),
            "psar_down": indicator.psar_down(),
            "psar_up_indicator": indicator.psar_up_indicator(),
            "psar_down_indicator": indicator.psar_down_indicator(),
        }

    def stc(
        self,
        fast: int = 23,
        slow: int = 50,
        cycle: int = 10,
        smooth1: int = 3,
        smooth2: int = 3,
    ) -> pd.Series:
        """Schaff Trend Cycle."""
        indicator = STCIndicator(
            self.close,
            window_slow=slow,
            window_fast=fast,
            cycle=cycle,
            smooth1=smooth1,
            smooth2=smooth2,
        )
        return indicator.stc()

    def trix(self, period: int = 15) -> pd.Series:
        """TRIX Indicator."""
        indicator = TRIXIndicator(self.close, window=period)
        return indicator.trix()

    def vortex(self, period: int = 14) -> dict[str, pd.Series]:
        """Vortex Indicator."""
        indicator = VortexIndicator(
            self.high, self.low, self.close, window=period
        )
        return {
            "vi_pos": indicator.vortex_indicator_pos(),
            "vi_neg": indicator.vortex_indicator_neg(),
            "vi_diff": indicator.vortex_indicator_diff(),
        }

    # ==========================================================================
    # Momentum Indicators
    # ==========================================================================

    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        indicator = RSIIndicator(self.close, window=period)
        return indicator.rsi()

    def stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_window: int = 3,
    ) -> dict[str, pd.Series]:
        """Stochastic Oscillator."""
        indicator = StochasticOscillator(
            self.high, self.low, self.close,
            window=k_period,
            smooth_window=smooth_window,
        )
        return {
            "stoch_k": indicator.stoch(),
            "stoch_d": indicator.stoch_signal(),
        }

    def stoch_rsi(
        self,
        period: int = 14,
        smooth1: int = 3,
        smooth2: int = 3,
    ) -> dict[str, pd.Series]:
        """Stochastic RSI."""
        indicator = StochRSIIndicator(
            self.close,
            window=period,
            smooth1=smooth1,
            smooth2=smooth2,
        )
        return {
            "stoch_rsi": indicator.stochrsi(),
            "stoch_rsi_k": indicator.stochrsi_k(),
            "stoch_rsi_d": indicator.stochrsi_d(),
        }

    def williams_r(self, period: int = 14) -> pd.Series:
        """Williams %R."""
        indicator = WilliamsRIndicator(
            self.high, self.low, self.close, lbp=period
        )
        return indicator.williams_r()

    def roc(self, period: int = 12) -> pd.Series:
        """Rate of Change."""
        indicator = ROCIndicator(self.close, window=period)
        return indicator.roc()

    def ao(self, short: int = 5, long: int = 34) -> pd.Series:
        """Awesome Oscillator."""
        indicator = AwesomeOscillatorIndicator(
            self.high, self.low, window1=short, window2=long
        )
        return indicator.awesome_oscillator()

    def kama(self, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman's Adaptive Moving Average."""
        indicator = KAMAIndicator(
            self.close, window=period, pow1=fast, pow2=slow
        )
        return indicator.kama()

    def ppo(self, slow: int = 26, fast: int = 12, signal: int = 9) -> dict[str, pd.Series]:
        """Percentage Price Oscillator."""
        indicator = PPOIndicator(
            self.close, window_slow=slow, window_fast=fast, window_sign=signal
        )
        return {
            "ppo": indicator.ppo(),
            "signal": indicator.ppo_signal(),
            "histogram": indicator.ppo_hist(),
        }

    def pvo(self, slow: int = 26, fast: int = 12, signal: int = 9) -> dict[str, pd.Series]:
        """Percentage Volume Oscillator."""
        indicator = PercentageVolumeOscillator(
            self.volume, window_slow=slow, window_fast=fast, window_sign=signal
        )
        return {
            "pvo": indicator.pvo(),
            "signal": indicator.pvo_signal(),
            "histogram": indicator.pvo_hist(),
        }

    def tsi(self, slow: int = 25, fast: int = 13) -> pd.Series:
        """True Strength Index."""
        indicator = TSIIndicator(self.close, window_slow=slow, window_fast=fast)
        return indicator.tsi()

    def uo(
        self,
        short: int = 7,
        medium: int = 14,
        long: int = 28,
    ) -> pd.Series:
        """Ultimate Oscillator."""
        indicator = UltimateOscillator(
            self.high, self.low, self.close,
            window1=short, window2=medium, window3=long
        )
        return indicator.ultimate_oscillator()

    # ==========================================================================
    # Volatility Indicators
    # ==========================================================================

    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        indicator = AverageTrueRange(
            self.high, self.low, self.close, window=period
        )
        return indicator.average_true_range()

    def bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> dict[str, pd.Series]:
        """Bollinger Bands."""
        indicator = BollingerBands(
            self.close, window=period, window_dev=std_dev
        )
        return {
            "upper": indicator.bollinger_hband(),
            "middle": indicator.bollinger_mavg(),
            "lower": indicator.bollinger_lband(),
            "pband": indicator.bollinger_pband(),
            "wband": indicator.bollinger_wband(),
            "hband_indicator": indicator.bollinger_hband_indicator(),
            "lband_indicator": indicator.bollinger_lband_indicator(),
        }

    def donchian_channel(self, period: int = 20) -> dict[str, pd.Series]:
        """Donchian Channel."""
        indicator = DonchianChannel(
            self.high, self.low, self.close, window=period
        )
        return {
            "upper": indicator.donchian_channel_hband(),
            "middle": indicator.donchian_channel_mband(),
            "lower": indicator.donchian_channel_lband(),
            "pband": indicator.donchian_channel_pband(),
            "wband": indicator.donchian_channel_wband(),
        }

    def keltner_channel(
        self,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> dict[str, pd.Series]:
        """Keltner Channel."""
        indicator = KeltnerChannel(
            self.high, self.low, self.close,
            window=period,
            window_atr=atr_period,
            multiplier=multiplier,
        )
        return {
            "upper": indicator.keltner_channel_hband(),
            "middle": indicator.keltner_channel_mband(),
            "lower": indicator.keltner_channel_lband(),
            "pband": indicator.keltner_channel_pband(),
            "wband": indicator.keltner_channel_wband(),
            "hband_indicator": indicator.keltner_channel_hband_indicator(),
            "lband_indicator": indicator.keltner_channel_lband_indicator(),
        }

    def ulcer_index(self, period: int = 14) -> pd.Series:
        """Ulcer Index."""
        indicator = UlcerIndex(self.close, window=period)
        return indicator.ulcer_index()

    # ==========================================================================
    # Volume Indicators
    # ==========================================================================

    def obv(self) -> pd.Series:
        """On-Balance Volume."""
        indicator = OnBalanceVolumeIndicator(self.close, self.volume)
        return indicator.on_balance_volume()

    def adi(self) -> pd.Series:
        """Accumulation/Distribution Index."""
        indicator = AccDistIndexIndicator(
            self.high, self.low, self.close, self.volume
        )
        return indicator.acc_dist_index()

    def cmf(self, period: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        indicator = ChaikinMoneyFlowIndicator(
            self.high, self.low, self.close, self.volume, window=period
        )
        return indicator.chaikin_money_flow()

    def eom(self, period: int = 14) -> dict[str, pd.Series]:
        """Ease of Movement."""
        indicator = EaseOfMovementIndicator(
            self.high, self.low, self.volume, window=period
        )
        return {
            "eom": indicator.ease_of_movement(),
            "sma_eom": indicator.sma_ease_of_movement(),
        }

    def force_index(self, period: int = 13) -> pd.Series:
        """Force Index."""
        indicator = ForceIndexIndicator(self.close, self.volume, window=period)
        return indicator.force_index()

    def mfi(self, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        indicator = MFIIndicator(
            self.high, self.low, self.close, self.volume, window=period
        )
        return indicator.money_flow_index()

    def nvi(self) -> pd.Series:
        """Negative Volume Index."""
        indicator = NegativeVolumeIndexIndicator(self.close, self.volume)
        return indicator.negative_volume_index()

    def vpt(self) -> pd.Series:
        """Volume Price Trend."""
        indicator = VolumePriceTrendIndicator(self.close, self.volume)
        return indicator.volume_price_trend()

    def vwap(self) -> pd.Series:
        """Volume Weighted Average Price."""
        indicator = VolumeWeightedAveragePrice(
            self.high, self.low, self.close, self.volume
        )
        return indicator.volume_weighted_average_price()

    # ==========================================================================
    # Custom Indicators
    # ==========================================================================

    def typical_price(self) -> pd.Series:
        """Typical Price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    def true_range(self) -> pd.Series:
        """True Range."""
        high_low = self.high - self.low
        high_close = (self.high - self.close.shift(1)).abs()
        low_close = (self.low - self.close.shift(1)).abs()
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    def pivot_points(self) -> dict[str, float]:
        """Calculate pivot points (for most recent bar)."""
        h = float(self.high.iloc[-1])
        l = float(self.low.iloc[-1])
        c = float(self.close.iloc[-1])

        pivot = (h + l + c) / 3

        return {
            "pivot": pivot,
            "r1": 2 * pivot - l,
            "r2": pivot + (h - l),
            "r3": h + 2 * (pivot - l),
            "s1": 2 * pivot - h,
            "s2": pivot - (h - l),
            "s3": l - 2 * (h - pivot),
        }

    def fibonacci_retracements(
        self,
        lookback: int = 50,
    ) -> dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        high = float(self.high.tail(lookback).max())
        low = float(self.low.tail(lookback).min())
        diff = high - low

        return {
            "0.0": high,
            "0.236": high - diff * 0.236,
            "0.382": high - diff * 0.382,
            "0.5": high - diff * 0.5,
            "0.618": high - diff * 0.618,
            "0.786": high - diff * 0.786,
            "1.0": low,
        }

    def heikin_ashi(self) -> pd.DataFrame:
        """Calculate Heikin-Ashi candles."""
        ha_close = (self.open + self.high + self.low + self.close) / 4

        ha_open = pd.Series(index=self.df.index, dtype=float)
        ha_open.iloc[0] = (float(self.open.iloc[0]) + float(self.close.iloc[0])) / 2

        for i in range(1, len(self.df)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

        ha_high = pd.concat([self.high, ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([self.low, ha_open, ha_close], axis=1).min(axis=1)

        return pd.DataFrame({
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        })

    # ==========================================================================
    # Signal Generation
    # ==========================================================================

    def generate_signals(self) -> dict[str, IndicatorResult]:
        """Generate trading signals from multiple indicators.

        Returns:
            Dictionary of indicator results with signals
        """
        signals = {}

        # RSI
        rsi = self.rsi()
        rsi_value = float(rsi.iloc[-1])
        rsi_signal = "buy" if rsi_value < 30 else "sell" if rsi_value > 70 else "neutral"
        signals["rsi"] = IndicatorResult(
            name="RSI",
            value=rsi_value,
            signal=rsi_signal,
            metadata={"period": 14, "overbought": 70, "oversold": 30},
        )

        # MACD
        macd_data = self.macd()
        macd_value = float(macd_data["macd"].iloc[-1])
        macd_signal_value = float(macd_data["signal"].iloc[-1])
        macd_hist = macd_value - macd_signal_value
        macd_signal = "buy" if macd_hist > 0 else "sell" if macd_hist < 0 else "neutral"
        signals["macd"] = IndicatorResult(
            name="MACD",
            value=macd_value,
            signal=macd_signal,
            metadata={"signal_line": macd_signal_value, "histogram": macd_hist},
        )

        # Bollinger Bands
        bb = self.bollinger_bands()
        close = float(self.close.iloc[-1])
        upper = float(bb["upper"].iloc[-1])
        lower = float(bb["lower"].iloc[-1])
        bb_signal = "buy" if close < lower else "sell" if close > upper else "neutral"
        signals["bollinger"] = IndicatorResult(
            name="Bollinger Bands",
            value=close,
            signal=bb_signal,
            metadata={"upper": upper, "lower": lower, "middle": float(bb["middle"].iloc[-1])},
        )

        # ADX
        adx_data = self.adx()
        adx_value = float(adx_data["adx"].iloc[-1])
        di_pos = float(adx_data["di_pos"].iloc[-1])
        di_neg = float(adx_data["di_neg"].iloc[-1])
        if adx_value > 25:
            adx_signal = "buy" if di_pos > di_neg else "sell"
        else:
            adx_signal = "neutral"
        signals["adx"] = IndicatorResult(
            name="ADX",
            value=adx_value,
            signal=adx_signal,
            metadata={"di_pos": di_pos, "di_neg": di_neg, "trend_strength": adx_value > 25},
        )

        # MFI
        mfi = self.mfi()
        mfi_value = float(mfi.iloc[-1])
        mfi_signal = "buy" if mfi_value < 20 else "sell" if mfi_value > 80 else "neutral"
        signals["mfi"] = IndicatorResult(
            name="MFI",
            value=mfi_value,
            signal=mfi_signal,
            metadata={"overbought": 80, "oversold": 20},
        )

        # Stochastic
        stoch = self.stochastic()
        stoch_k = float(stoch["stoch_k"].iloc[-1])
        stoch_d = float(stoch["stoch_d"].iloc[-1])
        stoch_signal = "buy" if stoch_k < 20 else "sell" if stoch_k > 80 else "neutral"
        signals["stochastic"] = IndicatorResult(
            name="Stochastic",
            value=stoch_k,
            signal=stoch_signal,
            metadata={"k": stoch_k, "d": stoch_d},
        )

        return signals

    def calculate_all(self) -> dict[str, Any]:
        """Calculate all indicators and return as dictionary.

        Returns:
            Dictionary with all indicator values
        """
        result = {}

        # Trend
        result["sma_20"] = self.sma(20).iloc[-1]
        result["sma_50"] = self.sma(50).iloc[-1]
        result["sma_200"] = self.sma(200).iloc[-1] if len(self.df) >= 200 else None
        result["ema_12"] = self.ema(12).iloc[-1]
        result["ema_26"] = self.ema(26).iloc[-1]

        macd = self.macd()
        result["macd"] = macd["macd"].iloc[-1]
        result["macd_signal"] = macd["signal"].iloc[-1]
        result["macd_histogram"] = macd["histogram"].iloc[-1]

        adx = self.adx()
        result["adx"] = adx["adx"].iloc[-1]
        result["di_pos"] = adx["di_pos"].iloc[-1]
        result["di_neg"] = adx["di_neg"].iloc[-1]

        # Momentum
        result["rsi"] = self.rsi().iloc[-1]

        stoch = self.stochastic()
        result["stoch_k"] = stoch["stoch_k"].iloc[-1]
        result["stoch_d"] = stoch["stoch_d"].iloc[-1]

        result["williams_r"] = self.williams_r().iloc[-1]
        result["cci"] = self.cci().iloc[-1]
        result["roc"] = self.roc().iloc[-1]

        # Volatility
        result["atr"] = self.atr().iloc[-1]

        bb = self.bollinger_bands()
        result["bb_upper"] = bb["upper"].iloc[-1]
        result["bb_middle"] = bb["middle"].iloc[-1]
        result["bb_lower"] = bb["lower"].iloc[-1]
        result["bb_pband"] = bb["pband"].iloc[-1]
        result["bb_wband"] = bb["wband"].iloc[-1]

        # Volume
        result["obv"] = self.obv().iloc[-1]
        result["mfi"] = self.mfi().iloc[-1]
        result["cmf"] = self.cmf().iloc[-1]
        result["vwap"] = self.vwap().iloc[-1]

        # Pivot points
        pivots = self.pivot_points()
        for key, value in pivots.items():
            result[f"pivot_{key}"] = value

        # Fibonacci
        fibs = self.fibonacci_retracements()
        for key, value in fibs.items():
            result[f"fib_{key}"] = value

        return result


def calculate_indicators(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate specified indicators and add to DataFrame.

    Args:
        df: OHLCV DataFrame
        indicators: List of indicator names to calculate (default: all)

    Returns:
        DataFrame with indicator columns added
    """
    ti = TechnicalIndicators(df)
    result = df.copy()

    # Default indicators
    if indicators is None:
        indicators = [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "macd", "rsi", "stochastic", "adx",
            "atr", "bollinger", "obv", "mfi", "vwap",
        ]

    for ind in indicators:
        if ind == "sma_20":
            result["sma_20"] = ti.sma(20)
        elif ind == "sma_50":
            result["sma_50"] = ti.sma(50)
        elif ind == "ema_12":
            result["ema_12"] = ti.ema(12)
        elif ind == "ema_26":
            result["ema_26"] = ti.ema(26)
        elif ind == "macd":
            macd = ti.macd()
            result["macd"] = macd["macd"]
            result["macd_signal"] = macd["signal"]
            result["macd_histogram"] = macd["histogram"]
        elif ind == "rsi":
            result["rsi"] = ti.rsi()
        elif ind == "stochastic":
            stoch = ti.stochastic()
            result["stoch_k"] = stoch["stoch_k"]
            result["stoch_d"] = stoch["stoch_d"]
        elif ind == "adx":
            adx = ti.adx()
            result["adx"] = adx["adx"]
            result["di_pos"] = adx["di_pos"]
            result["di_neg"] = adx["di_neg"]
        elif ind == "atr":
            result["atr"] = ti.atr()
        elif ind == "bollinger":
            bb = ti.bollinger_bands()
            result["bb_upper"] = bb["upper"]
            result["bb_middle"] = bb["middle"]
            result["bb_lower"] = bb["lower"]
        elif ind == "obv":
            result["obv"] = ti.obv()
        elif ind == "mfi":
            result["mfi"] = ti.mfi()
        elif ind == "vwap":
            result["vwap"] = ti.vwap()

    return result
