"""Technical indicator helpers.

Comprehensive set of technical indicators for trading strategies.
Includes: RSI, MFI, EMA, SMA, ATR, BB (Bollinger Bands), Fib (Fibonacci)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index - volume-weighted RSI"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    delta = typical_price.diff()
    positive_flow = pd.Series(np.where(delta > 0, raw_money_flow, 0), index=close.index)
    negative_flow = pd.Series(np.where(delta < 0, raw_money_flow, 0), index=close.index)
    
    # Calculate money flow ratio
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    # Avoid division by zero
    mfr = positive_mf / negative_mf.replace(0, 0.000001)
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range - volatility indicator"""
    # Calculate True Range
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(period).mean()
    return atr


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2) -> tuple:
    """Bollinger Bands - returns (upper_band, middle_band, lower_band)"""
    middle_band = sma(series, period)
    std = series.rolling(period).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def fibonacci_retracement(high: float, low: float) -> dict:
    """Fibonacci retracement levels"""
    diff = high - low
    
    levels = {
        '0.0%': high,
        '23.6%': high - (diff * 0.236),
        '38.2%': high - (diff * 0.382),
        '50.0%': high - (diff * 0.5),
        '61.8%': high - (diff * 0.618),
        '78.6%': high - (diff * 0.786),
        '100.0%': low
    }
    
    return levels


def fibonacci_extension(high: float, low: float, swing_low: float) -> dict:
    """Fibonacci extension levels for targets"""
    diff = high - low
    
    levels = {
        '61.8%': high + (diff * 0.618),
        '100.0%': high + diff,
        '161.8%': high + (diff * 1.618),
        '261.8%': high + (diff * 2.618),
        '423.6%': high + (diff * 4.236)
    }
    
    return levels


def apply_indicators(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with the requested indicators."""

    result = df.copy()
    for name, params in spec.items():
        if name == "sma":
            result[f"sma_{params['period']}"] = sma(result[params.get("column", "close")], params["period"])
        elif name == "ema":
            result[f"ema_{params['period']}"] = ema(result[params.get("column", "close")], params["period"])
        elif name == "rsi":
            result[f"rsi_{params.get('period',14)}"] = rsi(result[params.get("column", "close")], params.get("period", 14))
        elif name == "mfi" and all(col in result.columns for col in ['high', 'low', 'close', 'volume']):
            result[f"mfi_{params.get('period',14)}"] = mfi(
                result['high'], result['low'], result['close'], result['volume'], 
                params.get("period", 14)
            )
        elif name == "atr" and all(col in result.columns for col in ['high', 'low', 'close']):
            result[f"atr_{params.get('period',14)}"] = atr(
                result['high'], result['low'], result['close'], 
                params.get("period", 14)
            )
        elif name == "bb":
            period = params.get('period', 20)
            num_std = params.get('num_std', 2)
            upper, middle, lower = bollinger_bands(result[params.get("column", "close")], period, num_std)
            result[f"bb_upper_{period}"] = upper
            result[f"bb_middle_{period}"] = middle
            result[f"bb_lower_{period}"] = lower
    return result


__all__ = [
    "sma", "ema", "rsi", "mfi", "atr", "bollinger_bands", 
    "fibonacci_retracement", "fibonacci_extension", "apply_indicators"
]
