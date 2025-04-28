# modules/technical_indicators.py

import math
from typing import List, Tuple, Dict, Optional

def moving_average(data: List[float], period: int) -> Optional[float]:
    """
    Simple Moving Average (SMA) of the last `period` values.
    """
    if data is None or period <= 0 or len(data) < period:
        return None
    return sum(data[-period:]) / period


def exponential_moving_average(data: List[float], period: int) -> Optional[float]:
    """
    Exponential Moving Average (EMA) over `period` values.
    """
    if data is None or period <= 0 or len(data) < period:
        return None
    # Start EMA with SMA of first `period` points
    ema = sum(data[:period]) / period
    multiplier = 2.0 / (period + 1)
    for price in data[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def adx(high: List[float], low: List[float], close: List[float],
        period: int = 14) -> Optional[float]:
    """
    Average Directional Index, a trend-strength indicator.
    Returns the latest ADX value.
    """
    n = period
    length = len(close)
    if not (high and low and close) or length < 2 * n:
        return None

    # True Range, +DM, -DM
    tr = [0.0] * length
    pdm = [0.0] * length
    ndm = [0.0] * length
    for i in range(1, length):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        pdm[i] = up if up > down and up > 0 else 0.0
        ndm[i] = down if down > up and down > 0 else 0.0

    # Wilder's smoothing
    atr = sum(tr[1:n+1]) / n
    pdm_sum = sum(pdm[1:n+1])
    ndm_sum = sum(ndm[1:n+1])

    if atr == 0:
        return 0.0

    pdi = (pdm_sum / atr) * 100.0
    ndi = (ndm_sum / atr) * 100.0
    prev_adx = abs(pdi - ndi) / (pdi + ndi) * 100.0 if (pdi + ndi) else 0.0

    for i in range(n+1, length):
        atr = (atr * (n - 1) + tr[i]) / n
        pdm_sum = (pdm_sum * (n - 1) + pdm[i]) / n
        ndm_sum = (ndm_sum * (n - 1) + ndm[i]) / n
        pdi = (pdm_sum / atr) * 100.0 if atr else 0.0
        ndi = (ndm_sum / atr) * 100.0 if atr else 0.0
        dx = abs(pdi - ndi) / (pdi + ndi) * 100.0 if (pdi + ndi) else 0.0
        prev_adx = ((prev_adx * (n - 1)) + dx) / n

    return prev_adx


def cci(high: List[float], low: List[float], close: List[float],
        period: int = 20) -> Optional[float]:
    """
    Commodity Channel Index (CCI) for the latest data point.
    """
    length = len(close)
    if not (high and low and close) or length < period:
        return None

    # Typical Price
    tp = [(high[i] + low[i] + close[i]) / 3.0 for i in range(length)]
    tp_window = tp[-period:]
    sma_tp = sum(tp_window) / period
    mean_dev = sum(abs(x - sma_tp) for x in tp_window) / period
    if mean_dev == 0:
        return 0.0
    return (tp[-1] - sma_tp) / (0.015 * mean_dev)


def williams_r(high: List[float], low: List[float], close: List[float],
               period: int = 14) -> Optional[float]:
    """
    Williams %R oscillator (range 0 to -100).
    """
    length = len(close)
    if not (high and low and close) or length < period:
        return None

    hh = max(high[-period:])
    ll = min(low[-period:])
    if hh == ll:
        return 0.0
    return -100.0 * (hh - close[-1]) / (hh - ll)


def obv(close: List[float], volume: List[float]) -> Optional[List[float]]:
    """
    On-Balance Volume (OBV) series.
    """
    if close is None or volume is None or len(close) != len(volume):
        return None

    length = len(close)
    obv_series = [0.0] * length
    for i in range(1, length):
        if close[i] > close[i-1]:
            obv_series[i] = obv_series[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv_series[i] = obv_series[i-1] - volume[i]
        else:
            obv_series[i] = obv_series[i-1]
    return obv_series


def stochastic_oscillator(
    high: List[float],
    low: List[float],
    close: List[float],
    k_period: int = 14,
    d_period: int = 3
) -> Optional[Tuple[List[Optional[float]], List[Optional[float]]]]:
    """
    %K and %D lines of the Stochastic Oscillator.
    Pads initial values with None.
    """
    length = len(close)
    if not (high and low and close) or length < k_period:
        return None

    k_line: List[Optional[float]] = [None] * length
    d_line: List[Optional[float]] = [None] * length

    for i in range(length):
        if i + 1 >= k_period:
            low_min = min(low[i+1-k_period:i+1])
            high_max = max(high[i+1-k_period:i+1])
            denom = (high_max - low_min) or 1e-12
            k_val = (close[i] - low_min) / denom * 100.0
            k_line[i] = k_val

            if i + 1 >= k_period + d_period - 1:
                window = [v for v in k_line[i+1-d_period:i+1] if v is not None]
                if window:
                    d_line[i] = sum(window) / len(window)
    return k_line, d_line


def fibonacci_retracement(
    high: List[float],
    low: List[float],
    close: List[float],
    lookback: int = 14
) -> Dict[str, float]:
    """
    Standard Fibonacci retracement levels over last `lookback` bars.
    Returns a dict: "0%", "23.6%", "38.2%", "50%", "61.8%", "100%".
    """
    length = len(close)
    if not (high and low and close) or length < lookback:
        return {}

    window_high = max(high[-lookback:])
    window_low = min(low[-lookback:])
    diff = window_high - window_low

    return {
        "0%": window_high,
        "23.6%": window_high - 0.236 * diff,
        "38.2%": window_high - 0.382 * diff,
        "50%": window_high - 0.50 * diff,
        "61.8%": window_high - 0.618 * diff,
        "100%": window_low
    }


def market_regime(
    close: List[float],
    short_period: int = 50,
    long_period: int = 200
) -> Optional[List[Optional[int]]]:
    """
    Market regime series: +1 (bull) if short SMA > long SMA, -1 if short < long.
    Pads initial values with None.
    """
    length = len(close)
    if not close or length < long_period:
        return None

    regime: List[Optional[int]] = [None] * length
    for i in range(length):
        if i + 1 >= long_period:
            short_sma = sum(close[i+1-short_period:i+1]) / short_period if i+1>=short_period else None
            long_sma  = sum(close[i+1-long_period:i+1]) / long_period
            if short_sma is not None:
                regime[i] = 1 if short_sma > long_sma else -1
    return regime


def entropy_volatility(
    close: List[float],
    period: int = 14
) -> Optional[List[Optional[float]]]:
    """
    Rolling entropy of absolute returns over `period`, as a proxy for volatility.
    Pads first (period) entries with None.
    """
    length = len(close)
    if not close or length < period + 1:
        return None

    # compute returns
    returns = [0.0] * length
    for i in range(1, length):
        prev = close[i-1] or 1e-12
        returns[i] = (close[i] - close[i-1]) / prev

    ent_series: List[Optional[float]] = [None] * length
    for i in range(period, length):
        window = returns[i+1-period:i+1]
        abs_w = [abs(r) for r in window]
        total = sum(abs_w) or 1e-12
        entropy = 0.0
        for r in window:
            p = abs(r) / total
            if p > 0:
                entropy -= p * math.log(p)
        ent_series[i] = entropy

    return ent_series
