# modules/technical_indicators.py

import math
from typing import List, Tuple, Dict, Optional

# NOTE: This module exposes a *function* API (sma, ema, rsi, atr, etc.)
# AND a wrapper class `TechnicalIndicators` with pandas-aware static methods.
# - Functions: operate on plain Python lists, return scalars/series-likes.
# - Class methods: accept pandas Series (preferred) or lists; return pandas Series.


# ─────────────────────────────
# Core moving averages (list API)
# ─────────────────────────────
def sma(data: List[float], window: int) -> Optional[float]:
    """Simple Moving Average of the last `window` values."""
    if not data or window <= 0 or len(data) < window:
        return None
    return sum(data[-window:]) / window


def ema(data: List[float], window: int) -> Optional[float]:
    """Exponential Moving Average over `window` values (Wilder-style)."""
    if not data or window <= 0 or len(data) < window:
        return None
    seed = sum(data[:window]) / window
    k = 2.0 / (window + 1)
    ema_val = seed
    for price in data[window:]:
        ema_val = (price - ema_val) * k + ema_val
    return ema_val


# Backwards-compat aliases
def moving_average(data: List[float], period: int) -> Optional[float]:
    return sma(data, period)


def exponential_moving_average(data: List[float], period: int) -> Optional[float]:
    return ema(data, period)


# ─────────────────────────────
# Momentum / Oscillators (list API)
# ─────────────────────────────
def rsi(close: List[float], window: int = 14) -> Optional[float]:
    """Relative Strength Index (last value)."""
    if not close or len(close) < window + 1 or window <= 0:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, window + 1):
        diff = close[i] - close[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / window
    avg_loss = losses / window

    for i in range(window + 1, len(close)):
        diff = close[i] - close[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def stochastic_oscillator(
    high: List[float],
    low: List[float],
    close: List[float],
    k_period: int = 14,
    d_period: int = 3
) -> Optional[Tuple[List[Optional[float]], List[Optional[float]]]]:
    """%K and %D lines (list API). Pads initial values with None."""
    length = len(close)
    if not (high and low and close) or length < k_period:
        return None
    k_line: List[Optional[float]] = [None] * length
    d_line: List[Optional[float]] = [None] * length
    for i in range(length):
        if i + 1 >= k_period:
            low_min = min(low[i + 1 - k_period:i + 1])
            high_max = max(high[i + 1 - k_period:i + 1])
            denom = (high_max - low_min) or 1e-12
            k_val = (close[i] - low_min) / denom * 100.0
            k_line[i] = k_val
            if i + 1 >= k_period + d_period - 1:
                window = [v for v in k_line[i + 1 - d_period:i + 1] if v is not None]
                if window:
                    d_line[i] = sum(window) / len(window)
    return k_line, d_line


def williams_r(high: List[float], low: List[float], close: List[float],
               period: int = 14) -> Optional[float]:
    """Williams %R oscillator (range 0 to -100)."""
    length = len(close)
    if not (high and low and close) or length < period:
        return None
    hh = max(high[-period:])
    ll = min(low[-period:])
    if hh == ll:
        return 0.0
    return -100.0 * (hh - close[-1]) / (hh - ll)


# ─────────────────────────────
# Trend / Volatility (list API)
# ─────────────────────────────
def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]:
    """Average True Range (last value, Wilder smoothing)."""
    if period <= 0:
        return None
    n = period
    length = len(close)
    if not (high and low and close) or length < n + 1:
        return None
    tr = [0.0] * length
    for i in range(1, length):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    atr_val = sum(tr[1:n + 1]) / n
    for i in range(n + 1, length):
        atr_val = (atr_val * (n - 1) + tr[i]) / n
    return atr_val


def adx(high: List[float], low: List[float], close: List[float],
        period: int = 14) -> Optional[float]:
    """Average Directional Index (trend strength), last value."""
    n = period
    length = len(close)
    if not (high and low and close) or length < 2 * n:
        return None
    tr = [0.0] * length
    pdm = [0.0] * length
    ndm = [0.0] * length
    for i in range(1, length):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up if (up > down and up > 0) else 0.0
        ndm[i] = down if (down > up and down > 0) else 0.0
    atr_seed = sum(tr[1:n + 1]) / n
    p_seed = sum(pdm[1:n + 1])
    n_seed = sum(ndm[1:n + 1])
    if atr_seed == 0:
        return 0.0
    pdi = (p_seed / atr_seed) * 100.0
    ndi = (n_seed / atr_seed) * 100.0
    prev_adx = (abs(pdi - ndi) / (pdi + ndi) * 100.0) if (pdi + ndi) else 0.0
    atr_s = atr_seed
    p_s = p_seed
    n_s = n_seed
    for i in range(n + 1, length):
        atr_s = (atr_s * (n - 1) + tr[i]) / n
        p_s = (p_s * (n - 1) + pdm[i]) / n
        n_s = (n_s * (n - 1) + ndm[i]) / n
        pdi = (p_s / atr_s) * 100.0 if atr_s else 0.0
        ndi = (n_s / atr_s) * 100.0 if atr_s else 0.0
        dx = (abs(pdi - ndi) / (pdi + ndi) * 100.0) if (pdi + ndi) else 0.0
        prev_adx = ((prev_adx * (n - 1)) + dx) / n
    return prev_adx


def cci(high: List[float], low: List[float], close: List[float],
        period: int = 20) -> Optional[float]:
    """Commodity Channel Index (last value)."""
    length = len(close)
    if not (high and low and close) or length < period:
        return None
    tp = [(high[i] + low[i] + close[i]) / 3.0 for i in range(length)]
    tp_window = tp[-period:]
    sma_tp = sum(tp_window) / period
    mean_dev = sum(abs(x - sma_tp) for x in tp_window) / period
    if mean_dev == 0:
        return 0.0
    return (tp[-1] - sma_tp) / (0.015 * mean_dev)


def bollinger_bands(close: List[float], period: int = 20, num_std: float = 2.0) -> Optional[Tuple[float, float, float]]:
    """Bollinger Bands (middle SMA, upper, lower) for the last bar."""
    if not close or len(close) < period:
        return None
    window = close[-period:]
    mid = sum(window) / period
    var = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(var)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def entropy_volatility(close: List[float], period: int = 14) -> Optional[List[Optional[float]]]:
    """Rolling entropy of absolute returns over `period` (list API)."""
    length = len(close)
    if not close or length < period + 1:
        return None
    returns = [0.0] * length
    for i in range(1, length):
        prev = close[i - 1] or 1e-12
        returns[i] = (close[i] - close[i - 1]) / prev
    ent_series: List[Optional[float]] = [None] * length
    for i in range(period, length):
        window = returns[i + 1 - period:i + 1]
        abs_w = [abs(r) for r in window]
        total = sum(abs_w) or 1e-12
        entropy = 0.0
        for r in window:
            p = abs(r) / total
            if p > 0:
                entropy -= p * math.log(p)
        ent_series[i] = entropy
    return ent_series


def obv(close: List[float], volume: List[float]) -> Optional[List[float]]:
    """On-Balance Volume series (list API)."""
    if close is None or volume is None or len(close) != len(volume):
        return None
    length = len(close)
    obv_series = [0.0] * length
    for i in range(1, length):
        if close[i] > close[i - 1]:
            obv_series[i] = obv_series[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv_series[i] = obv_series[i - 1] - volume[i]
        else:
            obv_series[i] = obv_series[i - 1]
    return obv_series


def fibonacci_retracement(
    high: List[float],
    low: List[float],
    close: List[float],
    lookback: int = 14
) -> Dict[str, float]:
    """Standard Fibonacci retracement levels over last `lookback` bars."""
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


def market_regime(close: List[float], short_period: int = 50, long_period: int = 200) -> Optional[List[Optional[int]]]:
    """Market regime series (list API). +1 if short SMA > long SMA, else -1."""
    length = len(close)
    if not close or length < long_period:
        return None
    regime: List[Optional[int]] = [None] * length
    for i in range(length):
        if i + 1 >= long_period:
            short_sma = sum(close[i + 1 - short_period:i + 1]) / short_period if i + 1 >= short_period else None
            long_sma = sum(close[i + 1 - long_period:i + 1]) / long_period
            if short_sma is not None:
                regime[i] = 1 if short_sma > long_sma else -1
    return regime


# ─────────────────────────────
# Pandas-aware wrapper class (single definition)
# ─────────────────────────────
class TechnicalIndicators:
    sma = staticmethod(sma)
    ema = staticmethod(ema)
    rsi = staticmethod(rsi)
    atr = staticmethod(atr)
    adx = staticmethod(adx)
    cci = staticmethod(cci)
    williams_r = staticmethod(williams_r)
    obv = staticmethod(obv)
    stochastic_oscillator = staticmethod(stochastic_oscillator)
    bollinger_bands = staticmethod(bollinger_bands)
    fibonacci_retracement = staticmethod(fibonacci_retracement)
    market_regime = staticmethod(market_regime)
    entropy_volatility = staticmethod(entropy_volatility)

    @staticmethod
    def _to_series(x):
        try:
            import pandas as pd  # lazy import
            if hasattr(x, "rolling"):  # already a Series/DataFrame column
                return x
            return pd.Series(list(x))
        except Exception:
            return None

    @staticmethod
    def sma(series, window: int):
        s = TechnicalIndicators._to_series(series)
        if s is None:
            return None
        try:
            return s.rolling(window=window, min_periods=window).mean()
        except Exception:
            return None

    @staticmethod
    def ema(series, window: int):
        s = TechnicalIndicators._to_series(series)
        if s is None:
            return None
        try:
            return s.ewm(span=window, adjust=False, min_periods=window).mean()
        except Exception:
            return None

    @staticmethod
    def rsi(series, window: int = 14):
        s = TechnicalIndicators._to_series(series)
        if s is None:
            return None
        try:
            delta = s.diff()
            up = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
            down = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
            rs = up / down.replace(0, float("inf"))
            return 100 - (100 / (1 + rs))
        except Exception:
            return None

