# modules/technical_indicators.py
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class TechnicalIndicators:
    @staticmethod
    def moving_average(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average over specified window."""
        return series.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def ema(data: pd.DataFrame, window: int = 20, price_col: str = 'close') -> pd.Series:
        """Exponential Moving Average on specified price column."""
        try:
            return data[price_col].ewm(span=window, adjust=False, min_periods=window).mean()
        except Exception as e:
            logger.error(f"EMA error: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26,
             signal: int = 9, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, histogram."""
        try:
            fast_ema = TechnicalIndicators.ema(data, fast, price_col)
            slow_ema = TechnicalIndicators.ema(data, slow, price_col)
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            hist = macd_line - signal_line
            return macd_line, signal_line, hist
        except Exception as e:
            logger.error(f"MACD error: {e}")
            empty = pd.Series(dtype=float)
            return empty, empty, empty

    @staticmethod
    def bollinger_bands(data: pd.DataFrame, window: int = 20,
                        std_dev: float = 2.0, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: middle, upper, lower."""
        try:
            mid = TechnicalIndicators.moving_average(data[price_col], window)
            std = data[price_col].rolling(window=window, min_periods=window).std()
            upper = mid + std * std_dev
            lower = mid - std * std_dev
            return mid, upper, lower
        except Exception as e:
            logger.error(f"Bollinger Bands error: {e}")
            empty = pd.Series(dtype=float)
            return empty, empty, empty

    @staticmethod
    def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range."""
        try:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.ewm(alpha=1/window, min_periods=window).mean()
        except Exception as e:
            logger.error(f"ATR error: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def obv(data: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
        """On-Balance Volume."""
        try:
            sign = np.sign(data[price_col].diff())
            obv = (sign * data[volume_col]).cumsum()
            return obv
        except Exception as e:
            logger.error(f"OBV error: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def stochastic_oscillator(data: pd.DataFrame, k_window: int = 14,
                              d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic %K and %D."""
        try:
            low_min = data['low'].rolling(window=k_window, min_periods=k_window).min()
            high_max = data['high'].rolling(window=k_window, min_periods=k_window).max()
            k_line = 100 * (data['close'] - low_min) / (high_max - low_min)
            d_line = k_line.rolling(window=d_window, min_periods=d_window).mean()
            return k_line, d_line
        except Exception as e:
            logger.error(f"Stochastic error: {e}")
            empty = pd.Series(dtype=float)
            return empty, empty

    @staticmethod
    def fibonacci_retracement(data: pd.DataFrame, lookback: int = 30) -> Dict[str, float]:
        """Fibonacci levels."""
        try:
            max_p = data['high'].rolling(window=lookback, min_periods=lookback).max().iloc[-1]
            min_p = data['low'].rolling(window=lookback, min_periods=lookback).min().iloc[-1]
            diff = max_p - min_p
            return {
                '0.0': max_p,
                '0.236': max_p - diff * 0.236,
                '0.382': max_p - diff * 0.382,
                '0.5': max_p - diff * 0.5,
                '0.618': max_p - diff * 0.618,
                '1.0': min_p
            }
        except Exception as e:
            logger.error(f"Fibonacci error: {e}")
            return {}

    @staticmethod
    def adv_volume_indicators(data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Volume SMA and ROC."""
        try:
            vol_sma = data['volume'].rolling(window=window, min_periods=window).mean()
            vol_roc = (data['volume'] / data['volume'].shift(window) - 1) * 100
            return vol_sma, vol_roc
        except Exception as e:
            logger.error(f"Volume indicators error: {e}")
            empty = pd.Series(dtype=float)
            return empty, empty

    @staticmethod
    def market_regime(data: pd.DataFrame, short_window: int = 50,
                      long_window: int = 200) -> pd.Series:
        """EMA crossover regime: +1 bull, -1 bear."""
        try:
            short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
            long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
            return pd.Series(np.where(short_ema > long_ema, 1, -1), index=data.index)
        except Exception as e:
            logger.error(f"Market regime error: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def entropy_volatility(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Rolling entropy of returns."""
        try:
            returns = data['close'].pct_change().dropna()
            return returns.rolling(window, min_periods=window).apply(
                lambda x: -np.sum(x * np.log(np.abs(x) + 1e-12)), raw=True)
        except Exception as e:
            logger.error(f"Entropy volatility error: {e}")
            return pd.Series(dtype=float)

# -----------------------------------------------------------------------------
# Module-level list-based functions for basic tests
# -----------------------------------------------------------------------------

def moving_average(data: List[float], period: int) -> Optional[float]:
    """SMA of last 'period' points."""
    if data is None or period <= 0 or len(data) < period:
        return None
    return sum(data[-period:]) / period

def exponential_moving_average(data: List[float], period: int) -> Optional[float]:
    """EMA over list."""
    if data is None or period <= 0 or len(data) < period:
        return None
    ema_val = sum(data[:period]) / period
    k = 2 / (period + 1)
    for price in data[period:]:
        ema_val = (price - ema_val) * k + ema_val
    return ema_val

def adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]:
    """List-based ADX."""
    n = period
    if not (high and low and close) or len(close) < 2 * n:
        return None
    length = len(close)
    tr = [0.0] * length
    pdm = [0.0] * length
    ndm = [0.0] * length

    # True Range and Directional Movements
    for i in range(1, length):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up if up > down and up > 0 else 0.0
        ndm[i] = down if down > up and down > 0 else 0.0

    # Initial Wilder sums
    tr_sum = sum(tr[1:n + 1])
    pdm_sum = sum(pdm[1:n + 1])
    ndm_sum = sum(ndm[1:n + 1])

    def safe_div(a, b):
        return a / b if b else 0.0

    plus = safe_div(pdm_sum, tr_sum) * 100
    minus = safe_div(ndm_sum, tr_sum) * 100
    first_dx = safe_div(abs(plus - minus), plus + minus) * 100 if (plus + minus) else 0.0

    dx_list = [None] * length
    dx_list[n] = first_dx

    # Wilder smoothing for DX
    for i in range(n + 1, length):
        tr_sum = tr_sum - tr_sum / n + tr[i]
        pdm_sum = pdm_sum - pdm_sum / n + pdm[i]
        ndm_sum = ndm_sum - ndm_sum / n + ndm[i]
        plus = safe_div(pdm_sum, tr_sum) * 100
        minus = safe_div(ndm_sum, tr_sum) * 100
        dx_list[i] = safe_div(abs(plus - minus), plus + minus) * 100 if (plus + minus) else 0.0

    # Initial ADX (average of first n DX)
    valid_dx = [d for d in dx_list[n:2 * n] if d is not None]
    initial_adx = sum(valid_dx) / len(valid_dx) if valid_dx else 0.0

    adx_list = [None] * length
    adx_list[2 * n - 1] = initial_adx

    # Wilder smoothing for ADX
    for j in range(2 * n, length):
        prev = adx_list[j - 1] if adx_list[j - 1] is not None else initial_adx
        adx_list[j] = (prev * (n - 1) + dx_list[j]) / n

    return adx_list[-1]

def cci(high: List[float], low: List[float], close: List[float], period: int = 20) -> Optional[float]:
    """List-based CCI."""
    if not (high and low and close) or len(close) < period:
        return None
    tp = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
    sma_tp = sum(tp[-period:]) / period
    mean_dev = sum(abs(x - sma_tp) for x in tp[-period:]) / period
    if mean_dev == 0:
        return 0.0
    return (tp[-1] - sma_tp) / (0.015 * mean_dev)

def williams_r(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]:
    """List-based Williams %R."""
    if not (high and low and close) or len(close) < period:
        return None
    highest = max(high[-period:])
    lowest = min(low[-period:])
    if highest == lowest:
        return 0.0
    return (highest - close[-1]) / (highest - lowest) * -100

