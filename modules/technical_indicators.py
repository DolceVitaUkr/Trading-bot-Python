# modules/technical_indicators.py
import pandas as pd
import numpy as np
import logging
from scipy.stats import linregress
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    @staticmethod
    def moving_average(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window).mean()


    @staticmethod
    def ema(data: pd.DataFrame, window: int = 20, 
           price_col: str = 'close') -> pd.Series:
        """Exponential Moving Average with proper error handling"""
        try:
            return data[price_col].ewm(span=window, adjust=False, min_periods=window).mean()
        except Exception as e:
            logger.error(f"EMA error: {str(e)}", exc_info=True)
            return pd.Series()

    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs)).fillna(50)
    
    @staticmethod
    def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
            signal: int = 9, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD with histogram output"""
        try:
            fast_ema = data[price_col].ewm(span=fast, adjust=False).mean()
            slow_ema = data[price_col].ewm(span=slow, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"MACD error: {str(e)}", exc_info=True)
            return (pd.Series(), pd.Series(), pd.Series())

    @staticmethod
    def bollinger_bands(data: pd.DataFrame, window: int = 20, 
                       std_dev: float = 2.0, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands with dynamic standard deviation"""
        try:
            sma = data[price_col].rolling(window).mean()
            rolling_std = data[price_col].rolling(window).std()
            upper = sma + (rolling_std * std_dev)
            lower = sma - (rolling_std * std_dev)
            return sma, upper, lower
        except Exception as e:
            logger.error(f"Bollinger Bands error: {str(e)}", exc_info=True)
            return (pd.Series(), pd.Series(), pd.Series())

    @staticmethod
    def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range for volatility measurement"""
        try:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.ewm(alpha=1/window, min_periods=window).mean()
        except Exception as e:
            logger.error(f"ATR error: {str(e)}", exc_info=True)
            return pd.Series()

    @staticmethod
    def obv(data: pd.DataFrame, price_col: str = 'close', 
           volume_col: str = 'volume') -> pd.Series:
        """On-Balance Volume indicator"""
        try:
            obv = (np.sign(data[price_col].diff()) * data[volume_col])
            return obv.cumsum()
        except Exception as e:
            logger.error(f"OBV error: {str(e)}", exc_info=True)
            return pd.Series()
        except Exception as e:
            logger.error(f"OBV error: {str(e)}", exc_info=True)
            return pd.Series()

    @staticmethod
    def ichimoku_cloud(data: pd.DataFrame, conversion: int = 9, 
                      base: int = 26, leading: int = 52, 
                      displacement: int = 26) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Ichimoku Cloud with all components"""
        try:
            conversion_line = (data['high'].rolling(conversion).max() + 
                              data['low'].rolling(conversion).min()) / 2
            base_line = (data['high'].rolling(base).max() + 
                        data['low'].rolling(base).min()) / 2
            leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
            leading_span_b = ((data['high'].rolling(leading).max() + 
                             data['low'].rolling(leading).min()) / 2).shift(displacement)
            lagging_span = data['close'].shift(-displacement)
            return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span
        except Exception as e:
            logger.error(f"Ichimoku error: {str(e)}", exc_info=True)
            return (pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series())

    @staticmethod
    def stochastic_oscillator(data: pd.DataFrame, k_window: int = 14, 
                             d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator %K and %D"""
        try:
            low_min = data['low'].rolling(k_window).min()
            high_max = data['high'].rolling(k_window).max()
            k_line = 100 * ((data['close'] - low_min) / (high_max - low_min))
            d_line = k_line.rolling(d_window).mean()
            return k_line, d_line
        except Exception as e:
            logger.error(f"Stochastic error: {str(e)}", exc_info=True)
            return (pd.Series(), pd.Series())

    @staticmethod
    def fibonacci_retracement(data: pd.DataFrame, lookback: int = 30) -> dict:
        """Fibonacci Retracement Levels"""
        try:
            max_price = data['high'].rolling(lookback).max().iloc[-1]
            min_price = data['low'].rolling(lookback).min().iloc[-1]
            diff = max_price - min_price
            
            return {
                '0.0': max_price,
                '0.236': max_price - diff * 0.236,
                '0.382': max_price - diff * 0.382,
                '0.5': max_price - diff * 0.5,
                '0.618': max_price - diff * 0.618,
                '1.0': min_price
            }
        except Exception as e:
            logger.error(f"Fibonacci error: {str(e)}", exc_info=True)
            return {}

    @staticmethod
    def adv_volume_indicators(data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Advanced volume indicators (Volume SMA and Volume ROC)"""
        try:
            vol_sma = data['volume'].rolling(window).mean()
            vol_roc = (data['volume'] / data['volume'].shift(window) - 1) * 100
            return vol_sma, vol_roc
        except Exception as e:
            logger.error(f"Volume indicators error: {str(e)}", exc_info=True)
            return (pd.Series(), pd.Series())

    @staticmethod
    def market_regime(data: pd.DataFrame, short_window: int = 50, 
                     long_window: int = 200) -> pd.Series:
        """Market Regime Detection using EMA crossovers"""
        try:
            short_ema = data['close'].ewm(span=short_window).mean()
            long_ema = data['close'].ewm(span=long_window).mean()
            return np.where(short_ema > long_ema, 1, -1)
        except Exception as e:
            logger.error(f"Market regime error: {str(e)}", exc_info=True)
            return pd.Series()

    @staticmethod
    def entropy_volatility(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Information-theoretic volatility measure"""
        try:
            returns = data['close'].pct_change().dropna()
            rolling_entropy = returns.rolling(window).apply(
                lambda x: -np.sum(x * np.log(np.abs(x) + 1e-12))
            )
            return rolling_entropy
        except Exception as e:
            logger.error(f"Entropy volatility error: {str(e)}", exc_info=True)
            return pd.Series()
        except Exception as e:
            logger.error(f"Entropy volatility error: {str(e)}", exc_info=True)
            return pd.Series()

# Example usage:
if __name__ == "__main__":
    # Load sample data
    data = pd.read_csv('sample_data.csv', parse_dates=['date'])
    
    # Calculate indicators
    ti = TechnicalIndicators()
    data['sma_20'] = ti.moving_average(data)
    data['rsi_14'] = ti.rsi(data)
    data['macd'], data['signal'], data['histogram'] = ti.macd(data)
    data['atr_14'] = ti.atr(data)
    print("Technical indicators calculated successfully.")