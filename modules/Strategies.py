# modules/Strategies.py

"""
This module defines the individual trading strategy classes.

Each class encapsulates the logic for a specific strategy, such as Mean Reversion
or Trend Following. The StrategyManager will instantiate and run the appropriate
strategy based on its regime detection.
"""
import pandas as pd
import config
from modules.technical_indicators import TechnicalIndicators

class BaseStrategy:
    """
    Abstract base class for all trading strategies.
    It defines the common interface for strategies.
    """
    def __init__(self, params: dict):
        self.params = params

    def check_entry_conditions(self, df_5m, regime_context: dict):
        """
        Checks for entry signals based on 5m data and 15m context.

        :param df_5m: The 5-minute DataFrame for entry signals.
        :param regime_context: A dictionary with data from the 15m regime detection.
        :return: A signal dictionary {"side": "buy/sell", "tp": float, "sl": float} or None.
        """
        raise NotImplementedError

class TrendFollowStrategy(BaseStrategy):
    """
    Implements the Trend Following strategy.
    """
    def check_entry_conditions(self, df_5m: pd.DataFrame, regime_context: dict):
        params = self.params
        ti = TechnicalIndicators

        # 15m context checks
        close_15m = regime_context.get('close_15m')
        ema_100_15m = regime_context.get('ema_100_15m')
        ema_200_15m = regime_context.get('ema_200_15m')

        if not all([close_15m, ema_100_15m, ema_200_15m]):
            return None

        is_bullish_context = close_15m > ema_100_15m and close_15m > ema_200_15m
        is_bearish_context = close_15m < ema_100_15m and close_15m < ema_200_15m

        # 5m entry indicators
        close = df_5m['close']
        ema_short_1 = ti.ema(close, window=params['ema_short_period_1'][0])
        ema_short_2 = ti.ema(close, window=params['ema_short_period_2'][0])
        rsi = ti.rsi(close, window=params['rsi_period'][0])
        atr = ti.atr(df_5m['high'], df_5m['low'], close, period=params['atr_period'][0])

        if ema_short_1 is None or ema_short_2 is None or rsi is None or atr is None or len(ema_short_1) < 2:
            return None

        # Check for crossover in the last 2 candles
        cross_up = (ema_short_1.iloc[-2] < ema_short_2.iloc[-2]) and (ema_short_1.iloc[-1] > ema_short_2.iloc[-1])
        cross_down = (ema_short_1.iloc[-2] > ema_short_2.iloc[-2]) and (ema_short_1.iloc[-1] < ema_short_2.iloc[-1])

        last_rsi = rsi.iloc[-1]
        rsi_neutral_level = params['rsi_level_neutral'][1] # Use mid-point

        signal = None

        # Long entry
        if is_bullish_context and cross_up and last_rsi > rsi_neutral_level:
            signal = {"side": "buy"}
            sl_multiplier = params['sl_atr_multiplier_trend'][0]
            tp_multiplier = params['tp_atr_multiplier_trend'][0]
            signal['sl'] = close.iloc[-1] - (atr.iloc[-1] * sl_multiplier)
            signal['tp'] = close.iloc[-1] + (atr.iloc[-1] * tp_multiplier)

        # Short entry
        elif is_bearish_context and cross_down and last_rsi < rsi_neutral_level:
            signal = {"side": "sell"}
            sl_multiplier = params['sl_atr_multiplier_trend'][0]
            tp_multiplier = params['tp_atr_multiplier_trend'][0]
            signal['sl'] = close.iloc[-1] + (atr.iloc[-1] * sl_multiplier)
            signal['tp'] = close.iloc[-1] - (atr.iloc[-1] * tp_multiplier)

        return signal


class MeanReversionStrategy(BaseStrategy):
    """
    Implements the Mean Reversion trading strategy.
    """
    def check_entry_conditions(self, df_5m: pd.DataFrame, regime_context: dict):
        """
        Checks for Mean Reversion entry signals on the 5m timeframe.

        :param df_5m: The 5-minute DataFrame for entry signals.
        :param regime_context: A dictionary with data from the 15m regime detection.
        :return: A signal dictionary or None.
        """
        params = self.params

        # 1. 15m Context Filter
        close_15m = regime_context.get('close_15m')
        ema_100_15m = regime_context.get('ema_100_15m')
        atr_15m = regime_context.get('atr_15m')

        if not all([close_15m, ema_100_15m, atr_15m]):
            return None # Missing context data

        # Condition: |price−EMA100| ≤ 1×ATR
        if abs(close_15m - ema_100_15m) > atr_15m:
            return None # Context filter not met

        # 2. Calculate 5m indicators
        ti = TechnicalIndicators
        close = df_5m['close']

        # Bollinger Bands
        middle_bb, upper_bb, lower_bb = ti.bollinger_bands(
            close, window=params['bb_period'][0], num_std=params['bb_std'][0]
        )

        # RSI
        rsi = ti.rsi(close, window=params['rsi_period'][0])

        # Z-score of EMA
        ema_for_zscore = ti.ema(close, window=params['zscore_lookback'][0])
        z_score = ti.z_score(ema_for_zscore, window=params['zscore_lookback'][0])

        # ATR for TP/SL
        atr = ti.atr(
            df_5m['high'], df_5m['low'], close, period=params['atr_period'][0]
        )

        if upper_bb is None or rsi is None or z_score is None or atr is None:
            return None

        # Get the latest values
        last_close = close.iloc[-1]
        last_upper_bb = upper_bb.iloc[-1]
        last_lower_bb = lower_bb.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_z_score = z_score.iloc[-1]
        last_atr = atr.iloc[-1]

        if pd.isna(last_upper_bb) or pd.isna(last_rsi) or pd.isna(last_z_score) or pd.isna(last_atr):
            return None

        # --- Entry Conditions ---
        # Long Entry: Price below lower BB, RSI oversold, Z-score shows deviation
        is_long_entry = (
            last_close < last_lower_bb and
            last_rsi <= params['rsi_os_level'][1] and # Using the upper bound of the OS level
            abs(last_z_score) >= params['zscore_entry_threshold'][0]
        )

        # Short Entry: Price above upper BB, RSI overbought, Z-score shows deviation
        is_short_entry = (
            last_close > last_upper_bb and
            last_rsi >= params['rsi_ob_level'][0] and # Using the lower bound of the OB level
            abs(last_z_score) >= params['zscore_entry_threshold'][0]
        )

        signal = None
        if is_long_entry:
            signal = {"side": "buy"}
            sl_multiplier = params['sl_atr_multiplier_mr'][0]
            tp_multiplier = params['tp_atr_multiplier_mr'][0]
            signal['sl'] = last_close - (last_atr * sl_multiplier)
            signal['tp'] = last_close + (last_atr * tp_multiplier)
            # For MR, TP can also be the BB midline
            # signal['tp'] = middle_bb.iloc[-1]

        elif is_short_entry:
            signal = {"side": "sell"}
            sl_multiplier = params['sl_atr_multiplier_mr'][0]
            tp_multiplier = params['tp_atr_multiplier_mr'][0]
            signal['sl'] = last_close + (last_atr * sl_multiplier)
            signal['tp'] = last_close - (last_atr * tp_multiplier)
            # For MR, TP can also be the BB midline
            # signal['tp'] = middle_bb.iloc[-1]

        return signal
