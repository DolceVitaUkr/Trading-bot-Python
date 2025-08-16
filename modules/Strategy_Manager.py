import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import config
from modules.data_manager import DataManager
from modules.technical_indicators import TechnicalIndicators
from modules.Strategies import TrendFollowStrategy, MeanReversionStrategy, BaseStrategy

logger = logging.getLogger(__name__)

@dataclass
class Decision:
    """Data class to hold the outcome of a strategy decision."""
    signal: str
    sl: float
    tp: Optional[float]
    meta: Dict[str, Any] = field(default_factory=dict)

class StrategyManager:
    """
    Selects a strategy based on market regime and produces a trading decision,
    including a signal score and a "good setup" flag.
    """

    def __init__(self, data_provider: DataManager):
        """
        Initializes the StrategyManager.
        """
        self.data_provider = data_provider
        self.indicators = TechnicalIndicators()

        strategy_params = getattr(config, "OPTIMIZATION_PARAMETERS", {})
        self.strategy_map = {
            "Trend": TrendFollowStrategy(params=strategy_params),
            "MeanReversion": MeanReversionStrategy(params=strategy_params)
        }
        self.good_setup_score_threshold = 0.75

    def _calculate_signal_score(self, regime: str, context: Dict) -> float:
        """
        Calculates a signal score (0-1) based on the regime and its context.
        """
        score = 0.0
        adx = context.get('adx', 0)

        if regime == "Trend":
            if adx < 25:
                score = 0.5 * (adx / 25.0)
            else:
                score = 0.5 + 0.5 * min((adx - 25) / 35.0, 1.0)

        elif regime == "MeanReversion":
            bbw_pct = context.get('bbw_percentile', 0)
            adx_score = 1.0 - min(adx / 20.0, 1.0)
            bbw_score = min(bbw_pct / 0.9, 1.0)
            score = (adx_score * 0.5) + (bbw_score * 0.5)

        return round(np.clip(score, 0.0, 1.0), 2)

    def _determine_regime(self, df_15m) -> Tuple[str, Dict]:
        """
        Determines the market regime based on 15m data.
        """
        regime_params = config.STRATEGY_MODES['regime_detection']
        ti = self.indicators

        close = df_15m['close']
        adx = ti.adx(df_15m['high'], df_15m['low'], close, period=regime_params['adx_period_15m'])
        bbw_percentile = ti.bb_width_percentile(
            close,
            bb_window=regime_params['bb_period_15m'],
            bb_std=regime_params['bb_std_15m'],
            percentile_lookback=regime_params['bb_width_lookback']
        )
        atr = ti.atr(df_15m['high'], df_15m['low'], close, period=regime_params['atr_period_15m'])
        ema_100 = ti.ema(close, window=regime_params['ema_macro_periods_15m'][0])
        ema_200 = ti.ema(close, window=regime_params['ema_macro_periods_15m'][1])

        if adx is None or bbw_percentile is None or bbw_percentile.empty or atr is None or ema_100 is None or ema_200 is None:
            return "Neutral", {}

        last_close = close.iloc[-1]
        last_adx = adx.iloc[-1]
        last_bbw_percentile = bbw_percentile.iloc[-1]
        last_atr = atr.iloc[-1]
        last_ema_100 = ema_100.iloc[-1]
        last_ema_200 = ema_200.iloc[-1]

        context = {
            "adx": last_adx, "bbw_percentile": last_bbw_percentile,
            "atr_15m": last_atr, "ema_100_15m": last_ema_100, "ema_200_15m": last_ema_200,
            "close_15m": last_close
        }

        adx_trend_min = regime_params['adx_trend_threshold'][0]
        adx_mr_max = regime_params['adx_mr_max'][1]
        bbw_mr_max = regime_params['bb_width_percentile_mr'][1]
        ema_dist_atr_mult = regime_params['ema_distance_atr_mult_switch'][0]

        if last_adx >= adx_trend_min:
            logger.info(f"Regime detected: Trend (ADX: {last_adx:.2f})")
            return "Trend", context

        is_mr_vol_contraction = last_adx < adx_mr_max and last_bbw_percentile <= bbw_mr_max
        price_dist_from_ema = min(abs(last_close - last_ema_100), abs(last_close - last_ema_200))
        is_mr_price_extension = (price_dist_from_ema / last_atr) >= ema_dist_atr_mult if last_atr > 0 else False

        if is_mr_vol_contraction or is_mr_price_extension:
            logger.info(f"Regime detected: MeanReversion (ADX: {last_adx:.2f}, BBW%: {last_bbw_percentile:.2f}, Price Extension: {is_mr_price_extension})")
            return "MeanReversion", context

        return "Neutral", {}

    def select_mode(self, regime: str, session: str, asset: str) -> str:
        """
        Selects the trading mode based on regime, session, and asset class.
        """
        # In a real implementation, this would use a policy file.
        if regime == "Trend":
            return "TREND"
        if regime == "MeanReversion":
            return "MEAN_REVERSION"
        return "NEUTRAL"

    def decide(self, symbol: str, context: Dict[str, Any]) -> Optional[Decision]:
        """
        Makes a trading decision for a given symbol and context.
        """
        regime = context.get("regime")
        strategy: BaseStrategy = self.strategy_map.get(regime)
        if not strategy:
            return None

        df_5m = self.data_provider.load_historical_data(symbol, "5m", backfill_bars=200)
        if df_5m.empty:
            logger.warning(f"No 5m data for {symbol}, cannot make a decision.")
            return None

        # The context passed to the strategy should probably include the dataframe
        # Let's add it.
        full_context = {**context, "df_5m": df_5m}

        signal_result = strategy.check_entry_conditions(df_5m, full_context)
        if not (signal_result and signal_result.get("side")):
            return None

        logger.info(f"Potential signal found for {symbol}: {signal_result}")

        signal_score = self._calculate_signal_score(regime, context)
        good_setup = signal_score >= self.good_setup_score_threshold

        decision = Decision(
            signal=signal_result["side"],
            sl=signal_result["sl"],
            tp=signal_result.get("tp"),
            meta={
                "symbol": symbol,
                "regime": regime,
                "signal_score": signal_score,
                "good_setup": good_setup,
                **context
            }
        )
        return decision
