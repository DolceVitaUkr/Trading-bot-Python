import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import config
from modules.data_manager import DataManager
from modules.technical_indicators import TechnicalIndicators
from modules.Strategies import TrendFollowStrategy, MeanReversionStrategy, BaseStrategy
# Import new modules for integration
from modules.Validation_Manager import ValidationManager
from modules.News_Agent import NewsAgent
from modules.Portfolio_Manager import PortfolioManager


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
    Integrates validation, news, liquidity, and correlation filters.
    """

    def __init__(self,
                 data_provider: DataManager,
                 validation_manager: ValidationManager,
                 news_agent: NewsAgent,
                 portfolio_manager: PortfolioManager):
        """
        Initializes the StrategyManager.
        """
        self.data_provider = data_provider
        self.validation_manager = validation_manager
        self.news_agent = news_agent
        self.portfolio_manager = portfolio_manager
        self.indicators = TechnicalIndicators()

        strategy_params = getattr(config, "OPTIMIZATION_PARAMETERS", {})
        self.strategy_map = {
            "TrendFollowStrategy": TrendFollowStrategy(params=strategy_params),
            "MeanReversionStrategy": MeanReversionStrategy(params=strategy_params)
        }
        self.good_setup_score_threshold = 0.75

        # Simple correlation map for the filter
        self.correlation_map = {
            'BTCUSDT': ['ETHUSDT', 'SOLUSDT'],
            'ETHUSDT': ['BTCUSDT', 'MATICUSDT'],
            # Add other correlated pairs as needed
        }

    def _is_overexposed_by_correlation(self, symbol: str, open_symbols: list) -> bool:
        """
        Checks if opening a new position would lead to over-exposure in correlated assets.
        Placeholder implementation.
        """
        correlated_assets = self.correlation_map.get(symbol, [])
        if not correlated_assets:
            return False

        count = 0
        for s in open_symbols:
            if s in correlated_assets:
                count += 1

        # Max 3 correlated positions (including the new one)
        if count >= 2:
            logger.warning(f"Correlation check failed for {symbol}. Found {count} correlated open positions.")
            return True
        return False

    def _calculate_signal_score(self, regime: str, context: Dict, news_bias: str = 'neutral') -> float:
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

        # Boost or penalize score based on news
        if news_bias == 'long':
            score *= 1.10 # 10% boost
        elif news_bias == 'short':
            score *= 0.90 # 10% penalty

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
        Makes a filtered trading decision for a given symbol and context.
        """
        asset_class = context.get("asset_class")
        if not asset_class:
            logger.error("Asset class must be provided in the context.")
            return None

        # 1. Liquidity Filter
        # TODO: Implement get_daily_volume in a real data_provider
        if hasattr(self.data_provider, 'get_daily_volume'):
            daily_volume = self.data_provider.get_daily_volume(symbol)
            if daily_volume < 5_000_000:
                reason = f"Low liquidity (Volume: ${daily_volume:,.0f})"
                logger.warning(f"Trade rejected for {symbol}: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
                return None

        # 2. Correlation Filter
        # TODO: Implement get_open_positions in portfolio_manager
        # open_positions = self.portfolio_manager.get_open_positions()
        open_positions = [] # Placeholder
        if self._is_overexposed_by_correlation(symbol, open_positions):
            reason = "Over-exposed to correlated assets"
            logger.warning(f"Trade rejected for {symbol}: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
            return None

        # 3. News Agent Filter (High-Impact Events)
        is_event, event_name = self.news_agent.is_high_impact_event_imminent()
        if is_event:
            reason = f"High-impact event '{event_name}' is imminent"
            logger.warning(f"Trade rejected for {symbol}: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
            return None

        # 4. Determine Regime and select Strategy
        regime = context.get("regime")
        strategy_name = "TrendFollowStrategy" if regime == "Trend" else "MeanReversionStrategy"
        strategy: BaseStrategy = self.strategy_map.get(strategy_name)
        if not strategy:
            logger.warning(f"No strategy found for regime '{regime}'.")
            return None

        # 5. Validation Manager Filter
        if not self.validation_manager.is_strategy_approved(strategy_name, asset_class):
            reason = f"Strategy '{strategy_name}' not approved for asset class '{asset_class}'"
            logger.warning(f"Trade rejected for {symbol}: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
            return None

        # 6. Get Signal from Strategy
        df_5m = self.data_provider.load_historical_data(symbol, "5m", backfill_bars=200)
        if df_5m.empty:
            logger.warning(f"No 5m data for {symbol}, cannot make a decision.")
            return None

        full_context = {**context, "df_5m": df_5m}
        signal_result = strategy.check_entry_conditions(df_5m, full_context)

        if not (signal_result and signal_result.get("side")):
            return None

        # 7. News Sentiment Filter
        news_bias = self.news_agent.get_news_bias(symbol)
        if (news_bias == 'short' and signal_result["side"] == 'buy') or \
           (news_bias == 'long' and signal_result["side"] == 'sell'):
            reason = f"Conflicting news sentiment ('{news_bias}') for {signal_result['side']} signal"
            logger.warning(f"Trade for {symbol} rejected: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
            return None

        # 8. Calculate final signal score and create decision
        signal_score = self._calculate_signal_score(regime, context, news_bias)
        good_setup = signal_score >= self.good_setup_score_threshold

        logger.info(f"Potential signal found for {symbol}: {signal_result} with score {signal_score}")

        decision = Decision(
            signal=signal_result["side"],
            sl=signal_result["sl"],
            tp=signal_result.get("tp"),
            meta={
                "symbol": symbol,
                "regime": regime,
                "signal_score": signal_score,
                "good_setup": good_setup,
                "news_bias": news_bias,
                **context
            }
        )
        return decision
