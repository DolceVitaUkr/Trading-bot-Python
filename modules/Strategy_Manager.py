import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import config
from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from modules.reward_system import RewardSystem
from modules.risk_management import RiskManager
from modules.technical_indicators import TechnicalIndicators
from modules.top_pairs import TopPairs
from modules.trade_executor import TradeExecutor
from modules.Strategies import TrendFollowStrategy, MeanReversionStrategy, BaseStrategy

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Orchestrates the trading process by selecting and executing strategies
    based on market regime detection.
    """

    def __init__(self, data_provider: DataManager,
                 error_handler: ErrorHandler,
                 reward_system: RewardSystem,
                 risk_manager: RiskManager, **kwargs):
        """
        Initializes the StrategyManager.
        """
        self.data_provider = data_provider
        self.error_handler = error_handler
        self.reward_system = reward_system
        self.risk_manager = risk_manager

        self.default_symbol: str = kwargs.get("symbol", "BTC/USDT")
        self.last_pairs_update = datetime.utcnow() - timedelta(minutes=61)
        self.open_pos_check_interval = timedelta(minutes=1)

        self.executor = TradeExecutor(
            simulation_mode=config.USE_SIMULATION
        )
        self.indicators = TechnicalIndicators()
        self.top_pairs = TopPairs()
        self.top_symbols: List[str] = [self.default_symbol]

        self.strategy_map = {
            "Trend": TrendFollowStrategy,
            "MeanReversion": MeanReversionStrategy
        }

    def _determine_regime(self, df_15m) -> Tuple[str, Dict]:
        """
        Determines the market regime based on 15m data.

        :param df_15m: 15-minute DataFrame.
        :return: A tuple of (regime_name, context_dict).
        """
        regime_params = config.STRATEGY_MODES['regime_detection']
        ti = self.indicators

        # Calculate all necessary 15m indicators
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

        # Ensure we have values to work with
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

        # --- Regime Logic ---
        adx_trend_min = regime_params['adx_trend_threshold'][0]
        adx_mr_max = regime_params['adx_mr_max'][1]
        bbw_mr_max = regime_params['bb_width_percentile_mr'][1]
        ema_dist_atr_mult = regime_params['ema_distance_atr_mult_switch'][0]

        # 1. Trend Regime
        if last_adx >= adx_trend_min:
            logger.info(f"Regime detected: Trend (ADX: {last_adx:.2f})")
            return "Trend", context

        # 2. Mean-Reversion Regime
        is_mr_vol_contraction = last_adx < adx_mr_max and last_bbw_percentile <= bbw_mr_max

        price_dist_from_ema = min(abs(last_close - last_ema_100), abs(last_close - last_ema_200))
        is_mr_price_extension = (price_dist_from_ema / last_atr) >= ema_dist_atr_mult if last_atr > 0 else False

        if is_mr_vol_contraction or is_mr_price_extension:
            logger.info(f"Regime detected: MeanReversion (ADX: {last_adx:.2f}, BBW%: {last_bbw_percentile:.2f}, Price Extension: {is_mr_price_extension})")
            return "MeanReversion", context

        return "Neutral", {}

    def _check_spread(self, symbol: str) -> bool:
        """Checks if the bid-ask spread is within an acceptable range."""
        spread_guard_bp = config.OPTIMIZATION_PARAMETERS['spread_guard_bp'][0]
        order_book = self.data_provider.exchange.get_order_book(symbol)
        if not order_book or not order_book['bids'] or not order_book['asks']:
            logger.warning(f"Could not retrieve order book for {symbol} to check spread.")
            return False # Fail safe

        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        spread = (best_ask - best_bid) / best_bid
        spread_bp = spread * 10000

        if spread_bp > spread_guard_bp:
            logger.warning(f"Spread for {symbol} is too wide: {spread_bp:.2f} bp > {spread_guard_bp} bp. Skipping trade.")
            return False
        return True

    def _check_volume(self, df_5m: 'pd.DataFrame') -> bool:
        """Checks if the recent volume is above its moving average."""
        if 'volume' not in df_5m.columns or len(df_5m) < 20:
            return False # Not enough data to check

        volume_sma_20 = df_5m['volume'].rolling(window=20).mean().iloc[-1]
        last_volume = df_5m['volume'].iloc[-1]

        if last_volume < volume_sma_20:
            logger.warning(f"Volume {last_volume} is below 20-period SMA {volume_sma_20:.2f}. Skipping trade.")
            return False
        return True

    def _select_and_run_strategy(self, symbol: str, regime: str, context: Dict):
        """
        Selects and runs the appropriate strategy for the given regime,
        including all risk checks.
        """
        # 1. Select strategy based on regime
        strategy_class = self.strategy_map.get(regime)
        if not strategy_class:
            return

        strategy: BaseStrategy = strategy_class(params=config.OPTIMIZATION_PARAMETERS)

        # 2. Get data for entry checks
        df_5m = self.data_provider.load_historical_data(
            symbol, "5m", backfill_bars=200)

        if df_5m.empty:
            logger.warning(f"No 5m data for {symbol}, cannot run strategy.")
            return

        # 3. Check for entry signal
        signal = strategy.check_entry_conditions(df_5m, context)
        if not (signal and signal.get("side")):
            return

        logger.info(f"Potential signal found for {symbol}: {signal}")

        # 4. Perform all risk and pre-trade checks
        if not self.risk_manager.is_trade_allowed():
            logger.warning(f"Trade for {symbol} disallowed by master risk controls (cooldown/daily loss).")
            return

        if not self._check_spread(symbol):
            return

        if not self._check_volume(df_5m):
            return

        # 5. Calculate position size based on risk
        entry_price = df_5m['close'].iloc[-1]
        sl_price = signal.get('sl')

        quantity, dollar_risk = self.risk_manager.calculate_position_size(symbol, entry_price, sl_price)

        if quantity is None or quantity <= 0:
            logger.warning(f"Position size calculation for {symbol} resulted in zero or invalid quantity.")
            return

        # 6. Execute order
        logger.info(f"Executing order for {symbol}: side={signal['side']}, quantity={quantity:.6f}, dollar_risk={dollar_risk:.2f}")

        # This is a temporary solution. Ideally, the executor would confirm the trade
        # and then the risk manager would be updated.
        self.risk_manager.register_position(symbol, quantity, entry_price, dollar_risk)

        try:
            self.executor.execute_order(
                symbol=symbol,
                side=signal["side"],
                quantity=quantity,
                price=entry_price,
                order_type="market",
                sl=sl_price,
                tp=signal.get('tp')
            )
        except Exception as e:
            # If order fails, unregister the position to correct the risk manager's state
            logger.error(f"Order execution for {symbol} failed. Unregistering position. Error: {e}")
            self.risk_manager.unregister_position(symbol)
            self.error_handler.handle_error(e, f"Order execution for {symbol}")

    def _evaluate_symbol(self, symbol: str):
        """
        Evaluates a single symbol for trading opportunities.
        """
        if symbol in self.executor.exchange._sim_positions:
            logger.debug(f"Position already open for {symbol}. Skipping.")
            return

        df_15m = self.data_provider.load_historical_data(
            symbol, "15m", backfill_bars=300) # Need enough for lookbacks

        if df_15m.empty or len(df_15m) < 200: # Basic data check
            logger.warning(f"Not enough 15m data for {symbol} to determine regime.")
            return

        regime, context = self._determine_regime(df_15m)

        if regime != "Neutral":
            self._select_and_run_strategy(symbol, regime, context)

    def refresh_top_pairs(self):
        """
        Refreshes the list of top pairs to trade.
        """
        if datetime.utcnow() - self.last_pairs_update >= timedelta(minutes=60):
            self.top_symbols = self.top_pairs.get_top_pairs(force=True)
            self.last_pairs_update = datetime.utcnow()
            logger.info(f"Top pairs list refreshed: {self.top_symbols}")

    def run(self):
        """
        Main bot loop, called from a thread in main.py.
        """
        logger.info("StrategyManager started.")
        while True:
            try:
                self.refresh_top_pairs()
                for symbol in self.top_symbols:
                    self._evaluate_symbol(symbol)

                time.sleep(max(5, float(config.LIVE_LOOP_INTERVAL)))
            except Exception as e:
                logger.exception(f"Error in StrategyManager run loop: {e}")
                self.error_handler.handle_error(e, "StrategyManager Run Loop")
                time.sleep(15)
