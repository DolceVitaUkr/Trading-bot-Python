import logging
import time
from datetime import datetime, timedelta
import config
from modules.top_pairs import TopPairs
from modules.trade_executor import TradeExecutor
from modules.data_manager import DataManager
from modules.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class SelfLearningBot:
    def __init__(self):
        self.executor = TradeExecutor(simulation_mode=config.USE_SIMULATION)
        self.data_manager = DataManager()
        self.indicators = TechnicalIndicators()
        self.top_pairs = TopPairs()
        self.last_pairs_update = datetime.utcnow() - timedelta(minutes=61)
        self.open_pos_check_interval = timedelta(minutes=1)

    def refresh_top_pairs(self):
        if datetime.utcnow() - self.last_pairs_update >= timedelta(minutes=60):
            self.top_pairs.update_top_pairs()
            self.last_pairs_update = datetime.utcnow()
            logger.info("Top pairs list refreshed.")

    def run(self):
        while True:
            self.refresh_top_pairs()
            pairs = self.top_pairs.get_pairs(limit=config.MAX_SIMULATION_PAIRS)

            for symbol in pairs:
                if symbol in self.executor.open_positions:
                    continue

                df_15m = self.data_manager.get_klines(symbol, "15m", limit=200)
                df_5m = self.data_manager.get_klines(symbol, "5m", limit=200)

                if df_15m is None or df_5m is None:
                    continue

                signal = self.indicators.generate_signal(df_15m, df_5m)
                if signal:
                    self.executor.execute_trade(
                        symbol,
                        side=signal["side"],
                        size_usd=config.MIN_TRADE_AMOUNT_USD,
                        tp=signal.get("tp"),
                        sl=signal.get("sl")
                    )

            self.executor.check_positions()
            time.sleep(config.LIVE_LOOP_INTERVAL)
