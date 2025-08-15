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
    def __init__(self, data_provider, error_handler, reward_system, risk_manager, state_size, action_size, hidden_dims, batch_size, gamma, learning_rate, exploration_max, exploration_min, exploration_decay, memory_size, tau, training, timeframe, symbol):
        self.data_provider = data_provider
        self.error_handler = error_handler
        self.reward_system = reward_system
        self.risk_manager = risk_manager
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.memory_size = memory_size
        self.tau = tau
        self.training = training
        self.timeframe = timeframe
        self.default_symbol = symbol
        self.last_pairs_update = datetime.utcnow() - timedelta(minutes=61)
        self.open_pos_check_interval = timedelta(minutes=1)
        self.top_symbols = []
        self.ui_hook = None

    def refresh_top_pairs(self):
        if datetime.utcnow() - self.last_pairs_update >= timedelta(minutes=60):
            self.top_pairs.update_top_pairs()
            self.last_pairs_update = datetime.utcnow()
            logger.info("Top pairs list refreshed.")

    def act_and_learn(self, symbol, timestamp):
        pass

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
