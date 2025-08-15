import itertools
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import config
from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from modules.reward_system import RewardSystem
from modules.risk_management import RiskManager
from modules.technical_indicators import TechnicalIndicators
from modules.top_pairs import TopPairs
from modules.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)

STRATEGY_MEMORY_FILE = "strategy_memory.json"


class SelfLearningBot:
    """
    A self-learning trading bot that uses a genetic algorithm to optimize
    its trading strategy.
    """

    def __init__(self, data_provider: DataManager,
                 error_handler: ErrorHandler,
                 reward_system: RewardSystem,
                 risk_manager: RiskManager, **kwargs):
        """
        Initializes the SelfLearningBot.
        """
        self.data_provider = data_provider
        self.error_handler = error_handler
        self.reward_system = reward_system
        self.risk_manager = risk_manager

        self.state_size: int = kwargs.get("state_size", 5)
        self.action_size: int = kwargs.get("action_size", 3)
        self.training: bool = kwargs.get("training", True)
        self.timeframe: str = kwargs.get("timeframe", "5m")
        self.default_symbol: str = kwargs.get("symbol", "BTC/USDT")

        self.last_pairs_update = datetime.utcnow() - timedelta(minutes=61)
        self.open_pos_check_interval = timedelta(minutes=1)

        self.executor = TradeExecutor(
            simulation_mode=True)  # Always simulate for learning
        self.indicators = TechnicalIndicators()
        self.top_pairs = TopPairs()

        self.top_symbols: List[str] = [self.default_symbol]
        self.portfolio_value: float = float(config.SIMULATION_START_BALANCE)
        self.ui_hook: Optional[Any] = None

        # -- Self-Learning Components --
        # A list of strategy variations to test during training.
        self.strategy_variations = self._define_strategy_variations()
        # The best strategy parameters loaded from memory.
        self.best_strategy_params = self.load_best_strategy()

    def _define_strategy_variations(self) -> List[Dict]:
        """
        Dynamically defines a set of strategy parameters to test during training.
        """
        param_grid = config.OPTIMIZATION_PARAMETERS
        keys = list(param_grid.keys())

        variations = []
        # Create all possible combinations of parameters
        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))

            # --- Add constraints to filter invalid combinations ---
            if (params.get("ema_short_period_1", 0) >=
                    params.get("ema_short_period_2", 1)):
                continue
            if (params.get("ema_short_period_2", 0) >=
                    params.get("ema_long_period", 1)):
                continue

            # Generate a unique ID for the strategy variation
            param_str = "_".join(
                f"{k.replace('_period', '')}{v}"
                for k, v in sorted(params.items()))
            strategy_id = f"strat_{param_str}"

            variations.append({
                "id": strategy_id,
                "params": params
            })

        logger.info(
            f"Generated {len(variations)} strategy variations for testing.")
        if not variations:
            logger.warning(
                "No strategy variations were generated. "
                "Check config.OPTIMIZATION_PARAMETERS.")
            # Fallback to a default strategy to prevent errors
            default_params = {
                "ema_long_period": 50, "ema_short_period_1": 8,
                "ema_short_period_2": 21, "rsi_period": 14, "rsi_level": 50,
                "atr_period": 14, "tp_atr_multiplier": 2.0,
                "sl_atr_multiplier": 1.5
            }
            variations.append(
                {"id": "default_fallback", "params": default_params})

        return variations

    def save_best_strategy(self, params: Dict):
        """Saves the best performing strategy parameters to a file."""
        try:
            with open(STRATEGY_MEMORY_FILE, 'w') as f:
                json.dump(params, f, indent=4)
            logger.info(f"Saved best strategy to {STRATEGY_MEMORY_FILE}")
            self.best_strategy_params = params
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")

    def load_best_strategy(self) -> Optional[Dict]:
        """Loads the best strategy parameters from a file."""
        try:
            with open(STRATEGY_MEMORY_FILE, 'r') as f:
                params = json.load(f)
            logger.info(f"Loaded best strategy from {STRATEGY_MEMORY_FILE}")
            return params
        except FileNotFoundError:
            logger.warning(
                f"{STRATEGY_MEMORY_FILE} not found. Using default strategy.")
            return None
        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
            return None

    def _log_strategy_performance(self, results: Dict):
        """Appends the results of a training run to the performance log."""
        log_file = "strategy_performance_log.json"
        try:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                log_data = []

            timestamp = datetime.now(timezone.utc).isoformat()
            for strategy_id, result in results.items():
                log_entry = {
                    "timestamp": timestamp,
                    "strategy_id": strategy_id,
                    "params": result["params"],
                    "pnl": result["pnl"],
                    "win_rate": result["win_rate"],
                    "score": result["score"]
                }
                log_data.append(log_entry)

            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=4)

            logger.info(
                f"Logged {len(results)} strategy performance records to "
                f"{log_file}")

        except Exception as e:
            logger.error(f"Failed to log strategy performance: {e}")

    def train_and_select_strategy(self):
        """
        This is the core of the self-learning mechanism.
        It simulates each strategy variation for a fixed number of trades.
        It then evaluates the PnL of each variation and saves the best one.
        """
        logger.info(
            f"Starting training process with {len(self.strategy_variations)} "
            f"variations.")
        if self.ui_hook:
            self.ui_hook.log(
                f"Training started: {len(self.strategy_variations)} "
                f"variations to test.",
                level="SUCCESS")

        results = {}
        for variation in self.strategy_variations:
            strategy_id = variation["id"]
            params = variation["params"]
            self.executor.reset_simulation()  # Reset for each new strategy test
            trade_count = 0

            logger.info(f"Testing strategy: {strategy_id}")
            if self.ui_hook:
                self.ui_hook.log(f"Testing strategy: {strategy_id}")

            while trade_count < 100:
                symbol = self.top_symbols[0]
                self.run_strategy(symbol, params)
                if self.executor.get_total_trades() > trade_count:
                    trade_count = self.executor.get_total_trades()
                time.sleep(0.1)  # Speed up simulation

            pnl = self.executor.get_pnl()
            win_rate = self.executor.get_win_rate()
            score = pnl * win_rate

            results[strategy_id] = {
                "pnl": pnl, "win_rate": win_rate,
                "score": score, "params": params}
            log_msg = (
                f"Strategy {strategy_id} finished with PnL: {pnl:.2f}, "
                f"Win Rate: {win_rate:.2%}, Score: {score:.2f}")
            logger.info(log_msg)
            if self.ui_hook:
                self.ui_hook.log(
                    f"  - {strategy_id} PnL: {pnl:.2f}, "
                    f"Win Rate: {win_rate:.2%}")

        self._log_strategy_performance(results)

        if not results:
            logger.error("No results from strategy training. Aborting.")
            self.training = False
            return

        best_strategy_id = max(results, key=lambda k: results[k]["score"])
        best_result = results[best_strategy_id]
        best_params = best_result["params"]

        log_msg = (
            f"Training complete. Best strategy is {best_strategy_id} "
            f"with Score: {best_result['score']:.2f} "
            f"(PnL: {best_result['pnl']:.2f}, "
            f"Win Rate: {best_result['win_rate']:.2%})")
        logger.info(log_msg)
        if self.ui_hook:
            self.ui_hook.log(
                f"Training complete. Best strategy: {best_strategy_id}",
                level="SUCCESS")

        self.save_best_strategy(best_params)
        self.training = False  # Switch to trading mode after training

    def run_strategy(self, symbol: str, params: Optional[Dict] = None):
        """
        Runs the trading strategy for a given symbol.
        """
        if symbol in self.executor.exchange._sim_positions:
            return

        df_15m = self.data_provider.load_historical_data(
            symbol, "15m", backfill_bars=200)
        df_5m = self.data_provider.load_historical_data(
            symbol, "5m", backfill_bars=200)

        if df_15m.empty or df_5m.empty:
            return

        strategy_params = params if params is not None else self.best_strategy_params
        # TODO: This needs to be implemented
        # signal = self.indicators.generate_signal(
        #     df_15m, df_5m, strategy_params)
        signal = None

        if signal and signal.get("side"):
            self.executor.execute_order(
                symbol,
                side=signal["side"],
                quantity=config.MIN_TRADE_AMOUNT_USD / df_5m['close'].iloc[-1],
                price=df_5m['close'].iloc[-1],
                order_type="market"
            )

    def refresh_top_pairs(self):
        """
        Refreshes the list of top pairs to trade.
        """
        if datetime.utcnow() - self.last_pairs_update >= timedelta(minutes=60):
            self.top_symbols = self.top_pairs.get_top_pairs(force=True)
            self.last_pairs_update = datetime.utcnow()
            logger.info("Top pairs list refreshed.")

    def act_and_learn(self, symbol, timestamp):
        """
        This is the main entry point from the agent loop in main.py.
        """
        self.run_strategy(symbol)

    def run(self):
        """
        Main bot loop, called from a thread in main.py.
        """
        if self.training:
            self.train_and_select_strategy()

        logger.info(
            "Bot switching to trading mode with the best learned strategy.")

        while True:
            try:
                self.refresh_top_pairs()
                for symbol in self.top_symbols:
                    self.run_strategy(symbol)
                time.sleep(max(5, float(config.LIVE_LOOP_INTERVAL)))
            except Exception as e:
                logger.exception(f"Error in bot run loop: {e}")
                time.sleep(15)
