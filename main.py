# main.py

import os
import sys
import signal
import asyncio
import logging
from datetime import datetime, timezone

import config
from modules.data_manager import DataManager
from modules.exchange import ExchangeAPI
from modules.self_learning import SelfLearningBot
from modules.trade_executor import TradeExecutor
from modules.top_pairs import PairManager
from modules.error_handler import ErrorHandler, OrderExecutionError, RiskViolationError
from modules.parameter_optimization import ParameterOptimizer
from modules.trade_simulator import TradeSimulator
from modules.ui import TradingUI

# ensure repo root on path
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        # load config
        self.environment = config.ENVIRONMENT.lower()
        simulation = (self.environment == "simulation")

        # core components
        self.data_manager = DataManager(test_mode=simulation)
        self.exchange      = ExchangeAPI()
        self.executor      = TradeExecutor(simulation_mode=simulation)
        self.error_handler = ErrorHandler()
        self.pair_manager  = PairManager()
        self.simulator     = TradeSimulator(data_provider=self.data_manager)
        self.optimizer     = ParameterOptimizer()
        self.sl_bot        = SelfLearningBot(
            data_provider=self.data_manager,
            error_handler=self.error_handler,
            training=(self.environment=="simulation")
        )

        # UI
        self.ui = TradingUI()
        self._register_ui_handlers()

        # internal state
        self._running = False
        self._loop    = asyncio.get_event_loop()

    def _register_ui_handlers(self):
        # Map UI actions to our methods
        self.ui.add_action_handler("start_training", lambda: asyncio.run_coroutine_threadsafe(self._run_simulation(), self._loop))
        self.ui.add_action_handler("stop_training",  lambda: self._stop())
        self.ui.add_action_handler("start_trading", lambda: asyncio.run_coroutine_threadsafe(self._run_live_trading(), self._loop))
        self.ui.add_action_handler("stop_trading",  lambda: self._shutdown())
        self.ui.add_action_handler("optimize",      lambda: asyncio.run_coroutine_threadsafe(self._run_optimization(), self._loop))

    async def _run_simulation(self):
        """Run a full backtest/simulation and show results in UI."""
        try:
            self._running = True
            balance, points = await self.simulator.run()
            # inform UI
            self.ui.update_simulation_results(final_balance=balance, total_points=points)
        except Exception as e:
            self.error_handler.handle(e)
        finally:
            self._running = False

    async def _run_live_trading(self):
        """Continuous live trading loop using the self-learning bot + risk checks."""
        symbol = config.DEFAULT_SYMBOL  # e.g. "BTC/USDT"
        interval = config.LIVE_LOOP_INTERVAL or 5  # seconds

        self._running = True
        # ensure we don’t reopen existing positions
        open_pos = self.exchange.get_position(symbol)
        if open_pos:
            logger.info(f"Resuming with open position: {open_pos}")

        while self._running:
            try:
                # 1) Fetch latest data, compute state
                df = self.data_manager.load_historical_data(symbol)
                latest = df.iloc[-1]
                state = {
                    "symbol":        symbol,
                    "price":         latest["close"],
                    "wallet_balance": self.executor.get_balance(),  # implement in TradeExecutor
                    # add any other features you need, e.g. indicators
                }

                # 2) Let the RL agent act & learn
                self.sl_bot.act_and_learn(state, datetime.now(timezone.utc))

                # 3) Update UI metrics
                metrics = {
                    "balance": self.executor.get_balance(),
                    "equity":  self.executor.get_balance() + self.executor.unrealized_pnl(symbol),
                    "price":   state["price"]
                }
                self.ui.update_live_metrics(metrics)

            except (OrderExecutionError, RiskViolationError) as e:
                # critical – shutdown
                self.error_handler.handle(e)
                await self._shutdown()
                break

            except Exception as e:
                # log but keep going
                self.error_handler.handle(e)

            # pause before next tick
            await asyncio.sleep(interval)

    async def _run_optimization(self):
        """Run parameter optimization with a default objective and show results."""
        def default_objective(params):
            # You’ll need to define how to evaluate a param set:
            # e.g., run a short simulation and return final return_pct
            self.simulator.reset()  # if you implement a reset
            self.simulator.strategy.update_params(params)
            result = asyncio.get_event_loop().run_until_complete(self.simulator.run())
            return result.return_pct

        try:
            best_params = self.optimizer.run(default_objective)
            self.ui.update_optimization_results(best_params)
        except Exception as e:
            self.error_handler.handle(e)

    def run(self):
        """Start the UI (which will kick off asyncio tasks via handlers)."""
        # 1) Setup shutdown on SIGINT/SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: asyncio.ensure_future(self._shutdown()))

        # 2) Start UI (this blocks until window closes)
        self.ui.set_title(f"Trading Bot ({self.environment.title()})")
        self.ui.run()

    async def _shutdown(self):
        """Gracefully stop all tasks and close resources."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down trading bot...")
        # give live loop a chance to exit
        await asyncio.sleep(0.1)
        # close async components if any
        await self.ui.shutdown()
        await self.exchange.close()  # implement close() if you have websockets
        logger.info("Shutdown complete.")

    def _stop(self):
        """Stop training or live trading without closing UI."""
        self._running = False

if __name__ == "__main__":
    # configure root logger
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    bot = TradingBot()
    bot.run()
