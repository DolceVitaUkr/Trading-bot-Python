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
from modules.runtime_state import RuntimeState
from modules.rollout_manager import RolloutManager

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

        # runtime persistent state + rollout manager
        self.state = RuntimeState()
        self.rollout = RolloutManager(self.state)
        logger.info(f"Loaded runtime state (stage={self.state.get_stage()})")

        # Safe-restart normalization
        self.rollout.reconcile_on_boot()

        # Ensure paper wallet baseline in simulation
        if simulation and self.state.get_paper_wallet("Crypto_Paper") == 0.0:
            self.state.set_paper_wallet("Crypto_Paper", config.SIMULATION_START_BALANCE)

        # core components
        self.data_manager = DataManager(test_mode=simulation)
        self.exchange = ExchangeAPI()
        self.executor = TradeExecutor(simulation_mode=simulation)
        self.error_handler = ErrorHandler()
        self.pair_manager = PairManager()
        self.simulator = TradeSimulator(data_provider=self.data_manager)
        self.optimizer = ParameterOptimizer()
        self.sl_bot = SelfLearningBot(
            data_provider=self.data_manager,
            error_handler=self.error_handler,
            training=simulation
        )

        # UI
        self.ui = TradingUI()
        self._register_ui_handlers()

        # internal state
        self._running = False
        self._loop = asyncio.get_event_loop()

    def _register_ui_handlers(self):
        # Core actions
        self.ui.add_action_handler(
            "start_training",
            lambda: asyncio.run_coroutine_threadsafe(self._run_simulation(), self._loop)
        )
        self.ui.add_action_handler("stop_training", lambda: self._stop())
        self.ui.add_action_handler(
            "start_trading",
            lambda: asyncio.run_coroutine_threadsafe(self._run_live_trading(), self._loop)
        )
        self.ui.add_action_handler(
            "stop_trading",
            lambda: asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        )
        self.ui.add_action_handler(
            "optimize",
            lambda: asyncio.run_coroutine_threadsafe(self._run_optimization(), self._loop)
        )

        # Optional: expose stage change in UI (confirmations should be inside UI)
        # self.ui.add_action_handler("set_stage_1", lambda: self.rollout.set_stage(1))
        # self.ui.add_action_handler("set_stage_2", lambda: self.rollout.set_stage(2))
        # ...

    async def _run_simulation(self):
        """Run a full backtest/simulation and show results in UI."""
        # Stage guard: Crypto paper requires stage >=1 (always true unless someone set 0)
        if not self.rollout.guard_paper("crypto", ui=self.ui):
            return
        try:
            self._running = True
            self.state._event("simulation.start", "Backtest started")
            balance, points = await self.simulator.run()
            self.state._event("simulation.end", f"Balance={balance}, Points={points}")
            self.state.save()
            self.ui.update_simulation_results(final_balance=balance, total_points=points)
        except Exception as e:
            self.error_handler.handle(e)
        finally:
            self._running = False

    async def _run_live_trading(self):
        """
        Continuous live trading loop using the self-learning bot + risk checks.
        Live Crypto allowed from Stage >= 2.
        """
        if not self.rollout.guard_live("crypto", ui=self.ui):
            return

        symbol = config.DEFAULT_SYMBOL
        interval = config.LIVE_LOOP_INTERVAL or 5
        self._running = True

        # turn crypto live flag on (rollout) and record start
        if not self.rollout.enable_crypto_live():
            # guard already shouted
            self._running = False
            return

        self.state._event("trading.start", f"Symbol={symbol}")

        open_pos = self.exchange.get_position(symbol)
        if open_pos:
            logger.info(f"Resuming with open position: {open_pos}")
            self.state.upsert_open_position("crypto", symbol, open_pos)

        while self._running:
            try:
                # 1) Live data fetch
                df = self.data_manager.load_historical_data(symbol)
                latest = df.iloc[-1]
                state = {
                    "symbol": symbol,
                    "price": latest["close"],
                    "wallet_balance": self.executor.get_balance(),
                }

                # 2) Learn & act
                self.sl_bot.act_and_learn(state, datetime.now(timezone.utc))

                # 3) UI metrics + persist balance
                metrics = {
                    "balance": self.executor.get_balance(),
                    "equity": self.executor.get_balance() + self.executor.unrealized_pnl(symbol),
                    "price": state["price"]
                }
                self.ui.update_live_metrics(metrics)
                self.state.set_last_seen_balance("crypto", "spot", metrics["balance"])

            except (OrderExecutionError, RiskViolationError) as e:
                self.error_handler.handle(e)
                await self._shutdown()
                break
            except Exception as e:
                self.error_handler.handle(e)

            await asyncio.sleep(interval)

    async def _run_optimization(self):
        """Run parameter optimization and show results."""
        def default_objective(params):
            self.simulator.reset()
            self.simulator.strategy.update_params(params)
            result = asyncio.get_event_loop().run_until_complete(self.simulator.run())
            return result.return_pct

        try:
            self.state._event("optimization.start", "")
            best_params = self.optimizer.run(default_objective)
            self.state._event("optimization.end", f"Best={best_params}")
            self.state.save()
            self.ui.update_optimization_results(best_params)
        except Exception as e:
            self.error_handler.handle(e)

    def run(self):
        """Start the UI."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: asyncio.ensure_future(self._shutdown()))

        self.ui.set_title(f"Trading Bot ({self.environment.title()})")
        self.ui.run()

    async def _shutdown(self):
        """Gracefully stop all tasks and close resources."""
        if not self._running:
            return
        self._running = False
        self.state._event("bot.shutdown", "")
        # crypto domain off
        self.rollout.disable_crypto_live()
        self.state.save()
        logger.info("Shutting down trading bot...")
        await asyncio.sleep(0.1)
        await self.ui.shutdown()
        await self.exchange.close()
        logger.info("Shutdown complete.")

    def _stop(self):
        """Stop training or live trading without closing UI."""
        self._running = False
        self.state._event("bot.stop", "")
        self.state.save()


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    bot = TradingBot()
    bot.run()
