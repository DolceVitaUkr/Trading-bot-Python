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
        # ────────────────────────────────────────────────────────────────────────
        # Environment & defaults
        # ────────────────────────────────────────────────────────────────────────
        self.environment = config.ENVIRONMENT.lower()
        self.simulation = (self.environment == "simulation")
        self.timeframe = getattr(config, "PRIMARY_TIMEFRAME", "15m")
        self.confirm_timeframe = getattr(config, "CONFIRM_TIMEFRAME", "5m")
        self.current_symbol = getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")

        # ────────────────────────────────────────────────────────────────────────
        # Runtime persistent state + rollout
        # ────────────────────────────────────────────────────────────────────────
        self.state = RuntimeState()
        self.rollout = RolloutManager(self.state)
        logger.info(f"Loaded runtime state (stage={self.state.get_stage()})")

        # Safe-restart normalization
        self.rollout.reconcile_on_boot()

        # Ensure paper wallet baseline in simulation
        if self.simulation and self.state.get_paper_wallet("Crypto_Paper") == 0.0:
            self.state.set_paper_wallet("Crypto_Paper", config.SIMULATION_START_BALANCE)

        # ────────────────────────────────────────────────────────────────────────
        # Core components
        # ────────────────────────────────────────────────────────────────────────
        self.data_manager = DataManager(test_mode=self.simulation)
        self.exchange = ExchangeAPI()  # NOTE: implements get_position(), close()
        self.executor = TradeExecutor(simulation_mode=self.simulation)
        self.error_handler = ErrorHandler()
        self.pair_manager = PairManager()
        self.simulator = TradeSimulator()  # uses simple synthetic SIM logic
        self.optimizer = ParameterOptimizer()
        self.sl_bot = SelfLearningBot(
            data_provider=self.data_manager,
            error_handler=self.error_handler,
            training=self.simulation
        )

        # ────────────────────────────────────────────────────────────────────────
        # UI
        # ────────────────────────────────────────────────────────────────────────
        # Pass a reference to this bot (current ui.py expects a bot)
        self.ui = TradingUI(self)
        self._register_ui_handlers()

        # ────────────────────────────────────────────────────────────────────────
        # Bot status (read by UI)
        # ────────────────────────────────────────────────────────────────────────
        self.is_connected = True
        self.is_training = False
        self.is_trading = False
        self.current_balance = config.SIMULATION_START_BALANCE if self.simulation else 0.0
        self.portfolio_value = self.current_balance

        # internal
        self._running = False
        self._loop = asyncio.get_event_loop()

    # ────────────────────────────────────────────────────────────────────────────
    # UI wiring
    # ────────────────────────────────────────────────────────────────────────────
    def _register_ui_handlers(self):
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

    # ────────────────────────────────────────────────────────────────────────────
    # Simulation / Backtest (sync simulator -> run in executor thread)
    # ────────────────────────────────────────────────────────────────────────────
    async def _run_simulation(self):
        """Run a simple backtest pass and push results to UI."""
        if not self.rollout.guard_paper("crypto", ui=self.ui):
            return
        try:
            self._running = True
            self.is_training = True
            self.state._event("simulation.start", f"symbol={self.current_symbol}, tf={self.timeframe}")

            # Use DataManager to get candles for selected symbol/tf
            df = self.data_manager.load_historical_data(self.current_symbol, self.timeframe)
            # Convert to kline list: [ts, open, high, low, close, volume]
            market_data = [
                [int(ts.value // 1_000_000), row["open"], row["high"], row["low"], row["close"], row["volume"]]
                for ts, row in df.iterrows()
            ]

            # Run simulator in a thread to avoid blocking loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.simulator.run, market_data)

            balance = self.simulator.wallet_balance
            points = self.simulator.points
            self.current_balance = balance
            self.portfolio_value = self.simulator.portfolio_value

            self.state._event("simulation.end", f"Balance={balance:.2f}, Points={points:.2f}")
            self.state.save()

            # UI hook (exists in your current UI)
            self.ui.update_simulation_results(balance=balance, points=points)

        except Exception as e:
            self.error_handler.handle(e)
        finally:
            self.is_training = False
            self._running = False

    # ────────────────────────────────────────────────────────────────────────────
    # Live trading loop
    # ────────────────────────────────────────────────────────────────────────────
    async def _run_live_trading(self):
        """
        Continuous live trading using the self-learning bot.
        Live Crypto allowed from Stage >= 2.
        """
        if not self.rollout.guard_live("crypto", ui=self.ui):
            return

        symbol = self.current_symbol
        interval = getattr(config, "LIVE_LOOP_INTERVAL", 5) or 5

        self._running = True
        self.is_trading = True

        if not self.rollout.enable_crypto_live():
            self._running = False
            self.is_trading = False
            return

        self.state._event("trading.start", f"Symbol={symbol}")
        self.state.save()

        # Attempt to resume known position
        try:
            open_pos = self.exchange.get_position(symbol)
            if open_pos:
                logger.info(f"Resuming with open position: {open_pos}")
                self.state.upsert_open_position("crypto", symbol, open_pos)
        except Exception as e:
            self.error_handler.handle(e)

        while self._running:
            try:
                # 1) Fetch latest data (primary timeframe)
                df = self.data_manager.load_historical_data(symbol, self.timeframe)
                latest = df.iloc[-1]
                price = float(latest["close"])

                # 2) RL agent act & learn (symbol-based API)
                self.sl_bot.act_and_learn(symbol, datetime.now(timezone.utc))

                # 3) Update metrics exposed to UI
                bal = getattr(self.executor, "get_balance", lambda: self.current_balance)()
                eq = bal
                try:
                    eq += self.executor.unrealized_pnl(symbol)
                except Exception:
                    pass

                self.current_balance = float(bal)
                self.portfolio_value = float(eq)
                self.state.set_last_seen_balance("crypto", "spot", self.current_balance)

            except (OrderExecutionError, RiskViolationError) as e:
                self.error_handler.handle(e)
                await self._shutdown()
                break
            except Exception as e:
                self.error_handler.handle(e)

            await asyncio.sleep(interval)

    # ────────────────────────────────────────────────────────────────────────────
    # Optimization (runs default objective inside optimizer)
    # ────────────────────────────────────────────────────────────────────────────
    async def _run_optimization(self):
        try:
            self.state._event("optimization.start", f"symbol={self.current_symbol}, tf={self.timeframe}")
            loop = asyncio.get_event_loop()
            # Run optimizer in thread pool
            best_params = await loop.run_in_executor(None, self.optimizer.run, self.current_symbol, self.timeframe, None)
            self.state._event("optimization.end", f"Best={best_params}")
            self.state.save()
            # UI might not have a dedicated method yet; log to UI panel
            try:
                self.ui.log(f"Optimization complete. Best params: {best_params}", level="SUCCESS")
            except Exception:
                pass
        except Exception as e:
            self.error_handler.handle(e)

    # ────────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────────────
    def run(self):
        """Start the UI (blocking)."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: asyncio.ensure_future(self._shutdown()))

        # If your UI provides set_title(), use it; else the window title is already set inside UI
        try:
            if hasattr(self.ui, "set_title"):
                self.ui.set_title(f"Trading Bot ({self.environment.title()})")
        except Exception:
            pass

        self.ui.run()

    async def _shutdown(self):
        """Gracefully stop all tasks and close resources."""
        if not self._running and not self.is_trading and not self.is_training:
            # still save a heartbeat event
            self.state._event("bot.shutdown", "idle")
            self.state.save()
            return

        self._running = False
        self.is_trading = False
        self.is_training = False
        self.state._event("bot.shutdown", "")
        self.rollout.disable_crypto_live()
        self.state.save()

        logger.info("Shutting down trading bot...")
        await asyncio.sleep(0.1)

        # UI may or may not support async shutdown; guard it
        try:
            if hasattr(self.ui, "shutdown"):
                await self.ui.shutdown()
        except Exception:
            pass

        try:
            await self.exchange.close()
        except Exception:
            pass

        logger.info("Shutdown complete.")

    def _stop(self):
        """Stop training or live trading without closing UI."""
        self._running = False
        self.is_trading = False
        self.is_training = False
        self.state._event("bot.stop", "")
        self.state.save()


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, str(config.LOG_LEVEL), logging.INFO)
              if isinstance(config.LOG_LEVEL, str) else config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    bot = TradingBot()
    bot.run()
