# main.py

import os
import sys
import signal
import asyncio
import logging
import time
import threading
from typing import Dict, Any
from datetime import datetime, timezone

import config
from modules.data_manager import DataManager
from modules.exchange import ExchangeAPI
from modules.self_learning import SelfLearningBot
from modules.trade_executor import TradeExecutor
from modules.top_pairs import PairManager
from modules.error_handler import ErrorHandler, OrderExecutionError, RiskViolationError
from modules.trade_simulator import TradeSimulator
from modules.ui import TradingUI
from state.runtime_state import RuntimeState
from modules.notification_manager import NotificationManager
from modules.telegram_bot import TelegramNotifier

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        # --- environment ---
        self.environment = config.ENVIRONMENT.lower()
        self.simulation = (self.environment == "simulation")
        self.timeframe = getattr(config, "PRIMARY_TIMEFRAME", "15m")
        self.current_symbol = getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")

        # --- state & core modules ---
        self.state = RuntimeState()
        self.data_manager = DataManager(test_mode=self.simulation)
        self.exchange = ExchangeAPI()
        self.executor = TradeExecutor(simulation_mode=self.simulation)
        self.error_handler = ErrorHandler()
        self.pair_manager = PairManager()
        self.simulator = TradeSimulator()

        # reward/risk are optional
        try:
            from modules.reward_system import RewardSystem
            from modules.risk_management import RiskManager
            reward = RewardSystem()
            risk = RiskManager(account_balance=getattr(config, "SIMULATION_START_BALANCE", 1000.0))
        except Exception:
            reward, risk = None, None

        self.sl_bot = SelfLearningBot(
            data_provider=self.data_manager,
            error_handler=self.error_handler,
            reward_system=reward,
            risk_manager=risk,
            state_size=5,
            training=self.simulation,
            timeframe=self.timeframe,
            symbol=self.current_symbol,
        )

        # --- notifications ---
        self.notifier = TelegramNotifier(disable_async=not getattr(config, "ASYNC_TELEGRAM", True))
        self.notifications = NotificationManager(
            notifier=self.notifier,
            paper_recap_minutes=getattr(config, "TELEGRAM_PAPER_RECAP_MIN", 60),
            live_alert_level=getattr(config, "TELEGRAM_LIVE_ALERT_LEVEL", "normal"),
            heartbeat_minutes=getattr(config, "TELEGRAM_HEARTBEAT_MIN", 10),
        )

        # --- UI ---
        self.ui = TradingUI(self)

        # --- UI-visible flags ---
        self.is_connected = True
        self.is_training = False
        self.is_trading = False
        self.current_balance = getattr(config, "SIMULATION_START_BALANCE", 1000.0) if self.simulation else 0.0
        self.portfolio_value = self.current_balance
        self.last_heartbeat = None

        # --- lifecycle flags ---
        self._running = False

        # --- background asyncio loop for bot tasks ---
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="bot-asyncio", daemon=True)
        self._loop_thread.start()

        # after loop exists, register UI actions that schedule work on it
        self._register_ui_handlers()

    # ----- UI hooking -----
    def _register_ui_handlers(self):
        self.ui.add_action_handler(
            "start_training",
            lambda: asyncio.run_coroutine_threadsafe(self._run_simulation(), self._loop)
        )
        self.ui.add_action_handler(
            "stop_training",
            self._stop
        )
        self.ui.add_action_handler(
            "start_trading",
            lambda: asyncio.run_coroutine_threadsafe(self._run_live_trading(), self._loop)
        )
        self.ui.add_action_handler(
            "stop_trading",
            lambda: asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        )

    def apply_notification_prefs(self, prefs: Dict[str, Any]):
        self.notifications.apply_prefs(prefs)

    # ----- simulation -----
    async def _run_simulation(self):
        try:
            self._running = True
            self.is_training = True
            df = self.data_manager.load_historical_data(self.current_symbol, self.timeframe)
            if df is None or len(df) == 0:
                self.notifications.notify_status("No historical data available for simulation.", "info")
                return

            # Convert to [ms, o, h, l, c, v]
            market_data = []
            for ts, r in df.iterrows():
                # pandas Timestamp .value is ns since epoch
                ms = int(int(getattr(ts, "value", 0)) // 1_000_000)
                market_data.append([ms, r["open"], r["high"], r["low"], r["close"], r["volume"]])

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.simulator.run, market_data)

            balance = float(getattr(self.simulator, "wallet_balance", self.current_balance))
            points = float(getattr(self.simulator, "points", 0.0))
            self.current_balance = balance
            self.portfolio_value = float(getattr(self.simulator, "portfolio_value", balance))
            self.ui.update_simulation_results(final_balance=balance, total_points=points)
            self.notifications.notify_status(f"Simulation complete. Balance={balance:.2f}, Points={points:.2f}")
        except Exception as e:
            self.error_handler.handle(e)
        finally:
            self.is_training = False
            self._running = False

    # ----- live loop -----
    async def _run_live_trading(self):
        symbol = self.current_symbol
        interval = float(getattr(config, "LIVE_LOOP_INTERVAL", 5) or 5)
        self._running = True
        self.is_trading = True
        self.notifications.notify_status("Live trading loop started ✅")

        while self._running:
            try:
                df = self.data_manager.load_historical_data(symbol, self.timeframe)
                if df is None or len(df) == 0:
                    await asyncio.sleep(interval)
                    continue

                latest = df.iloc[-1]
                price = float(latest["close"])

                # agent acts
                self.sl_bot.act_and_learn(symbol, datetime.now(timezone.utc))

                # metrics → UI + notifications snapshot
                bal = float(self.executor.get_balance())
                eq = bal
                try:
                    eq += float(self.executor.unrealized_pnl(symbol))
                except Exception:
                    pass
                self.current_balance = bal
                self.portfolio_value = eq

                self.notifications.update_metrics_snapshot(
                    price=price,
                    symbol=symbol,
                    equity=self.portfolio_value,
                    balance=self.current_balance
                )
                self.notifications.tick()
                self.last_heartbeat = time.time()

            except (OrderExecutionError, RiskViolationError) as e:
                self.error_handler.handle(e)
                await self._shutdown()
                break
            except Exception as e:
                self.error_handler.handle(e)

            await asyncio.sleep(interval)

    # ----- lifecycle -----
    def run(self):
        # OS signals → schedule shutdown on our background loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self._loop.call_soon_threadsafe(asyncio.create_task, self._shutdown()))

        try:
            if hasattr(self.ui, "set_title"):
                self.ui.set_title(f"Trading Bot ({self.environment.title()})")
        except Exception:
            pass

        self.ui.run()

    async def _shutdown(self):
        if not (self._running or self.is_trading or self.is_training):
            return
        self._running = False
        self.is_trading = False
        self.is_training = False
        self.notifications.notify_status("Trading bot stopping…")

        # let UI settle
        await asyncio.sleep(0.1)
        try:
            if hasattr(self.ui, "shutdown"):
                maybe_coro = self.ui.shutdown()
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
        except Exception:
            pass

        try:
            maybe_close = getattr(self.exchange, "close", None)
            if callable(maybe_close):
                res = maybe_close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass

        self.notifications.notify_status("Trading bot stopped ❌")
        self.notifier.graceful_shutdown()

    def _stop(self):
        self._running = False
        self.is_trading = False
        self.is_training = False


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, str(config.LOG_LEVEL), logging.INFO)
              if isinstance(config.LOG_LEVEL, str) else config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    bot = TradingBot()
    bot.run()

