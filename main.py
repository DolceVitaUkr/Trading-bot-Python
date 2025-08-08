# main.py

import os
import sys
import signal
import asyncio
import logging
import time
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
from modules.notification_manager import NotificationManager, TradeEvent
from modules.telegram_bot import TelegramNotifier

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.environment = config.ENVIRONMENT.lower()
        self.simulation = (self.environment == "simulation")
        self.timeframe = getattr(config, "PRIMARY_TIMEFRAME", "15m")
        self.current_symbol = getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")

        self.state = RuntimeState()
        self.data_manager = DataManager(test_mode=self.simulation)
        self.exchange = ExchangeAPI()
        self.executor = TradeExecutor(simulation_mode=self.simulation)
        self.error_handler = ErrorHandler()
        self.pair_manager = PairManager()
        self.simulator = TradeSimulator()
        # simple reward/risk stubs if you have them; else pass None and bot will size by default notional
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

        # Notifier + notification policy
        self.notifier = TelegramNotifier(disable_async=not getattr(config, "ASYNC_TELEGRAM", True))
        self.notifications = NotificationManager(
            notifier=self.notifier,
            mode="paper" if self.simulation else "live",
            paper_recap_min=getattr(config, "TELEGRAM_PAPER_RECAP_MIN", 60),
            live_alert_level=getattr(config, "TELEGRAM_LIVE_ALERT_LEVEL", "normal"),
            heartbeat_min=getattr(config, "TELEGRAM_HEARTBEAT_MIN", 10),
        )

        # UI
        self.ui = TradingUI(self)
        self._register_ui_handlers()

        # UI-visible flags
        self.is_connected = True
        self.is_training = False
        self.is_trading = False
        self.current_balance = getattr(config, "SIMULATION_START_BALANCE", 1000.0) if self.simulation else 0.0
        self.portfolio_value = self.current_balance
        self.last_heartbeat = None

        self._running = False
        self._loop = asyncio.get_event_loop()

    # ----- UI hooking -----

    def _register_ui_handlers(self):
        self.ui.add_action_handler("start_training", lambda: asyncio.run_coroutine_threadsafe(self._run_simulation(), self._loop))
        self.ui.add_action_handler("stop_training", lambda: self._stop())
        self.ui.add_action_handler("start_trading", lambda: asyncio.run_coroutine_threadsafe(self._run_live_trading(), self._loop))
        self.ui.add_action_handler("stop_trading", lambda: asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop))

    def apply_notification_prefs(self, prefs: Dict[str, Any]):
        self.notifications.apply_prefs(prefs)

    # ----- simulation -----

    async def _run_simulation(self):
        try:
            self._running = True
            self.is_training = True
            df = self.data_manager.load_historical_data(self.current_symbol, self.timeframe)
            market_data = [[int(ts.value // 1_000_000), r["open"], r["high"], r["low"], r["close"], r["volume"]] for ts, r in df.iterrows()]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.simulator.run, market_data)
            balance = self.simulator.wallet_balance
            points = self.simulator.points
            self.current_balance = balance
            self.portfolio_value = self.simulator.portfolio_value
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
        interval = getattr(config, "LIVE_LOOP_INTERVAL", 5) or 5
        self._running = True
        self.is_trading = True
        self.notifications.notify_status("Live trading loop started ✅")

        while self._running:
            try:
                df = self.data_manager.load_historical_data(symbol, self.timeframe)
                if len(df) == 0:
                    await asyncio.sleep(interval)
                    continue
                latest = df.iloc[-1]
                price = float(latest["close"])

                # agent acts
                self.sl_bot.act_and_learn(symbol, datetime.now(timezone.utc))

                # metrics → UI + notifications snapshot
                bal = self.executor.get_balance()
                eq = bal
                try:
                    eq += self.executor.unrealized_pnl(symbol)
                except Exception:
                    pass
                self.current_balance = float(bal)
                self.portfolio_value = float(eq)

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
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: asyncio.ensure_future(self._shutdown()))
        try:
            if hasattr(self.ui, "set_title"):
                self.ui.set_title(f"Trading Bot ({self.environment.title()})")
        except Exception:
            pass
        self.ui.run()

    async def _shutdown(self):
        if not self._running and not self.is_trading and not self.is_training:
            return
        self._running = False
        self.is_trading = False
        self.is_training = False
        self.notifications.notify_status("Trading bot stopping…")
        await asyncio.sleep(0.1)
        try:
            if hasattr(self.ui, "shutdown"):
                await self.ui.shutdown()
        except Exception:
            pass
        try:
            await self.exchange.close()
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
