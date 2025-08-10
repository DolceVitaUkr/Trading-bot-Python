# main.py

import asyncio
import signal
import sys
import argparse
import logging
from datetime import datetime, timezone

import config
from utils.utilities import configure_logging
from modules.error_handler import ErrorHandler
from modules.data_manager import DataManager
from modules.reward_system import RewardSystem
from modules.risk_management import RiskManager
from modules.self_learning import SelfLearningBot
from modules.scheduler import LightweightScheduler
from modules.ui import TradingUI


class AppState:
    """Small container to expose flags the UI expects."""
    is_connected: bool = False
    is_training: bool = False
    is_trading: bool = False
    last_heartbeat: float | None = None
    current_balance: float | None = None
    portfolio_value: float | None = None
    current_symbol: str | None = None
    timeframe: str | None = None


async def bot_tick(bot: SelfLearningBot, app_state: AppState):
    """
    One iteration: call bot.step(), update heartbeat + a few public metrics the UI reads.
    """
    bot.step()  # consumes live (incremental) data, executes in simulation, handles learning
    now = datetime.now(timezone.utc)
    app_state.last_heartbeat = now.timestamp()

    # UI metrics
    try:
        bal = bot.executor.get_balance()
    except Exception:
        bal = None

    app_state.current_balance = bal
    # Portfolio value: best-effort — use risk manager current equity if present, else wallet
    if bot.risk_manager:
        try:
            bot.risk_manager.update_equity(bal if bal is not None else 0.0)
            app_state.portfolio_value = bot.risk_manager.current_equity
        except Exception:
            app_state.portfolio_value = bal
    else:
        app_state.portfolio_value = bal

    # Expose symbol/timeframe hints for the UI header
    try:
        if bot.current_universe:
            app_state.current_symbol = bot.current_universe[0]
    except Exception:
        pass
    app_state.timeframe = bot.tf_entry


def build_bot() -> tuple[SelfLearningBot, AppState, LightweightScheduler, TradingUI]:
    # ----- logging ----- #
    configure_logging(
        level=getattr(config, "LOG_LEVEL", "INFO"),
        log_file=getattr(config, "LOG_FILE", "bot.log"),
    )
    logger = logging.getLogger("main")

    # ----- core objects ----- #
    data_mgr = DataManager(
        base_path=getattr(config, "HISTORICAL_DATA_PATH", "historical_data"),
        # The DataManager should internally use Bybit REST+WS and perform incremental appends.
        exchange="bybit",
        use_websocket=True,
    )
    err = ErrorHandler()
    rs = RewardSystem()

    rm = RiskManager(
        account_balance=getattr(config, "SIMULATION_START_BALANCE", 1000.0),
        min_rr=1.5,
        atr_mult_sl=1.5,
        atr_mult_tp=3.0,
    )

    bot = SelfLearningBot(
        data_provider=data_mgr,
        error_handler=err,
        reward_system=rs,
        risk_manager=rm,
        training=True,  # can be toggled by UI
        timeframe_entry="5m",
        timeframe_setup="15m",
        base_symbol=getattr(config, "DEFAULT_SYMBOL", "BTC/USDT"),
    )

    # app state the UI reads
    app_state = AppState()
    app_state.is_connected = True
    app_state.is_training = False
    app_state.is_trading = False
    app_state.timeframe = "5m"

    # Attach a couple of attrs the UI polls directly from `bot`
    # (UI.refresh reads attributes off `bot`, so we mirror AppState on `bot` too)
    bot.is_connected = True
    bot.is_training = False
    bot.is_trading = False
    bot.last_heartbeat = None
    bot.current_balance = None
    bot.portfolio_value = None
    bot.current_symbol = getattr(config, "DEFAULT_SYMBOL", "BTC/USDT")
    bot.timeframe = "5m"
    bot.app_state = app_state  # handy link

    # Lightweight scheduler
    scheduler = LightweightScheduler()

    # UI
    ui = TradingUI(bot)

    # Wire UI handlers
    def start_training():
        if bot.is_training:
            logger.info("Training already running.")
            return
        bot.is_training = True
        app_state.is_training = True
        # Schedule bot tick
        scheduler.add_job(
            name="bot_tick",
            coro=lambda: bot_tick(bot, app_state),
            interval_seconds=float(getattr(config, "LIVE_LOOP_INTERVAL", 5.0)),
        )
        # Periodic heartbeat to reflect in the bot object for the UI
        def _mirror_state():
            bot.last_heartbeat = app_state.last_heartbeat
            bot.current_balance = app_state.current_balance
            bot.portfolio_value = app_state.portfolio_value
            bot.current_symbol = app_state.current_symbol or bot.current_symbol
            bot.timeframe = app_state.timeframe or bot.timeframe

        scheduler.add_job(
            name="mirror_ui_state",
            coro=lambda: asyncio.to_thread(_mirror_state),
            interval_seconds=2.0,
        )
        logger.info("Training started.")

    def stop_training():
        if not bot.is_training:
            logger.info("Training not running.")
            return
        bot.is_training = False
        app_state.is_training = False
        scheduler.cancel_job("bot_tick")
        scheduler.cancel_job("mirror_ui_state")
        logger.info("Training stopped.")

    # Keep these for UI parity; they operate in sim as well
    def start_trading():
        if bot.is_trading:
            logger.info("Trading already running.")
            return
        bot.is_trading = True
        app_state.is_trading = True
        # In this design, “trading” shares the same bot loop as training (sim execution + learning).
        if not bot.is_training:
            start_training()

    def stop_trading():
        if not bot.is_trading:
            logger.info("Trading not running.")
            return
        bot.is_trading = False
        app_state.is_trading = False
        # keep training loop if user started it explicitly; otherwise stop it
        stop_training()

    ui.add_action_handler("start_training", start_training)
    ui.add_action_handler("stop_training", stop_training)
    ui.add_action_handler("start_trading", start_trading)
    ui.add_action_handler("stop_trading", stop_trading)

    ui.set_title("AI Trading Terminal (Simulation)")

    return bot, app_state, scheduler, ui


async def main_async(args):
    bot, app_state, scheduler, ui = build_bot()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_stop():
        try:
            scheduler.stop()
        except Exception:
            pass
        stop_event.set()

    # OS signals (ignore on Windows if unsupported)
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                loop.add_signal_handler(sig, _handle_stop)
            except NotImplementedError:
                pass

    # Optionally auto-start training
    if args.autostart:
        # mimic clicking the UI button
        for name, cb in ui._action_handlers.items():
            if name == "start_training":
                cb()
                break

    # Run UI in a thread to keep asyncio loop free
    def _run_ui():
        ui.run()

    ui_task = asyncio.to_thread(_run_ui)
    scheduler.start()

    await stop_event.wait()
    await ui.shutdown()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Learning Trading Bot (Simulation)")
    p.add_argument("--autostart", action="store_true", help="Start training immediately")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
