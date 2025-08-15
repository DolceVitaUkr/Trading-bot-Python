# main.py

import argparse
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import config
from utils.utilities import configure_logging
from modules.data_manager import DataManager
from modules.exchange import ExchangeAPI
from modules.trade_executor import TradeExecutor
from modules.risk_management import RiskManager
from modules.self_learning import SelfLearningBot
from modules.top_pairs import TopPairs
from modules.ui import TradingUI
from modules.telegram_bot import TelegramNotifier
from modules.reward_system import RewardSystem
from scheduler import JobScheduler


def build_risk_manager(account_balance: float) -> RiskManager:
    """
    Builds a RiskManager instance based on the configuration.

    Args:
        account_balance: The current account balance.

    Returns:
        A RiskManager instance.
    """
    caps = config.RISK_CAPS.get(
        "crypto_spot" if config.EXCHANGE_PROFILE == "spot" else "perp",
        {"per_pair_pct": 0.15, "portfolio_concurrent_pct": 0.30},
    )
    rm = RiskManager(
        account_balance=account_balance,
        max_drawdown_limit=config.KPI_TARGETS.get("max_drawdown", 0.15),
        per_pair_cap_pct=caps["per_pair_pct"],
        portfolio_cap_pct=caps["portfolio_concurrent_pct"],
        base_risk_per_trade_pct=config.TRADE_SIZE_PERCENT,
        min_rr=1.5,
        atr_mult_sl=1.5,
        atr_mult_tp=3.0,
    )
    return rm


def run_bot(args: argparse.Namespace,
            test_mode: bool = False,
            stop_event: Optional[threading.Event] = None) -> int:
    """
    Initializes and runs the trading bot.

    Args:
        args: The command-line arguments.
        test_mode: Whether to run in test mode.
        stop_event: An event to stop the bot.

    Returns:
        The exit code.
    """
    configure_logging(config.LOG_LEVEL, config.LOG_FILE)
    log = logging.getLogger("main")
    log.info("Booting Self-Learning Trading Bot…")

    # Exchange + Data
    exchange = ExchangeAPI()
    exchange.load_markets()

    notifier = TelegramNotifier(disable_async=not config.ASYNC_TELEGRAM)

    # Wallets
    starting_balance = float(config.SIMULATION_START_BALANCE)
    trade_executor = TradeExecutor(
        simulation_mode=True,  # Keep execution in simulation as requested
        notifier=notifier,
        notifications=None,
    )

    # Risk
    risk_manager = build_risk_manager(starting_balance)

    # Data Manager
    dm = DataManager(exchange=exchange.client)

    # Top pairs manager
    top_pairs = TopPairs(
        exchange=exchange,
        quote="USDT",
        max_pairs=config.MAX_SIMULATION_PAIRS,
        ttl_sec=60 * 60,  # re-scan hourly
        min_volume_usd_24h=5_000_000,
    )

    # Reward system
    reward = RewardSystem()

    # Agent
    bot = SelfLearningBot(
        data_provider=dm,
        error_handler=exchange.error_handler,
        reward_system=reward,
        risk_manager=risk_manager,
        state_size=5,
        action_size=6,
        hidden_dims=[128, 64, 32],
        batch_size=64,
        gamma=0.99,
        learning_rate=1e-3,
        exploration_max=1.0,
        exploration_min=0.05,
        exploration_decay=0.995,
        memory_size=100_000,
        tau=0.005,
        training=True,
        timeframe="5m",  # Primary loop on 5m
        symbol=config.DEFAULT_SYMBOL,  # Fallback if top-pairs empty
    )

    # UI
    ui = TradingUI(bot=bot)
    ui.set_title("AI Trading Terminal (Simulation)")
    bot.ui_hook = ui  # let the agent push UI metrics

    # Expose a couple of handlers for UI buttons
    def start_training():
        bot.training = True
        ui.log("Training started.", level="SUCCESS")
        ui.set_button_active("start_training")

    def stop_training():
        bot.training = False
        ui.log("Training stopped.", level="WARN")
        ui.set_button_active("stop_training")

    def start_trading():
        bot.training = True
        ui.log("Trading loop (simulation) started.", level="SUCCESS")
        ui.set_button_active("start_trading")

    def stop_trading():
        bot.training = False
        ui.log("Trading loop (simulation) stopped.", level="WARN")
        ui.set_button_active("stop_trading")

    ui.add_action_handler("start_training", start_training)
    ui.add_action_handler("stop_training", stop_training)
    ui.add_action_handler("start_trading", start_trading)
    ui.add_action_handler("stop_trading", stop_trading)

    # Scheduler jobs
    scheduler = JobScheduler()

    # 1) Hourly top pairs refresh (to detect spikes and changes)
    def refresh_pairs_job():
        try:
            pairs = top_pairs.get_top_pairs()  # triggers refresh if stale
            bot.top_symbols = pairs or [config.DEFAULT_SYMBOL]
            ui.log(f"Top pairs refreshed: {', '.join(bot.top_symbols)}",
                   level="INFO")
        except Exception as e:
            ui.log(f"Top pairs refresh failed: {e}", level="ERROR")

    scheduler.every(minutes=60, name="hourly_top_pairs",
                    func=refresh_pairs_job)

    # 2) 15m setup scan (secondary timeframe)
    def fifteen_scan_job():
        try:
            symbols = getattr(bot, "top_symbols",
                                None) or [config.DEFAULT_SYMBOL]
            for sym in symbols:
                dm.load_historical_data(
                    sym, "15m", backfill_bars=300)  # light backfill
                # You can add setup-detection hooks here if needed
        except Exception as e:
            ui.log(f"15m scan error: {e}", level="ERROR")

    scheduler.every(minutes=15, name="scan_15m_setup", func=fifteen_scan_job)

    # 3) Heartbeat → UI metrics
    def heartbeat_job():
        try:
            # live balance (if we had real, keep sim for now)
            live_balance = trade_executor.get_balance()
            # sim portfolio value from agent perspective
            portfolio_val = getattr(bot, "portfolio_value",
                                    float(starting_balance))
            reward_pts = getattr(bot.reward_system, "total_points", 0.0)
            ui.update_live_metrics({
                "balance": live_balance,
                "equity": portfolio_val,
                "symbol": bot.symbol or config.DEFAULT_SYMBOL,
                "timeframe": bot.timeframe,
            })
            ui.update_timeseries(wallet=live_balance,
                                  vwallet=portfolio_val,
                                  points=reward_pts)
        except Exception as e:
            logging.getLogger("main").debug(f"Heartbeat err: {e}")

    scheduler.every(seconds=max(5, int(config.LIVE_LOOP_INTERVAL)),
                    name="heartbeat", func=heartbeat_job)

    # Start scheduler thread
    scheduler_thread = threading.Thread(target=scheduler.run_forever,
                                        daemon=True)
    scheduler_thread.start()

    # Agent background loop (5m tick loop, sim execution, live data)
    def agent_loop():
        while not (stop_event and stop_event.is_set()):
            try:
                symbols = getattr(bot, "top_symbols",
                                    None) or [config.DEFAULT_SYMBOL]
                for sym in symbols:
                    # Keep data fresh & light: append-only small pulls
                    dm.load_historical_data(sym, "5m", incremental=True)
                    # Act & learn using the latest state (sim execution)
                    bot.act_and_learn(sym,
                                      timestamp=datetime.now(timezone.utc))
                # Pace loop
                time.sleep(max(5, float(config.LIVE_LOOP_INTERVAL)))

                if test_mode:
                    break  # Run only once in test mode
            except Exception as e:
                logging.getLogger("main").exception(f"agent_loop error: {e}")
                time.sleep(5)

    threading.Thread(target=agent_loop, daemon=True).start()

    # Initial top pairs warmup
    try:
        refresh_pairs_job()
    except Exception:
        pass

    if not test_mode:
        # Start UI (blocking)
        ui.run_ui()

    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parses command-line arguments.

    Args:
        argv: The list of command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Self-Learning AI Trading Bot")
    p.add_argument("--mode",
                   choices=["simulation", "production"],
                   default=config.ENVIRONMENT)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_bot(parse_args()))
