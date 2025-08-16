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
from modules.Strategy_Manager import StrategyManager
from modules.top_pairs import TopPairs
from modules.telegram_bot import TelegramNotifier
from modules.reward_system import RewardSystem
from scheduler import JobScheduler


def build_risk_manager(account_balance: float, notifier: TelegramNotifier) -> RiskManager:
    """
    Builds a RiskManager instance based on the configuration.

    Args:
        account_balance: The current account balance.
        notifier: The Telegram notifier instance.

    Returns:
        A RiskManager instance.
    """
    rm = RiskManager(
        account_balance=account_balance,
        notifier=notifier
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
    if not test_mode:
        exchange.load_markets()

    notifier = TelegramNotifier(disable_async=not config.ASYNC_TELEGRAM)

    # Wallets
    starting_balance = float(config.SIMULATION_START_BALANCE)
    # Risk
    risk_manager = build_risk_manager(starting_balance, notifier)

    trade_executor = TradeExecutor(
        simulation_mode=True,  # Keep execution in simulation as requested
        notifier=notifier,
        risk_manager=risk_manager,
        notifications=None,
    )

    # Data Manager
    dm = DataManager(exchange=exchange)

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
    bot = StrategyManager(
        data_provider=dm,
        error_handler=exchange.error_handler,
        reward_system=reward,
        risk_manager=risk_manager,
        symbol=config.DEFAULT_SYMBOL,
    )

    # The new StrategyManager has its own run loop.
    # We just need to start it in a thread.
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()

    if not test_mode:
        # In a real application, you would have a UI or some other way to
        # interact with the bot.
        while True:
            time.sleep(1)

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
