# main.py

import time
import logging
import config

from modules.telegram_bot import TelegramNotifier
from modules.notification_manager import NotificationManager, TradeEvent
from modules.trade_executor import TradeExecutor
from modules.market_scanner import MarketScanner
from modules.portfolio_manager import PortfolioManager
from modules.strategy_loader import StrategyLoader
from runtime_state import RuntimeState

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class TradingBot:
    def __init__(self):
        logger.info("Initializing TradingBot...")

        # Core runtime state
        self.state = RuntimeState()

        # Notifier & notification manager
        self.notifier = TelegramNotifier(disable_async=not getattr(config, "ASYNC_TELEGRAM", True))
        self.notifications = NotificationManager(
            notifier=self.notifier,
            mode="paper" if getattr(config, "SIMULATION_MODE", True) else "live",
            paper_recap_min=getattr(config, "TELEGRAM_PAPER_RECAP_MIN", 60),
            live_alert_level=getattr(config, "TELEGRAM_LIVE_ALERT_LEVEL", "normal"),
            heartbeat_min=getattr(config, "TELEGRAM_HEARTBEAT_MIN", 10)
        )

        # Other core components
        self.scanner = MarketScanner()
        self.portfolio = PortfolioManager()
        self.strategy_loader = StrategyLoader()
        self.executor = TradeExecutor(
            portfolio_manager=self.portfolio,
            notifier=self.notifications
        )

        self.running = True
        self.loop_delay = getattr(config, "BOT_LOOP_DELAY", 15)  # seconds

    def run(self):
        logger.info("Starting TradingBot loop...")
        self.notifications.notify_status("Trading bot started ✅")

        while self.running:
            try:
                loop_start = time.time()

                # Scan market & get candidate trades
                scan_results = self.scanner.scan_top_symbols()

                # Execute trades
                for trade_signal in scan_results:
                    result = self.executor.execute_trade(trade_signal)
                    if result:  # Trade executed
                        trade_event = TradeEvent(
                            symbol=result["symbol"],
                            side=result["side"],
                            qty=result["qty"],
                            price=result["price"],
                            pnl=result.get("pnl"),
                            return_pct=result.get("return_pct"),
                            leverage=result.get("leverage"),
                            opened=result.get("opened"),
                            closed=result.get("closed"),
                            status=result.get("status"),
                            meta=result.get("meta", {})
                        )
                        self.notifications.notify_trade(trade_event)

                # Update metrics snapshot for heartbeat/digest
                self.notifications.update_metrics_snapshot(
                    price=self.scanner.last_price,
                    symbol=self.scanner.last_symbol,
                    equity=self.portfolio.current_equity(),
                    balance=self.portfolio.current_balance()
                )

                # Tick notifications manager (heartbeat + digest)
                self.notifications.tick()

                # Loop delay
                elapsed = time.time() - loop_start
                if elapsed < self.loop_delay:
                    time.sleep(self.loop_delay - elapsed)

            except KeyboardInterrupt:
                logger.info("Stopping TradingBot...")
                self.running = False
            except Exception as e:
                logger.exception("Error in main loop: %s", e)
                self.notifications.notify_error(f"Main loop error: {e}")

        self.shutdown()

    def shutdown(self):
        logger.info("Shutting down TradingBot...")
        self.notifications.notify_status("Trading bot stopped ❌")
        self.notifier.graceful_shutdown()


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
