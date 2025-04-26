# modules/error_handler.py

import logging
import traceback
from datetime import datetime, timezone
import config
from modules.exceptions import (TradingBotError,
                                NetworkError,
                                APIError,
                                DataIntegrityError,
                                StrategyError,
                                RiskViolationError,
                                OrderExecutionError,
                                ConfigurationError,
                                NotificationError)
from modules.telegram_bot import TelegramNotifier

class ErrorHandler:
    def __init__(self):
        logging.basicConfig(filename=config.LOG_FILE,
                            level=getattr(logging, config.LOG_LEVEL),
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            force=True)
        self.logger = logging.getLogger("ErrorHandler")
        self.telegram_bot = TelegramNotifier(disable_async=True)
        self.error_counts = {}

    def handle(self, error: Exception, context=None):
        code = getattr(error, "code", 0)
        self._log_error(error, context)
        if code in getattr(config, "CRITICAL_ERROR_CODES", []):
            self._send_critical_alert(error)

    def _log_error(self, error, context):
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.logger.error(f"[{getattr(error,'code',0)}] {error} Context:{context}\n{tb}")

    def _send_critical_alert(self, error: Exception):
        msg = (
            f"🚨 CRITICAL ERROR 🚨\n"
            f"Code: {getattr(error,'code','?')}\n"
            f"Type: {error.__class__.__name__}\n"
            f"Message: {error}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}"
        )
        self.telegram_bot.send_message(msg)
