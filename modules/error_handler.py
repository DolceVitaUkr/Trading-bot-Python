# modules/error_handler.py

import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
import sys
import os

# Add project root to Python path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.telegram_bot import TelegramNotifier

class TradingBotError(Exception):
    """Base exception class for all trading bot errors."""
    def __init__(self, message: str = "", code: int = 0, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.context = context or {}
        self.module = self.__class__.__module__

    def __str__(self):
        return f"[{self.code}] {super().__str__()} (Module: {self.module}, Time: {self.timestamp})"

class NetworkError(TradingBotError):
    def __init__(self, message: str = "Network operation failed", code: int = 1000, **kwargs):
        super().__init__(message, code, **kwargs)

class APIError(TradingBotError):
    def __init__(self, message: str = "API communication failed", code: int = 2000, **kwargs):
        super().__init__(message, code, **kwargs)

class DataIntegrityError(TradingBotError):
    def __init__(self, message: str = "Data integrity issue", code: int = 3000, **kwargs):
        super().__init__(message, code, **kwargs)

class StrategyError(TradingBotError):
    def __init__(self, message: str = "Strategy violation", code: int = 4000, **kwargs):
        super().__init__(message, code, **kwargs)

class RiskViolationError(TradingBotError):
    def __init__(self, message: str = "Risk limit exceeded", code: int = 5000, **kwargs):
        super().__init__(message, code, **kwargs)

class OrderExecutionError(TradingBotError):
    def __init__(self, message: str = "Order execution failed", code: int = 6000, **kwargs):
        super().__init__(message, code, **kwargs)

class ConfigurationError(TradingBotError):
    def __init__(self, message: str = "Configuration error", code: int = 7000, **kwargs):
        super().__init__(message, code, **kwargs)

class NotificationError(TradingBotError):
    def __init__(self, message: str = "Notification failed", code: int = 8000, **kwargs):
        super().__init__(message, code, **kwargs)

class ErrorHandler:
    """
    Unified error handling pipeline:
    - Logs detailed tracebacks.
    - Tracks error counts and activates circuit breakers.
    - Sends critical alerts via Telegram.
    - Can signal the main bot to stop trading.
    """
    def __init__(self, stop_callback: Optional[Callable[[], None]] = None):
        self.telegram_bot = TelegramNotifier(disable_async=True)
        self.error_counts: Dict[int, int] = {}
        self.circuit_breakers: Dict[int, datetime] = {}
        self.error_log: list = []
        self.stop_callback = stop_callback

        # Configure our dedicated logger
        self.logger = logging.getLogger("ErrorHandler")
        self._configure_logger()

        # In debug_mode, we also print the traceback to console for dev purposes
        self.debug_mode = True

    def _configure_logger(self):
        """Set up a FileHandler for error logs if none exists."""
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            fh = logging.FileHandler(config.LOG_FILE)
            fh.setLevel(getattr(logging, config.LOG_LEVEL, logging.ERROR))
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.ERROR))

    def set_stop_callback(self, callback: Callable[[], None]):
        """Register a function to call when trading should be halted."""
        self.stop_callback = callback

    def log_error(self,
                  error: Exception,
                  context: Optional[Dict[str, Any]] = None,
                  extra: Optional[Dict[str, Any]] = None) -> None:
        """Log detailed error info and full traceback."""
        error_code = getattr(error, 'code', 0)
        context = context or {}
        extra = extra or {}

        error_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': error.__class__.__name__,
            'code': error_code,
            'message': str(error),
            'module': getattr(error, 'module', 'unknown'),
            'context': context,
            'extra': extra,
            'traceback': traceback.format_exception(type(error), error, error.__traceback__)
        }
        self.error_log.append(error_info)

        if self.debug_mode:
            print("\n🔴 ERROR DETAILS:")
            print(f"Type: {error_info['type']}")
            print(f"Message: {error_info['message']}")
            if context:
                print(f"Context: {context}")
            print("Traceback:")
            print("".join(error_info['traceback']))

        self.logger.error(
            "System Error [%(code)d]: %(message)s\n"
            "Type: %(type)s\n"
            "Module: %(module)s\n"
            "Context: %(context)s\n"
            "Extra: %(extra)s\n"
            "Traceback:\n%(traceback)s",
            {
                'code': error_code,
                'type': error_info['type'],
                'message': error_info['message'],
                'module': error_info['module'],
                'context': context,
                'extra': extra,
                'traceback': "".join(error_info['traceback'])
            }
        )

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Main entry point: log, update stats, attempt recovery, alert if critical."""
        context = context or {}
        error_code = getattr(error, 'code', 0)

        # 1) Update counts and potentially trip circuit breaker
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        if self.error_counts[error_code] > config.ERROR_RATE_THRESHOLD:
            self._activate_circuit_breaker(error_code)

        # 2) Always log the error
        self.log_error(error, context)

        # 3) Attempt recovery
        self._execute_recovery(error, context)

        # 4) If critical, send alert
        if error_code in config.CRITICAL_ERROR_CODES:
            self._send_critical_alert(error, context)

    def _activate_circuit_breaker(self, error_code: int):
        """Record the circuit-breaker trip and disable trading."""
        now = datetime.now(timezone.utc)
        self.circuit_breakers[error_code] = now
        self.logger.warning("Circuit breaker activated for error code: %d", error_code)
        self._disable_trading()

    def _disable_trading(self):
        """Emergency stop: log and invoke the stop callback if provided."""
        self.logger.critical("Trading operations disabled")
        if self.stop_callback:
            try:
                self.stop_callback()
            except Exception as e:
                self.logger.error("Error in stop_callback: %s", e, exc_info=True)

    def _execute_recovery(self, error: Exception, context: Dict[str, Any]):
        """Route to the specific recovery strategy based on error type."""
        if isinstance(error, NetworkError):
            self._handle_network_error(error, context)
        elif isinstance(error, APIError):
            self._handle_api_error(error, context)
        elif isinstance(error, RiskViolationError):
            self._handle_risk_violation(error, context)
        elif isinstance(error, OrderExecutionError):
            self._handle_order_error(error, context)
        # Other error types can be added here as needed

    def _handle_network_error(self, error: NetworkError, context: Dict[str, Any]):
        max_retries = 3
        attempt = context.get('retry_count', 0)
        if attempt < max_retries:
            self.logger.info("Retrying network operation (%d/%d)", attempt + 1, max_retries)
            context['retry_count'] = attempt + 1
        else:
            self.logger.error("Network operation failed after %d retries", max_retries)

    def _handle_api_error(self, error: APIError, context: Dict[str, Any]):
        self.logger.error("API failure: %s", error)

    def _handle_risk_violation(self, error: RiskViolationError, context: Dict[str, Any]):
        self.logger.critical("Risk violation detected: %s", error)
        # Already tripped circuit breaker in handle()

    def _handle_order_error(self, error: OrderExecutionError, context: Dict[str, Any]):
        self.logger.critical("Order execution failure: %s", error)
        # Already tripped circuit breaker in handle()

    def _send_critical_alert(self, error: Exception, context: Dict[str, Any]):
        """Send a critical alert via Telegram."""
        try:
            msg = (
                f"🚨 CRITICAL ERROR 🚨\n"
                f"Code: {getattr(error, 'code', 'UNKNOWN')}\n"
                f"Type: {error.__class__.__name__}\n"
                f"Message: {str(error)}\n"
                f"Time: {datetime.now(timezone.utc).isoformat()}"
            )
            # note: TelegramNotifier has an async-safe send_message_sync method
            self.telegram_bot.send_message_sync(msg)
        except Exception as e:
            self.logger.error("Failed to send critical alert: %s", e, exc_info=True)
