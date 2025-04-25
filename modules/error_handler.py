# modules/error_handler.py
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.telegram_bot import TelegramNotifier

class TradingBotError(Exception):
    """Base exception class for all trading bot errors"""
    def __init__(self, message="", code=0000, context=None):
        super().__init__(message)
        self.code = code
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.context = context or {}
        self.module = self.__class__.__module__
        
    def __str__(self):
        return f"[{self.code}] {super().__str__()} (Module: {self.module}, Time: {self.timestamp})"

# Specialized Exceptions
class NetworkError(TradingBotError):
    """Network-related errors (retryable)"""
    def __init__(self, message="Network operation failed", code=1000, **kwargs):
        super().__init__(message, code, **kwargs)

class APIError(TradingBotError):
    """Exchange API errors"""
    def __init__(self, message="API communication failed", code=2000, **kwargs):
        super().__init__(message, code, **kwargs)

class DataIntegrityError(TradingBotError):
    """Data validation/processing errors"""
    def __init__(self, message="Data integrity issue", code=3000, **kwargs):
        super().__init__(message, code, **kwargs)

class StrategyError(TradingBotError):
    """Trading strategy-related errors"""
    def __init__(self, message="Strategy violation", code=4000, **kwargs):
        super().__init__(message, code, **kwargs)

class RiskViolationError(TradingBotError):
    """Risk management rule violations"""
    def __init__(self, message="Risk limit exceeded", code=5000, **kwargs):
        super().__init__(message, code, **kwargs)

class OrderExecutionError(TradingBotError):
    """Order placement/execution failures"""
    def __init__(self, message="Order execution failed", code=6000, **kwargs):
        super().__init__(message, code, **kwargs)

class ConfigurationError(TradingBotError):
    """Invalid configuration errors"""
    def __init__(self, message="Configuration error", code=7000, **kwargs):
        super().__init__(message, code, **kwargs)

class NotificationError(TradingBotError):
    """Alert/notification system errors"""
    def __init__(self, message="Notification failed", code=8000, **kwargs):
        super().__init__(message, code, **kwargs)

class ErrorHandler:
    def __init__(self):
        self.telegram_bot = TelegramNotifier(disable_async=True)
        self.error_counts = {}
        self.circuit_breakers = {}
        self.error_log = []
        self.logger = logging.getLogger("ErrorHandler")
        self._configure_logging()
        self.debug_mode = True  # Enable detailed logging

    def _configure_logging(self):
        """Set up structured logging"""
        logging.basicConfig(
            filename=config.LOG_FILE,
            level=getattr(logging, config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            force=True
        )

    def log_error(self, 
                error: Exception, 
                context: Optional[Dict[str, Any]] = None,
                extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Enhanced error logging with complete traceback
        """
        error_code = getattr(error, 'code', 0000)
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
            'traceback': traceback.format_exception(
                type(error), error, error.__traceback__
            )
        }

        self.error_log.append(error_info)

        if self.debug_mode:
            print("\n🔴 ERROR DETAILS:")
            print(f"Type: {error_info['type']}")
            print(f"Message: {error_info['message']}")
            if context: print(f"Context: {context}")
            print(f"Traceback:\n{error_info['traceback']}")

        self.logger.error(
            "System Error [%(code)s]: %(message)s\n"
            "Type: %(type)s\n"
            "Module: %(module)s\n"
            "Context: %(context)s\n"
            "Extra: %(extra)s\n"
            "Traceback: %(traceback)s",
            {
                'code': error_code,
                'type': error_info['type'],
                'message': error_info['message'],
                'module': error_info['module'],
                'context': context,
                'extra': extra,
                'traceback': "".join(error_info['traceback']).strip()
            },
            extra={'error_info': error_info}
        )

    def _get_traceback(self, error):
        import traceback
        return ''.join(traceback.format_exception(type(error), error, error.__traceback__))

    def handle(self, error: Exception, context: Dict[str, Any] = None):
        """Main error handling pipeline"""
        context = context or {}
        error_code = getattr(error, 'code', 0000)

        self._update_error_stats(error_code)
        self.log_error(error, context)
        self._execute_recovery(error, context)
        
        if error_code in config.CRITICAL_ERROR_CODES:
            self._send_critical_alert(error, context)

    def _update_error_stats(self, error_code: int):
        """Update error counters and check thresholds"""
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        if self.error_counts[error_code] > config.ERROR_RATE_THRESHOLD:
            self._activate_circuit_breaker(error_code)

    def _activate_circuit_breaker(self, error_code: int):
        """Implement circuit breaker pattern"""
        self.circuit_breakers[error_code] = datetime.now(timezone.utc)
        self.logger.warning("Circuit breaker activated for error code: %s", error_code)
        
        if error_code in config.CRITICAL_ERROR_CODES:
            self._disable_trading()

    def _execute_recovery(self, error: Exception, context: Dict[str, Any]):
        """Route to appropriate recovery strategy"""
        if isinstance(error, NetworkError):
            self._handle_network_error(error, context)
        elif isinstance(error, APIError):
            self._handle_api_error(error, context)
        elif isinstance(error, RiskViolationError):
            self._handle_risk_violation(error, context)
        elif isinstance(error, OrderExecutionError):
            self._handle_order_error(error, context)

    def _send_critical_alert(self, error: Exception, context: Dict[str, Any]):
        """Send formatted alert through multiple channels"""
        try:
            message = (
                f"🚨 CRITICAL ERROR 🚨\n"
                f"Code: {getattr(error, 'code', 'UNKNOWN')}\n"
                f"Type: {error.__class__.__name__}\n"
                f"Message: {str(error)}\n"
                f"Module: {getattr(error, 'module', 'unknown')}\n"
                f"Time: {datetime.now(timezone.utc).isoformat()}"
            )
            self.telegram_bot.send_message_sync(message)
        except Exception as e:
            self.logger.error("Failed to send critical alert: %s", str(e))

    def _handle_network_error(self, error, context):
        """Network error recovery strategy"""
        max_retries = 3
        current_retry = context.get('retry_count', 0)
        
        if current_retry < max_retries:
            self.logger.info("Retrying network operation (attempt %d/%d)", 
                            current_retry+1, max_retries)
            context['retry_count'] = current_retry + 1
        else:
            self.logger.error("Network operation failed after %d retries", max_retries)

    def _handle_api_error(self, error, context):
        """API error recovery strategy"""
        self.logger.error("API failure: %s", error)

    def _handle_risk_violation(self, error, context):
        """Risk violation recovery strategy"""
        self.logger.critical("Risk violation detected: %s", error)
        self._disable_trading()
        self._trigger_manual_review()

    def _handle_order_error(self, error, context):
        """Order execution error handling"""
        self.logger.critical("Order execution failure: %s", error)
        self._disable_trading()

    def _disable_trading(self):
        """Emergency stop trading operations"""
        self.logger.critical("Trading operations disabled")

    def _trigger_manual_review(self):
        """Flag for human intervention"""
        self.logger.critical("Manual review required")

# Example usage remains the same
if __name__ == "__main__":
    handler = ErrorHandler()
    
    try:
        raise OrderExecutionError("Failed to execute limit order", 
                                context={"order_id": 123, "symbol": "BTC/USDT"})
    except Exception as e:
        handler.handle(e, {"stage": "order_execution"})