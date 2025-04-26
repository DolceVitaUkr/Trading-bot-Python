# modules/exceptions.py

import traceback
from datetime import datetime, timezone

class TradingBotError(Exception):
    """Base exception class for all trading bot errors."""
    def __init__(self, message: str = "", code: int = 0, context=None):
        super().__init__(message)
        self.code = code
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
    def __str__(self):
        return f"[{self.code}] {super().__str__()} (Context: {self.context}, Time: {self.timestamp})"

class NetworkError(TradingBotError):
    """Network-related errors (retryable)."""
    def __init__(self, message="Network operation failed", code=1000, context=None):
        super().__init__(message, code, context)

class APIError(TradingBotError):
    """Exchange API errors."""
    def __init__(self, message="API communication failed", code=2000, context=None):
        super().__init__(message, code, context)

class DataIntegrityError(TradingBotError):
    """Data validation/processing errors."""
    def __init__(self, message="Data integrity issue", code=3000, context=None):
        super().__init__(message, code, context)

class StrategyError(TradingBotError):
    """Trading strategy-related errors."""
    def __init__(self, message="Strategy violation", code=4000, context=None):
        super().__init__(message, code, context)

class RiskViolationError(TradingBotError):
    """Risk management rule violations."""
    def __init__(self, message="Risk limit exceeded", code=5000, context=None):
        super().__init__(message, code, context)

class OrderExecutionError(TradingBotError):
    """Order placement/execution failures."""
    def __init__(self, message="Order execution failed", code=6000, context=None):
        super().__init__(message, code, context)

class ConfigurationError(TradingBotError):
    """Invalid configuration errors."""
    def __init__(self, message="Configuration error", code=7000, context=None):
        super().__init__(message, code, context)

class NotificationError(TradingBotError):
    """Alert/notification system errors."""
    def __init__(self, message="Notification failed", code=8000, context=None):
        super().__init__(message, code, context)
