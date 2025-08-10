# modules/error_handler.py

import logging
from typing import Any, Dict, Optional


class RiskViolationError(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class OrderExecutionError(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class APIError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.context = context or {}


class ErrorHandler:
    """
    Central error pipeline:
      - Structured logging with context
      - Optional notifier (Telegram or higher-level notifications manager)
      - Safe 'handle' wrapper used across the app
    """

    def __init__(self, notifier: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        self.notifier = notifier
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    # Lightweight console + optional notifier
    def log_error(self, exc: Exception, context: Optional[Dict[str, Any]] = None):
        ctx = context or {}
        try:
            self.logger.error(f"{exc.__class__.__name__}: {exc} | ctx={ctx}")
        except Exception:
            pass
        self._notify_error(exc, ctx)

    def handle(self, exc: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Use in try/except blocks to consistently log & optionally notify.
        """
        self.log_error(exc, context)

    # Hook-friendly alerting
    def _notify_error(self, exc: Exception, context: Dict[str, Any]):
        """
        Notifier can be:
          - Telegram notifier with .send_message_sync(...)
          - A notifications manager with .notify_error(...)
        """
        if not self.notifier:
            return
        try:
            # Prefer notifications manager shape
            if hasattr(self.notifier, "notify_error"):
                self.notifier.notify_error({"type": "ERROR", "error": str(exc), "context": context})
                return
            # Fallback simple Telegram-like
            if hasattr(self.notifier, "send_message_sync"):
                payload = {"type": "ERROR", "error": str(exc), "context": context}
                self.notifier.send_message_sync(payload, format="alert")
        except Exception:
            # Never fail due to notifier pipeline
            pass
