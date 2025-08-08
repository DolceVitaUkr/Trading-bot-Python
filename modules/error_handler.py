# modules/error_handler.py
from __future__ import annotations

import logging
import traceback
import sys
import os
import time
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable

# Ensure project root on path for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.telegram_bot import TelegramNotifier

# Optional: runtime state for persistence (errors/events)
try:
    from modules.runtime_state import RuntimeState
except Exception:
    RuntimeState = None  # type: ignore


# ────────────────────────────────────────────────────────────────────────────────
# Exception taxonomy (unchanged API)
# ────────────────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────────────────
# Error Handler
# ────────────────────────────────────────────────────────────────────────────────
class ErrorHandler:
    """
    Unified error handling pipeline:
    - Logs detailed tracebacks.
    - Tracks error rates in a rolling window and trips circuit breakers.
    - Sends critical alerts via Telegram.
    - Signals stop/flatten via injected callbacks.
    - Persists to RuntimeState if available.
    """
    # Rolling window (seconds) for error budget/rate check
    ERROR_RATE_WINDOW_SEC = 120

    def __init__(
        self,
        stop_callback: Optional[Callable[[], None]] = None,
        flatten_all_callback: Optional[Callable[[str], None]] = None,
        runtime_state: Optional["RuntimeState"] = None,
    ):
        self.telegram_bot = TelegramNotifier(disable_async=True)
        self._error_times: Dict[int, deque] = defaultdict(deque)  # per-code timestamps
        self._window = self.ERROR_RATE_WINDOW_SEC
        self._threshold = int(getattr(config, "ERROR_RATE_THRESHOLD", 5))
        self._critical = set(getattr(config, "CRITICAL_ERROR_CODES", {5000, 6000, 7000, 8000}))

        self.circuit_breakers: Dict[int, datetime] = {}
        self.error_log: list = []

        self.stop_callback = stop_callback
        self.flatten_all_callback = flatten_all_callback

        # Runtime state (optional)
        self.runtime_state: Optional["RuntimeState"] = runtime_state

        # Dedicated logger
        self.logger = logging.getLogger("ErrorHandler")
        self._configure_logger()

        # Dev-friendly console details
        self.debug_mode = True

    # Public wiring helpers
    def set_stop_callback(self, callback: Callable[[], None]):
        self.stop_callback = callback

    def set_flatten_callback(self, callback: Callable[[str], None]):
        self.flatten_all_callback = callback

    def set_runtime_state(self, runtime_state: "RuntimeState"):
        self.runtime_state = runtime_state

    # ──────────────────────────────────────────────────────────────────────
    # Core flow
    # ──────────────────────────────────────────────────────────────────────
    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Main entry point: log, update stats, attempt recovery, alert if critical."""
        context = context or {}
        code = int(getattr(error, "code", 0))

        # 1) Always log to file (and console in debug)
        self.log_error(error, context)

        # 2) Persist to runtime_state (if available)
        self._persist_event("error", {
            "code": code,
            "type": error.__class__.__name__,
            "message": str(error),
            "context": context,
        })

        # 3) Update rate counters and maybe trip breaker
        if self._should_trip_breaker(code):
            self._activate_circuit_breaker(code)

        # 4) Attempt recovery
        self._execute_recovery(error, context)

        # 5) Critical alert path
        if code in self._critical:
            self._send_critical_alert(error, context)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log detailed error info and full traceback."""
        error_code = int(getattr(error, 'code', 0))
        context = context or {}
        extra = extra or {}

        tb_list = traceback.format_exception(type(error), error, error.__traceback__)
        tb_compact = "".join(tb_list[-10:])  # last 10 lines, keep alerts readable

        error_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': error.__class__.__name__,
            'code': error_code,
            'message': str(error),
            'module': getattr(error, 'module', 'unknown'),
            'context': context,
            'extra': extra,
            'traceback': tb_list,
        }
        self.error_log.append(error_info)

        if self.debug_mode:
            print("\n🔴 ERROR DETAILS:")
            print(f"Type: {error_info['type']}")
            print(f"Message: {error_info['message']}")
            if context:
                print(f"Context: {context}")
            print("Traceback (compact):")
            print(tb_compact)

        self.logger.error(
            "System Error [%(code)d]: %(message)s\n"
            "Type: %(type)s | Module: %(module)s\n"
            "Context: %(context)s | Extra: %(extra)s\n"
            "Traceback (tail):\n%(tb)s",
            {
                'code': error_code,
                'type': error_info['type'],
                'message': error_info['message'],
                'module': error_info['module'],
                'context': context,
                'extra': extra,
                'tb': tb_compact
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    # Recovery routing
    # ──────────────────────────────────────────────────────────────────────
    def _execute_recovery(self, error: Exception, context: Dict[str, Any]):
        if isinstance(error, NetworkError):
            self._handle_network_error(error, context)
        elif isinstance(error, APIError):
            self._handle_api_error(error, context)
        elif isinstance(error, RiskViolationError):
            self._handle_risk_violation(error, context)
        elif isinstance(error, OrderExecutionError):
            self._handle_order_error(error, context)
        elif isinstance(error, DataIntegrityError):
            self._handle_data_error(error, context)
        elif isinstance(error, StrategyError):
            self._handle_strategy_error(error, context)
        # extend as needed

    def _handle_network_error(self, error: NetworkError, context: Dict[str, Any]):
        attempt = int(context.get('retry_count', 0))
        max_retries = 3
        if attempt < max_retries:
            self.logger.info("Retrying network operation (%d/%d)", attempt + 1, max_retries)
            context['retry_count'] = attempt + 1
        else:
            self.logger.error("Network operation failed after %d retries", max_retries)

    def _handle_api_error(self, error: APIError, context: Dict[str, Any]):
        self.logger.error("API failure: %s", error)

    def _handle_data_error(self, error: DataIntegrityError, context: Dict[str, Any]):
        self.logger.error("Data integrity issue: %s", error)
        # mark to reconcile on next boot
        self._persist_event("reconcile_required", {"reason": "data_integrity"})

    def _handle_strategy_error(self, error: StrategyError, context: Dict[str, Any]):
        self.logger.error("Strategy error: %s", error)

    def _handle_risk_violation(self, error: RiskViolationError, context: Dict[str, Any]):
        self.logger.critical("Risk violation detected: %s", error)
        # breaker trip handled by rate engine; still force stop
        self._disable_trading("risk_violation")

    def _handle_order_error(self, error: OrderExecutionError, context: Dict[str, Any]):
        self.logger.critical("Order execution failure: %s", error)
        self._disable_trading("order_execution")

    # ──────────────────────────────────────────────────────────────────────
    # Circuit breaker & disaster
    # ──────────────────────────────────────────────────────────────────────
    def _should_trip_breaker(self, code: int) -> bool:
        now = time.time()
        q = self._error_times[code]
        q.append(now)
        # drop old
        while q and (now - q[0] > self._window):
            q.popleft()
        return len(q) > self._threshold

    def _activate_circuit_breaker(self, error_code: int):
        now = datetime.now(timezone.utc)
        self.circuit_breakers[error_code] = now
        self.logger.warning("Circuit breaker activated for error code: %d", error_code)
        # record in runtime and notify
        self._persist_event("circuit_break", {"code": error_code, "time": now.isoformat()})

    def _disable_trading(self, reason: str = "unknown"):
        self.logger.critical("Trading operations disabled (reason=%s)", reason)
        self._persist_event("trading_disabled", {"reason": reason})
        if self.stop_callback:
            try:
                self.stop_callback()
            except Exception as e:
                self.logger.error("Error in stop_callback: %s", e, exc_info=True)

    def request_disaster_flatten(self, reason: str = "manual_disaster"):
        """
        One-click 'flatten all across domains', then stop.
        The actual position-closing is delegated to injected callback.
        """
        self.logger.critical("DISASTER MODE: flatten all positions (%s)", reason)
        self._persist_event("disaster_mode", {"reason": reason})
        # Telegram heads-up
        try:
            self.telegram_bot.send_message_sync(
                f"🛑 Disaster mode triggered: flatten ALL positions now. Reason: {reason}"
            )
        except Exception as e:
            self.logger.error("Failed to notify Telegram (disaster): %s", e, exc_info=True)

        if self.flatten_all_callback:
            try:
                self.flatten_all_callback(reason)
            except Exception as e:
                self.logger.error("Error in flatten_all_callback: %s", e, exc_info=True)
        # Always stop after attempting flatten
        self._disable_trading("disaster_mode")

    def reset_error_counters(self):
        """Clear rolling counters and breakers (e.g., after maintenance)."""
        self._error_times.clear()
        self.circuit_breakers.clear()
        self._persist_event("error_counters_reset", {})

    # ──────────────────────────────────────────────────────────────────────
    # Alerts & logging plumbing
    # ──────────────────────────────────────────────────────────────────────
    def _send_critical_alert(self, error: Exception, context: Dict[str, Any]):
        """Send a critical alert via Telegram."""
        try:
            code = getattr(error, 'code', 'UNKNOWN')
            msg = (
                f"🚨 *CRITICAL ERROR* 🚨\n"
                f"*Code*: `{code}`\n"
                f"*Type*: `{error.__class__.__name__}`\n"
                f"*Message*: {str(error)}\n"
                f"*When*: {datetime.now(timezone.utc).isoformat()}\n"
                f"*Context*: `{self._compact_context(context)}`"
            )
            self.telegram_bot.send_message_sync(msg)
        except Exception as e:
            self.logger.error("Failed to send critical alert: %s", e, exc_info=True)

    def _configure_logger(self):
        """Set up a FileHandler for error logs if none exists."""
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            fh = logging.FileHandler(config.LOG_FILE)
            fh.setLevel(getattr(logging, config.LOG_LEVEL, logging.ERROR))
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.ERROR))

    def _compact_context(self, ctx: Dict[str, Any]) -> str:
        try:
            # keep it short for Telegram
            return ", ".join(f"{k}={v}" for k, v in list(ctx.items())[:6])
        except Exception:
            return "<unrepr>"

    def _persist_event(self, kind: str, payload: Dict[str, Any]):
        if self.runtime_state and hasattr(self.runtime_state, "append_event"):
            try:
                self.runtime_state.append_event(kind, payload)
            except Exception as e:
                self.logger.error("Failed to persist event '%s': %s", kind, e, exc_info=True)
