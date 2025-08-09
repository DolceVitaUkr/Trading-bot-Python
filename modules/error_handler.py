# modules/error_handler.py
from __future__ import annotations

import logging
import traceback
import sys
import os
import time
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable

# Ensure project root on path for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Telegram is optional (don’t crash if not configured)
from modules.telegram_bot import TelegramNotifier

# Try both runtime_state locations
RuntimeStateType = None
try:
    from state.runtime_state import RuntimeState as RuntimeStateType
except Exception:
    try:
        from modules.runtime_state import RuntimeState as RuntimeStateType  # type: ignore
    except Exception:
        RuntimeStateType = None  # type: ignore


class TradingBotError(Exception):
    def __init__(self, message: str = "", code: int = 0, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.context = context or {}
        self.module = self.__class__.__module__

    def __str__(self):
        return f"[{self.code}] {super().__str__()} (Module: {self.module}, Time: {self.timestamp})"


class NetworkError(TradingBotError):        pass
class APIError(TradingBotError):            pass
class DataIntegrityError(TradingBotError):  pass
class StrategyError(TradingBotError):       pass
class RiskViolationError(TradingBotError):  pass
class OrderExecutionError(TradingBotError): pass
class ConfigurationError(TradingBotError):  pass
class NotificationError(TradingBotError):   pass


class ErrorHandler:
    ERROR_RATE_WINDOW_SEC = 120

    def __init__(
        self,
        stop_callback: Optional[Callable[[], None]] = None,
        flatten_all_callback: Optional[Callable[[str], None]] = None,
        runtime_state: Optional["RuntimeStateType"] = None,
    ):
        self._telegram: Optional[TelegramNotifier] = None
        try:
            # may raise if no token/chat configured — that’s fine, we’ll just skip alerts
            self._telegram = TelegramNotifier(disable_async=True)
        except Exception:
            self._telegram = None

        self._error_times: Dict[int, deque] = defaultdict(deque)
        self._window = self.ERROR_RATE_WINDOW_SEC
        self._threshold = int(getattr(config, "ERROR_RATE_THRESHOLD", 5))
        self._critical = set(getattr(config, "CRITICAL_ERROR_CODES", {5000, 6000, 7000, 8000}))

        self.circuit_breakers: Dict[int, datetime] = {}
        self.error_log: list = []

        self.stop_callback = stop_callback
        self.flatten_all_callback = flatten_all_callback

        self.runtime_state = runtime_state

        self.logger = logging.getLogger("ErrorHandler")
        self._configure_logger()

        self.debug_mode = True

    def set_stop_callback(self, callback: Callable[[], None]):
        self.stop_callback = callback

    def set_flatten_callback(self, callback: Callable[[str], None]):
        self.flatten_all_callback = callback

    def set_runtime_state(self, runtime_state: "RuntimeStateType"):
        self.runtime_state = runtime_state

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        code = int(getattr(error, "code", 0))

        self.log_error(error, context)

        self._persist_event("error", {
            "code": code,
            "type": error.__class__.__name__,
            "message": str(error),
            "context": context,
        })

        if self._should_trip_breaker(code):
            self._activate_circuit_breaker(code)

        self._execute_recovery(error, context)

        if code in self._critical:
            self._send_critical_alert(error, context)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        error_code = int(getattr(error, 'code', 0))
        context = context or {}
        extra = extra or {}

        tb_list = traceback.format_exception(type(error), error, error.__traceback__)
        tb_compact = "".join(tb_list[-10:])

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

    def _execute_recovery(self, error: Exception, context: Dict[str, Any]):
        if isinstance(error, NetworkError):
            attempt = int(context.get('retry_count', 0))
            if attempt < 3:
                self.logger.info("Retrying network operation (%d/3)", attempt + 1)
                context['retry_count'] = attempt + 1
        elif isinstance(error, APIError):
            self.logger.error("API failure: %s", error)
        elif isinstance(error, DataIntegrityError):
            self.logger.error("Data integrity issue: %s", error)
            self._persist_event("reconcile_required", {"reason": "data_integrity"})
        elif isinstance(error, StrategyError):
            self.logger.error("Strategy error: %s", error)
        elif isinstance(error, RiskViolationError):
            self.logger.critical("Risk violation detected: %s", error)
            self._disable_trading("risk_violation")
        elif isinstance(error, OrderExecutionError):
            self.logger.critical("Order execution failure: %s", error)
            self._disable_trading("order_execution")

    def _should_trip_breaker(self, code: int) -> bool:
        now = time.time()
        q = self._error_times[code]
        q.append(now)
        while q and (now - q[0] > self._window):
            q.popleft()
        return len(q) > self._threshold

    def _activate_circuit_breaker(self, error_code: int):
        now = datetime.now(timezone.utc)
        self.circuit_breakers[error_code] = now
        self.logger.warning("Circuit breaker activated for error code: %d", error_code)
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
        self.logger.critical("DISASTER MODE: flatten all positions (%s)", reason)
        self._persist_event("disaster_mode", {"reason": reason})
        try:
            if self._telegram:
                self._telegram.send_message_sync(f"🛑 Disaster mode triggered: flatten ALL positions now. Reason: {reason}")
        except Exception as e:
            self.logger.error("Failed to notify Telegram (disaster): %s", e, exc_info=True)

        if self.flatten_all_callback:
            try:
                self.flatten_all_callback(reason)
            except Exception as e:
                self.logger.error("Error in flatten_all_callback: %s", e, exc_info=True)
        self._disable_trading("disaster_mode")

    def reset_error_counters(self):
        self._error_times.clear()
        self.circuit_breakers.clear()
        self._persist_event("error_counters_reset", {})

    def _send_critical_alert(self, error: Exception, context: Dict[str, Any]):
        if not self._telegram:
            return
        try:
            code = getattr(error, 'code', 'UNKNOWN')
            msg = (
                "🚨 CRITICAL ERROR 🚨\n"
                f"Code: {code}\n"
                f"Type: {error.__class__.__name__}\n"
                f"Message: {str(error)}\n"
                f"When: {datetime.now(timezone.utc).isoformat()}\n"
                f"Context: {self._compact_context(context)}"
            )
            self._telegram.send_message_sync(msg)
        except Exception as e:
            self.logger.error("Failed to send critical alert: %s", e, exc_info=True)

    def _configure_logger(self):
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            fh = logging.FileHandler(config.LOG_FILE)
            level = getattr(logging, str(getattr(config, "LOG_LEVEL", "ERROR")), logging.ERROR)
            fh.setLevel(level)
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        level = getattr(logging, str(getattr(config, "LOG_LEVEL", "ERROR")), logging.ERROR)
        self.logger.setLevel(level)

    def _compact_context(self, ctx: Dict[str, Any]) -> str:
        try:
            return ", ".join(f"{k}={v}" for k, v in list(ctx.items())[:6])
        except Exception:
            return "<unrepr>"

    def _persist_event(self, kind: str, payload: Dict[str, Any]):
        if self.runtime_state and hasattr(self.runtime_state, "append_event"):
            try:
                self.runtime_state.append_event(kind, payload)
            except Exception as e:
                self.logger.error("Failed to persist event '%s': %s", kind, e, exc_info=True)

