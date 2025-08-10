# modules/telegram_bot.py

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Union

import requests

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


class TelegramNotifier:
    """
    Minimal Telegram notifier with:
      - send_message_sync(text|dict, format="status")
      - optional async mode (fire-and-forget)
      - simple rate limiting & backoff
      - helpers: notify_trade / notify_error / notify_status

    Env/config:
      TELEGRAM_BOT_TOKEN
      TELEGRAM_CHAT_ID
      ASYNC_TELEGRAM (bool)
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[Union[str, int]] = None,
        disable_async: bool = None,  # legacy knob; if True forces sync
        async_mode: Optional[bool] = None,
        parse_mode: str = "HTML",
        rate_limit_msgs_per_min: int = 30,
        timeout: float = 10.0,
    ):
        self.token = bot_token or getattr(config, "TELEGRAM_BOT_TOKEN", "")
        self.chat_id = str(chat_id or getattr(config, "TELEGRAM_CHAT_ID", "")).strip()
        # async toggle precedence: explicit arg > config > legacy flag
        if async_mode is None:
            async_mode = bool(getattr(config, "ASYNC_TELEGRAM", True))
        if disable_async is True:
            async_mode = False
        self.async_mode = async_mode

        self.parse_mode = parse_mode
        self.timeout = timeout
        self.api_url = self.API_URL.format(token=self.token)

        # simple rate limiting
        self._rl_capacity = max(1, rate_limit_msgs_per_min)
        self._rl_window = 60.0
        self._rl_lock = threading.Lock()
        self._sent_timestamps = []

        # async worker
        self._q: list[Dict[str, Any]] = []
        self._q_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        if self.async_mode:
            self._start_worker()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def send_message_sync(self, message: Union[str, Dict[str, Any]], *, format: str = "status") -> bool:
        """
        Send a message right away (blocking). If async_mode=True, the call still
        blocks—the async path is send_message() below.
        """
        if not self._enabled():
            return False

        text = self._format_payload(message, format)
        if not text:
            return False

        if not self._rate_allow():
            logger.debug("Telegram rate limit reached; dropping message.")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode

        try:
            r = requests.post(self.api_url, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning(f"Telegram send failed {r.status_code}: {r.text[:200]}")
                return False
            return True
        except Exception as e:
            logger.debug(f"Telegram send exception: {e}")
            return False

    def send_message(self, message: Union[str, Dict[str, Any]], *, format: str = "status") -> None:
        """
        Enqueue message for async delivery; falls back to sync if async disabled.
        """
        if not self.async_mode:
            self.send_message_sync(message, format=format)
            return
        if not self._enabled():
            return
        text = self._format_payload(message, format)
        if not text:
            return
        with self._q_lock:
            self._q.append({"text": text})

    # Convenience helpers used around the codebase
    def notify_trade(self, event: Dict[str, Any]) -> None:
        txt = self._render_trade(event)
        if txt:
            self.send_message(txt, format="trade")

    def notify_error(self, error: Union[str, Dict[str, Any]]) -> None:
        self.send_message(error, format="alert")

    def notify_status(self, info: Union[str, Dict[str, Any]]) -> None:
        self.send_message(info, format="status")

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────
    def _enabled(self) -> bool:
        if not self.token or not self.chat_id or self.chat_id == "0":
            logger.debug("Telegram not configured; skipping send.")
            return False
        return True

    def _format_payload(self, message: Union[str, Dict[str, Any]], fmt: str) -> str:
        if isinstance(message, str):
            return message
        # dict
        try:
            if fmt == "trade":
                # already rendered by _render_trade upstream
                return message.get("text") or json.dumps(message, ensure_ascii=False, indent=2)
            if fmt == "alert":
                return self._render_alert(message)
            # default
            return message.get("message") or json.dumps(message, ensure_ascii=False, indent=2)
        except Exception:
            return json.dumps(message, ensure_ascii=False)

    def _render_trade(self, event: Dict[str, Any]) -> Optional[str]:
        try:
            side = str(event.get("side", "")).upper()
            sym = event.get("symbol", "N/A")
            qty = event.get("qty", event.get("quantity", ""))
            price = event.get("price", event.get("entry_price", ""))
            status = event.get("status", "open")
            pnl = event.get("pnl")
            ret = event.get("return_pct")
            parts = [f"📊 <b>TRADE {side}</b>",
                     f"Pair: <code>{sym}</code>",
                     f"Amount: <code>{qty}</code>",
                     f"Price: <code>{price}</code>",
                     f"Status: <b>{status}</b>"]
            if pnl is not None:
                parts.append(f"PnL: <b>{float(pnl):.4f}</b>")
            if ret is not None:
                parts.append(f"Return: <b>{float(ret):.2f}%</b>")
            return "\n".join(parts)
        except Exception as e:
            logger.debug(f"_render_trade failed: {e}")
            return None

    def _render_alert(self, payload: Dict[str, Any]) -> str:
        try:
            msg = payload.get("message") or payload.get("error") or "Alert"
            ctx = payload.get("context")
            if isinstance(ctx, dict) and ctx:
                ctx_json = json.dumps(ctx, ensure_ascii=False)
                return f"⚠️ <b>ALERT</b>\n{msg}\n\n<code>{ctx_json}</code>"
            return f"⚠️ <b>ALERT</b>\n{msg}"
        except Exception:
            return f"⚠️ <b>ALERT</b>\n{str(payload)}"

    def _rate_allow(self) -> bool:
        with self._rl_lock:
            now = time.time()
            # drop old timestamps
            self._sent_timestamps = [t for t in self._sent_timestamps if now - t < self._rl_window]
            if len(self._sent_timestamps) >= self._rl_capacity:
                return False
            self._sent_timestamps.append(now)
            return True

    def _start_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def _run_worker(self):
        while True:
            item = None
            with self._q_lock:
                if self._q:
                    item = self._q.pop(0)
            if not item:
                time.sleep(0.1)
                continue
            if not self._rate_allow():
                time.sleep(1.0)
                continue
            payload = {
                "chat_id": self.chat_id,
                "text": item["text"],
                "disable_web_page_preview": True,
            }
            if self.parse_mode:
                payload["parse_mode"] = self.parse_mode
            try:
                requests.post(self.api_url, json=payload, timeout=self.timeout)
            except Exception:
                # swallow
                pass
