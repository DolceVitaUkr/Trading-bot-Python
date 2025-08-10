# modules/telegram_bot.py

import logging
import json
from typing import Union, Dict, Any, Optional

import config

try:
    import requests
except Exception:  # optional dependency
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


class TelegramNotifier:
    """
    Minimal Telegram notifier (sync by default).
    Set config.ASYNC_TELEGRAM to True to allow async in future (not used here).
    """

    def __init__(self, disable_async: bool = True):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        if not self.token or not self.chat_id:
            logger.info("Telegram credentials missing; notifier disabled.")
        self.enabled = bool(self.token and self.chat_id)

    def _post(self, text: str) -> None:
        if not self.enabled:
            logger.debug(f"[telegram] (disabled) {text}")
            return
        if requests is None:
            logger.warning("requests not available; cannot send Telegram messages")
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code != 200:
                logger.warning(f"Telegram send failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.warning(f"Telegram send exception: {e}")

    def send_message_sync(self, text: Union[str, Dict[str, Any]], *, format: str = "text"):
        if isinstance(text, dict):
            pretty = json.dumps(text, indent=2, ensure_ascii=False)
            self._post(f"<pre>{pretty}</pre>")
        else:
            self._post(str(text))
