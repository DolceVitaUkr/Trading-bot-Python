# modules/telegram_bot.py

import os
import asyncio
import threading
import logging
from datetime import datetime, timezone
from typing import Optional, Union, Dict

from telegram import Bot, InputFile
from telegram.error import TelegramError, RetryAfter

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TelegramNotifier:
    """
    Unified Telegram notification client.

    - Reads credentials from config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID.
    - Async queue by default (non-blocking); can be forced into sync mode with disable_async=True.
    - send_message()           → enqueues (async) or sends (sync).
    - send_message_sync()      → blocking send (use for critical alerts).
    - send_file()              → send a document/image with optional caption.
    - graceful_shutdown()      → stop worker & close session.

    Notes:
    * Works with python-telegram-bot v13/v20+. We detect if send_message is coroutine and await accordingly.
    * No silent failures: all exceptions are logged; 3 retries with exponential backoff; RetryAfter respected.
    """

    def __init__(self, disable_async: bool = False):
        token = os.getenv("TELEGRAM_BOT_TOKEN", config.TELEGRAM_BOT_TOKEN)
        chat_id = os.getenv("TELEGRAM_CHAT_ID", str(config.TELEGRAM_CHAT_ID))

        if not token or not chat_id:
            raise ValueError("Telegram bot token and chat ID must be set in config or environment")

        self._bot = Bot(token=token)
        self._chat_id = int(chat_id)
        self._disable_async = bool(disable_async)

        # Async infra
        self._shutdown = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None

        if not self._disable_async:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    # ──────────────────────────────────────────────────────────────
    # Event loop & worker
    # ──────────────────────────────────────────────────────────────
    def _run_loop(self):
        """Owns an event loop in a background thread and runs a consumer task."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._queue = asyncio.Queue()
            self._worker = self._loop.create_task(self._consumer())
            self._loop.run_forever()
        except Exception:
            logger.exception("Telegram notifier event loop crashed")
        finally:
            try:
                if self._worker and not self._worker.done():
                    self._worker.cancel()
                pending = asyncio.all_tasks(loop=self._loop) if self._loop else set()
                for t in pending:
                    t.cancel()
                if self._loop and not self._loop.is_closed():
                    self._loop.stop()
                    self._loop.close()
            except Exception:
                pass

    async def _consumer(self):
        """Process queued messages one-by-one, honoring rate limits."""
        assert self._queue is not None
        while not self._shutdown:
            payload = await self._queue.get()
            try:
                if isinstance(payload, dict) and payload.get("_kind") == "file":
                    await self._do_send_file(payload["path"], payload.get("caption", ""))
                else:
                    await self._do_send_text(payload["text"], payload.get("parse_mode"))
            finally:
                self._queue.task_done()
                # Small delay to be nice to Telegram; real rate limits handled via RetryAfter.
                await asyncio.sleep(0.5)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────
    def send_message(self, content: Union[str, Dict], format: str = "text"):
        """
        Enqueue a message (async mode) or send immediately (sync mode).
        format: 'text' | 'log' | 'trade' | 'alert'
        """
        text = self._format(content, format)
        text = self._truncate(text)

        if self._disable_async:
            self.send_message_sync(text, format="text")  # already formatted
            return

        if not self._loop or not self._queue:
            logger.warning("Telegram notifier loop not started; falling back to sync send")
            self.send_message_sync(text, format="text")
            return

        self._loop.call_soon_threadsafe(self._queue.put_nowait, {"text": text, "parse_mode": "HTML"})

    def send_message_sync(self, content: Union[str, Dict], format: str = "text"):
        """Blocking send with retry. Safe to call from any thread."""
        text = content if isinstance(content, str) else self._format(content, format)
        text = self._truncate(text)
        self._run_blocking(self._do_send_text(text, parse_mode="HTML"))

    def send_file(self, file_path: str, caption: str = ""):
        """
        Send a file (document). In async mode, enqueue. In sync, block.
        """
        if not os.path.exists(file_path):
            logger.error("File to send not found: %s", file_path)
            return

        if self._disable_async or not self._loop or not self._queue:
            self._run_blocking(self._do_send_file(file_path, caption))
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, {"_kind": "file", "path": file_path, "caption": caption})

    def graceful_shutdown(self):
        """Stop background worker and close HTTP session."""
        self._shutdown = True
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)
        try:
            # Close underlying HTTP session if present (best-effort).
            sess = getattr(self._bot, "session", None)
            if sess:
                sess.close()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # Internal send helpers (awaitable)
    # ──────────────────────────────────────────────────────────────
    async def _do_send_text(self, text: str, parse_mode: Optional[str] = None):
        """Send text with retry & RetryAfter handling."""
        for attempt in range(3):
            try:
                maybe_coro = self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode or "HTML",
                    disable_web_page_preview=True,
                )
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
                # v13 blocking returns Message; v20+ returns coroutine awaited above
                return
            except RetryAfter as e:
                delay = getattr(e, "retry_after", 1)
                logger.warning("Telegram rate limited; sleeping %ss", delay)
                await asyncio.sleep(delay)
            except TelegramError as e:
                logger.warning("Telegram API error on send (attempt %d/3): %s", attempt + 1, e)
                await asyncio.sleep(2 ** attempt)
            except Exception:
                logger.exception("Unexpected Telegram error")
                break
        logger.error("Failed to send Telegram message after retries")

    async def _do_send_file(self, file_path: str, caption: str = ""):
        """Send document with retry logic."""
        for attempt in range(3):
            try:
                maybe_coro = self._bot.send_document(
                    chat_id=self._chat_id,
                    document=InputFile(file_path),
                    caption=self._truncate(caption, 1000),
                )
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
                return
            except RetryAfter as e:
                delay = getattr(e, "retry_after", 1)
                logger.warning("Telegram rate limited (file); sleeping %ss", delay)
                await asyncio.sleep(delay)
            except TelegramError as e:
                logger.warning("Telegram API error on file send (attempt %d/3): %s", attempt + 1, e)
                await asyncio.sleep(2 ** attempt)
            except Exception:
                logger.exception("Unexpected Telegram error (file)")
                break
        logger.error("Failed to send Telegram file after retries")

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────
    def _format(self, content: Union[str, Dict], fmt: str) -> str:
        """Prepare message text based on a simple schema."""
        if fmt == "text" or isinstance(content, str):
            return str(content)

        if fmt == "log" and isinstance(content, dict):
            lvl = content.get("level", "INFO").upper()
            msg = content.get("message", "")
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            return f"<b>{lvl}</b> [{ts}]<br/>{msg}"

        if fmt == "trade" and isinstance(content, dict):
            return (
                f"📊 <b>TRADE {content.get('side','').upper()}</b>\n"
                f"Pair: <code>{content.get('symbol')}</code>\n"
                f"Amount: <code>{content.get('amount')}</code>\n"
                f"Price: <code>{content.get('price')}</code>\n"
                f"Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}"
            )

        if fmt == "alert" and isinstance(content, dict):
            typ = content.get("type", "ALERT").upper()
            msg = content.get("message", "")
            return f"🚨 <b>{typ}</b> 🚨\n{msg}"

        # Fallback
        return str(content)

    def _truncate(self, text: str, max_length: int = 4000) -> str:
        """Truncate to Telegram limits (preserve word boundary)."""
        if len(text) <= max_length:
            return text
        cut = text[: max_length - 3]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return cut + "..."

    def _run_blocking(self, coro: "asyncio.Future"):
        """Run an awaitable to completion from any context."""
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Run in a private loop in a temp thread to avoid interfering with caller's loop
                result_holder = {}

                def _worker():
                    priv = asyncio.new_event_loop()
                    asyncio.set_event_loop(priv)
                    try:
                        result_holder["result"] = priv.run_until_complete(coro)
                    finally:
                        priv.close()

                t = threading.Thread(target=_worker, daemon=True)
                t.start()
                t.join()
                return result_holder.get("result")
            else:
                # No running loop in this thread
                return asyncio.run(coro)
        except Exception:
            logger.exception("Failed to run blocking Telegram operation")
            return None
