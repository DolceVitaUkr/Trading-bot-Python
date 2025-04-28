# modules/telegram_bot.py

import os
import asyncio
import threading
import logging
from datetime import datetime
from typing import Optional, Union, Dict, List

from telegram import Bot, InputFile
from telegram.error import TelegramError, RetryAfter

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TelegramNotifier:
    """
    Unified Telegram notification client.

    - Reads credentials from config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID.
    - Supports async sending by default; can be forced into sync mode.
    - Exposes send_message() for async use and send_message_sync() for blocking calls.
    - Queues messages to respect rate limits and retries transient failures.
    """

    def __init__(self, disable_async: bool = False):
        token = os.getenv("TELEGRAM_BOT_TOKEN", config.TELEGRAM_BOT_TOKEN)
        chat_id = os.getenv("TELEGRAM_CHAT_ID", config.TELEGRAM_CHAT_ID)
        if not token or not chat_id:
            raise ValueError("Telegram bot token and chat ID must be set in config or environment")

        self._bot = Bot(token=token)
        self._chat_id = int(chat_id)
        self._disable_async = disable_async

        # Async machinery
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue = asyncio.Queue(loop=self._loop)
        self._worker: Optional[asyncio.Task] = None
        self._shutdown = False

        if not self._disable_async:
            self._thread = threading.Thread(target=self._start_loop, daemon=True)
            self._thread.start()
            # Schedule the consumer
            def _schedule_consumer():
                self._worker = self._loop.create_task(self._consumer())
            self._loop.call_soon_threadsafe(_schedule_consumer)

    def _start_loop(self):
        """Run the asyncio event loop in its own thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _consumer(self):
        """Process the message queue, sending messages one by one."""
        while not self._shutdown:
            msg, kwargs = await self._queue.get()
            await self._do_send(msg, **kwargs)
            self._queue.task_done()
            # Respect Telegram rate limits
            await asyncio.sleep(1)

    async def _do_send(self, text: str, parse_mode: Optional[str] = None):
        """Actual send call with retry logic."""
        for attempt in range(3):
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )
                return
            except RetryAfter as e:
                logger.warning(f"Telegram rate limit, sleeping {e.retry_after}s")
                await asyncio.sleep(e.retry_after)
            except TelegramError as e:
                logger.warning(f"Telegram error on send: {e} (attempt {attempt+1}/3)")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected Telegram error: {e}", exc_info=True)
                break
        logger.error("Failed to send Telegram message after retries")

    def send_message(self, content: Union[str, Dict], format: str = 'text'):
        """
        Enqueue a message for sending.

        :param content: str or dict (for structured formats 'log', 'trade', 'alert')
        :param format: 'text' | 'log' | 'trade' | 'alert'
        """
        if self._disable_async:
            # In sync mode, delegate to blocking send
            self.send_message_sync(content, format=format)
            return

        # Prepare text
        text = self._format(content, format)
        # Truncate if necessary
        text = self._truncate(text)
        # Put in queue
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (text, {"parse_mode": "HTML"}))

    def send_message_sync(self, content: Union[str, Dict], format: str = 'text'):
        """
        Blocking send (useful for critical alerts).
        """
        text = self._format(content, format)
        text = self._truncate(text)
        try:
            self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Synchronous Telegram send failed: {e}", exc_info=True)

    def send_file(self, file_path: str, caption: str = ""):
        """
        Send a file (document) to Telegram.
        """
        if not os.path.exists(file_path):
            logger.error(f"File to send not found: {file_path}")
            return
        size = os.path.getsize(file_path)
        if size > 50 * 1024 * 1024:
            logger.error(f"File too large to send: {file_path}")
            return

        async def _send():
            try:
                await self._bot.send_document(
                    chat_id=self._chat_id,
                    document=InputFile(file_path),
                    caption=self._truncate(caption, max_length=1000)
                )
            except Exception as e:
                logger.error(f"Telegram send_file failed: {e}", exc_info=True)

        if self._disable_async:
            # Blocking
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_send())
            loop.close()
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (_send, {}))

    def _format(self, content: Union[str, Dict], format: str) -> str:
        """
        Prepare the message text based on format.
        """
        if format == 'text':
            return str(content)
        elif format == 'log' and isinstance(content, dict):
            lvl = content.get('level', 'INFO').upper()
            msg = content.get('message', '')
            ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            return f"<b>{lvl}</b> [{ts}]\n{msg}"
        elif format == 'trade' and isinstance(content, dict):
            return (
                f"📊 <b>TRADE {content.get('side','').upper()}</b>\n"
                f"Pair: {content.get('symbol')}\n"
                f"Amount: {content.get('amount')}\n"
                f"Price: {content.get('price')}\n"
                f"Time: {datetime.utcnow().strftime('%H:%M:%S')}"
            )
        elif format == 'alert' and isinstance(content, dict):
            typ = content.get('type', 'ALERT').upper()
            msg = content.get('message', '')
            return f"🚨 <b>{typ}</b> 🚨\n{msg}"
        else:
            raise ValueError(f"Unknown telegram format '{format}' or bad content")

    def _truncate(self, text: str, max_length: int = 4000) -> str:
        """
        Truncate text to Telegram limit, preserving words.
        """
        if len(text) <= max_length:
            return text
        cut = text[: max_length - 3]
        if ' ' in cut:
            cut = cut.rsplit(' ', 1)[0]
        return cut + "..."

    def graceful_shutdown(self):
        """
        Stop the background consumer and close bot.
        """
        self._shutdown = True
        if self._worker:
            self._worker.cancel()
        if not self._disable_async:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1)
        try:
            self._bot.session.close()
        except Exception:
            pass
