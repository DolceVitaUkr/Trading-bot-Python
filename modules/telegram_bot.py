# modules/telegram_bot.py

import logging
import asyncio
import os
from datetime import datetime
from typing import Optional, Union, Dict, List

import config
from telegram import Bot, InputFile
from telegram.error import TelegramError, RetryAfter

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class TelegramNotifier:
    def __init__(self, disable_async: bool = False):
        """
        :param disable_async: If True, sends messages synchronously; otherwise, uses background event loop.
        """
        self._disable_async = disable_async
        self._bot: Bot
        self._chat_id: int
        self._loop: asyncio.AbstractEventLoop
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._rate_limit_delay = 0.0
        self._max_retries = 3

        self._validate_config()
        if not self._disable_async:
            # start dedicated event loop
            self._loop = asyncio.new_event_loop()
            self._thread = asyncio.Thread(target=self._run_event_loop, daemon=True)
            self._thread.start()
        else:
            # use default event loop for synchronous sends
            self._loop = asyncio.get_event_loop()

    def _validate_config(self):
        token = getattr(config, 'TELEGRAM_BOT_TOKEN', None)
        chat = getattr(config, 'TELEGRAM_CHAT_ID', None)
        if not token or not chat:
            logger.critical("Telegram token or chat ID not configured")
            raise ValueError("Invalid Telegram configuration")
        self._bot = Bot(token=str(token))
        try:
            self._chat_id = int(chat)
        except Exception as e:
            logger.critical(f"Invalid chat ID: {e}")
            raise

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._worker_task = self._loop.create_task(self._message_consumer())
        self._loop.run_forever()

    async def _message_consumer(self):
        """
        Consume queued messages and send them with rate-limit handling.
        """
        while not self._shutting_down:
            message, kwargs = await self._message_queue.get()
            await self._safe_send_message(message, **kwargs)
            await asyncio.sleep(self._rate_limit_delay)
            self._message_queue.task_done()

    async def _safe_send_message(self, message: str, parse_mode: str = 'HTML'):
        """
        Try sending a message with retries and rate-limit handling.
        """
        retry = 0
        last_error = None
        while retry < self._max_retries and not self._shutting_down:
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )
                return
            except RetryAfter as e:
                self._rate_limit_delay = e.retry_after
                await asyncio.sleep(e.retry_after)
            except TelegramError as e:
                last_error = e
                await asyncio.sleep(2 ** retry)
                retry += 1
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected send error: {e}")
                break
        if last_error:
            logger.error(f"Failed to send message after retries: {last_error}")

    def send_message(self, content: Union[str, Dict], format: str = 'text'):
        """
        Public method to send a message. In async mode, queues it; in sync mode, sends immediately.
        :param content: message content or structured dict
        :param format: 'text', 'log', 'trade', or 'alert'
        """
        formatted, kwargs = self._prepare_message(content, format)
        if self._disable_async:
            try:
                result = self._safe_send_message(formatted, **kwargs)
                if asyncio.iscoroutine(result):
                    asyncio.run(result)
            except Exception as e:
                logger.error(f"Synchronous send failed: {e}")
        else:
            self._loop.call_soon_threadsafe(
                self._message_queue.put_nowait, (formatted, kwargs)
            )

    def _prepare_message(self, content: Union[str, Dict], fmt: str):
        """
        Format content according to specified template and return (message, kwargs).
        """
        if fmt == 'text':
            msg = str(content)
        elif fmt == 'log':
            msg = self._format_log(content)
        elif fmt == 'trade':
            msg = self._format_trade(content)
        elif fmt == 'alert':
            msg = self._format_alert(content)
        else:
            raise ValueError(f"Unknown format: {fmt}")
        return msg, {'parse_mode': 'HTML'}

    def _format_log(self, data: Dict) -> str:
        if not isinstance(data, dict) or 'level' not in data or 'message' not in data:
            raise ValueError("Log format requires {'level','message'} keys")
        level = data['level'].upper()
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{ts}] {level}: {data['message']}"

    def _format_trade(self, data: Dict) -> str:
        if not isinstance(data, dict) or not all(k in data for k in ('symbol','side','amount','price')):
            raise ValueError("Trade format requires {'symbol','side','amount','price'} keys")
        total = float(data['amount']) * float(data['price'])
        ts = datetime.now().strftime('%H:%M:%S')
        return (
            f"TRADE {data['side'].upper()} {data['symbol']}"
            f" qty={data['amount']:.6f} @ {data['price']:.2f}"
            f" total={total:.2f} time={ts}"
        )

    def _format_alert(self, data: Dict) -> str:
        if not isinstance(data, dict) or 'message' not in data:
            raise ValueError("Alert format requires {'message'} key")
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"ALERT: {data['message']} at {ts}"

    async def send_file(self, file_path: str, caption: str = ''):
        """
        Send a file via Telegram, enforcing size limits.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
        if os.path.getsize(file_path) > 50 * 1024 * 1024:
            logger.error(f"File too large (>50MB): {file_path}")
            return
        await self._bot.send_document(
            chat_id=self._chat_id,
            document=InputFile(file_path),
            caption=caption[:1000]
        )

    async def test_connection(self) -> bool:
        """
        Test bot connectivity with get_me().
        """
        try:
            await self._bot.get_me()
            return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False

    async def graceful_shutdown(self):
        """
        Gracefully stop consumer and close the Bot.
        """
        self._shutting_down = True
        if not self._disable_async:
            # Drain queue
            await self._message_queue.join()
            if self._worker_task:
                self._worker_task.cancel()
            self._loop.stop()
        try:
            await self._bot.close()
        except Exception as e:
            logger.error(f"Error during Telegram shutdown: {e}")
