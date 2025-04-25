# modules/telegram_bot.py
# modules/telegram_bot.py
import logging
import config
import asyncio
import os
import threading
from datetime import datetime
from telegram import Bot, InputFile
from telegram.error import TelegramError, RetryAfter
from typing import Optional, Union, Dict, List

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, disable_async: bool = False):
        self._bot = None
        self._chat_id = None
        self._message_queue = asyncio.Queue()
        self._worker_task = None
        self._rate_limit_delay = 0
        self._max_retries = 3
        self._disable_async = disable_async
        self._shutting_down = False
        self._loop_thread = None
        self._loop = asyncio.new_event_loop()

        self._validate_config()
        
        if not self._disable_async:
            self._start_event_loop()

    def _start_event_loop(self):
        """Start dedicated event loop thread"""
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="TelegramEventLoop"
        )
        self._loop_thread.start()

    def _run_event_loop(self):
        """Run event loop in dedicated thread"""
        asyncio.set_event_loop(self._loop)
        try:
            self._worker_task = self._loop.create_task(self._message_consumer())
            self._loop.run_forever()
        finally:
            self._loop.close()

    def _validate_config(self):
        """Validate configuration with enhanced security checks"""
        required = {
            'TELEGRAM_BOT_TOKEN': config.TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': config.TELEGRAM_CHAT_ID
        }
        
        for key, value in required.items():
            # Allow numeric chat ID but ensure string type
            if not value or (not isinstance(value, (str, int))):
                logger.critical(f"Invalid Telegram config: {key}")
                raise ValueError(f"Invalid Telegram configuration: {key}")

        try:
            self._bot = Bot(token=str(config.TELEGRAM_BOT_TOKEN))
            self._chat_id = int(str(config.TELEGRAM_CHAT_ID))  # Convert to int here
        except ValueError as e:
            logger.critical(f"Invalid Telegram chat ID format: {str(e)}")
            raise
        except Exception as e:
            logger.critical(f"Telegram config validation failed: {str(e)}")
            raise

    async def _message_consumer(self):
        """Robust message consumer with improved error handling"""
        while not self._shutting_down:
            try:
                message, kwargs = await self._message_queue.get()
                await self._safe_send_message(message, **kwargs)
                await asyncio.sleep(self._rate_limit_delay)
                self._message_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Message consumer task cancelled")
                break
            except Exception as e:
                logger.error(f"Message consumer error: {str(e)}", exc_info=True)
                await asyncio.sleep(1)

    async def _safe_send_message(self, 
                               message: str, 
                               parse_mode: Optional[str] = None):
        """Enhanced send message with circuit breaker pattern"""
        retry_count = 0
        last_error = None
        
        while retry_count < self._max_retries and not self._shutting_down:
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )
                logger.debug(f"Message sent successfully: {message[:50]}...")
                return
            except RetryAfter as e:
                logger.warning(f"Rate limited, retrying in {e.retry_after}s")
                self._rate_limit_delay = e.retry_after
                await asyncio.sleep(e.retry_after)
            except TelegramError as e:
                last_error = e
                logger.warning(f"Telegram error ({str(e)}), retry {retry_count + 1}/{self._max_retries}")
                await asyncio.sleep(2 ** retry_count)
                retry_count += 1
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error sending message: {str(e)}", exc_info=True)
                break

        if last_error and not self._shutting_down:
            logger.error(f"Permanent send failure after {self._max_retries} attempts: {str(last_error)}")

    def send_message(self, content: Union[str, Dict], format: str = 'text'):
        """Thread-safe message submission with validation"""
        if self._shutting_down:
            logger.warning("Rejecting message during shutdown")
            return

        try:
            formatted = self._format_message(content, format)
            self._loop.call_soon_threadsafe(
                self._message_queue.put_nowait, 
                (formatted, {'parse_mode': 'HTML'})
            )
        except ValueError as e:
            logger.error(f"Invalid message format: {str(e)}")
        except Exception as e:
            logger.error(f"Message submission failed: {str(e)}", exc_info=True)

    async def graceful_shutdown(self):
        """Enhanced shutdown with queue draining"""
        logger.info("Initiating graceful shutdown...")
        self._shutting_down = True

        # Drain remaining messages
        try:
            while not self._message_queue.empty():
                message, kwargs = await self._message_queue.get()
                await self._safe_send_message(message, **kwargs)
                self._message_queue.task_done()
        except Exception as e:
            logger.error(f"Error draining queue: {str(e)}")

        # Cancel consumer task
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Close bot connection
        try:
            await self._bot.close()
        except Exception as e:
            logger.error(f"Error closing bot: {str(e)}")

        # Stop event loop
        if self._loop.is_running():
            self._loop.stop()

        logger.info("Shutdown completed successfully")

    def _format_message(self, content: Union[str, Dict], format: str) -> str:
        """Format messages based on type with strict format validation"""
        if format == 'log':
            if not isinstance(content, dict):
                raise ValueError("Log format requires dictionary content")
            required_keys = ['level', 'message']
            self._validate_content_keys(content, required_keys, 'log')
            return self._format_log_message(content)
        elif format == 'trade':
            if not isinstance(content, dict):
                raise ValueError("Trade format requires dictionary content")
            required_keys = ['symbol', 'side', 'amount', 'price']
            self._validate_content_keys(content, required_keys, 'trade')
            return self._format_trade_message(content)
        elif format == 'alert':
            if not isinstance(content, dict):
                raise ValueError("Alert format requires dictionary content")
            required_keys = ['type', 'message']
            self._validate_content_keys(content, required_keys, 'alert')
            return self._format_alert_message(content)
        elif format == 'text':
            return self._truncate_message(str(content))
        else:
            raise ValueError(f"Invalid message format: {format}")

    def _validate_content_keys(self, content: Dict, required_keys: List[str], format_name: str):
        """Validate required keys in content dictionary"""
        missing = [key for key in required_keys if key not in content]
        if missing:
            raise ValueError(f"Missing required keys for {format_name} format: {', '.join(missing)}")

    def _format_log_message(self, log_data: Dict) -> str:
        """Format structured log messages with level-specific emojis"""
        level_emojis = {
            'CRITICAL': '🔴',
            'ERROR': '🟠',
            'WARNING': '🟡',
            'INFO': '🔵',
            'DEBUG': '⚪'
        }
        emoji = level_emojis.get(log_data['level'].upper(), '⚪')
        tags = ' '.join([f'#{tag.strip()}' for tag in log_data.get('tags', '').split(',') if tag.strip()])
        
        return f"""
        {emoji} <b>{log_data['level'].upper()}</b> {emoji}
        ────────────────
        📅 <b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        📝 <b>Message:</b> {log_data['message']}
        🏷 <b>Tags:</b> {tags or 'None'}
        """.strip()

    def _format_trade_message(self, trade_data: Dict) -> str:
            """Format trade execution messages with calculated values"""
            total = float(trade_data['amount']) * float(trade_data['price'])
            return f"""
        📊 <b>TRADE EXECUTED</b> 📊
        ────────────────
        🪙 <b>Pair:</b> {trade_data['symbol']}
        📈 <b>Direction:</b> {trade_data['side'].upper()}
        💰 <b>Amount:</b> {float(trade_data['amount']):.8f}
        🔢 <b>Price:</b> {float(trade_data['price']):.8f}
        💵 <b>Total:</b> {total:.8f}
        ⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
        """.strip()

    def _format_alert_message(self, alert_data: Dict) -> str:
        """Format system alerts with type-specific formatting"""
        alert_emojis = {
            'INFO': '🔵',
            'WARNING': '🟠',
            'CRITICAL': '🔴',
            'SUCCESS': '🟢'
        }
        emoji = alert_emojis.get(alert_data['type'].upper(), '🔔')
        return f"""
{emoji} <b>{alert_data['type'].upper()} ALERT</b> {emoji}
────────────────
📌 <b>Message:</b> {alert_data['message']}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()

    async def send_file(self, file_path: str, caption: str = ''):
        """Send files with caption including existence and size check"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
            
        if os.path.getsize(file_path) > 50 * 1024 * 1024:  # 50MB limit
            logger.error(f"File too large (max 50MB): {file_path}")
            return

        try:
            with open(file_path, 'rb') as f:
                await self._bot.send_document(
                    chat_id=self._chat_id,
                    document=InputFile(f),
                    caption=self._truncate_message(caption, max_length=1000)
                )
        except Exception as e:
            logger.error(f"Error sending file: {str(e)}")

    def _truncate_message(self, text: str, max_length: int = 4000) -> str:
        """Smart truncation that preserves whole words"""
        if len(text) <= max_length:
            return text
            
        # Find the last space within the limit
        truncated = text[:max_length-3]
        if ' ' in truncated:
            truncated = truncated.rsplit(' ', 1)[0]
        return truncated + '...'

    async def test_connection(self) -> bool:
        """Verify bot connectivity"""
        try:
            await self._bot.get_me()
            logger.info("Telegram connection test successful")
            return True
        except Exception as e:
            logger.error(f"Telegram connection failed: {str(e)}")
            return False

    async def graceful_shutdown(self):
        """Final async cleanup"""
        logger.info("Shutting down Telegram notifier...")
        try:
            # Cancel all tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Close connections
            await self._bot.close()
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")
        finally:
            logger.info("Telegram notifier stopped")

    def __del__(self):
        """Safety net for resource cleanup"""
        if not self._shutting_down:
            logger.warning("Telegram Notifier destroyed without graceful shutdown!")
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

        
if __name__ == "__main__":
    async def main():
        notifier = TelegramNotifier()
        try:
                await notifier.test_connection()
                
                notifier.send_message({
                    'type': 'INFO',
                    'message': 'System startup completed'
                }, format='alert')
                
                await asyncio.sleep(1)  # Allow time for message processing
        finally:
                await notifier.graceful_shutdown()
    
    
    asyncio.run(main())