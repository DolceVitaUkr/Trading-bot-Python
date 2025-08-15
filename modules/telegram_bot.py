# modules/telegram_bot.py

import json
import logging
from typing import Dict, Any

from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext, filters, MessageHandler

import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Minimalist Telegram bot for sending notifications.
    - Supports synchronous and asynchronous sending.
    - Can be used directly or within a NotificationManager.
    """

    def __init__(
        self,
        token: str = config.TELEGRAM_BOT_TOKEN,
        chat_id: str = config.TELEGRAM_CHAT_ID,
        disable_async: bool = False
    ):
        self.token = token
        self.chat_id = chat_id
        if not self.token or not self.chat_id:
            logger.warning("[Telegram] bot token or chat ID not set; notifications will be disabled.")
            self.bot = None
            return

        self.bot = Bot(token=self.token)
        self.use_async = not disable_async

    def send_message_sync(self, text: str, format: str = "text"):
        if not self.bot:
            return
        try:
            # Simple formatter for dicts
            if isinstance(text, dict):
                text = json.dumps(text, indent=2)
            self.bot.send_message(chat_id=self.chat_id, text=str(text))
        except Exception as e:
            logger.error(f"[Telegram] Failed to send sync message: {e}")

    async def send_message_async(self, text: str, format: str = "text"):
        if not self.bot:
            return
        try:
            if isinstance(text, dict):
                text = json.dumps(text, indent=2)
            await self.bot.send_message(chat_id=self.chat_id, text=str(text))
        except Exception as e:
            logger.error(f"[Telegram] Failed to send async message: {e}")


class TelegramBot(TelegramNotifier):
    """
    Full-fledged bot with command handlers.
    (This part is optional if you only need one-way notifications).
    """

    def __init__(self, token: str, chat_id: str):
        super().__init__(token, chat_id)
        if not self.bot:
            return
        self.updater = Updater(token=self.token, use_context=True)
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(CommandHandler("help", self.help))
        dp.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.echo))
        dp.add_error_handler(self.error)

    def start(self, update: Update, context: CallbackContext):
        update.message.reply_text("Bot started! Use /help for commands.")

    def help(self, update: Update, context: CallbackContext):
        update.message.reply_text("Available commands: /start, /help, /status")

    def echo(self, update: Update, context: CallbackContext):
        update.message.reply_text(f"Echo: {update.message.text}")

    def error(self, update: Update, context: CallbackContext):
        logger.warning(f'Update "{update}" caused error "{context.error}"')

    def run(self):
        if not self.bot:
            return
        logger.info("Telegram bot is now polling...")
        self.updater.start_polling()
        self.updater.idle()
