import json
import logging

from telegram import Bot, Update
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

from trading_bot.core.configmanager import config_manager

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Minimalist Telegram bot for sending notifications.
    - Supports synchronous and asynchronous sending.
    - Can be used directly or within a NotificationManager.
    """

    def __init__(
        self,
        token: str = config_manager.get_config().get('telegram', {}).get('token'),
        chat_id: str = config_manager.get_config().get('telegram', {}).get('chat_id'),
        disable_async: bool = False
    ):
        """
        Initializes the TelegramNotifier.

        Args:
            token: The Telegram bot token.
            chat_id: The Telegram chat ID to send notifications to.
            disable_async: Whether to disable asynchronous sending.
        """
        self.token = token
        self.chat_id = chat_id
        if not self.token or not self.chat_id:
            logger.warning(
                "[Telegram] bot token or chat ID not set; "
                "notifications will be disabled.")
            self.bot = None
            return

        self.bot = Bot(token=self.token)
        self.use_async = not disable_async

    async def send_message_async(self, text: str, format: str = "text"):
        """
        Sends a message asynchronously.

        Args:
            text: The text of the message to send.
            format: The format of the message.
        """
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
        """
        Initializes the TelegramBot.

        Args:
            token: The Telegram bot token.
            chat_id: The Telegram chat ID.
        """
        super().__init__(token, chat_id)
        if not self.bot:
            return
        self.application = Application.builder().token(token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self.echo))
        self.application.add_error_handler(self.error)

    async def start(self, update: Update, context: CallbackContext):
        """
        Sends a message when the command /start is issued.
        """
        if update.message:
            await update.message.reply_text("Bot started! Use /help for commands.")

    async def help(self, update: Update, context: CallbackContext):
        """
        Sends a message when the command /help is issued.
        """
        if update.message:
            await update.message.reply_text(
                "Available commands: /start, /help, /status")

    async def echo(self, update: Update, context: CallbackContext):
        """
        Echo the user message.
        """
        if update.message:
            await update.message.reply_text(f"Echo: {update.message.text}")

    async def error(self, update: Update, context: CallbackContext):
        """
        Log Errors caused by Updates.
        """
        logger.warning(f'Update "{update}" caused error "{context.error}"')

    async def run(self):
        """
        Run the bot.
        """
        if not self.bot:
            return
        logger.info("Telegram bot is now polling...")
        await self.application.initialize()
        await self.application.start()
        await self.application.idle()
