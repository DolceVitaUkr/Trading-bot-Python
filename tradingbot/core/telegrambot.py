from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path
import os, json
import httpx

import logging

import pandas as pd
from telegram import Bot, Update
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

from tradingbot.core.configmanager import config_manager

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Backwards-compatible notifier.

    - If token/chat_id are not provided, it will try to load them from:
      1) config JSON (default: tradingbot/config/config.json), keys:
         - top-level: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
         - or nested:  { "telegram": { "token": ..., "chat_id": ... } }
      2) environment variables: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
    - If still missing, it stays disabled and send_message_async becomes a no-op.
    """
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, timeout: float = 5.0, config_path: Optional[str] = None):
        self.timeout = timeout
        cfg_token = None
        cfg_chat = None
        # Resolve config path
        cand = []
        if config_path:
            cand.append(Path(config_path))
        # relative to package root
        cand.append(Path("tradingbot/config/config.json"))
        # relative to this file (../config/config.json)
        cand.append(Path(__file__).resolve().parents[1] / "config" / "config.json")
        cfg = {}
        for p in cand:
            try:
                if Path(p).exists():
                    cfg = json.loads(Path(p).read_text(encoding="utf-8"))
                    break
            except Exception:
                pass
        if cfg:
            cfg_token = cfg.get("TELEGRAM_TOKEN")
            cfg_chat = cfg.get("TELEGRAM_CHAT_ID")
            if not cfg_token or not cfg_chat:
                tg = cfg.get("telegram") or {}
                cfg_token = cfg_token or tg.get("token")
                cfg_chat = cfg_chat or tg.get("chat_id")
        env_token = os.getenv("TELEGRAM_TOKEN")
        env_chat = os.getenv("TELEGRAM_CHAT_ID")
        self.token = token or cfg_token or env_token
        self.chat_id = chat_id or cfg_chat or env_chat
        self.enabled = bool(self.token and self.chat_id)

    async def send_message_async(self, text: str, parse_mode: Optional[str] = "Markdown") -> Dict[str, Any]:
        if not self.enabled:
            return {"sent": False, "error": "not_configured"}
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if parse_mode: payload["parse_mode"] = parse_mode
            r = await client.post(url, json=payload)
            try:
                j = r.json()
            except Exception:
                j = {"ok": False, "description": r.text}
            if j.get("ok"):
                return {"sent": True, "response": j}
            payload.pop("parse_mode", None)
            r2 = await client.post(url, json=payload)
            try:
                j2 = r2.json()
                return {"sent": j2.get("ok", False), "response": j2, "error": None if j2.get("ok") else j.get("description")}
            except Exception:
                return {"sent": False, "error": r2.text}


class TelegramBot(TelegramNotifier):
    """
    Full-fledged bot with command handlers.
    (This part is optional if you only need one-way notifications).
    """

    def __init__(self, token: str = None, chat_id: str = None):
        """
        Initializes the TelegramBot.

        Args:
            token: The Telegram bot token.
            chat_id: The Telegram chat ID.
        """
        super().__init__(token, chat_id)
        if not self.enabled:
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
        if not self.enabled:
            return
        logger.info("Telegram bot is now polling...")
        await self.application.initialize()
        await self.application.start()
        await self.application.idle()

    def Notify_Telegram_Update(self, asset: str, strategy: str, validation_result: Dict[str, Any], 
                              result_path: str) -> Dict[str, Any]:
        """
        Send comprehensive Telegram notification about strategy validation results.
        
        Args:
            asset: Asset type (crypto_spot, crypto_futures, forex, forex_options)
            strategy: Strategy name/identifier
            validation_result: Complete validation package from Prepare_Validator_Package
            result_path: Path to result files
            
        Returns:
            Dictionary containing notification status and details
        """
        notification_result = {
            'sent': False,
            'message_content': '',
            'timestamp': pd.Timestamp.now().isoformat(),
            'error': None
        }
        
        try:
            # Extract key information from validation result
            final_verdict = validation_result.get('final_verdict', {})
            validation_scores = validation_result.get('validation_scores', {})
            key_metrics = validation_result.get('summary_report', {}).get('key_metrics', {})
            
            # Determine message status emoji and header
            approved = final_verdict.get('approved', False)
            status_emoji = "✅" if approved else "❌"
            status_text = "APPROVED" if approved else "REJECTED"
            
            # Build comprehensive message
            message_lines = [
                f"{status_emoji} **Strategy Validation {status_text}**",
                "",
                f"**Strategy:** {strategy}",
                f"**Asset:** {asset.replace('_', ' ').title()}",
                f"**Final Score:** {validation_scores.get('final_score', 0):.1%}",
                "",
                "📊 **Key Performance Metrics:**",
                f"• Sharpe Ratio: {key_metrics.get('sharpe_ratio', 0):.2f}",
                f"• Max Drawdown: {key_metrics.get('max_drawdown', 0):.1%}",
                f"• Profit Factor: {key_metrics.get('profit_factor', 0):.2f}",
                f"• Win Rate: {key_metrics.get('win_rate', 0):.1%}",
                "",
                "🎯 **Validation Scores:**",
                f"• KPI Score: {validation_scores.get('kpi_score', 0):.1%}",
                f"• Baseline Score: {validation_scores.get('baseline_score', 0):.1%}",
                f"• Robustness Score: {validation_scores.get('robustness_score', 0):.1%}",
                f"• Compliance Score: {validation_scores.get('compliance_score', 0):.1%}",
                ""
            ]
            
            # Add detailed reasoning
            reasoning = final_verdict.get('reasoning', 'No detailed reasoning provided')
            message_lines.extend([
                "💭 **Reasoning:**",
                reasoning,
                ""
            ])
            
            # Add critical information for rejected strategies
            if not approved:
                critical_failures = final_verdict.get('critical_failures', [])
                if critical_failures:
                    message_lines.extend([
                        "⚠️ **Critical Issues:**",
                        *[f"• {failure}" for failure in critical_failures],
                        ""
                    ])
                
                # Add specific validation failures
                validation_status = validation_result.get('validation_status', {})
                failed_checks = []
                for check, status in validation_status.items():
                    if not status.get('passes', True):
                        score = status.get('score', 0)
                        threshold = status.get('threshold', 0)
                        failed_checks.append(f"{check}: {score:.1%} < {threshold:.1%}")
                
                if failed_checks:
                    message_lines.extend([
                        "📉 **Failed Validations:**",
                        *[f"• {check}" for check in failed_checks],
                        ""
                    ])
            
            # Add approval status for approved strategies
            if approved:
                message_lines.extend([
                    "🚀 **Next Steps:**",
                    "• Strategy ready for live deployment",
                    "• All validation criteria satisfied",
                    "• Risk compliance verified",
                    ""
                ])
            
            # Add file information
            files_info = validation_result.get('files', {})
            message_lines.extend([
                "📁 **Files Generated:**",
                f"• Validation Package: `validation_package.json`",
                f"• Summary Report: `validation_summary.json`",
                f"• Result Path: `{result_path}`",
                "",
                f"⏰ **Completed:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            ])
            
            # Join message
            message_content = "\n".join(message_lines)
            notification_result['message_content'] = message_content
            
            # Send notification asynchronously
            import asyncio
            
            async def send_notification():
                await self.send_message_async(message_content, parse_mode="Markdown")
            
            # Try to send the message
            try:
                asyncio.create_task(send_notification())
                notification_result['sent'] = True
                logger.info(f"Telegram notification sent for strategy {strategy} ({asset})")
            except Exception as send_error:
                notification_result['error'] = str(send_error)
                logger.warning(f"Failed to send Telegram notification: {send_error}")
                
                # Fallback: try without markdown formatting
                try:
                    # Remove markdown formatting for fallback
                    plain_message = message_content.replace("**", "").replace("*", "").replace("`", "")
                    asyncio.create_task(self.send_message_async(plain_message))
                    notification_result['sent'] = True
                    notification_result['error'] = f"Sent as plain text due to formatting error: {send_error}"
                    logger.info(f"Telegram notification sent as plain text for strategy {strategy}")
                except Exception as fallback_error:
                    notification_result['error'] = f"Both markdown and plain text failed: {send_error}, {fallback_error}"
                    logger.error(f"Complete Telegram notification failure: {fallback_error}")
            
        except Exception as e:
            notification_result['error'] = str(e)
            logger.error(f"Error preparing Telegram notification: {e}")
        
        return notification_result
