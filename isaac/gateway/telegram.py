"""Telegram gateway adapter."""
from __future__ import annotations

import logging
import os
from typing import Any

from isaac.gateway.base import GatewayAdapter, InboundMessage, OutboundMessage

log = logging.getLogger(__name__)


class TelegramGateway(GatewayAdapter):
    """Telegram Bot API gateway."""

    name = "telegram"

    def __init__(self) -> None:
        self._callback: Any = None
        self._app: Any = None

    async def start(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import ApplicationBuilder, MessageHandler, filters
        except ImportError:
            raise RuntimeError("Install python-telegram-bot: pip install 'isaac[gateway]'")

        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var")

        self._app = ApplicationBuilder().token(token).build()

        async def handle_message(update: Update, context: Any) -> None:
            if not update.message or not update.message.text:
                return
            if not self._callback:
                return

            inbound = InboundMessage(
                text=update.message.text,
                sender_id=str(update.effective_user.id) if update.effective_user else "",
                sender_name=update.effective_user.first_name if update.effective_user else "",
                channel=str(update.effective_chat.id) if update.effective_chat else "",
                thread_id=str(update.message.message_thread_id or ""),
            )

            response = await self._callback(inbound)
            if response and update.message:
                # Split long messages (Telegram 4096 char limit)
                for i in range(0, len(response), 4000):
                    await update.message.reply_text(response[i:i + 4000])

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        log.info("Starting Telegram gateway...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send(self, msg: OutboundMessage) -> None:
        if self._app and msg.channel:
            await self._app.bot.send_message(chat_id=msg.channel, text=msg.text)

    def on_message(self, callback: Any) -> None:
        self._callback = callback
