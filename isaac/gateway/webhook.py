"""Generic webhook gateway — for WhatsApp, iMessage bridges, or custom integrations."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from isaac.gateway.base import GatewayAdapter, InboundMessage, OutboundMessage

log = logging.getLogger(__name__)


class WebhookGateway(GatewayAdapter):
    """HTTP webhook gateway. Receives POST /message, returns response.

    Works with:
    - WhatsApp via Twilio/Meta webhook
    - iMessage via BlueBubbles/Beeper bridge
    - Any HTTP-based chat provider
    """

    name = "webhook"

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._callback: Any = None
        self._server: Any = None

    async def start(self) -> None:
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise RuntimeError("Install flask: pip install 'isaac[gateway]'")

        # We use a simple aiohttp server instead of flask for async
        import asyncio
        from aiohttp import web

        app = web.Application()

        async def handle_message(req: web.Request) -> web.Response:
            try:
                body = await req.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON"}, status=400)

            inbound = InboundMessage(
                text=body.get("text", ""),
                sender_id=body.get("sender_id", "anonymous"),
                sender_name=body.get("sender_name", ""),
                channel=body.get("channel", "webhook"),
                thread_id=body.get("thread_id", ""),
                raw=body,
            )

            if not self._callback:
                return web.json_response({"error": "No handler registered"}, status=503)

            response = await self._callback(inbound)
            return web.json_response({"response": response})

        async def health(req: web.Request) -> web.Response:
            return web.json_response({"status": "ok", "gateway": "webhook"})

        app.router.add_post("/message", handle_message)
        app.router.add_get("/health", health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._server = runner
        log.info(f"Webhook gateway listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._server:
            await self._server.cleanup()

    async def send(self, msg: OutboundMessage) -> None:
        # Webhook is request/response — outbound happens in handle_message
        log.debug(f"Outbound message to {msg.channel}: {msg.text[:100]}")

    def on_message(self, callback: Any) -> None:
        self._callback = callback
