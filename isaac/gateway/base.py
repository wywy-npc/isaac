"""Gateway adapter base — interface for chat providers (Telegram, WhatsApp, etc.)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class InboundMessage:
    text: str
    sender_id: str
    sender_name: str = ""
    channel: str = ""
    thread_id: str = ""
    attachments: list[str] | None = None
    raw: dict[str, Any] | None = None


@dataclass
class OutboundMessage:
    text: str
    channel: str = ""
    thread_id: str = ""
    reply_to: str = ""


class GatewayAdapter(ABC):
    """Base class for chat provider gateways."""

    name: str = "base"

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the gateway."""

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to the chat provider."""

    @abstractmethod
    def on_message(self, callback: Any) -> None:
        """Register a callback for incoming messages: async (InboundMessage) -> str."""
