"""LLM client interface — Protocol that any provider must implement.

The orchestrator talks to this interface, never to anthropic/openai/litellm directly.
Implementations: AnthropicClient (native, prompt caching), LiteLLMClient (universal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


@dataclass
class StreamEvent:
    """A single event from an LLM stream."""
    type: str  # "text_delta", "tool_use_start", "tool_input_delta", "tool_use_end", "done"
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    tool_input_json: str = ""


@dataclass
class LLMResult:
    """Final result from an LLM call (streamed or non-streamed)."""
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"id", "name", "input"}]
    stop_reason: str = "end_turn"  # "end_turn", "tool_use", "max_tokens"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class LLMClient(Protocol):
    """Protocol any LLM provider must implement."""

    async def create_stream(
        self,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        betas: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent | LLMResult]:
        """Stream an LLM call. Yields StreamEvents, then a final LLMResult."""
        ...

    async def create(
        self,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        betas: list[str] | None = None,
    ) -> LLMResult:
        """Non-streaming LLM call. Used for summarization, simple queries."""
        ...
