"""Native Anthropic LLM client — preserves prompt caching, extended thinking, betas.

Use this for Anthropic models (Haiku, Sonnet, Opus). For everything else, use LiteLLMClient.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import anthropic

from isaac.core.config import get_env
from isaac.core.llm import LLMResult, StreamEvent


class AnthropicClient:
    """LLMClient implementation wrapping anthropic.AsyncAnthropic.

    Extracted from orchestrator.py — all streaming logic, usage tracking, cache handling.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or get_env("ANTHROPIC_API_KEY"),
        )

    async def create_stream(
        self,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        betas: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent | LLMResult]:
        """Stream an Anthropic LLM call. Yields StreamEvents, then a final LLMResult."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if betas:
            kwargs["betas"] = betas

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        current_tool_input = ""
        current_tool_id = ""
        current_tool_name = ""

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            current_tool_id = event.content_block.id
                            current_tool_name = event.content_block.name
                            current_tool_input = ""
                            yield StreamEvent(
                                type="tool_use_start",
                                tool_id=current_tool_id,
                                tool_name=current_tool_name,
                            )
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        text_parts.append(event.delta.text)
                        yield StreamEvent(type="text_delta", text=event.delta.text)
                    elif hasattr(event.delta, "partial_json"):
                        current_tool_input += event.delta.partial_json
                        yield StreamEvent(
                            type="tool_input_delta",
                            tool_input_json=event.delta.partial_json,
                        )
                elif event.type == "content_block_stop":
                    if current_tool_name:
                        try:
                            tool_input = json.loads(current_tool_input) if current_tool_input else {}
                        except json.JSONDecodeError:
                            tool_input = {}
                        tool_calls.append({
                            "id": current_tool_id,
                            "name": current_tool_name,
                            "input": tool_input,
                        })
                        yield StreamEvent(type="tool_use_end", tool_id=current_tool_id)
                        current_tool_name = ""
                        current_tool_input = ""
                        current_tool_id = ""

            response = await stream.get_final_message()

        stop = "end_turn"
        if response.stop_reason == "tool_use":
            stop = "tool_use"
        elif response.stop_reason == "max_tokens":
            stop = "max_tokens"

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0

        yield LLMResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )

    async def create(
        self,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        betas: list[str] | None = None,
    ) -> LLMResult:
        """Non-streaming Anthropic call."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if betas:
            kwargs["betas"] = betas

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        stop = "end_turn"
        if response.stop_reason == "tool_use":
            stop = "tool_use"
        elif response.stop_reason == "max_tokens":
            stop = "max_tokens"

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0

        return LLMResult(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )
