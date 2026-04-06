"""Universal LLM client via litellm — supports 100+ models.

Use for: OpenAI, Gemini, Mistral, local (ollama), AWS Bedrock, GCP Vertex, etc.
For Anthropic models, prefer AnthropicClient (preserves prompt caching).
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from isaac.core.llm import LLMResult, StreamEvent


class LiteLLMClient:
    """LLMClient implementation via litellm. Universal model access."""

    def __init__(self, **default_kwargs: Any) -> None:
        self._defaults = default_kwargs

    async def create_stream(
        self,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
        betas: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent | LLMResult]:
        """Stream via litellm. Translates to OpenAI-format streaming."""
        import litellm

        # Convert Anthropic-style system prompt to OpenAI-style
        oai_messages = _to_openai_messages(system, messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "stream": True,
            **self._defaults,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        text_parts: list[str] = []
        tool_calls_by_idx: dict[int, dict[str, Any]] = {}

        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Text content
            if delta.content:
                text_parts.append(delta.content)
                yield StreamEvent(type="text_delta", text=delta.content)

            # Tool calls (OpenAI format: indexed, streamed)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_by_idx:
                        tool_calls_by_idx[idx] = {
                            "id": tc.id or f"tc_{idx}",
                            "name": "",
                            "input_json": "",
                        }
                        if tc.function and tc.function.name:
                            tool_calls_by_idx[idx]["name"] = tc.function.name
                            yield StreamEvent(
                                type="tool_use_start",
                                tool_id=tool_calls_by_idx[idx]["id"],
                                tool_name=tc.function.name,
                            )
                    if tc.function and tc.function.arguments:
                        tool_calls_by_idx[idx]["input_json"] += tc.function.arguments

        # Build final tool calls
        final_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls_by_idx):
            tc_data = tool_calls_by_idx[idx]
            try:
                parsed_input = json.loads(tc_data["input_json"]) if tc_data["input_json"] else {}
            except json.JSONDecodeError:
                parsed_input = {}
            final_tool_calls.append({
                "id": tc_data["id"],
                "name": tc_data["name"],
                "input": parsed_input,
            })

        stop = "tool_use" if final_tool_calls else "end_turn"

        # litellm doesn't give granular token counts in streaming, estimate
        text = "".join(text_parts)
        yield LLMResult(
            text=text,
            tool_calls=final_tool_calls,
            stop_reason=stop,
            input_tokens=0,  # filled by callback if available
            output_tokens=0,
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
        """Non-streaming call via litellm."""
        import litellm

        oai_messages = _to_openai_messages(system, messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            **self._defaults,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        response = await litellm.acompletion(**kwargs)
        choice = response.choices[0]

        text = choice.message.content or ""
        tool_calls: list[dict[str, Any]] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    parsed = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    parsed = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "input": parsed})

        stop = "tool_use" if tool_calls else "end_turn"
        usage = response.usage
        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


def _to_openai_messages(
    system: str | list[dict[str, Any]], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert Anthropic-style system+messages to OpenAI format."""
    result: list[dict[str, Any]] = []

    # System prompt
    if isinstance(system, str):
        if system:
            result.append({"role": "system", "content": system})
    elif isinstance(system, list):
        # Anthropic cache-control format: extract text blocks
        text_parts = []
        for block in system:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        if text_parts:
            result.append({"role": "system", "content": "\n\n".join(text_parts)})

    result.extend(messages)
    return result


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool format to OpenAI function-calling format."""
    oai_tools: list[dict[str, Any]] = []
    for tool in tools:
        # Skip Anthropic-specific tool types (computer_use)
        if tool.get("type") in ("computer_20250124",):
            continue
        oai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return oai_tools
