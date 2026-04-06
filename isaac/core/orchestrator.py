"""Agentic orchestrator — the brain. while stop_reason != end_turn, up to N iterations.

Now uses LLMClient interface (no direct anthropic import) and ToolExecutor for concurrency.
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from isaac.core.context import (
    COMPACTION_SUMMARY_MAX_TOKENS,
    build_cached_system_prompt,
    build_overflow_preview,
    compact_messages,
    estimate_tokens,
    proactive_compact_on_resume,
    should_compact,
    truncate_tool_result,
)
from isaac.core.executor import ToolExecutor
from isaac.core.llm import LLMClient, LLMResult, StreamEvent
from isaac.core.permissions import PermissionGate
from isaac.core.soul import load_soul
from isaac.core.types import (
    AgentConfig,
    Handoff,
    Message,
    PermissionLevel,
    Role,
    SessionState,
    StopReason,
    ToolCall,
    ToolDef,
    ToolResult,
)

# Pricing per 1M tokens (input/output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    # OpenAI (for litellm routing)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.0, 60.0),
    # Gemini
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
}


class Event:
    """Stream event from orchestrator to terminal."""
    pass


class TextEvent(Event):
    def __init__(self, text: str) -> None:
        self.text = text


class TextDeltaEvent(Event):
    """Streaming text chunk — arrives as the LLM generates."""
    def __init__(self, delta: str) -> None:
        self.delta = delta


class ThinkingEvent(Event):
    """Signals the LLM is processing (shown as a spinner)."""
    def __init__(self, active: bool = True) -> None:
        self.active = active


class ToolCallEvent(Event):
    def __init__(self, tool_call: ToolCall) -> None:
        self.tool_call = tool_call


class ToolResultEvent(Event):
    def __init__(self, result: ToolResult, tool_name: str) -> None:
        self.result = result
        self.tool_name = tool_name


class ApprovalEvent(Event):
    def __init__(self, tool_call: ToolCall, tool_def: ToolDef) -> None:
        self.tool_call = tool_call
        self.tool_def = tool_def


class ErrorEvent(Event):
    def __init__(self, error: str) -> None:
        self.error = error


class CostEvent(Event):
    def __init__(
        self, input_tokens: int, output_tokens: int, cost: float,
        cache_read: int = 0, cache_create: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost
        self.cache_read = cache_read
        self.cache_create = cache_create


class ProgressEvent(Event):
    def __init__(self, iteration: int, elapsed_seconds: int, cost: float, tools_used: list[str] | None = None) -> None:
        self.iteration = iteration
        self.elapsed_seconds = elapsed_seconds
        self.cost = cost
        self.tools_used = tools_used or []


class CompactEvent(Event):
    def __init__(self, summary: str) -> None:
        self.summary = summary


class ModelRouteEvent(Event):
    def __init__(self, model: str, reason: str) -> None:
        self.model = model
        self.reason = reason


class Orchestrator:
    """The agentic loop. Streams events. Handles tool calls, permissions, compaction.

    Now takes an LLMClient (no direct anthropic dependency) and uses ToolExecutor
    for concurrent tool execution.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_registry: dict[str, tuple[ToolDef, Any]],  # name -> (def, handler)
        permission_gate: PermissionGate,
        llm_client: LLMClient | None = None,
        memory_fn: Any = None,
        approval_fn: Any = None,
        soul_mode: str = "full",  # "full" for interactive, "minimal" for heartbeats
        soul_override: str = "",  # inject custom soul text (e.g. hatch mode)
        connector_registry: Any = None,  # ConnectorRegistry for soul awareness
    ) -> None:
        self.config = agent_config
        self.tools = tool_registry
        self.gate = permission_gate
        self.memory_fn = memory_fn  # async (query) -> str
        self.soul_mode = soul_mode
        self.soul_override = soul_override
        self.approval_fn = approval_fn  # async (ToolCall) -> bool
        self.connector_registry = connector_registry
        self.executor = ToolExecutor()

        # LLM client — injected or auto-resolved
        if llm_client:
            self.llm = llm_client
        else:
            # Backward compat: auto-create Anthropic client
            from isaac.core.llm_anthropic import AnthropicClient
            self.llm = AnthropicClient()

    def _build_api_tools(self) -> list[dict[str, Any]]:
        """Convert tool defs to Anthropic API format."""
        api_tools: list[dict[str, Any]] = []
        for name, (tdef, _) in self.tools.items():
            api_tools.append({
                "name": name,
                "description": tdef.description,
                "input_schema": tdef.input_schema,
            })
        # Anthropic computer use beta tool
        if self.config.computer_use:
            api_tools.append({
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": 1280,
                "display_height_px": 800,
                "display_number": 0,
            })
        return api_tools

    def _messages_to_api(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to Anthropic API format."""
        api_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                # System messages injected as user messages with context
                api_msgs.append({
                    "role": "user",
                    "content": f"[System context]: {msg.content}",
                })
                api_msgs.append({
                    "role": "assistant",
                    "content": "Understood.",
                })
                continue

            if msg.tool_calls:
                # Assistant message with tool use
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                api_msgs.append({"role": "assistant", "content": content})
            elif msg.tool_results:
                # Tool result message — validate that each tool_result has a matching
                # tool_use in the previous assistant message (prevents API errors after compaction)
                valid_tool_ids: set[str] = set()
                if api_msgs:
                    prev = api_msgs[-1]
                    if prev.get("role") == "assistant" and isinstance(prev.get("content"), list):
                        for block in prev["content"]:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                valid_tool_ids.add(block["id"])

                content_blocks: list[dict[str, Any]] = []
                for tr in msg.tool_results:
                    # Skip orphaned tool results (their tool_call was compacted away)
                    if valid_tool_ids and tr.tool_call_id not in valid_tool_ids:
                        continue
                    content_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": truncate_tool_result(tr.content),
                        **({"is_error": True} if tr.is_error else {}),
                    })
                if content_blocks:
                    api_msgs.append({"role": "user", "content": content_blocks})
            else:
                api_msgs.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return api_msgs

    async def _execute_tool(self, tc: ToolCall) -> ToolResult:
        """Execute a single tool call (computer use or registry)."""
        # Computer use tool — dispatched to ComputerController
        if tc.name == "computer" and self.config.computer_use:
            try:
                from isaac.tools.computer_use import ComputerController
                if not hasattr(self, "_computer"):
                    self._computer = ComputerController()
                action = tc.input.get("action", "screenshot")
                result = await self._computer.execute(action, **tc.input)
                if result.get("type") == "image":
                    return ToolResult(
                        tool_call_id=tc.id,
                        content=json.dumps([{
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": result["data"]},
                        }]),
                    )
                content = json.dumps(result)
                return ToolResult(tool_call_id=tc.id, content=content)
            except Exception as e:
                return ToolResult(tool_call_id=tc.id, content=f"Computer use error: {e}", is_error=True)

        # Delegate to executor for registry tools
        return await self.executor._exec_one(tc, self.tools)

    async def _summarize(self, text: str) -> str:
        """Use a fast model for summarization (self-chat pattern)."""
        result = await self.llm.create(
            model="claude-haiku-4-5-20251001",
            system=(
                "You are a conversation summarizer. Produce a concise summary of the "
                "key points, decisions, and context from this conversation excerpt. "
                "Focus on information the assistant would need to continue helping "
                "effectively. Be brief."
            ),
            messages=[{"role": "user", "content": text}],
            max_tokens=COMPACTION_SUMMARY_MAX_TOKENS,
        )
        return result.text

    async def run(
        self, user_message: str, state: SessionState
    ) -> AsyncIterator[Event]:
        """Run the agentic loop. Yields events for consumers to render.

        Token efficiency patterns ported from self-chat:
        1. Proactive compaction on session resume (before first LLM call)
        2. 3-layer cached system prompt (soul+tools cached, memory fresh)
        3. Memory context injected on iteration 1 ONLY (not every iteration)
        4. Tool results use overflow preview (not just truncation)
        5. Real API token counts drive compaction, not estimates
        6. Conversation summary lives in session layer (cached between turns)
        7. Streaming LLM calls for real-time output
        8. Concurrent tool execution (safe tools run in parallel)
        """
        # --- PROACTIVE COMPACTION ON RESUME (self-chat pattern) ---
        is_resume = len(state.messages) > 0

        if is_resume and proactive_compact_on_resume(state):
            state.messages, summary = await compact_messages(
                state.messages, self._summarize
            )
            if summary:
                state.summary = summary
                yield CompactEvent(summary)

        # Build scout query — on resume, enrich with recent conversation context
        scout_query = user_message
        if is_resume:
            recent_texts: list[str] = []
            for msg in state.messages[-4:]:
                if msg.content:
                    recent_texts.append(msg.content[:200])
            if recent_texts:
                scout_query = f"{user_message[:300]} | context: {' | '.join(recent_texts)[:200]}"

        # Add user message to history
        state.messages.append(Message(role=Role.USER, content=user_message))
        state.turn_count += 1

        # --- LOAD SOUL ---
        if self.soul_override:
            soul = self.soul_override
        else:
            soul = load_soul(
                self.config.soul,
                self.config.name,
                agent_config=self.config,
                active_tools=list(self.tools.keys()),
                mode=self.soul_mode,
                connector_registry=self.connector_registry,
            )

        # --- MEMORY SCOUT (first iteration only — self-chat pattern) ---
        memory_context = ""
        if self.memory_fn:
            try:
                memory_context = await self.memory_fn(scout_query)
            except Exception:
                pass

        tool_defs = [
            {"name": name, "description": tdef.description}
            for name, (tdef, _) in self.tools.items()
        ]
        api_tools = self._build_api_tools()

        # Track real token usage from API for accurate compaction
        real_input_tokens = 0
        total_cache_read = 0
        total_cache_create = 0
        loop_start = time.time()

        tools_used_since_checkin: list[str] = []

        # --- AGENTIC LOOP ---
        for iteration in range(self.config.max_iterations):
            elapsed = time.time() - loop_start

            # --- PROGRESS CHECK-IN (every 3 iterations) ---
            if iteration > 0 and iteration % 3 == 0:
                yield ProgressEvent(
                    iteration=iteration,
                    elapsed_seconds=int(elapsed),
                    cost=state.total_cost,
                    tools_used=tools_used_since_checkin,
                )
                tools_used_since_checkin = []

            # Check compaction
            if real_input_tokens > 0:
                if real_input_tokens > self.config.context_budget * 0.8:
                    state.messages, summary = await compact_messages(
                        state.messages, self._summarize
                    )
                    state.summary = summary
                    yield CompactEvent(summary)
                    real_input_tokens = 0
            elif should_compact(state, self.config.context_budget):
                state.messages, summary = await compact_messages(
                    state.messages, self._summarize
                )
                state.summary = summary
                yield CompactEvent(summary)

            # Build system prompt — memory only on iteration 1
            system_prompt = build_cached_system_prompt(
                soul=soul,
                tool_defs=tool_defs,
                memory_context=memory_context if iteration == 0 else "",
                conversation_summary=state.summary,
            )

            api_messages = self._messages_to_api(state.messages)

            # --- MODEL ROUTING ---
            from isaac.core.router import route_model
            has_pending = iteration > 0
            routed_model = route_model(
                config_model=self.config.model,
                iteration=iteration,
                user_message=user_message,
                has_pending_tool_calls=has_pending,
            )
            if routed_model != self.config.model:
                yield ModelRouteEvent(routed_model, f"iter={iteration}")

            # --- STREAMING LLM CALL (via LLMClient interface) ---
            yield ThinkingEvent(active=True)
            text = ""
            tool_calls: list[ToolCall] = []
            stop_reason = StopReason.END_TURN
            in_tok = out_tok = cache_read = cache_create = 0

            betas = ["computer-use-2025-01-24"] if self.config.computer_use else None

            try:
                async for stream_event in self.llm.create_stream(
                    model=routed_model,
                    system=system_prompt,
                    messages=api_messages,
                    tools=api_tools or None,
                    betas=betas,
                ):
                    if isinstance(stream_event, StreamEvent):
                        if stream_event.type == "text_delta":
                            yield TextDeltaEvent(stream_event.text)
                    elif isinstance(stream_event, LLMResult):
                        text = stream_event.text
                        tool_calls = [
                            ToolCall(id=tc["id"], name=tc["name"], input=tc["input"])
                            for tc in stream_event.tool_calls
                        ]
                        if stream_event.stop_reason == "tool_use":
                            stop_reason = StopReason.TOOL_USE
                        elif stream_event.stop_reason == "max_tokens":
                            stop_reason = StopReason.MAX_TOKENS
                        in_tok = stream_event.input_tokens
                        out_tok = stream_event.output_tokens
                        cache_read = stream_event.cache_read_tokens
                        cache_create = stream_event.cache_creation_tokens
            except Exception as e:
                yield ThinkingEvent(active=False)
                yield ErrorEvent(f"API error: {e}")
                return

            yield ThinkingEvent(active=False)

            real_input_tokens = in_tok
            total_cache_read += cache_read
            total_cache_create += cache_create

            # Track cost
            pricing = MODEL_PRICING.get(routed_model, (3.0, 15.0))
            base_input_cost = (in_tok - cache_read - cache_create) * pricing[0]
            cache_read_cost = cache_read * pricing[0] * 0.1
            cache_write_cost = cache_create * pricing[0] * 1.25
            output_cost = out_tok * pricing[1]
            cost = (base_input_cost + cache_read_cost + cache_write_cost + output_cost) / 1_000_000
            state.total_tokens += in_tok + out_tok
            state.total_cost += cost
            yield CostEvent(in_tok, out_tok, cost, cache_read, cache_create)

            # Record assistant message
            state.messages.append(Message(
                role=Role.ASSISTANT,
                content=text,
                tool_calls=tool_calls,
            ))

            # Done?
            if stop_reason == StopReason.END_TURN or not tool_calls:
                return

            # --- EXECUTE TOOLS (concurrent safe, serial exclusive) ---
            results: list[ToolResult] = []

            def _permission_check(tc: ToolCall) -> PermissionLevel:
                entry = self.tools.get(tc.name)
                if not entry:
                    return PermissionLevel.AUTO
                tdef, _ = entry
                return self.gate.check(tdef)

            async for event_type, payload in self.executor.execute_batch(
                tool_calls, self.tools, _permission_check, self.approval_fn
            ):
                if event_type == "call":
                    yield ToolCallEvent(payload)
                elif event_type == "approval":
                    entry = self.tools.get(payload.name)
                    if entry:
                        yield ApprovalEvent(payload, entry[0])
                elif event_type == "result":
                    # Overflow preview
                    preview, overflowed = build_overflow_preview(payload.content)
                    if overflowed:
                        payload = ToolResult(
                            tool_call_id=payload.tool_call_id,
                            content=preview,
                            is_error=payload.is_error,
                        )
                    results.append(payload)
                    # Find the tool name for the event
                    tool_name = ""
                    for tc in tool_calls:
                        if tc.id == payload.tool_call_id:
                            tool_name = tc.name
                            break
                    tools_used_since_checkin.append(tool_name)
                    yield ToolResultEvent(payload, tool_name)

            # Record tool results
            state.messages.append(Message(
                role=Role.USER,
                content="",
                tool_results=results,
            ))

        yield ErrorEvent(f"Hit max iterations ({self.config.max_iterations})")

    def generate_handoff(self, state: SessionState) -> Handoff:
        """Generate a handoff when session ends or context runs out."""
        recent = state.messages[-4:] if state.messages else []
        summary_parts: list[str] = []
        for m in recent:
            if m.content:
                summary_parts.append(f"[{m.role.value}] {m.content[:200]}")

        return Handoff(
            summary=state.summary or "\n".join(summary_parts),
            continuation_prompt="Continue from where you left off.",
            memory_keys=[],
        )
