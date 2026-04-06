"""
Context Engine — surgical context assembly that never wastes tokens.

Ported from self-chat (~/self/self-chat/server/services/chat/context_engine.py).

Three-layer prompt composition (Claude Code pattern):

LAYER 1 — STATIC (cached across turns, shared within session)
├── Soul (platform + agent)
├── Tool schemas (descriptions only, capped at 150 chars)
└── [cache_control: ephemeral]

LAYER 2 — SESSION (cached within conversation, changes rarely)
├── Conversation summary (auto-compacted older turns)
└── [cache_control: ephemeral]

LAYER 3 — TURN (fresh every request, injected on iteration 1 ONLY)
├── Memory Scout results (token-budgeted)
└── Recent messages (last N turns)

Static + session sections use Anthropic's prompt caching (90% savings).
Memory context only on first iteration to avoid wasting tokens on multi-step tool use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from isaac.core.types import Message, Role, SessionState

# ----- Token estimation -----
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(m.content)
        for tc in m.tool_calls:
            total += estimate_tokens(json.dumps(tc.input))
        for tr in m.tool_results:
            total += estimate_tokens(tr.content)
    return total


# ----- Configurable thresholds (from self-chat settings) -----
# These can be overridden via AgentConfig in the future

CONTEXT_BUDGET_TOKENS = 180_000     # total context budget
COMPACTION_THRESHOLD = 0.8          # compact when context hits 80% of budget
MEMORY_SCOUT_BUDGET = 2000          # tokens for agent/company memory retrieval
PERSONAL_SCOUT_BUDGET = 1000        # tokens for personal memory retrieval
TOOL_RESULT_MAX_TOKENS = 2000       # overflow to preview above this
MAX_RECENT_TURNS = 20               # keep last N turns uncompacted
TOOL_DESC_MAX_CHARS = 150           # truncate tool descriptions in static layer
COMPACTION_SUMMARY_MAX_TOKENS = 500 # max tokens for summarizer output


# ----- Context layers -----

@dataclass
class ContextLayers:
    """Assembled context layers ready for the LLM call."""
    system_prompt: list[dict[str, Any]]  # Anthropic system format with cache_control
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    static_tokens: int = 0
    session_tokens: int = 0
    turn_tokens: int = 0
    message_tokens: int = 0
    total_tokens: int = 0
    was_compacted: bool = False


# ----- Prompt building -----

def build_system_prompt(soul: str, tool_defs: list[dict[str, Any]], memory_context: str) -> str:
    """Build flat system prompt string (for simple/test use cases)."""
    parts: list[str] = [soul]
    if tool_defs:
        tool_lines = [f"- **{t['name']}**: {_truncate_desc(t['description'])}" for t in tool_defs]
        parts.append("## Available Tools\n" + "\n".join(tool_lines))
    if memory_context:
        parts.append(f"## Relevant Memory\n{memory_context}")
    return "\n\n".join(parts)


def build_cached_system_prompt(
    soul: str,
    tool_defs: list[dict[str, Any]],
    memory_context: str,
    conversation_summary: str = "",
) -> list[dict[str, Any]]:
    """Build 3-layer system prompt with cache_control breakpoints.

    Layer 1 (static): Soul + tool descriptions. Cached across entire session.
    Layer 2 (session): Conversation summary. Cached within conversation.
    Layer 3 (turn): Memory context. Fresh per-turn, NOT cached.

    On multi-step tool use iterations, memory_context should be "" to save tokens
    (it was already injected on iteration 1).
    """
    blocks: list[dict[str, Any]] = []

    # --- LAYER 1: STATIC (soul + tools) ---
    static_parts = [soul]
    if tool_defs:
        tool_lines = [f"- **{t['name']}**: {_truncate_desc(t['description'])}" for t in tool_defs]
        static_parts.append("## Available Tools\n" + "\n".join(tool_lines))

    blocks.append({
        "type": "text",
        "text": "\n\n".join(static_parts),
        "cache_control": {"type": "ephemeral"},
    })

    # --- LAYER 2: SESSION (conversation summary) ---
    if conversation_summary:
        blocks.append({
            "type": "text",
            "text": f"## Earlier in This Conversation\n{conversation_summary}",
            "cache_control": {"type": "ephemeral"},
        })

    # --- LAYER 3: TURN (memory context, fresh, NOT cached) ---
    if memory_context:
        blocks.append({
            "type": "text",
            "text": f"## Relevant Memory\n{memory_context}",
        })

    return blocks


# ----- Compaction -----

@dataclass
class CompactionResult:
    messages: list[Message]
    summary: Optional[str] = None
    turns_compacted: int = 0


def should_compact(state: SessionState, budget: int = CONTEXT_BUDGET_TOKENS) -> bool:
    """Check if we've hit the compaction threshold."""
    used = estimate_messages_tokens(state.messages)
    threshold = int(budget * COMPACTION_THRESHOLD)
    return used > threshold and len(state.messages) > MAX_RECENT_TURNS


async def compact_messages(
    messages: list[Message], summarizer: Any, keep_count: int = MAX_RECENT_TURNS,
) -> tuple[list[Message], str]:
    """Summarize older messages, keep last N verbatim.

    Ported from self-chat: uses a dedicated summarizer prompt that focuses
    on preserving information needed to continue the task.

    Args:
        messages: full message history
        summarizer: async callable(text) -> summary string
        keep_count: how many recent messages to preserve verbatim

    Returns:
        (compacted_messages, summary)
    """
    if len(messages) <= keep_count:
        return messages, ""

    # Find a safe split point — never split between a tool_call and its tool_result.
    # Walk backward from the target split point until we find a clean boundary
    # (a message that is NOT a tool_result response).
    split = len(messages) - keep_count
    while split > 0 and split < len(messages):
        msg = messages[split]
        # If this message has tool_results, it's a response to the previous message's
        # tool_calls — don't split here, include the pair in recent.
        if msg.tool_results:
            split -= 1
        else:
            break

    if split <= 0:
        return messages, ""

    old = messages[:split]
    recent = messages[split:]

    # Build text from old messages (capped at 500 chars each)
    old_text_parts: list[str] = []
    for msg in old:
        role = msg.role.value
        content = msg.content
        if content:
            old_text_parts.append(f"{role}: {content[:500]}")
        for tc in msg.tool_calls:
            old_text_parts.append(f"{role}: [tool_call: {tc.name}({json.dumps(tc.input)[:200]})]")
        for tr in msg.tool_results:
            old_text_parts.append(f"tool_result: {tr.content[:300]}")

    old_text = "\n".join(old_text_parts)

    # Self-chat's summarizer prompt — focuses on continuation context
    summarizer_prompt = (
        "You are a conversation summarizer. Produce a concise summary of the key points, "
        "decisions, and context from this conversation excerpt. Focus on information the "
        "assistant would need to continue helping effectively. Be brief.\n\n"
        + old_text
    )

    try:
        summary = await summarizer(summarizer_prompt)
    except Exception:
        summary = f"[{len(old)} earlier messages truncated]"

    summary_msg = Message(
        role=Role.SYSTEM,
        content=f"[Conversation summary]: {summary}",
    )
    return [summary_msg] + recent, summary


def proactive_compact_on_resume(state: SessionState) -> bool:
    """Check if session needs proactive compaction on resume (before first LLM call).

    Self-chat pattern: when resuming a session with a long history, compact
    BEFORE adding the new message, so the first LLM call doesn't blow the budget.
    """
    return len(state.messages) > MAX_RECENT_TURNS


# ----- Tool result overflow -----

def build_overflow_preview(full_result: str, max_tokens: int = TOOL_RESULT_MAX_TOKENS) -> tuple[str, bool]:
    """If a tool result exceeds max_tokens, return a preview + overflow flag.

    Ported from self-chat's build_overflow_preview. The full result can be
    stored for on-demand retrieval via a get_full_result tool.
    """
    result_tokens = estimate_tokens(full_result)
    if result_tokens <= max_tokens:
        return full_result, False

    preview_chars = max_tokens * CHARS_PER_TOKEN
    preview = full_result[:preview_chars]
    preview += (
        f"\n\n[... truncated. Full result: {result_tokens} tokens. "
        f"Use `get_full_result` tool to retrieve.]"
    )
    return preview, True


def truncate_tool_result(content: str, max_chars: int = 8000) -> str:
    """Simple truncation for tool results (head + tail). Fallback for non-overflow path."""
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return content[:half] + "\n\n[... truncated ...]\n\n" + content[-half:]


# ----- Helpers -----

def _truncate_desc(description: str) -> str:
    """Truncate tool descriptions for the static prompt layer (self-chat pattern)."""
    if len(description) <= TOOL_DESC_MAX_CHARS:
        return description
    return description[:TOOL_DESC_MAX_CHARS - 3] + "..."
