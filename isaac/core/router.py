"""Model router — heuristic-based model selection per iteration.

Three tiers:
  Haiku  — fast ops: summarization, simple follow-ups, yes/no
  Sonnet — standard work: tool-use loops, coding, most tasks
  Opus   — heavy reasoning: architecture, complex debugging, first-pass planning

The router picks the cheapest model that can handle the current iteration.
The agent config's model field acts as the CEILING — if you configure an agent
with Sonnet, it will never escalate to Opus.

Cost savings: A typical 10-iteration tool-use session drops from ~$0.45 (all Opus)
to ~$0.08 (Opus first turn, Sonnet for tool loops).
"""
from __future__ import annotations

import re

# Model tier ordering (cheapest to most capable)
MODEL_TIERS = {
    "claude-haiku-4-5-20251001": 0,
    "claude-sonnet-4-6": 1,
    "claude-opus-4-6": 2,
}

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-6"

# Patterns that suggest a simple/short query
SIMPLE_PATTERNS = [
    r"^(yes|no|ok|sure|thanks|yep|nope|done|next|go|do it|proceed)\b",
    r"^(y|n)\b",
    r"^\w{1,5}$",  # single short word
]

# Patterns that suggest complex reasoning is needed
COMPLEX_PATTERNS = [
    r"architect",
    r"design.*system",
    r"debug.*complex",
    r"refactor.*entire",
    r"security.*audit",
    r"explain.*why.*(\w+\s+){10,}",  # long "explain why" queries
    r"review.*pull request",
    r"trade.?offs?",
    r"compare.*approaches",
]

_simple_re = [re.compile(p, re.IGNORECASE) for p in SIMPLE_PATTERNS]
_complex_re = [re.compile(p, re.IGNORECASE) for p in COMPLEX_PATTERNS]


def _classify_message(text: str) -> str:
    """Classify a user message as simple/standard/complex."""
    text = text.strip()

    # Short messages are simple
    if len(text) < 15:
        for pat in _simple_re:
            if pat.match(text):
                return "simple"

    # Check for complex patterns
    for pat in _complex_re:
        if pat.search(text):
            return "complex"

    # Long messages (>500 chars) with questions tend to be complex
    if len(text) > 500 and "?" in text:
        return "complex"

    return "standard"


def _cap_model(selected: str, ceiling: str) -> str:
    """Never exceed the configured model ceiling."""
    selected_tier = MODEL_TIERS.get(selected, 1)
    ceiling_tier = MODEL_TIERS.get(ceiling, 1)
    if selected_tier > ceiling_tier:
        return ceiling
    return selected


def route_model(
    config_model: str,
    iteration: int,
    user_message: str = "",
    has_pending_tool_calls: bool = False,
) -> str:
    """Pick the right model for this iteration.

    Args:
        config_model: The agent's configured model (acts as ceiling)
        iteration: Current iteration in the agentic loop (0-based)
        user_message: The user's original message (for complexity classification)
        has_pending_tool_calls: Whether we're continuing a tool-use loop

    Returns:
        Model name string to use for this LLM call
    """
    # Tool-use continuation (iterations 2+) — Sonnet handles tool loops fine
    if iteration > 0 and has_pending_tool_calls:
        return _cap_model(SONNET, config_model)

    # Iterations 2+ without tool calls (rare, usually means wrapping up)
    if iteration > 0:
        return _cap_model(SONNET, config_model)

    # First iteration (iteration 0) — classify the user message
    complexity = _classify_message(user_message)

    if complexity == "simple":
        return _cap_model(HAIKU, config_model)
    elif complexity == "complex":
        return _cap_model(OPUS, config_model)
    else:
        # Standard: use Sonnet
        return _cap_model(SONNET, config_model)
