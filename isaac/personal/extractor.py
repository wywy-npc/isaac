"""Personal fact extractor — auto-capture from conversations.

After each conversation turn, extracts personal facts about the user:
  - Preferences (likes, dislikes, tool choices, workflow patterns)
  - Context (role, company, goals, current projects)
  - Decisions (what they chose and why)
  - People mentioned (relationships, roles)
  - Recurring patterns

Uses Haiku for cost efficiency. Only writes genuinely new information.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from isaac.memory.store import MemoryStore, MemoryNode

EXTRACTION_SYSTEM = """You are a personal memory extractor. Given a conversation excerpt, \
extract facts ABOUT THE USER — their preferences, context, decisions, relationships, and \
recurring patterns. Only extract information the user has directly stated or clearly implied.

Rules:
- Only facts about the user, not general knowledge
- Skip trivial conversational content ("hi", "thanks", etc.)
- Each fact should be a concise, standalone statement
- Include the category: preference, context, decision, person, pattern
- If there are no personal facts worth extracting, return empty array

Return JSON array:
[
  {"fact": "User prefers Sonnet over Opus for most tasks", "category": "preference", "importance": 0.7},
  {"fact": "User works at Mercato Partners as a VC", "category": "context", "importance": 0.9}
]

Return [] if no personal facts found."""


async def extract_personal_facts(
    user_message: str,
    assistant_response: str,
    store: MemoryStore,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict[str, Any]]:
    """Extract personal facts from a conversation turn. Returns list of extracted facts."""
    # Skip very short messages (unlikely to contain personal info)
    if len(user_message) < 20:
        return []

    import anthropic
    client = anthropic.AsyncAnthropic()

    conversation = f"User: {user_message[:2000]}\n\nAssistant: {assistant_response[:1000]}"

    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=1000,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": conversation}],
        )
        text = resp.content[0].text.strip()

        # Parse JSON array from response
        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        facts = json.loads(text)
        if not isinstance(facts, list):
            return []

    except Exception:
        return []

    # Write each fact to personal memory
    written = []
    for fact in facts:
        if not isinstance(fact, dict) or "fact" not in fact:
            continue

        category = fact.get("category", "facts")
        importance = float(fact.get("importance", 0.5))
        fact_text = fact["fact"]

        # Check for duplicates — search existing personal memory
        existing = store.search(fact_text, max_results=3)
        is_duplicate = False
        for node in existing:
            if _similar_enough(fact_text, node.content):
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Write to appropriate path
        slug = _slugify(fact_text[:60])
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = f"{category}/{ts}-{slug}.md"

        content = f"# {fact_text}\n\nExtracted: {time.strftime('%Y-%m-%d %H:%M')}\n"
        meta = {
            "tags": [category, "auto-extracted"],
            "importance": importance,
        }

        store.write(path, content, meta)
        written.append({"path": path, "fact": fact_text, "category": category})

    return written


def _similar_enough(new_fact: str, existing_content: str) -> bool:
    """Quick similarity check to avoid writing duplicate facts."""
    new_words = set(new_fact.lower().split())
    existing_words = set(existing_content.lower().split())
    if not new_words:
        return False
    overlap = len(new_words & existing_words) / len(new_words)
    return overlap > 0.7


def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:40]
