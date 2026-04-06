"""Personal memory tools — remember, search, read.

The 'remember' tool is the explicit capture mechanism.
Search/read provide access to the personal memory store.
"""
from __future__ import annotations

import time
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef


def build_personal_tools() -> dict[str, tuple[ToolDef, Any]]:
    """Build personal memory tool registry."""
    registry: dict[str, tuple[ToolDef, Any]] = {}

    # --- remember (explicit capture) ---

    async def remember(
        fact: str,
        category: str = "facts",
        importance: float = 0.7,
    ) -> dict[str, Any]:
        """Explicitly save a personal fact or preference."""
        from isaac.personal.store import get_personal_store
        import re

        store = get_personal_store()

        slug = re.sub(r"[^\w\s-]", "", fact[:60].lower())
        slug = re.sub(r"[\s_]+", "-", slug).strip("-")
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = f"{category}/{ts}-{slug}.md"

        content = f"# {fact}\n\nSaved: {time.strftime('%Y-%m-%d %H:%M')}\n"
        meta = {
            "tags": [category, "explicit"],
            "importance": importance,
        }

        # Auto-link
        try:
            from isaac.memory.linker import auto_link
            content, stubs = auto_link(content, path, store)
        except Exception:
            stubs = []

        node = store.write(path, content, meta)
        return {"saved": path, "fact": fact, "category": category, "links": node.outgoing_links}

    registry["remember"] = (
        ToolDef(
            name="remember",
            description=(
                "Save a personal fact, preference, or context about the user. "
                "Use when the user says 'remember this', shares a preference, "
                "or provides important personal context."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact or preference to remember"},
                    "category": {
                        "type": "string",
                        "description": "Category: preference, context, decision, person, pattern, facts",
                        "default": "facts",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance 0-1 (0.7 default for explicit saves)",
                        "default": 0.7,
                    },
                },
                "required": ["fact"],
            },
            permission=PermissionLevel.AUTO,
        ),
        remember,
    )

    # --- personal_memory_search ---

    async def personal_memory_search(query: str) -> dict[str, Any]:
        from isaac.personal.store import get_personal_store
        store = get_personal_store()
        results = store.search(query, max_results=5)
        return {
            "results": [
                {"path": n.path, "content": n.content[:500], "tags": n.tags, "importance": n.importance}
                for n in results
            ]
        }

    registry["personal_memory_search"] = (
        ToolDef(
            name="personal_memory_search",
            description="Search personal memory for facts, preferences, and context about the user.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        personal_memory_search,
    )

    # --- personal_memory_read ---

    async def personal_memory_read(path: str) -> dict[str, Any]:
        from isaac.personal.store import get_personal_store
        store = get_personal_store()
        node = store.read(path)
        if not node:
            return {"error": f"Not found: {path}"}
        return {"path": node.path, "content": node.content, "meta": node.meta}

    registry["personal_memory_read"] = (
        ToolDef(
            name="personal_memory_read",
            description="Read a specific personal memory node by path.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Memory node path"}},
                "required": ["path"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        personal_memory_read,
    )

    return registry
