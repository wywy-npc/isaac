"""
Memory scout — multi-step search strategy for context retrieval.

Ported from self-chat (~/self/self-chat/server/services/memory/scout.py).

Key patterns:
- Token-budgeted: never exceeds MEMORY_SCOUT_BUDGET
- Full/summary inclusion: high-importance nodes get full content,
  low-importance get summary only. Falls back to summary if budget is tight.
- Structured output: nodes grouped by source category
"""
from __future__ import annotations

from typing import Any

from isaac.core.context import CHARS_PER_TOKEN, MEMORY_SCOUT_BUDGET
from isaac.memory.store import MemoryStore, MemoryNode


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _decide_inclusion(node: MemoryNode, is_direct_match: bool) -> str:
    """Decide full vs summary inclusion (ported from self-chat)."""
    if is_direct_match:
        return "full"
    importance = node.importance
    if importance >= 0.8:
        return "full"
    if importance <= 0.3:
        return "summary"
    return "full"


def _build_snippet(node: MemoryNode, inclusion: str) -> str:
    """Build a markdown snippet for a memory node."""
    tags_str = ", ".join(node.tags) if node.tags else ""
    meta_parts = [f"importance: {node.importance:.1f}"]
    if tags_str:
        meta_parts.append(f"tags: {tags_str}")
    meta_line = " | ".join(meta_parts)

    header = f"### {node.path}\n> {meta_line}\n"

    if inclusion == "summary":
        # Use first 200 chars as summary if no explicit summary in meta
        summary = node.meta.get("summary", node.content[:200])
        return f"{header}\n{summary}\n"

    return f"{header}\n{node.content}\n"


class MemoryScout:
    """Multi-step memory search with token budgeting and full/summary inclusion."""

    def __init__(self, store: MemoryStore, embedding_store: Any = None) -> None:
        self.store = store
        self.embedding_store = embedding_store

    async def search(self, query: str, budget: int = MEMORY_SCOUT_BUDGET) -> str:
        """Run the scout pipeline and return formatted memory context within budget."""
        budget_chars = budget * CHARS_PER_TOKEN
        tokens_used = 0
        collected: list[tuple[MemoryNode, str, str]] = []  # (node, inclusion, category)
        seen_paths: set[str] = set()

        # --- Step 1: Direct path match ---
        if "/" in query or query.endswith(".md"):
            node = self.store.read(query)
            if node and node.path not in seen_paths:
                snippet = _build_snippet(node, "full")
                snippet_tokens = _estimate_tokens(snippet)
                if tokens_used + snippet_tokens <= budget:
                    collected.append((node, "full", "Entity"))
                    seen_paths.add(node.path)
                    tokens_used += snippet_tokens

        # --- Step 1.5: Vector similarity search ---
        if self.embedding_store:
            try:
                similar = self.embedding_store.search_similar(query, top_k=5)
                for mem_path, score in similar:
                    if tokens_used >= budget or mem_path in seen_paths:
                        continue
                    node = self.store.read(mem_path)
                    if not node:
                        continue
                    inclusion = _decide_inclusion(node, is_direct_match=False)
                    snippet = _build_snippet(node, inclusion)
                    snippet_tokens = _estimate_tokens(snippet)
                    if tokens_used + snippet_tokens > budget and inclusion == "full":
                        summary_snippet = _build_snippet(node, "summary")
                        summary_tokens = _estimate_tokens(summary_snippet)
                        if tokens_used + summary_tokens <= budget:
                            collected.append((node, "summary", "Entity"))
                            seen_paths.add(node.path)
                            tokens_used += summary_tokens
                        continue
                    if tokens_used + snippet_tokens <= budget:
                        collected.append((node, inclusion, "Entity"))
                        seen_paths.add(node.path)
                        tokens_used += snippet_tokens
            except Exception:
                pass  # degrade gracefully if embeddings fail

        # --- Step 2: Full-text search ---
        search_results = self.store.search(query, max_results=8)
        primary_nodes: list[MemoryNode] = []

        for node in search_results:
            if tokens_used >= budget:
                break
            if node.path in seen_paths:
                continue

            inclusion = _decide_inclusion(node, is_direct_match=False)
            snippet = _build_snippet(node, inclusion)
            snippet_tokens = _estimate_tokens(snippet)

            # If full doesn't fit, try summary (self-chat pattern)
            if tokens_used + snippet_tokens > budget and inclusion == "full":
                summary_snippet = _build_snippet(node, "summary")
                summary_tokens = _estimate_tokens(summary_snippet)
                if tokens_used + summary_tokens <= budget:
                    collected.append((node, "summary", "Entity"))
                    seen_paths.add(node.path)
                    tokens_used += summary_tokens
                    primary_nodes.append(node)
                continue

            if tokens_used + snippet_tokens <= budget:
                collected.append((node, inclusion, "Entity"))
                seen_paths.add(node.path)
                tokens_used += snippet_tokens
                primary_nodes.append(node)

        # --- Step 3: Link traversal (from top results) ---
        for node in primary_nodes[:3]:
            if tokens_used >= budget:
                break
            for link in node.outgoing_links[:3]:
                if tokens_used >= budget:
                    break
                link_path = link if link.endswith(".md") else f"{link}.md"
                linked = self.store.read(link_path)
                if not linked or linked.path in seen_paths:
                    continue

                inclusion = _decide_inclusion(linked, is_direct_match=False)
                snippet = _build_snippet(linked, inclusion)
                snippet_tokens = _estimate_tokens(snippet)

                # Try summary if full doesn't fit
                if tokens_used + snippet_tokens > budget and inclusion == "full":
                    summary_snippet = _build_snippet(linked, "summary")
                    summary_tokens = _estimate_tokens(summary_snippet)
                    if tokens_used + summary_tokens <= budget:
                        collected.append((linked, "summary", "Related"))
                        seen_paths.add(linked.path)
                        tokens_used += summary_tokens
                    continue

                if tokens_used + snippet_tokens <= budget:
                    collected.append((linked, inclusion, "Related"))
                    seen_paths.add(linked.path)
                    tokens_used += snippet_tokens

        # --- Assemble structured output (self-chat pattern) ---
        return self._assemble_context(collected)

    @staticmethod
    def _assemble_context(collected: list[tuple[MemoryNode, str, str]]) -> str:
        """Assemble collected nodes into structured markdown, grouped by category."""
        if not collected:
            return ""

        sections: dict[str, list[tuple[MemoryNode, str]]] = {}
        for node, inclusion, category in collected:
            sections.setdefault(category, []).append((node, inclusion))

        lines = ["## Memory Context\n"]

        for category in ["Entity", "Related", "Agent Learnings", "Daily Log"]:
            items = sections.get(category)
            if not items:
                continue

            if category == "Related":
                lines.append("**Related:**")
                for node, inclusion in items:
                    tags_str = ", ".join(node.tags) if node.tags else ""
                    summary = node.meta.get("summary", node.content[:100])
                    lines.append(f"- **{node.path}** ({node.importance:.1f}): {summary}")
                lines.append("")
            else:
                if category != "Entity":
                    lines.append(f"### {category}")
                for node, inclusion in items:
                    lines.append(_build_snippet(node, inclusion))

        return "\n".join(lines).strip()
