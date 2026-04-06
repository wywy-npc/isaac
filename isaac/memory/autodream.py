"""autoDream — background memory consolidation.

4-phase pipeline:
  1. Orient  — scan recent memories
  2. Gather  — find related via embeddings + wiki-links
  3. Consolidate — merge clusters via Haiku
  4. Prune   — remove merged source nodes
"""
from __future__ import annotations

import time
from typing import Any

import anthropic

from isaac.core.config import get_env
from isaac.memory.store import MemoryStore, MemoryNode


class AutoDream:
    """Memory consolidation engine."""

    def __init__(self, store: MemoryStore, embedding_store: Any = None) -> None:
        self.store = store
        self.embedding_store = embedding_store
        self.client = anthropic.AsyncAnthropic(api_key=get_env("ANTHROPIC_API_KEY"))

    async def run(self, max_clusters: int = 10) -> dict[str, int]:
        """Run the full dream cycle. Returns a report dict."""
        # Phase 1: Orient — find recent memories
        recent = self._orient()
        if not recent:
            return {"scanned": 0, "clusters": 0, "consolidated": 0, "pruned": 0}

        # Phase 2: Gather — form clusters of related nodes
        clusters = self._gather(recent)

        # Phase 3: Consolidate — merge each cluster via Haiku
        consolidated = 0
        pruned = 0
        for anchor, related in clusters[:max_clusters]:
            if len(related) < 1:
                continue
            merged = await self._consolidate(anchor, related)
            if merged:
                # Write consolidated content back to anchor
                self.store.write(anchor.path, merged, anchor.meta)
                if self.embedding_store:
                    try:
                        self.embedding_store.embed_and_store(anchor.path, merged)
                    except Exception:
                        pass
                consolidated += 1

                # Phase 4: Prune — remove merged sources
                for node in related:
                    self.store.delete(node.path)
                    if self.embedding_store:
                        try:
                            self.embedding_store.delete(node.path)
                        except Exception:
                            pass
                    pruned += 1

        return {
            "scanned": len(recent),
            "clusters": len(clusters),
            "consolidated": consolidated,
            "pruned": pruned,
        }

    def _orient(self, max_nodes: int = 50) -> list[MemoryNode]:
        """Scan for recent memories (updated in last 7 days)."""
        cutoff = time.time() - 7 * 86400
        recent: list[MemoryNode] = []

        for path in self.store.list_all():
            node = self.store.read(path)
            if not node:
                continue
            # Parse timestamp from meta
            updated = node.meta.get("updated", "")
            if isinstance(updated, str) and "T" in updated:
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(updated)
                    if dt.timestamp() < cutoff:
                        continue
                except (ValueError, OSError):
                    pass
            recent.append(node)
            if len(recent) >= max_nodes:
                break

        return recent

    def _gather(self, nodes: list[MemoryNode]) -> list[tuple[MemoryNode, list[MemoryNode]]]:
        """Form clusters of related nodes. Anchor = highest importance in cluster."""
        seen: set[str] = set()
        clusters: list[tuple[MemoryNode, list[MemoryNode]]] = []

        for node in sorted(nodes, key=lambda n: n.importance, reverse=True):
            if node.path in seen:
                continue
            seen.add(node.path)
            related: list[MemoryNode] = []

            # Wiki-link relations
            for link in node.outgoing_links:
                link_path = link if link.endswith(".md") else f"{link}.md"
                linked = self.store.read(link_path)
                if linked and linked.path not in seen:
                    related.append(linked)
                    seen.add(linked.path)

            # Embedding-based relations
            if self.embedding_store:
                try:
                    similar = self.embedding_store.search_similar(node.content[:500], top_k=3)
                    for mem_path, score in similar:
                        if score > 0.7 and mem_path not in seen:
                            rel_node = self.store.read(mem_path)
                            if rel_node:
                                related.append(rel_node)
                                seen.add(mem_path)
                except Exception:
                    pass

            if related:
                clusters.append((node, related))

        return clusters

    async def _consolidate(self, anchor: MemoryNode, related: list[MemoryNode]) -> str | None:
        """Merge a cluster into a single consolidated memory via Haiku."""
        all_content = f"## Anchor: {anchor.path}\n{anchor.content}\n\n"
        for node in related:
            all_content += f"## Related: {node.path}\n{node.content}\n\n"

        try:
            response = await self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                system=(
                    "You are a memory consolidation agent. Merge the following related memories "
                    "into a single, coherent note. Preserve all unique facts. Remove redundancy. "
                    "Keep wiki-links ([[link]]). Output markdown only, no preamble."
                ),
                messages=[{"role": "user", "content": all_content}],
            )
            return response.content[0].text if response.content else None
        except Exception:
            return None
