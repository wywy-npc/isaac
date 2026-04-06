"""Memory store — file-backed with optional pgvector upgrade path.

Memory nodes are markdown files in ~/.isaac/memory/ with YAML frontmatter.
Scout searches by: path match, full-text, tags, and (future) embeddings.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaac.core.config import ISAAC_HOME

MEMORY_DIR = ISAAC_HOME / "memory"


@dataclass
class MemoryNode:
    path: str  # relative path within memory dir, e.g. "projects/isaac.md"
    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    # meta keys: type, name, importance (0-1), tags, summary
    outgoing_links: list[str] = field(default_factory=list)

    @property
    def importance(self) -> float:
        return float(self.meta.get("importance", 0.5))

    @property
    def tags(self) -> list[str]:
        return self.meta.get("tags", [])


class MemoryStore:
    """File-backed memory with wiki-link graph traversal."""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self.dir = memory_dir or MEMORY_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, path: str, content: str, meta: dict[str, Any] | None = None) -> MemoryNode:
        """Write a memory node."""
        full_path = self.dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Build frontmatter
        node_meta = meta or {}
        node_meta.setdefault("updated", time.strftime("%Y-%m-%dT%H:%M:%S"))

        fm_lines = ["---"]
        for k, v in node_meta.items():
            fm_lines.append(f"{k}: {json.dumps(v)}")
        fm_lines.append("---")

        # Parse wiki-links
        links = re.findall(r"\[\[([^\]]+)\]\]", content)

        full_content = "\n".join(fm_lines) + "\n\n" + content
        full_path.write_text(full_content)

        return MemoryNode(path=path, content=content, meta=node_meta, outgoing_links=links)

    def read(self, path: str) -> MemoryNode | None:
        """Read a memory node by path."""
        full_path = self.dir / path
        if not full_path.exists():
            return None

        text = full_path.read_text()
        meta, content = self._parse_frontmatter(text)
        links = re.findall(r"\[\[([^\]]+)\]\]", content)
        return MemoryNode(path=path, content=content, meta=meta, outgoing_links=links)

    def search(self, query: str, max_results: int = 10) -> list[MemoryNode]:
        """Full-text search across all memory nodes."""
        results: list[tuple[float, MemoryNode]] = []
        query_lower = query.lower()
        query_terms = query_lower.split()

        for md_file in self.dir.rglob("*.md"):
            try:
                text = md_file.read_text()
            except Exception:
                continue

            meta, content = self._parse_frontmatter(text)
            text_lower = content.lower()

            # Score: term frequency + importance boost + tag match
            score = 0.0
            for term in query_terms:
                score += text_lower.count(term) * 0.1

            # Tag match boost
            tags = meta.get("tags", [])
            for tag in tags:
                if any(t in tag.lower() for t in query_terms):
                    score += 2.0

            # Path match boost
            rel_path = str(md_file.relative_to(self.dir))
            for term in query_terms:
                if term in rel_path.lower():
                    score += 1.5

            # Importance boost
            score *= (1.0 + float(meta.get("importance", 0.5)))

            if score > 0:
                links = re.findall(r"\[\[([^\]]+)\]\]", content)
                results.append((
                    score,
                    MemoryNode(path=rel_path, content=content, meta=meta, outgoing_links=links),
                ))

        results.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in results[:max_results]]

    def list_all(self) -> list[str]:
        """List all memory node paths."""
        return [
            str(p.relative_to(self.dir))
            for p in self.dir.rglob("*.md")
        ]

    def delete(self, path: str) -> bool:
        full_path = self.dir / path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown."""
        if not text.startswith("---"):
            return {}, text

        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text

        meta: dict[str, Any] = {}
        for line in parts[1].strip().split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    meta[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    meta[key] = val

        return meta, parts[2].strip()
