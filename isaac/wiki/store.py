"""WikiStore — file-backed storage for personal knowledge bases.

Each wiki is a directory under ~/.isaac/wikis/<name>/ with:
  raw/          — immutable source documents
  pages/        — LLM-generated wiki pages
  index.md      — content catalog
  log.md        — append-only chronolog
  schema.md     — conventions (co-evolved with LLM)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from isaac.core.config import ISAAC_HOME

WIKIS_DIR = ISAAC_HOME / "wikis"


class WikiStore:
    """File I/O for a single wiki directory."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = WIKIS_DIR / name
        self.raw_dir = self.dir / "raw"
        self.pages_dir = self.dir / "pages"

    @property
    def exists(self) -> bool:
        return self.dir.exists()

    # --- Creation ---

    @classmethod
    def create(cls, name: str, description: str = "", schema_content: str = "") -> WikiStore:
        """Create a new wiki with directory structure and initial files."""
        store = cls(name)
        if store.dir.exists():
            raise FileExistsError(f"Wiki '{name}' already exists at {store.dir}")

        store.dir.mkdir(parents=True)
        store.raw_dir.mkdir()
        store.pages_dir.mkdir()

        # Schema
        if not schema_content:
            from isaac.wiki.schema import default_schema
            schema_content = default_schema(name, description)
        (store.dir / "schema.md").write_text(schema_content)

        # Index
        index = f"# {name}\n\n{description}\n\n## Pages\n\n_No pages yet. Run `isaac wiki compile {name}` after ingesting sources._\n"
        (store.dir / "index.md").write_text(index)

        # Log
        ts = time.strftime("%Y-%m-%d")
        log = f"## [{ts}] created | Wiki initialized\n\n"
        (store.dir / "log.md").write_text(log)

        return store

    @classmethod
    def list_wikis(cls) -> list[dict[str, Any]]:
        """List all wikis with basic stats."""
        if not WIKIS_DIR.exists():
            return []
        wikis = []
        for d in sorted(WIKIS_DIR.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            store = cls(d.name)
            raw_count = len(list(store.raw_dir.glob("*.md"))) if store.raw_dir.exists() else 0
            page_count = len(list(store.pages_dir.rglob("*.md"))) if store.pages_dir.exists() else 0
            wikis.append({
                "name": d.name,
                "raw_count": raw_count,
                "page_count": page_count,
                "path": str(d),
            })
        return wikis

    # --- Raw sources (immutable) ---

    def store_raw(self, filename: str, content: str, meta: str = "") -> str:
        """Store an immutable source document in raw/. Refuses overwrites."""
        path = self.raw_dir / filename
        if path.exists():
            raise FileExistsError(f"Raw source already exists: {filename}")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        full = meta + "\n\n" + content if meta else content
        path.write_text(full)
        return str(path.relative_to(self.dir))

    def list_raw(self) -> list[str]:
        """List all raw source filenames."""
        if not self.raw_dir.exists():
            return []
        return sorted(p.name for p in self.raw_dir.glob("*") if p.is_file())

    def read_raw(self, filename: str) -> str | None:
        """Read a raw source by filename."""
        path = self.raw_dir / filename
        if not path.exists():
            return None
        return path.read_text()

    # --- Wiki pages ---

    def write_page(self, path: str, content: str) -> str:
        """Write a wiki page. Path is relative to pages/ (e.g. 'overview.md')."""
        full_path = self.pages_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return str(full_path.relative_to(self.dir))

    def read_page(self, path: str) -> str | None:
        """Read a wiki page by path relative to pages/."""
        full_path = self.pages_dir / path
        if not full_path.exists():
            return None
        return full_path.read_text()

    def list_pages(self) -> list[str]:
        """List all wiki page paths relative to pages/."""
        if not self.pages_dir.exists():
            return []
        return sorted(
            str(p.relative_to(self.pages_dir))
            for p in self.pages_dir.rglob("*.md")
        )

    def delete_page(self, path: str) -> bool:
        """Delete a wiki page."""
        full_path = self.pages_dir / path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    # --- Navigation files ---

    def read_index(self) -> str:
        """Read index.md."""
        path = self.dir / "index.md"
        return path.read_text() if path.exists() else ""

    def update_index(self, content: str) -> None:
        """Overwrite index.md."""
        (self.dir / "index.md").write_text(content)

    def read_log(self) -> str:
        """Read log.md."""
        path = self.dir / "log.md"
        return path.read_text() if path.exists() else ""

    def append_log(self, operation: str, description: str) -> None:
        """Append a timestamped entry to log.md."""
        ts = time.strftime("%Y-%m-%d")
        entry = f"## [{ts}] {operation} | {description}\n\n"
        path = self.dir / "log.md"
        existing = path.read_text() if path.exists() else ""
        path.write_text(existing + entry)

    def read_schema(self) -> str:
        """Read schema.md."""
        path = self.dir / "schema.md"
        return path.read_text() if path.exists() else ""

    # --- General read ---

    def read(self, path: str) -> str | None:
        """Read any file in the wiki directory by relative path."""
        full_path = self.dir / path
        if not full_path.exists():
            return None
        return full_path.read_text()

    # --- Search (simple keyword, for when index.md isn't enough) ---

    def search_pages(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Simple keyword search across all wiki pages."""
        results: list[tuple[float, str, str]] = []
        query_lower = query.lower()
        terms = query_lower.split()

        for md_file in self.pages_dir.rglob("*.md"):
            content = md_file.read_text()
            text_lower = content.lower()

            score = 0.0
            for term in terms:
                score += text_lower.count(term) * 0.1

            # Title match boost (first line)
            first_line = content.split("\n", 1)[0].lower()
            for term in terms:
                if term in first_line:
                    score += 2.0

            # Path match
            rel = str(md_file.relative_to(self.pages_dir))
            for term in terms:
                if term in rel.lower():
                    score += 1.5

            if score > 0:
                results.append((score, rel, content))

        results.sort(key=lambda x: x[0], reverse=True)
        return [
            {"path": path, "score": round(score, 2), "content": content}
            for score, path, content in results[:max_results]
        ]
