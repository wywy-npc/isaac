"""Source ingestion — get material into raw/.

Replaces Obsidian Web Clipper. Supports:
  - Local files (.md, .txt, .pdf)
  - URLs (article extraction via trafilatura)
  - Raw text (paste with a title)
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from isaac.wiki.store import WikiStore


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:max_len]


def _date_prefix() -> str:
    return time.strftime("%Y-%m-%d")


async def ingest_file(store: WikiStore, file_path: str) -> dict:
    """Ingest a local file (.md, .txt, .pdf) into raw/."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    title = path.stem
    slug = _slugify(title)
    filename = f"{_date_prefix()}_{slug}.md"

    if path.suffix.lower() == ".pdf":
        content = _extract_pdf(path)
        if content is None:
            return {"error": "PDF extraction failed. Install pymupdf: pip install pymupdf"}
        meta = f"---\nsource: {path.name}\ntype: pdf\ningested: {_date_prefix()}\n---"
    else:
        content = path.read_text()
        meta = f"---\nsource: {path.name}\ntype: file\ningested: {_date_prefix()}\n---"

    raw_path = store.store_raw(filename, content, meta)
    store.append_log("ingest", f"File: {path.name} → {filename}")
    return {"raw_path": raw_path, "filename": filename, "chars": len(content)}


async def ingest_url(store: WikiStore, url: str) -> dict:
    """Ingest a web article into raw/ via trafilatura."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}

    # Fetch the page
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return {"error": f"Failed to fetch URL: {e}"}

    # Extract article content
    try:
        import trafilatura
        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="markdown",
        )
        if not content:
            content = trafilatura.extract(html, include_comments=False)
        if not content:
            return {"error": "Could not extract article content from URL"}
    except ImportError:
        # Fallback: strip HTML tags naively
        content = re.sub(r"<[^>]+>", "", html)
        content = re.sub(r"\s+", " ", content).strip()
        if len(content) > 100_000:
            content = content[:100_000] + "\n\n[... truncated ...]"

    # Extract title from content or URL
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
    else:
        # Use URL path as title
        from urllib.parse import urlparse
        parsed = urlparse(url)
        title = parsed.path.strip("/").split("/")[-1] or parsed.netloc

    slug = _slugify(title)
    filename = f"{_date_prefix()}_{slug}.md"

    meta = f"---\nsource: {url}\ntype: url\ningested: {_date_prefix()}\n---"

    try:
        raw_path = store.store_raw(filename, content, meta)
    except FileExistsError:
        # Append a counter
        for i in range(2, 10):
            alt = f"{_date_prefix()}_{slug}-{i}.md"
            try:
                raw_path = store.store_raw(alt, content, meta)
                filename = alt
                break
            except FileExistsError:
                continue
        else:
            return {"error": f"Could not store: filename collision for {slug}"}

    store.append_log("ingest", f"URL: {url} → {filename}")
    return {"raw_path": raw_path, "filename": filename, "chars": len(content), "title": title}


async def ingest_text(store: WikiStore, title: str, content: str) -> dict:
    """Ingest raw text with a given title into raw/."""
    slug = _slugify(title)
    filename = f"{_date_prefix()}_{slug}.md"

    meta = f"---\nsource: user-input\ntitle: {title}\ntype: text\ningested: {_date_prefix()}\n---"

    try:
        raw_path = store.store_raw(filename, content, meta)
    except FileExistsError:
        return {"error": f"Source already exists: {filename}"}

    store.append_log("ingest", f"Text: {title} → {filename}")
    return {"raw_path": raw_path, "filename": filename, "chars": len(content)}


def _extract_pdf(path: Path) -> str | None:
    """Extract text from a PDF using pymupdf."""
    try:
        import pymupdf  # noqa: F811
        doc = pymupdf.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n---\n\n".join(pages)
    except ImportError:
        return None
    except Exception:
        return None
