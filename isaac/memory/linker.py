"""Auto-linker — finds related memory nodes and creates stubs for new entities.

Runs before every memory_write to build the knowledge graph automatically.
No LLM calls — pure algorithmic matching via keyword search and embeddings.
"""
from __future__ import annotations

import re
import time
from typing import Any

from isaac.memory.store import MemoryStore

# Proper noun pattern: 2+ capitalized words in a row (e.g. "Mercato Partners")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Existing wikilinks in content
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Words to skip when extracting proper nouns (common headings, etc.)
_SKIP_NOUNS = frozenset({
    "Related", "Summary", "Overview", "Background", "Context",
    "Next Steps", "Action Items", "Key Takeaways", "Meeting Notes",
})

# Max stubs to create per write to avoid runaway growth
_MAX_STUBS = 3


def _extract_search_terms(content: str) -> str:
    """Extract key terms from content for searching related nodes."""
    terms: list[str] = []

    # First heading
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                terms.append(heading)
            break

    # Proper nouns
    for match in _PROPER_NOUN_RE.finditer(content):
        noun = match.group(1)
        if noun not in _SKIP_NOUNS:
            terms.append(noun)

    return " ".join(terms) if terms else content[:200]


def _extract_proper_nouns(content: str) -> list[str]:
    """Extract unique proper nouns from content for stub creation."""
    seen: set[str] = set()
    nouns: list[str] = []
    for match in _PROPER_NOUN_RE.finditer(content):
        noun = match.group(1)
        if noun not in _SKIP_NOUNS and noun not in seen:
            seen.add(noun)
            nouns.append(noun)
    return nouns


def _noun_to_path(noun: str) -> str:
    """Convert a proper noun to a memory path."""
    slug = noun.lower().replace(" ", "-")
    return f"entities/{slug}.md"


def _normalize_path(path: str) -> str:
    """Normalize path for comparison — ensure .md extension."""
    if not path.endswith(".md"):
        return path + ".md"
    return path


def _find_existing_links(content: str) -> set[str]:
    """Find all wikilink paths already in the content."""
    return {_normalize_path(m) for m in _WIKILINK_RE.findall(content)}


def _create_stub(
    store: MemoryStore,
    noun: str,
    stub_path: str,
    source_path: str,
) -> None:
    """Create a stub memory node for a proper noun entity."""
    content = (
        f"# {noun}\n\n"
        f"*Stub node — auto-created when referenced from [[{source_path}]].*\n"
        f"*Flesh this out with real content when you have it.*"
    )
    meta = {
        "name": noun,
        "type": "entity",
        "importance": 0.3,
        "tags": ["stub"],
        "summary": f"Auto-created stub — referenced from {source_path}",
    }
    store.write(stub_path, content, meta)


def auto_link(
    content: str,
    path: str,
    store: MemoryStore,
    embedding_store: Any = None,
    max_links: int = 5,
) -> tuple[str, list[str]]:
    """Auto-link content to related nodes and create stubs for new entities.

    Returns (augmented_content, list_of_created_stub_paths).
    """
    if not content.strip():
        return content, []

    self_path = _normalize_path(path)
    existing_links = _find_existing_links(content)

    # --- Find related existing nodes ---
    candidates: dict[str, float] = {}  # path -> score

    # Keyword search
    terms = _extract_search_terms(content)
    if terms:
        for node in store.search(terms, max_results=10):
            norm = _normalize_path(node.path)
            if norm != self_path and norm not in existing_links:
                candidates[norm] = max(candidates.get(norm, 0), node.importance + 0.5)

    # Embedding similarity search
    if embedding_store:
        try:
            similar = embedding_store.search_similar(content[:500], top_k=5)
            for mem_path, score in similar:
                norm = _normalize_path(mem_path)
                if norm != self_path and norm not in existing_links:
                    candidates[norm] = max(candidates.get(norm, 0), score)
        except Exception:
            pass  # graceful degradation

    # --- Create stubs for proper nouns without existing nodes ---
    created_stubs: list[str] = []
    nouns = _extract_proper_nouns(content)
    stubs_created = 0

    for noun in nouns:
        if stubs_created >= _MAX_STUBS:
            break

        stub_path = _noun_to_path(noun)

        # Skip if a node already exists at this path
        if store.read(stub_path) is not None:
            # Node exists — add to candidates if not already there
            if stub_path != self_path and stub_path not in existing_links:
                candidates.setdefault(stub_path, 0.4)
            continue

        # Skip if the stub path matches the node being written
        if stub_path == self_path:
            continue

        # Skip if already linked
        if stub_path in existing_links:
            continue

        _create_stub(store, noun, stub_path, path)
        created_stubs.append(stub_path)
        candidates[stub_path] = 0.4  # include stub in related links
        stubs_created += 1

    # --- Build related section ---
    if not candidates:
        return content, created_stubs

    # Sort by score, take top N
    sorted_links = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    links_to_add = [p for p, _ in sorted_links[:max_links]]

    if not links_to_add:
        return content, created_stubs

    # Build link lines
    link_lines = [f"- [[{p}]]" for p in links_to_add]

    # Append to existing Related section or create new one
    if "\n## Related" in content:
        # Append to existing section
        content = content.rstrip() + "\n" + "\n".join(link_lines)
    else:
        content = content.rstrip() + "\n\n## Related\n" + "\n".join(link_lines)

    return content, created_stubs
