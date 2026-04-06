"""Wiki search — simple BM25-style keyword search.

At small scale (~100 sources, hundreds of pages) the LLM reads index.md
directly (Karpathy's approach). This module kicks in when the wiki outgrows
that — proper keyword search with optional embedding similarity.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from isaac.wiki.store import WikiStore


def search(store: WikiStore, query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """BM25-style search over wiki pages.

    Also searches index.md and raw/ filenames for navigation hints.
    Returns results sorted by relevance score.
    """
    if not store.pages_dir.exists():
        return []

    # Tokenize query
    terms = _tokenize(query)
    if not terms:
        return []

    # Collect all documents
    docs: list[tuple[str, str]] = []  # (relative_path, content)
    for md_file in store.pages_dir.rglob("*.md"):
        try:
            content = md_file.read_text()
            rel = str(md_file.relative_to(store.pages_dir))
            docs.append((rel, content))
        except Exception:
            continue

    if not docs:
        return []

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    avg_dl = sum(len(_tokenize(c)) for _, c in docs) / len(docs) if docs else 1

    # Document frequency for each term
    df: dict[str, int] = {}
    for term in terms:
        df[term] = sum(1 for _, content in docs if term in _tokenize(content))

    n = len(docs)
    results: list[tuple[float, str, str]] = []

    for rel_path, content in docs:
        doc_tokens = _tokenize(content)
        dl = len(doc_tokens)
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for term in terms:
            if term not in tf:
                continue
            # IDF
            idf = math.log((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            # TF with BM25 saturation
            term_tf = tf[term]
            tf_norm = (term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm

        # Title boost
        first_line = content.split("\n", 1)[0].lower()
        for term in terms:
            if term in first_line:
                score *= 1.5

        # Path boost
        path_lower = rel_path.lower()
        for term in terms:
            if term in path_lower:
                score *= 1.3

        if score > 0:
            results.append((score, rel_path, content))

    results.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "path": path,
            "score": round(score, 3),
            "snippet": _snippet(content, terms),
            "content": content,
        }
        for score, path, content in results[:max_results]
    ]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def _snippet(content: str, terms: list[str], context: int = 150) -> str:
    """Extract a snippet around the first matching term."""
    lower = content.lower()
    best_pos = len(content)
    for term in terms:
        pos = lower.find(term)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos >= len(content):
        return content[:context * 2] + "..."

    start = max(0, best_pos - context)
    end = min(len(content), best_pos + context)
    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet
