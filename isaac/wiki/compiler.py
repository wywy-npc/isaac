"""WikiCompiler — the LLM compile pipeline.

This is the heart of Karpathy's system. The LLM reads raw sources and
incrementally builds/maintains the wiki. Three operations:
  - compile: raw sources → wiki pages (interactive, streaming)
  - query: question → search wiki → synthesize answer → optionally file back
  - lint: health-check the wiki for contradictions, orphans, stale claims
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, AsyncIterator

from isaac.wiki.store import WikiStore


# --- LLM client helper ---

def _get_client():
    """Get an Anthropic client."""
    import anthropic
    return anthropic.AsyncAnthropic()


# --- Prompts (Python constants per spec) ---

COMPILE_SYSTEM = """You are a wiki compiler. You maintain a personal knowledge wiki by reading \
raw source documents and integrating their key information into structured wiki pages.

You follow the schema conventions provided. Your job:
1. Extract key concepts, entities, facts, and claims from the source
2. Write or update wiki pages for each concept/entity
3. Maintain cross-references using [[page-name]] links
4. Keep pages concise but comprehensive
5. Always cite which raw source a fact came from

Output format: For each page you create or update, output a block:

=== PAGE: <filename.md> ===
<full markdown content of the page>
=== END PAGE ===

After all pages, output the updated index:

=== INDEX ===
<full index.md content>
=== END INDEX ==="""


QUERY_SYSTEM = """You are a wiki research assistant. You have access to a personal knowledge wiki. \
Given the user's question and the relevant wiki pages, synthesize a clear, comprehensive answer \
with citations to specific wiki pages using [[page-name]] references.

If the answer reveals a valuable synthesis or connection worth preserving, suggest filing it \
as a new wiki page by including:

=== NEW PAGE: <filename.md> ===
<page content>
=== END PAGE ==="""


LINT_SYSTEM = """You are a wiki quality auditor. Analyze the provided wiki pages and report:

1. **Contradictions**: Pages that make conflicting claims
2. **Stale claims**: Information that seems outdated or superseded
3. **Missing pages**: Concepts mentioned frequently but lacking their own page
4. **Missing cross-refs**: Pages about related topics that should link to each other
5. **Data gaps**: Areas where the wiki could be strengthened with additional sources
6. **Orphan pages**: Pages not referenced by any other page

For each issue, be specific: cite the page names and the problematic content.
Suggest concrete fixes. Prioritize by impact."""


# --- Compile operation ---

async def compile_source(
    store: WikiStore,
    raw_filename: str,
    model: str = "claude-haiku-4-5-20251001",
) -> AsyncIterator[dict[str, Any]]:
    """Compile a single raw source into wiki pages. Streams progress.

    Yields dicts with keys:
      - type: "status" | "page" | "index" | "done" | "error"
      - content: status message, page content, etc.
    """
    # Read source
    raw_content = store.read_raw(raw_filename)
    if raw_content is None:
        yield {"type": "error", "content": f"Raw source not found: {raw_filename}"}
        return

    # Read schema and index for context
    schema = store.read_schema()
    index = store.read_index()

    # Read existing pages for context (titles + first 200 chars)
    existing_pages = []
    for page_path in store.list_pages():
        content = store.read_page(page_path)
        if content:
            existing_pages.append(f"**{page_path}**: {content[:200]}")
    existing_context = "\n".join(existing_pages) if existing_pages else "_No existing pages._"

    yield {"type": "status", "content": f"Reading source: {raw_filename}"}

    # Build the compile prompt
    user_msg = f"""## Schema
{schema}

## Current Index
{index}

## Existing Pages (summaries)
{existing_context}

## New Source to Process: {raw_filename}
{raw_content}

---

Process this source following the schema's Ingest Workflow. Create/update wiki pages and \
regenerate the index. Remember to add [[cross-references]] between related pages."""

    client = _get_client()

    yield {"type": "status", "content": "Compiling with LLM..."}

    # Stream the response
    full_response = ""
    async with client.messages.stream(
        model=model,
        max_tokens=8192,
        system=COMPILE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        async for text in stream.text_stream:
            full_response += text

    # Parse output — extract pages and index
    pages_written = []
    page_pattern = re.compile(
        r"=== PAGE:\s*(.+?)\s*===\s*\n(.*?)\n=== END PAGE ===",
        re.DOTALL,
    )
    for match in page_pattern.finditer(full_response):
        filename = match.group(1).strip()
        content = match.group(2).strip()
        # Ensure .md extension
        if not filename.endswith(".md"):
            filename += ".md"
        store.write_page(filename, content)
        pages_written.append(filename)
        yield {"type": "page", "content": filename, "preview": content[:200]}

    # Extract and update index
    index_pattern = re.compile(
        r"=== INDEX ===\s*\n(.*?)\n=== END INDEX ===",
        re.DOTALL,
    )
    index_match = index_pattern.search(full_response)
    if index_match:
        new_index = index_match.group(1).strip()
        store.update_index(new_index)
        yield {"type": "index", "content": "index.md updated"}

    # Log the compile
    store.append_log("compile", f"{raw_filename} → {len(pages_written)} pages: {', '.join(pages_written)}")

    yield {
        "type": "done",
        "content": f"Compiled {raw_filename}: {len(pages_written)} pages written",
        "pages": pages_written,
    }


async def compile_all_new(
    store: WikiStore,
    model: str = "claude-haiku-4-5-20251001",
) -> AsyncIterator[dict[str, Any]]:
    """Compile all raw sources that haven't been compiled yet.

    Determines "new" by checking log.md for previous compile entries.
    """
    log = store.read_log()
    raw_files = store.list_raw()

    # Find already-compiled sources from log
    compiled = set()
    for line in log.split("\n"):
        if "compile |" in line.lower():
            # Extract filename from "compile | filename → ..."
            parts = line.split("|", 1)
            if len(parts) > 1:
                desc = parts[1].strip()
                # First word before " → " is the filename
                fname = desc.split("→")[0].strip().split(":")[-1].strip()
                if fname:
                    compiled.add(fname)

    new_sources = [f for f in raw_files if f not in compiled]

    if not new_sources:
        yield {"type": "status", "content": "No new sources to compile."}
        return

    yield {"type": "status", "content": f"Found {len(new_sources)} new sources to compile"}

    for filename in new_sources:
        async for event in compile_source(store, filename, model):
            yield event


# --- Query operation ---

async def query(
    store: WikiStore,
    question: str,
    file_back: bool = False,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Query the wiki — search, synthesize, optionally file back.

    Uses Sonnet for synthesis (higher quality than Haiku for analysis).
    """
    # Step 1: Read index to find relevant pages
    index = store.read_index()

    # Step 2: Also do keyword search for backup
    from isaac.wiki.search import search
    search_results = search(store, question, max_results=5)

    # Step 3: Read the relevant pages
    # From search results + any pages mentioned in index that match
    pages_to_read = set()
    for r in search_results:
        pages_to_read.add(r["path"])

    # Also scan index for relevant page names
    question_lower = question.lower()
    for line in index.split("\n"):
        # Look for markdown links like [page-name](pages/page-name.md) or [[page-name]]
        links = re.findall(r"\[\[([^\]]+)\]\]", line)
        for link in links:
            if any(term in link.lower() for term in question_lower.split()):
                pages_to_read.add(link if link.endswith(".md") else link + ".md")

    # Read page contents (cap at 10 pages)
    context_pages = []
    for page_path in list(pages_to_read)[:10]:
        content = store.read_page(page_path)
        if content:
            context_pages.append(f"### {page_path}\n{content}")

    if not context_pages:
        return {
            "answer": "No relevant wiki pages found for this question. Try ingesting more sources.",
            "pages_consulted": [],
            "filed_back": None,
        }

    wiki_context = "\n\n---\n\n".join(context_pages)

    # Step 4: Synthesize answer
    client = _get_client()
    user_msg = f"""## Wiki Index
{index}

## Relevant Pages
{wiki_context}

## Question
{question}"""

    resp = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=QUERY_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer = resp.content[0].text

    # Step 5: Check if the LLM suggested filing back
    filed_page = None
    new_page_match = re.search(
        r"=== NEW PAGE:\s*(.+?)\s*===\s*\n(.*?)\n=== END PAGE ===",
        answer,
        re.DOTALL,
    )
    if new_page_match and file_back:
        filename = new_page_match.group(1).strip()
        content = new_page_match.group(2).strip()
        if not filename.endswith(".md"):
            filename += ".md"
        store.write_page(filename, content)
        filed_page = filename
        # Clean the suggested page block from the answer shown to user
        answer = answer[:new_page_match.start()] + answer[new_page_match.end():]
        answer = answer.strip()

    # Log
    store.append_log("query", question[:80])

    return {
        "answer": answer,
        "pages_consulted": list(pages_to_read),
        "filed_back": filed_page,
    }


# --- Lint operation ---

async def lint(
    store: WikiStore,
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    """Run health checks on the wiki."""
    pages = store.list_pages()
    if not pages:
        return {"issues": [], "summary": "Wiki has no pages to lint."}

    # Collect all page contents
    all_pages = []
    all_links: dict[str, list[str]] = {}  # page → outgoing links
    inbound: dict[str, int] = {p: 0 for p in pages}

    for page_path in pages:
        content = store.read_page(page_path)
        if content:
            all_pages.append(f"### {page_path}\n{content}")
            links = re.findall(r"\[\[([^\]]+)\]\]", content)
            all_links[page_path] = links
            for link in links:
                link_path = link if link.endswith(".md") else link + ".md"
                if link_path in inbound:
                    inbound[link_path] += 1

    issues = []

    # 1. Orphan pages (no inbound links, not index/log/schema)
    for page_path, count in inbound.items():
        if count == 0 and page_path not in ("index.md", "log.md", "schema.md"):
            issues.append({
                "type": "orphan",
                "page": page_path,
                "description": f"No other page links to [[{page_path}]]",
            })

    # 2. Dead links (link targets that don't exist)
    for page_path, links in all_links.items():
        for link in links:
            link_path = link if link.endswith(".md") else link + ".md"
            if link_path not in pages:
                issues.append({
                    "type": "dead_link",
                    "page": page_path,
                    "target": link,
                    "description": f"[[{link}]] in {page_path} points to non-existent page",
                })

    # 3. Stale pages (not mentioned in recent log entries)
    # This is a heuristic — pages not updated recently may be stale

    # 4. LLM-powered checks (contradictions, missing concepts, gaps)
    if len(all_pages) > 0:
        wiki_dump = "\n\n---\n\n".join(all_pages[:30])  # Cap to avoid token overflow
        index = store.read_index()

        client = _get_client()
        resp = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=LINT_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"## Index\n{index}\n\n## All Pages\n{wiki_dump}",
            }],
        )
        llm_report = resp.content[0].text
        issues.append({
            "type": "llm_analysis",
            "description": llm_report,
        })

    store.append_log("lint", f"Found {len(issues)} issues across {len(pages)} pages")

    return {
        "issues": issues,
        "page_count": len(pages),
        "orphan_count": sum(1 for i in issues if i["type"] == "orphan"),
        "dead_link_count": sum(1 for i in issues if i["type"] == "dead_link"),
        "summary": f"Lint complete: {len(pages)} pages, {len(issues)} issues found",
    }
