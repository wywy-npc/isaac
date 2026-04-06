"""Wiki tools — agent tool definitions for personal knowledge bases.

Follows the same pattern as isaac/plugins/skills.py — returns a
dict[str, tuple[ToolDef, handler]] that gets merged into the registry.
"""
from __future__ import annotations

from typing import Any

from isaac.core.types import PermissionLevel, ToolDef


def build_wiki_tools() -> dict[str, tuple[ToolDef, Any]]:
    """Build the wiki tool registry."""
    registry: dict[str, tuple[ToolDef, Any]] = {}

    # --- wiki_list ---

    async def wiki_list() -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        wikis = WikiStore.list_wikis()
        return {"wikis": wikis, "count": len(wikis)}

    registry["wiki_list"] = (
        ToolDef(
            name="wiki_list",
            description="List all personal knowledge wikis with page and source counts.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        wiki_list,
    )

    # --- wiki_create ---

    async def wiki_create(name: str, description: str = "") -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        try:
            store = WikiStore.create(name, description)
            return {"created": name, "path": str(store.dir)}
        except FileExistsError:
            return {"error": f"Wiki '{name}' already exists"}

    registry["wiki_create"] = (
        ToolDef(
            name="wiki_create",
            description="Create a new personal knowledge wiki.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Wiki name (lowercase, hyphens, e.g. 'ai-agents')"},
                    "description": {"type": "string", "description": "One-line description of the wiki's topic"},
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
        ),
        wiki_create,
    )

    # --- wiki_ingest ---

    async def wiki_ingest(wiki: str, source: str, title: str = "") -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        store = WikiStore(wiki)
        if not store.exists:
            return {"error": f"Wiki '{wiki}' not found. Create it first with wiki_create."}

        # Detect source type
        if source.startswith("http://") or source.startswith("https://"):
            from isaac.wiki.ingest import ingest_url
            return await ingest_url(store, source)
        elif "/" in source or source.endswith((".md", ".txt", ".pdf")):
            from isaac.wiki.ingest import ingest_file
            return await ingest_file(store, source)
        else:
            # Raw text
            if not title:
                title = source[:50]
            from isaac.wiki.ingest import ingest_text
            return await ingest_text(store, title, source)

    registry["wiki_ingest"] = (
        ToolDef(
            name="wiki_ingest",
            description=(
                "Ingest a source into a wiki's raw/ collection. "
                "Accepts: URLs (web articles), file paths (.md, .txt, .pdf), or raw text. "
                "Sources are stored immutably. Run wiki_compile after to build wiki pages."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "wiki": {"type": "string", "description": "Wiki name"},
                    "source": {"type": "string", "description": "URL, file path, or raw text to ingest"},
                    "title": {"type": "string", "description": "Title for raw text input (optional for URLs/files)"},
                },
                "required": ["wiki", "source"],
            },
            permission=PermissionLevel.AUTO,
        ),
        wiki_ingest,
    )

    # --- wiki_compile ---

    async def wiki_compile(wiki: str, source: str = "") -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        from isaac.wiki.compiler import compile_source, compile_all_new
        store = WikiStore(wiki)
        if not store.exists:
            return {"error": f"Wiki '{wiki}' not found."}

        results = []
        if source:
            async for event in compile_source(store, source):
                results.append(event)
        else:
            async for event in compile_all_new(store):
                results.append(event)

        # Return summary
        pages = []
        errors = []
        for r in results:
            if r["type"] == "page":
                pages.append(r["content"])
            elif r["type"] == "error":
                errors.append(r["content"])

        return {
            "pages_written": pages,
            "page_count": len(pages),
            "errors": errors,
            "events": results,
        }

    registry["wiki_compile"] = (
        ToolDef(
            name="wiki_compile",
            description=(
                "Compile raw sources into wiki pages using an LLM. "
                "The LLM reads sources, writes summary pages, creates concept/entity pages, "
                "maintains cross-references, and updates the index. "
                "Optionally specify a single source filename, or compile all new sources."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "wiki": {"type": "string", "description": "Wiki name"},
                    "source": {"type": "string", "description": "Specific raw source filename to compile (optional — compiles all new if omitted)"},
                },
                "required": ["wiki"],
            },
            permission=PermissionLevel.AUTO,
        ),
        wiki_compile,
    )

    # --- wiki_query ---

    async def wiki_query(wiki: str, question: str, file_back: bool = False) -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        from isaac.wiki.compiler import query as wiki_query_fn
        store = WikiStore(wiki)
        if not store.exists:
            return {"error": f"Wiki '{wiki}' not found."}
        return await wiki_query_fn(store, question, file_back=file_back)

    registry["wiki_query"] = (
        ToolDef(
            name="wiki_query",
            description=(
                "Query a personal wiki. Searches index and pages, synthesizes an answer "
                "with citations. Set file_back=true to save valuable answers as new wiki pages."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "wiki": {"type": "string", "description": "Wiki name"},
                    "question": {"type": "string", "description": "Question to answer from the wiki"},
                    "file_back": {"type": "boolean", "description": "Save the answer as a new wiki page if valuable", "default": False},
                },
                "required": ["wiki", "question"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        wiki_query,
    )

    # --- wiki_read ---

    async def wiki_read(wiki: str, page: str) -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        store = WikiStore(wiki)
        if not store.exists:
            return {"error": f"Wiki '{wiki}' not found."}
        # Try pages/ first, then top-level
        content = store.read_page(page)
        if content is None:
            content = store.read(page)
        if content is None:
            return {"error": f"Page not found: {page}"}
        return {"wiki": wiki, "page": page, "content": content}

    registry["wiki_read"] = (
        ToolDef(
            name="wiki_read",
            description="Read a specific wiki page. Also reads index.md, log.md, schema.md.",
            input_schema={
                "type": "object",
                "properties": {
                    "wiki": {"type": "string", "description": "Wiki name"},
                    "page": {"type": "string", "description": "Page path (e.g. 'overview.md', 'index.md', 'log.md')"},
                },
                "required": ["wiki", "page"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        wiki_read,
    )

    # --- wiki_lint ---

    async def wiki_lint(wiki: str) -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        from isaac.wiki.compiler import lint
        store = WikiStore(wiki)
        if not store.exists:
            return {"error": f"Wiki '{wiki}' not found."}
        return await lint(store)

    registry["wiki_lint"] = (
        ToolDef(
            name="wiki_lint",
            description=(
                "Run health checks on a wiki. Finds contradictions, orphan pages, "
                "dead links, stale claims, missing cross-references, and data gaps."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "wiki": {"type": "string", "description": "Wiki name"},
                },
                "required": ["wiki"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        wiki_lint,
    )

    # --- wiki_search ---

    async def wiki_search_fn(wiki: str, query: str) -> dict[str, Any]:
        from isaac.wiki.store import WikiStore
        from isaac.wiki.search import search
        store = WikiStore(wiki)
        if not store.exists:
            return {"error": f"Wiki '{wiki}' not found."}
        results = search(store, query, max_results=10)
        return {
            "results": [
                {"path": r["path"], "score": r["score"], "snippet": r["snippet"]}
                for r in results
            ],
            "count": len(results),
        }

    registry["wiki_search"] = (
        ToolDef(
            name="wiki_search",
            description="Search wiki pages by keyword. Use when the wiki is too large for index.md browsing.",
            input_schema={
                "type": "object",
                "properties": {
                    "wiki": {"type": "string", "description": "Wiki name"},
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["wiki", "query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        wiki_search_fn,
    )

    return registry
