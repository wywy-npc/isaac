"""Built-in tools — the core toolset every agent gets."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef
from isaac.memory.store import MemoryStore


def build_builtin_tools(
    memory: MemoryStore, cwd: str | None = None, embedding_store: Any = None,
) -> dict[str, tuple[ToolDef, Any]]:
    """Build the registry of built-in tools."""
    working_dir = cwd or os.getcwd()
    registry: dict[str, tuple[ToolDef, Any]] = {}

    # --- Memory tools (auto-approved, read/write) ---

    async def memory_search(query: str) -> dict[str, Any]:
        results = memory.search(query, max_results=5)
        return {
            "results": [
                {"path": n.path, "content": n.content[:500], "tags": n.tags}
                for n in results
            ]
        }

    registry["memory_search"] = (
        ToolDef(
            name="memory_search",
            description="Search memory for relevant knowledge. Returns matching memory nodes.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        memory_search,
    )

    async def memory_read(path: str) -> dict[str, Any]:
        node = memory.read(path)
        if not node:
            return {"error": f"Not found: {path}"}
        return {"path": node.path, "content": node.content, "meta": node.meta}

    registry["memory_read"] = (
        ToolDef(
            name="memory_read",
            description="Read a specific memory node by path.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Memory node path"}},
                "required": ["path"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        memory_read,
    )

    async def memory_write(path: str, content: str, tags: list[str] | None = None, importance: float = 0.5) -> dict[str, Any]:
        from isaac.memory.linker import auto_link
        meta = {"tags": tags or [], "importance": importance}
        # Auto-link to related nodes and create stubs for new entities
        content, created_stubs = auto_link(content, path, memory, embedding_store)
        node = memory.write(path, content, meta)
        # Generate embedding if store available
        if embedding_store:
            try:
                embedding_store.embed_and_store(path, content)
            except Exception:
                pass  # never block memory writes on embedding failures
        return {"written": node.path, "links": node.outgoing_links, "created_stubs": created_stubs}

    registry["memory_write"] = (
        ToolDef(
            name="memory_write",
            description=(
                "Write or update a memory node. Auto-links to related existing nodes via [[wikilinks]] "
                "and creates stub nodes for new entities mentioned in the content. "
                "Format: use markdown with a # heading, body content, and [[path/to/node.md]] links. "
                "Organize paths: people/name.md, entities/name.md, projects/name.md, topics/name.md, "
                "deals/name.md, logs/date.md. Scout follows links automatically for context retrieval."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Memory node path (e.g. 'people/jane-doe.md', 'entities/acme-corp.md', 'projects/isaac.md')"},
                    "content": {"type": "string", "description": "Markdown content with # heading, body, and optional [[wikilinks]]"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                    "importance": {"type": "number", "description": "Importance score 0-1"},
                },
                "required": ["path", "content"],
            },
            permission=PermissionLevel.AUTO,
        ),
        memory_write,
    )

    # --- File tools ---

    async def file_read(path: str, offset: int = 0, limit: int = 0) -> dict[str, Any]:
        """Read a file. Supports pagination via offset (line number) and limit (line count)."""
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        if not target.exists():
            return {"error": f"File not found: {target}"}
        try:
            content = target.read_text()
            total_size = len(content)

            # Line-based pagination if offset or limit specified
            if offset > 0 or limit > 0:
                lines = content.split("\n")
                total_lines = len(lines)
                start = min(offset, total_lines)
                end = min(start + limit, total_lines) if limit > 0 else total_lines
                content = "\n".join(lines[start:end])
                return {
                    "path": str(target),
                    "content": content,
                    "offset": start,
                    "lines_returned": end - start,
                    "total_lines": total_lines,
                    "has_more": end < total_lines,
                }

            # Default: return up to 200KB
            if total_size > 200_000:
                content = content[:200_000]
                return {
                    "path": str(target),
                    "content": content + "\n\n[... truncated ...]",
                    "total_size": total_size,
                    "truncated": True,
                    "hint": "Use offset and limit params to read specific sections",
                }
            return {"path": str(target), "content": content}
        except Exception as e:
            return {"error": str(e)}

    registry["file_read"] = (
        ToolDef(
            name="file_read",
            description=(
                "Read a file from the filesystem. Supports pagination: "
                "use offset (line number) and limit (line count) for large files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (absolute or relative to cwd)"},
                    "offset": {"type": "integer", "description": "Start reading from this line number (0-based)", "default": 0},
                    "limit": {"type": "integer", "description": "Max lines to read (0 = all)", "default": 0},
                },
                "required": ["path"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        file_read,
    )

    async def file_write(path: str, content: str) -> dict[str, Any]:
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {"written": str(target), "bytes": len(content)}

    registry["file_write"] = (
        ToolDef(
            name="file_write",
            description="Write content to a file. Creates parent directories if needed.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
            permission=PermissionLevel.AUTO,
        ),
        file_write,
    )

    async def file_list(path: str = ".", pattern: str = "*") -> dict[str, Any]:
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        if not target.exists():
            return {"error": f"Directory not found: {target}"}
        files = sorted(str(p.relative_to(target)) for p in target.glob(pattern) if p.is_file())
        dirs = sorted(str(p.relative_to(target)) for p in target.iterdir() if p.is_dir())
        return {"files": files[:500], "dirs": dirs[:100]}

    registry["file_list"] = (
        ToolDef(
            name="file_list",
            description="List files and directories.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                    "pattern": {"type": "string", "description": "Glob pattern", "default": "*"},
                },
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        file_list,
    )

    async def file_search(pattern: str, path: str = ".", include: str = "", max_results: int = 50) -> dict[str, Any]:
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        try:
            cmd = ["grep", "-rl", "--binary-files=without-match"]
            if include:
                for ext in include.split(","):
                    ext = ext.strip()
                    if not ext.startswith("*."):
                        ext = f"*.{ext}"
                    cmd.extend(["--include", ext])
            cmd.extend([pattern, str(target)])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            files = stdout.decode().strip().split("\n")[:max_results]
            return {"matches": [f for f in files if f], "count": len([f for f in files if f])}
        except asyncio.TimeoutError:
            return {"error": "Search timed out", "matches": []}
        except Exception as e:
            return {"error": str(e)}

    registry["file_search"] = (
        ToolDef(
            name="file_search",
            description=(
                "Search file contents using grep. Returns matching file paths. "
                "Searches all text files by default. Use include to filter by extension (e.g. 'py,ts,go')."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex)"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."},
                    "include": {"type": "string", "description": "Comma-separated file extensions to search (e.g. 'py,ts,go'). Empty = all text files."},
                    "max_results": {"type": "integer", "description": "Max files to return", "default": 50},
                },
                "required": ["pattern"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        file_search,
    )

    # --- Shell tool ---

    async def bash(command: str, timeout: int = 120) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode()
            if len(output) > 30_000:
                output = output[:15_000] + "\n\n[... truncated ...]\n\n" + output[-15_000:]
            return {
                "exit_code": proc.returncode,
                "stdout": output,
                "stderr": stderr.decode()[:5000],
            }
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    registry["bash"] = (
        ToolDef(
            name="bash",
            description="Execute a shell command. Use for system operations, git, package management, etc.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
                },
                "required": ["command"],
            },
            permission=PermissionLevel.AUTO,
        ),
        bash,
    )

    # --- Web search (DuckDuckGo free, Brave if API key set) ---

    async def web_search(query: str, max_results: int = 10) -> dict[str, Any]:
        max_results = min(max(1, max_results), 20)
        # Prefer Brave if key is set
        brave_key = os.environ.get("BRAVE_API_KEY")
        if brave_key:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": max_results},
                        headers={"X-Subscription-Token": brave_key},
                    )
                    data = resp.json()
                    results = []
                    for r in data.get("web", {}).get("results", [])[:max_results]:
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("description", ""),
                        })
                    return {"results": results}
            except Exception as e:
                return {"results": [], "error": f"Brave search failed: {e}"}

        # Fallback: DuckDuckGo (free, no API key)
        try:
            from duckduckgo_search import DDGS
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: DDGS().text(query, max_results=max_results)
            )
            results = [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in (raw or [])
            ]
            return {"results": results}
        except Exception as e:
            return {"results": [], "error": f"Web search failed: {e}"}

    registry["web_search"] = (
        ToolDef(
            name="web_search",
            description="Search the web. Returns titles, URLs, and snippets.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (1-20)", "default": 10},
                },
                "required": ["query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        web_search,
    )

    # --- Web fetch (read any URL) ---

    async def web_fetch(url: str, extract: str = "text", max_bytes: int = 30000) -> dict[str, Any]:
        """Fetch a URL and return its content. Handles HTML, JSON, plain text."""
        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed. Run: pip install httpx"}
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                resp = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ISAAC/1.0; +https://github.com/isaac)",
                })
                content_type = resp.headers.get("content-type", "")
                status = resp.status_code

                if "application/json" in content_type:
                    return {"url": url, "status": status, "content_type": "json", "content": resp.text[:max_bytes]}

                if "text/plain" in content_type or "text/csv" in content_type:
                    return {"url": url, "status": status, "content_type": "text", "content": resp.text[:max_bytes]}

                # HTML → extract readable content
                raw_html = resp.text
                if extract == "raw":
                    return {"url": url, "status": status, "content_type": "html", "content": raw_html[:max_bytes]}

                # Lightweight HTML to text conversion
                import re as _re
                text = raw_html
                # Remove script and style blocks
                text = _re.sub(r'<script[^>]*>.*?</script>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
                text = _re.sub(r'<style[^>]*>.*?</style>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
                # Convert common elements
                text = _re.sub(r'<br\s*/?>', '\n', text, flags=_re.IGNORECASE)
                text = _re.sub(r'</(p|div|h[1-6]|li|tr)>', '\n', text, flags=_re.IGNORECASE)
                text = _re.sub(r'<(h[1-6])[^>]*>', r'\n## ', text, flags=_re.IGNORECASE)
                # Extract links if requested
                links = []
                if extract == "links":
                    links = _re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', raw_html, _re.IGNORECASE | _re.DOTALL)
                    links = [{"url": u, "text": _re.sub(r'<[^>]+>', '', t).strip()} for u, t in links[:100]]
                # Strip remaining tags
                text = _re.sub(r'<[^>]+>', '', text)
                # Collapse whitespace
                text = _re.sub(r'\n{3,}', '\n\n', text)
                text = _re.sub(r' {2,}', ' ', text)
                text = text.strip()

                result: dict[str, Any] = {"url": url, "status": status, "content_type": "html", "content": text[:max_bytes]}
                if links:
                    result["links"] = links
                if len(text) > max_bytes:
                    result["truncated"] = True
                    result["total_chars"] = len(text)
                return result
        except Exception as e:
            return {"error": str(e), "url": url}

    registry["web_fetch"] = (
        ToolDef(
            name="web_fetch",
            description=(
                "Fetch and read a URL. Returns page content as text (HTML converted to readable text). "
                "Use extract='links' to get all links, 'raw' for raw HTML. "
                "Handles HTML, JSON, plain text. For paginated reading, use max_bytes offset."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "extract": {"type": "string", "description": "Extraction mode: text (default), links, raw", "default": "text"},
                    "max_bytes": {"type": "integer", "description": "Max content bytes to return (default 30000)", "default": 30000},
                },
                "required": ["url"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        web_fetch,
    )

    # --- HTTP request (arbitrary API calls) ---

    async def http_request(
        url: str, method: str = "GET", headers: dict[str, str] | None = None,
        body: str = "", content_type: str = "application/json", timeout: int = 60,
    ) -> dict[str, Any]:
        """Make an HTTP request. Supports all methods, custom headers, JSON/form body."""
        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed. Run: pip install httpx"}
        try:
            method = method.upper()
            req_headers = dict(headers) if headers else {}
            kwargs: dict[str, Any] = {"headers": req_headers, "timeout": timeout, "follow_redirects": True}

            if body and method in ("POST", "PUT", "PATCH"):
                if content_type == "application/json":
                    kwargs["content"] = body
                    req_headers.setdefault("Content-Type", "application/json")
                elif content_type == "application/x-www-form-urlencoded":
                    kwargs["content"] = body
                    req_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
                else:
                    kwargs["content"] = body
                    req_headers.setdefault("Content-Type", content_type)

            async with httpx.AsyncClient() as client:
                resp = await client.request(method, url, **kwargs)
                resp_body = resp.text
                if len(resp_body) > 30_000:
                    resp_body = resp_body[:30_000] + "\n\n[... truncated ...]"
                return {
                    "status": resp.status_code,
                    "headers": dict(resp.headers),
                    "body": resp_body,
                    "url": str(resp.url),
                }
        except Exception as e:
            return {"error": str(e), "url": url}

    registry["http_request"] = (
        ToolDef(
            name="http_request",
            description=(
                "Make an HTTP request to any URL. Supports GET, POST, PUT, PATCH, DELETE. "
                "Use for REST API calls, webhooks, data fetching. "
                "JSON body by default; set content_type for form data or other formats."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Request URL"},
                    "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    "headers": {"type": "object", "description": "Request headers", "additionalProperties": {"type": "string"}},
                    "body": {"type": "string", "description": "Request body (for POST/PUT/PATCH)"},
                    "content_type": {"type": "string", "description": "Body content type", "default": "application/json"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
                },
                "required": ["url"],
            },
            permission=PermissionLevel.AUTO,
        ),
        http_request,
    )

    # --- File edit (surgical string replacement) ---

    async def file_edit(path: str, old_text: str, new_text: str, replace_all: bool = False) -> dict[str, Any]:
        """Surgical file edit — replace exact text without rewriting the whole file."""
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        if not target.exists():
            return {"error": f"File not found: {target}"}
        try:
            content = target.read_text()
            count = content.count(old_text)
            if count == 0:
                return {"error": "old_text not found in file", "path": str(target)}
            if count > 1 and not replace_all:
                return {
                    "error": f"old_text found {count} times. Set replace_all=true to replace all, or provide more context to make it unique.",
                    "count": count,
                    "path": str(target),
                }
            if replace_all:
                new_content = content.replace(old_text, new_text)
            else:
                new_content = content.replace(old_text, new_text, 1)
            target.write_text(new_content)
            return {"edited": str(target), "replacements": count if replace_all else 1}
        except Exception as e:
            return {"error": str(e)}

    registry["file_edit"] = (
        ToolDef(
            name="file_edit",
            description=(
                "Surgical file edit — replace exact text in a file without rewriting the whole thing. "
                "Provide old_text (must be unique in the file) and new_text. "
                "Set replace_all=true to replace all occurrences. "
                "Safer than file_write for modifying existing files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences", "default": False},
                },
                "required": ["path", "old_text", "new_text"],
            },
            permission=PermissionLevel.AUTO,
        ),
        file_edit,
    )

    # --- Agent delegation tool (stub — wired to real delegation in terminal.py) ---

    async def delegate_agent(agent_name: str, task: str) -> dict[str, Any]:
        return {
            "status": "delegated",
            "agent": agent_name,
            "task": task,
            "note": "Task delegated. Check agent output for results.",
        }

    registry["delegate_agent"] = (
        ToolDef(
            name="delegate_agent",
            description="Delegate a task to another agent in the constellation.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Name of the agent to delegate to"},
                    "task": {"type": "string", "description": "Task description for the agent"},
                },
                "required": ["agent_name", "task"],
            },
            permission=PermissionLevel.AUTO,
        ),
        delegate_agent,
    )

    # --- Continuation tool (clean-room handoff for heartbeats / long tasks) ---

    async def write_continuation(
        what_was_done: str,
        what_remains: str = "Nothing — task complete.",
        blocking_on: str = "Nothing.",
        artifacts: list[str] | None = None,
        next_priority: str = "",
    ) -> dict[str, Any]:
        from isaac.core.heartbeat import write_continuation as _write
        # Use the agent name from the working directory context
        agent_name = os.environ.get("ISAAC_AGENT", "default")
        path = _write(agent_name, what_was_done, what_remains, blocking_on, artifacts, next_priority)
        return {"written": path, "agent": agent_name}

    registry["write_continuation"] = (
        ToolDef(
            name="write_continuation",
            description="Write a structured handoff for the next run. Use when finishing a heartbeat or long task.",
            input_schema={
                "type": "object",
                "properties": {
                    "what_was_done": {"type": "string", "description": "Factual summary of actions taken"},
                    "what_remains": {"type": "string", "description": "Specific unfinished items"},
                    "blocking_on": {"type": "string", "description": "External dependencies or missing info"},
                    "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Paths to files/memory created"},
                    "next_priority": {"type": "string", "description": "What the next run should focus on first"},
                },
                "required": ["what_was_done"],
            },
            permission=PermissionLevel.AUTO,
        ),
        write_continuation,
    )

    # --- Get full result (retrieve overflow tool results) ---

    # Session-scoped overflow store — orchestrator writes here, agent reads back
    _overflow_store: dict[str, str] = {}

    async def get_full_result(tool_call_id: str, offset: int = 0, limit: int = 8000) -> dict[str, Any]:
        """Retrieve the full content of a truncated tool result."""
        if tool_call_id not in _overflow_store:
            return {"error": f"No overflow result found for tool_call_id '{tool_call_id}'. It may have been cleared or the ID is wrong."}
        full = _overflow_store[tool_call_id]
        chunk = full[offset:offset + limit]
        return {
            "content": chunk,
            "offset": offset,
            "limit": limit,
            "total_length": len(full),
            "has_more": offset + limit < len(full),
        }

    registry["get_full_result"] = (
        ToolDef(
            name="get_full_result",
            description=(
                "Retrieve the full content of a truncated tool result. "
                "When a tool result is too large, it gets truncated with a message pointing here. "
                "Use the tool_call_id from the truncation message. Supports pagination with offset/limit."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tool_call_id": {"type": "string", "description": "The tool_call_id from the truncated result"},
                    "offset": {"type": "integer", "description": "Character offset to start from", "default": 0},
                    "limit": {"type": "integer", "description": "Max characters to return", "default": 8000},
                },
                "required": ["tool_call_id"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        get_full_result,
    )

    # Expose the overflow store so the orchestrator can write to it
    registry["_overflow_store"] = _overflow_store  # type: ignore[assignment]

    # --- E2B sandbox tools (only if E2B_API_KEY is set) ---

    if os.environ.get("E2B_API_KEY"):
        try:
            from isaac.sandbox.e2b import E2BSandbox
            _sandbox = E2BSandbox()

            async def sandbox_execute(code: str, language: str = "python") -> dict[str, Any]:
                return await _sandbox.execute(code, language)

            registry["sandbox_execute"] = (
                ToolDef(
                    name="sandbox_execute",
                    description="Execute code in a cloud sandbox. Safe for untrusted code.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to execute"},
                            "language": {"type": "string", "description": "Language (python, javascript)", "default": "python"},
                        },
                        "required": ["code"],
                    },
                    permission=PermissionLevel.AUTO,
                ),
                sandbox_execute,
            )

            async def sandbox_upload(path: str) -> dict[str, Any]:
                return await _sandbox.upload(path)

            registry["sandbox_upload"] = (
                ToolDef(
                    name="sandbox_upload",
                    description="Upload a local file to the cloud sandbox.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Local file path to upload"}},
                        "required": ["path"],
                    },
                    permission=PermissionLevel.AUTO,
                ),
                sandbox_upload,
            )

            async def sandbox_download(path: str) -> dict[str, Any]:
                return await _sandbox.download(path)

            registry["sandbox_download"] = (
                ToolDef(
                    name="sandbox_download",
                    description="Download a file from the cloud sandbox.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Remote file path in sandbox"}},
                        "required": ["path"],
                    },
                    permission=PermissionLevel.AUTO,
                ),
                sandbox_download,
            )
        except ImportError:
            pass  # e2b-code-interpreter not installed

    # --- App tools (bolt any app as a tool) ---

    async def app_list() -> dict[str, Any]:
        from isaac.apps.manifest import list_manifests
        apps = list_manifests()
        return {"apps": apps, "count": len(apps)}

    registry["app_list"] = (
        ToolDef(
            name="app_list",
            description="List available external apps that can be run as tools.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        app_list,
    )

    async def app_run(app: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaac.apps.runner import AppRunner
        # Pass the full parent tool registry so the VM agent gets access to
        # web search, memory, MCP tools, connected apps — the full ISAAC surface
        runner = AppRunner(memory=memory, parent_tools=registry)
        result = await runner.run(app, inputs or {})
        return {
            "status": result.status,
            "summary": result.summary[:3000],
            "artifacts": result.artifacts,
            "duration": result.duration,
            "cost": result.cost,
            "error": result.error,
        }

    registry["app_run"] = (
        ToolDef(
            name="app_run",
            description="Run an external app on cloud GPU. Provisions compute, clones repo, runs the app, collects artifacts.",
            input_schema={
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "App name (from app_list)"},
                    "inputs": {
                        "type": "object",
                        "description": "App-specific inputs (see manifest for schema)",
                        "additionalProperties": True,
                    },
                },
                "required": ["app"],
            },
            permission=PermissionLevel.AUTO,
        ),
        app_run,
    )

    # --- Service connection tools (manage upstream MCP servers) ---

    async def list_services() -> dict[str, Any]:
        from isaac.mcp.connections import list_connections
        return {"services": list_connections()}

    registry["list_services"] = (
        ToolDef(
            name="list_services",
            description="List all connected external services (MCP servers, APIs).",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        list_services,
    )

    async def connect_service(
        name: str, command: str = "", args: str = "", url: str = "",
        transport: str = "stdio", description: str = "",
    ) -> dict[str, Any]:
        from isaac.mcp.connections import add_connection
        args_list = args.split() if args else []
        conn = add_connection(
            name=name, type="mcp", command=command, args=args_list,
            url=url, transport=transport, description=description,
        )
        return {
            "connected": name,
            "note": "Added to connections.yaml. Restart `isaac serve` to activate.",
        }

    registry["connect_service"] = (
        ToolDef(
            name="connect_service",
            description="Connect a new external service (MCP server). Saves to connections.yaml.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name (e.g. 'google', 'slack')"},
                    "command": {"type": "string", "description": "Command for stdio MCP server (e.g. 'npx')"},
                    "args": {"type": "string", "description": "Space-separated args (e.g. '@anthropic-ai/google-mcp')"},
                    "url": {"type": "string", "description": "URL for HTTP MCP server"},
                    "transport": {"type": "string", "description": "stdio or http", "default": "stdio"},
                    "description": {"type": "string", "description": "What this service does"},
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
        ),
        connect_service,
    )

    async def disconnect_service(name: str) -> dict[str, Any]:
        from isaac.mcp.connections import remove_connection
        if remove_connection(name):
            return {"disconnected": name}
        return {"error": f"Service '{name}' not found"}

    registry["disconnect_service"] = (
        ToolDef(
            name="disconnect_service",
            description="Remove an external service connection.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name to disconnect"},
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
        ),
        disconnect_service,
    )

    # --- Service catalog tools (discover + auto-connect) ---

    async def catalog_search(query: str) -> dict[str, Any]:
        """Search the catalog of known connectable services."""
        from isaac.mcp.catalog import search_catalog
        results = search_catalog(query)
        return {
            "results": [
                {
                    "name": e.name,
                    "description": e.description,
                    "category": e.category,
                    "auth_type": e.auth_type,
                    "provides": e.provides,
                    "setup_url": e.setup_url,
                }
                for e in results
            ]
        }

    registry["catalog_search"] = (
        ToolDef(
            name="catalog_search",
            description="Search available services that can be connected (Gmail, Slack, GitHub, etc.).",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search by name, description, or category (communication, development, search, database, productivity, automation)"},
                },
                "required": ["query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        catalog_search,
    )

    async def catalog_list() -> dict[str, Any]:
        """List all services in the catalog."""
        from isaac.mcp.catalog import get_full_catalog
        catalog = get_full_catalog()
        return {
            "services": [
                {
                    "name": e.name,
                    "description": e.description,
                    "category": e.category,
                    "auth_type": e.auth_type,
                }
                for e in catalog.values()
            ],
            "count": len(catalog),
        }

    registry["catalog_list"] = (
        ToolDef(
            name="catalog_list",
            description="List all known services in the catalog that can be connected.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        catalog_list,
    )

    async def catalog_setup(name: str) -> dict[str, Any]:
        """Get full setup instructions for connecting a service.

        Returns everything needed: command, env vars, auth steps, URLs.
        For services requiring API keys, the agent should tell the user
        exactly where to get the key and ask them to set it.
        """
        from isaac.mcp.catalog import get_full_catalog
        catalog = get_full_catalog()
        entry = catalog.get(name)
        if not entry:
            return {"error": f"Service '{name}' not in catalog. Use catalog_search to find it."}

        return {
            "name": entry.name,
            "description": entry.description,
            "command": entry.command,
            "args": entry.args,
            "auth_type": entry.auth_type,
            "env_vars": entry.env_vars,
            "setup_url": entry.setup_url,
            "setup_instructions": entry.setup_instructions,
            "provides": entry.provides,
            "npm_package": entry.npm_package,
            "next_steps": (
                "To connect this service:\n"
                "1. Ask the user to provide the required credentials listed in env_vars\n"
                "2. Use connect_service to add it to connections.yaml with the command and args shown\n"
                "3. The credentials go in env vars on the user's machine (never in the yaml file)\n"
                "4. Remind the user to restart the session or run /reload to activate"
            ),
        }

    registry["catalog_setup"] = (
        ToolDef(
            name="catalog_setup",
            description="Get full setup instructions for connecting a service from the catalog.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name from catalog (e.g. 'google', 'slack', 'github')"},
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        catalog_setup,
    )

    async def catalog_add(
        name: str, description: str, command: str, args: str = "",
        auth_type: str = "none", env_vars: str = "",
        provides: str = "", npm_package: str = "",
        setup_instructions: str = "",
    ) -> dict[str, Any]:
        """Add a new service to the catalog (Toolsmith can do this autonomously).

        This doesn't connect the service — it adds it to the catalog so
        any agent can later connect it via catalog_setup + connect_service.
        """
        from isaac.mcp.catalog import CatalogEntry, save_custom_catalog_entry

        env_dict = {}
        if env_vars:
            for pair in env_vars.split(","):
                if "=" in pair:
                    k, v = pair.strip().split("=", 1)
                    env_dict[k.strip()] = v.strip()

        entry = CatalogEntry(
            name=name,
            description=description,
            command=command,
            args=args.split() if args else [],
            auth_type=auth_type,
            env_vars=env_dict,
            provides=provides.split(",") if provides else [],
            npm_package=npm_package,
            setup_instructions=setup_instructions,
            category="custom",
        )
        save_custom_catalog_entry(entry)
        return {"added": name, "catalog": "custom"}

    registry["catalog_add"] = (
        ToolDef(
            name="catalog_add",
            description="Add a new service to the catalog. Use when Toolsmith discovers a new MCP server or builds a connector.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name"},
                    "description": {"type": "string", "description": "What this service does"},
                    "command": {"type": "string", "description": "Command to run (e.g. 'npx')"},
                    "args": {"type": "string", "description": "Space-separated args"},
                    "auth_type": {"type": "string", "description": "api_key, oauth, or none"},
                    "env_vars": {"type": "string", "description": "Comma-separated KEY=description pairs"},
                    "provides": {"type": "string", "description": "Comma-separated list of tool names this service provides"},
                    "npm_package": {"type": "string", "description": "NPM package name if applicable"},
                    "setup_instructions": {"type": "string", "description": "Human-readable setup steps"},
                },
                "required": ["name", "description", "command"],
            },
            permission=PermissionLevel.AUTO,
        ),
        catalog_add,
    )

    return registry
