"""Concurrent tool execution — Claude Code pattern.

Safe tools (is_read_only=True) run in parallel via asyncio.gather.
Exclusive tools (is_read_only=False) serialize after safe tools finish.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from isaac.core.types import PermissionLevel, ToolCall, ToolDef, ToolResult


class ToolExecutor:
    """Dispatches tool calls with concurrency control."""

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        registry: dict[str, tuple[ToolDef, Any]],
        permission_check: Any,  # (ToolCall) -> PermissionLevel
        approval_fn: Any | None = None,  # async (ToolCall) -> bool
    ) -> AsyncIterator[tuple[str, ToolCall | ToolResult]]:
        """Execute a batch of tool calls with concurrency.

        Yields tuples of (event_type, payload):
          ("call", ToolCall)       — tool is about to execute
          ("approval", ToolCall)   — tool needs user approval
          ("result", ToolResult)   — tool finished
        """
        # Partition into safe (parallel) and exclusive (serial)
        safe: list[ToolCall] = []
        exclusive: list[ToolCall] = []

        for tc in tool_calls:
            entry = registry.get(tc.name)
            if entry and entry[0].is_read_only:
                safe.append(tc)
            else:
                exclusive.append(tc)

        # Run safe tools in parallel
        if safe:
            tasks = []
            for tc in safe:
                yield ("call", tc)
                perm = permission_check(tc)
                if perm == PermissionLevel.DENY:
                    yield ("result", ToolResult(
                        tool_call_id=tc.id,
                        content=f"Permission denied for tool '{tc.name}'",
                        is_error=True,
                    ))
                    continue
                if perm == PermissionLevel.ASK:
                    yield ("approval", tc)
                    if approval_fn:
                        approved = await approval_fn(tc)
                        if not approved:
                            yield ("result", ToolResult(
                                tool_call_id=tc.id, content="User denied this tool call.", is_error=True,
                            ))
                            continue
                tasks.append((tc, self._exec_one(tc, registry)))

            if tasks:
                results = await asyncio.gather(
                    *[t[1] for t in tasks], return_exceptions=True
                )
                for (tc, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        yield ("result", ToolResult(
                            tool_call_id=tc.id,
                            content=f"Error executing {tc.name}: {result}",
                            is_error=True,
                        ))
                    else:
                        yield ("result", result)

        # Run exclusive tools sequentially
        for tc in exclusive:
            yield ("call", tc)
            perm = permission_check(tc)
            if perm == PermissionLevel.DENY:
                yield ("result", ToolResult(
                    tool_call_id=tc.id,
                    content=f"Permission denied for tool '{tc.name}'",
                    is_error=True,
                ))
                continue
            if perm == PermissionLevel.ASK:
                yield ("approval", tc)
                if approval_fn:
                    approved = await approval_fn(tc)
                    if not approved:
                        yield ("result", ToolResult(
                            tool_call_id=tc.id, content="User denied this tool call.", is_error=True,
                        ))
                        continue
            try:
                result = await self._exec_one(tc, registry)
                yield ("result", result)
            except Exception as e:
                yield ("result", ToolResult(
                    tool_call_id=tc.id,
                    content=f"Error executing {tc.name}: {e}",
                    is_error=True,
                ))

    async def _exec_one(
        self, tc: ToolCall, registry: dict[str, tuple[ToolDef, Any]]
    ) -> ToolResult:
        """Execute a single tool call."""
        entry = registry.get(tc.name)
        if not entry:
            return ToolResult(
                tool_call_id=tc.id,
                content=f"Error: unknown tool '{tc.name}'",
                is_error=True,
            )
        _, handler = entry
        result = await handler(**tc.input)
        content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        return ToolResult(tool_call_id=tc.id, content=content)
