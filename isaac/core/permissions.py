"""Permission model — Claude Code pattern.

The user sending a message IS the approval. Tools execute freely within
a turn's agentic loop. Only explicitly denied tools get blocked.

This is how Claude Code works:
- You send a message
- The agent runs tools to fulfill it
- You see results streaming in
- You do NOT get prompted for every file_write and bash

The deny list is for things you NEVER want the agent to do:
- rm -rf, format disk, etc.
- Configurable via agents.yaml or /deny command

If you want approval gates back for specific tools, add them
to the ask list explicitly.
"""
from __future__ import annotations

from isaac.core.types import PermissionLevel, ToolDef


class PermissionGate:
    """Manages tool permissions. Default: auto-approve everything.

    Only tools in the deny set get blocked. Only tools in the ask set
    get prompted. Everything else runs freely.
    """

    def __init__(self) -> None:
        self._deny: set[str] = set()
        self._ask: set[str] = set()
        self._session_allows: set[str] = set()

    def deny(self, tool_name: str) -> None:
        """Block a tool entirely."""
        self._deny.add(tool_name)
        self._ask.discard(tool_name)

    def require_approval(self, tool_name: str) -> None:
        """Require approval for a specific tool (opt-in, not default)."""
        self._ask.add(tool_name)

    def session_allow(self, tool_name: str) -> None:
        """Auto-approve a tool for the rest of this session (clears ask)."""
        self._session_allows.add(tool_name)
        self._ask.discard(tool_name)

    def set_override(self, tool_name: str, level: PermissionLevel) -> None:
        """Set a permission override (backward compat)."""
        if level == PermissionLevel.DENY:
            self.deny(tool_name)
        elif level == PermissionLevel.ASK:
            self.require_approval(tool_name)
        elif level == PermissionLevel.AUTO:
            self._session_allows.add(tool_name)

    def check(self, tool: ToolDef) -> PermissionLevel:
        """Resolve effective permission.

        Priority:
        1. Deny list → DENY (hard block, no override)
        2. Session allows → AUTO (user approved during this session)
        3. Ask list → ASK (explicit opt-in for this tool)
        4. Everything else → AUTO (Claude Code default)
        """
        if tool.name in self._deny:
            return PermissionLevel.DENY

        if tool.name in self._session_allows:
            return PermissionLevel.AUTO

        if tool.name in self._ask:
            return PermissionLevel.ASK

        # Default: auto-approve. The user's message is the approval.
        return PermissionLevel.AUTO

    def load_rules(self, rules: dict[str, str]) -> None:
        """Load permission rules from config.

        Format: {tool_name: 'auto'|'ask'|'deny'}
        """
        for name, level_str in rules.items():
            try:
                level = PermissionLevel(level_str)
                self.set_override(name, level)
            except ValueError:
                pass
