"""Core type definitions for ISAAC."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"


class PermissionLevel(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    permission: PermissionLevel = PermissionLevel.AUTO
    is_read_only: bool = False
    source: str = "built_in"  # "built_in", "plugin", "connector:{name}", "app", "skill"


@dataclass
class AgentConfig:
    name: str
    soul: str = "default"
    model: str = "claude-sonnet-4-6"
    tools: list[str] = field(default_factory=lambda: ["*"])
    mcp_servers: list[str] = field(default_factory=list)
    max_iterations: int = 25
    context_budget: int = 180_000
    cwd: str | None = None
    auto_start: bool = False
    expose_as_tool: bool = False
    tool_description: str = ""
    computer_use: bool = False
    max_time_seconds: int = 0  # Wall clock timeout per turn (0 = no limit, default)
    sandbox: str = "fly"  # "fly", "e2b" — all agents get a VM by default
    sandbox_template: str = "ubuntu-24"
    sandbox_size: str = ""  # Machine size: "shared-cpu-1x", "performance-4x", "a10", "a100-80gb"
    sandbox_disk_gb: int = 0  # Disk size (0 = default: 10GB cpu, 50GB gpu)
    scope: str = "cwd"  # "cwd" (current dir), "home" (~/), "system" (/) — controls tool reach
    computer_scope: bool = False  # Enable aggressive computer-scope tools (clipboard, process, gui, etc.)


@dataclass
class SessionState:
    session_id: str
    agent_name: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    turn_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0


@dataclass
class Handoff:
    summary: str
    open_questions: list[str] = field(default_factory=list)
    continuation_prompt: str = ""
    memory_keys: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
