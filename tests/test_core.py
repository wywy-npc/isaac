"""Core unit tests for ISAAC."""
import json
import tempfile
from pathlib import Path

import pytest

from isaac.core.types import AgentConfig, Message, Role, SessionState, ToolDef, PermissionLevel
from isaac.core.config import load_agents_config
from isaac.core.soul import load_soul, PLATFORM_SOUL
from isaac.core.context import (
    build_cached_system_prompt,
    build_overflow_preview,
    build_system_prompt,
    estimate_tokens,
    proactive_compact_on_resume,
    should_compact,
    truncate_tool_result,
    TOOL_DESC_MAX_CHARS,
    TOOL_RESULT_MAX_TOKENS,
    MAX_RECENT_TURNS,
)
from isaac.core.permissions import PermissionGate
from isaac.memory.store import MemoryStore
from isaac.memory.scout import MemoryScout
from isaac.agents.session import new_session, save_session, load_session, list_sessions
from isaac.agents.tools import build_builtin_tools
from isaac.mcp.tool_loader import ToolLoader


class TestTypes:
    def test_agent_config_defaults(self):
        cfg = AgentConfig(name="test")
        assert cfg.name == "test"
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.max_iterations == 0  # 0 = unlimited
        assert cfg.tools == ["*"]

    def test_message(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"
        assert msg.tool_calls == []


class TestConfig:
    def test_load_agents_from_yaml(self, tmp_path):
        cfg_file = tmp_path / "agents.yaml"
        cfg_file.write_text("""
agents:
  researcher:
    soul: research
    model: claude-opus-4-6
    tools: ["web_search", "memory_search"]
  ops:
    soul: ops
""")
        agents = load_agents_config(cfg_file)
        assert "researcher" in agents
        assert agents["researcher"].model == "claude-opus-4-6"
        assert agents["researcher"].soul == "research"
        assert "ops" in agents

    def test_load_default_when_missing(self, tmp_path):
        agents = load_agents_config(tmp_path / "nonexistent.yaml")
        assert "default" in agents


class TestSoul:
    def test_platform_soul_exists(self):
        assert "Isaac" in PLATFORM_SOUL

    def test_load_soul_returns_platform(self):
        soul = load_soul("nonexistent_soul")
        assert "Isaac" in soul

    def test_load_soul_with_file(self, tmp_path):
        from isaac.core import config as cfg
        original = cfg.SOULS_DIR
        cfg.SOULS_DIR = tmp_path

        soul_file = tmp_path / "test.md"
        soul_file.write_text("# Test Soul\nBe helpful.")

        soul = load_soul("test")
        assert "Test Soul" in soul
        assert "Isaac" in soul  # platform layer still there

        cfg.SOULS_DIR = original


class TestContext:
    def test_estimate_tokens(self):
        assert estimate_tokens("hello world") > 0

    def test_build_system_prompt(self):
        prompt = build_system_prompt(
            "You are helpful.",
            [{"name": "search", "description": "Search the web"}],
            "User prefers concise answers.",
        )
        assert "search" in prompt
        assert "helpful" in prompt
        assert "concise" in prompt

    def test_should_compact(self):
        state = SessionState(session_id="test", agent_name="test")
        # Empty — should not compact (not enough messages)
        assert not should_compact(state, 100_000)

        # Add lots of messages (must exceed MAX_RECENT_TURNS)
        for i in range(MAX_RECENT_TURNS + 50):
            state.messages.append(Message(role=Role.USER, content="x" * 5000))
        assert should_compact(state, 100_000)

    def test_truncate(self):
        short = "hello"
        assert truncate_tool_result(short) == short

        long = "x" * 20_000
        truncated = truncate_tool_result(long, max_chars=100)
        assert "truncated" in truncated
        assert len(truncated) < len(long)

    def test_tool_desc_truncation(self):
        """Tool descriptions capped at 150 chars in static layer (self-chat pattern)."""
        long_desc = "x" * 300
        prompt = build_system_prompt(
            "soul",
            [{"name": "tool", "description": long_desc}],
            "",
        )
        # The description in the prompt should be capped
        assert "x" * 200 not in prompt

    def test_cached_prompt_has_3_layers(self):
        blocks = build_cached_system_prompt(
            "soul text",
            [{"name": "t", "description": "d"}],
            "memory",
            "summary of earlier convo",
        )
        assert len(blocks) == 3
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}  # static
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}  # session
        assert "cache_control" not in blocks[2]  # turn (fresh)

    def test_cached_prompt_no_memory_block_when_empty(self):
        """On iteration 2+, memory is empty — no memory block emitted."""
        blocks = build_cached_system_prompt("soul", [{"name": "t", "description": "d"}], "")
        # Should only have static block (soul+tools combined), no memory
        for block in blocks:
            assert "Relevant Memory" not in block["text"]

    def test_overflow_preview(self):
        short = "hello"
        preview, overflowed = build_overflow_preview(short)
        assert not overflowed
        assert preview == short

        # Big result
        big = "x" * (TOOL_RESULT_MAX_TOKENS * 8)
        preview, overflowed = build_overflow_preview(big)
        assert overflowed
        assert "truncated" in preview
        assert "get_full_result" in preview

    def test_proactive_compact_on_resume(self):
        state = SessionState(session_id="test", agent_name="test")
        assert not proactive_compact_on_resume(state)  # empty

        for i in range(MAX_RECENT_TURNS + 1):
            state.messages.append(Message(role=Role.USER, content="msg"))
        assert proactive_compact_on_resume(state)  # needs compaction


class TestPermissions:
    def test_auto_for_readonly(self):
        gate = PermissionGate()
        tool = ToolDef(
            name="search", description="", input_schema={},
            permission=PermissionLevel.ASK, is_read_only=True,
        )
        assert gate.check(tool) == PermissionLevel.AUTO

    def test_override(self):
        gate = PermissionGate()
        gate.set_override("bash", PermissionLevel.DENY)
        tool = ToolDef(name="bash", description="", input_schema={})
        assert gate.check(tool) == PermissionLevel.DENY

    def test_session_allow(self):
        gate = PermissionGate()
        # Default is now AUTO (Claude Code pattern)
        tool = ToolDef(name="bash", description="", input_schema={})
        assert gate.check(tool) == PermissionLevel.AUTO

        # Explicit ask, then session allow overrides
        gate.require_approval("bash")
        assert gate.check(tool) == PermissionLevel.ASK
        gate.session_allow("bash")
        assert gate.check(tool) == PermissionLevel.AUTO

    def test_deny_overrides_everything(self):
        gate = PermissionGate()
        gate.deny("dangerous_tool")
        tool = ToolDef(name="dangerous_tool", description="", input_schema={})
        assert gate.check(tool) == PermissionLevel.DENY
        # Even session_allow can't override deny
        gate.session_allow("dangerous_tool")
        assert gate.check(tool) == PermissionLevel.DENY


class TestMemory:
    def test_write_and_read(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.write("test/note.md", "Hello world", {"tags": ["test"], "importance": 0.8})

        node = store.read("test/note.md")
        assert node is not None
        assert "Hello world" in node.content
        assert node.importance == 0.8

    def test_search(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.write("projects/alpha.md", "Alpha project about machine learning")
        store.write("projects/beta.md", "Beta project about web development")

        results = store.search("machine learning")
        assert len(results) >= 1
        assert any("alpha" in r.path for r in results)

    def test_wiki_links(self, tmp_path):
        store = MemoryStore(tmp_path)
        node = store.write("index.md", "See [[projects/alpha]] and [[projects/beta]]")
        assert "projects/alpha" in node.outgoing_links
        assert "projects/beta" in node.outgoing_links

    def test_list_and_delete(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.write("a.md", "content a")
        store.write("b.md", "content b")
        assert len(store.list_all()) == 2
        store.delete("a.md")
        assert len(store.list_all()) == 1


class TestScout:
    @pytest.mark.asyncio
    async def test_scout_search(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.write("notes/ai.md", "Artificial intelligence and deep learning research")
        store.write("notes/cooking.md", "Recipe for pasta carbonara")

        scout = MemoryScout(store)
        result = await scout.search("artificial intelligence")
        assert "intelligence" in result.lower()


class TestSession:
    def test_create_and_save(self, tmp_path):
        from isaac.core import config as cfg
        original = cfg.SESSIONS_DIR
        cfg.SESSIONS_DIR = tmp_path

        state = new_session("test_agent")
        state.messages.append(Message(role=Role.USER, content="hello"))
        state.messages.append(Message(role=Role.ASSISTANT, content="hi there"))
        state.turn_count = 1
        state.total_cost = 0.001

        path = save_session(state)
        assert path.exists()

        loaded = load_session("test_agent", state.session_id)
        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.total_cost == 0.001

        sessions = list_sessions("test_agent")
        assert len(sessions) == 1

        cfg.SESSIONS_DIR = original


class TestTools:
    def test_builtin_tools_created(self, tmp_path):
        store = MemoryStore(tmp_path)
        tools = build_builtin_tools(store)
        assert "memory_search" in tools
        assert "memory_write" in tools
        assert "file_read" in tools
        assert "bash" in tools
        assert "web_search" in tools
        assert "delegate_agent" in tools

    @pytest.mark.asyncio
    async def test_memory_tools_work(self, tmp_path):
        store = MemoryStore(tmp_path)
        tools = build_builtin_tools(store)

        # Write
        _, write_fn = tools["memory_write"]
        result = await write_fn(path="test.md", content="hello world")
        assert "written" in result

        # Search
        _, search_fn = tools["memory_search"]
        result = await search_fn(query="hello")
        assert len(result["results"]) > 0


class TestToolLoader:
    def test_scan_empty(self, tmp_path):
        loader = ToolLoader(tmp_path)
        tools = loader.scan()
        assert len(tools) == 0

    def test_load_tool_file(self, tmp_path):
        tool_file = tmp_path / "test_tool.py"
        tool_file.write_text("""
import asyncio

async def greet(name: str) -> dict:
    return {"greeting": f"Hello {name}"}

TOOLS = [
    {
        "name": "greet",
        "description": "Greet someone",
        "params": {"name": str},
        "handler": greet,
    }
]
""")
        loader = ToolLoader(tmp_path)
        tools = loader.scan()
        assert "greet" in tools
        assert tools["greet"].description == "Greet someone"
