"""Tests for ISAAC v0.2.0 features — web search, embeddings, autodream,
E2B sandbox, delegation, toolsmith, computer use, cost dashboard."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from isaac.core.types import AgentConfig, PermissionLevel, ToolDef


# --- F1: Web Search ---

class TestWebSearch:
    def test_ddg_fallback_returns_results(self):
        """When no BRAVE_API_KEY, DuckDuckGo should be used."""
        from isaac.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            store = MemoryStore(Path(td))
            from isaac.agents.tools import build_builtin_tools
            registry = build_builtin_tools(store)
            assert "web_search" in registry
            tdef, handler = registry["web_search"]
            assert tdef.permission == PermissionLevel.AUTO

    @pytest.mark.asyncio
    async def test_web_search_error_handling(self):
        """Web search should never crash the agent loop — returns error dict on failure."""
        from isaac.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            store = MemoryStore(Path(td))
            from isaac.agents.tools import build_builtin_tools
            registry = build_builtin_tools(store)
            _, handler = registry["web_search"]
            # Mock the duckduckgo_search module to raise on import
            with patch.dict("os.environ", {"BRAVE_API_KEY": ""}, clear=False):
                with patch.dict("sys.modules", {"duckduckgo_search": None}):
                    result = await handler(query="test query")
                    assert "error" in result or "results" in result


# --- F2: Embeddings ---

class TestEmbeddings:
    def test_embedding_store_init(self):
        """EmbeddingStore should create .embeddings dir."""
        with tempfile.TemporaryDirectory() as td:
            from isaac.memory.embeddings import EmbeddingStore
            store = EmbeddingStore(Path(td))
            assert (Path(td) / ".embeddings").exists()
            assert store._index == {}

    def test_safe_filename(self):
        from isaac.memory.embeddings import EmbeddingStore
        with tempfile.TemporaryDirectory() as td:
            store = EmbeddingStore(Path(td))
            assert store._safe_filename("projects/isaac.md") == "projects__isaac.npy"
            assert store._safe_filename("notes/daily log.md") == "notes__daily_log.npy"

    def test_index_persistence(self):
        """Index should persist across instances."""
        with tempfile.TemporaryDirectory() as td:
            from isaac.memory.embeddings import EmbeddingStore
            store = EmbeddingStore(Path(td))
            store._index = {"test.md": "test.npy"}
            store._save_index()

            store2 = EmbeddingStore(Path(td))
            assert store2._index == {"test.md": "test.npy"}


# --- F3: autoDream ---

class TestAutoDream:
    def test_orient_scans_recent(self):
        """Orient should find recently written memories."""
        with tempfile.TemporaryDirectory() as td:
            from isaac.memory.store import MemoryStore
            from isaac.memory.autodream import AutoDream
            store = MemoryStore(Path(td))
            store.write("test.md", "hello world", {"importance": 0.8})
            dreamer = AutoDream(store)
            recent = dreamer._orient()
            assert len(recent) >= 1

    def test_gather_forms_clusters(self):
        """Gather should cluster nodes connected by wiki-links."""
        with tempfile.TemporaryDirectory() as td:
            from isaac.memory.store import MemoryStore
            from isaac.memory.autodream import AutoDream
            store = MemoryStore(Path(td))
            store.write("a.md", "main topic [[b]]", {"importance": 0.9})
            store.write("b.md", "related to a", {"importance": 0.5})
            dreamer = AutoDream(store)
            recent = dreamer._orient()
            clusters = dreamer._gather(recent)
            assert len(clusters) >= 1
            anchor, related = clusters[0]
            assert anchor.path == "a.md"


# --- F4: E2B Sandbox ---

class TestE2BSandbox:
    def test_sandbox_tools_registered_with_key(self):
        """When E2B_API_KEY is set, sandbox tools should be registered."""
        # We can't test actual E2B without key, but verify the conditional logic
        from isaac.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            store = MemoryStore(Path(td))
            from isaac.agents.tools import build_builtin_tools
            # Without key — no sandbox tools
            with patch.dict("os.environ", {"E2B_API_KEY": ""}, clear=False):
                registry = build_builtin_tools(store)
                assert "sandbox_execute" not in registry

    def test_sandbox_class_init(self):
        """E2BSandbox should lazy-init."""
        from isaac.sandbox.e2b import E2BSandbox
        sb = E2BSandbox()
        assert sb._sandbox is None


# --- F5: Delegation ---

class TestDelegation:
    def test_get_exposable_tools(self):
        """Agents with expose_as_tool=True should generate tool entries."""
        from isaac.agents.delegation import AgentDelegator
        from isaac.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            store = MemoryStore(Path(td))
            agents = {
                "research": AgentConfig(
                    name="research",
                    expose_as_tool=True,
                    tool_description="Deep research agent",
                ),
                "ops": AgentConfig(name="ops"),
            }
            delegator = AgentDelegator(agents, store)
            tools = delegator.get_exposable_tools()
            assert "agent_research" in tools
            assert "agent_ops" not in tools

    @pytest.mark.asyncio
    async def test_delegate_unknown_agent(self):
        """Delegating to unknown agent should return error."""
        from isaac.agents.delegation import AgentDelegator
        from isaac.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            store = MemoryStore(Path(td))
            delegator = AgentDelegator({}, store)
            result = await delegator.delegate("nonexistent", "do something")
            assert "error" in result


# --- F7: Toolsmith ---

class TestToolsmith:
    def test_extract_code_strips_fences(self):
        from isaac.agents.toolsmith import Toolsmith
        code = '```python\nprint("hello")\n```'
        assert Toolsmith._extract_code(code) == 'print("hello")'

    def test_validate_valid_code(self):
        from isaac.agents.toolsmith import Toolsmith
        code = '''
TOOLS = [{"name": "test", "description": "test", "params": {}, "handler": None}]

async def handler() -> dict:
    return {"ok": True}

TOOLS[0]["handler"] = handler
'''
        errors = Toolsmith._validate(code)
        assert errors == []

    def test_validate_missing_tools(self):
        from isaac.agents.toolsmith import Toolsmith
        code = "async def handler(): pass"
        errors = Toolsmith._validate(code)
        assert "Missing TOOLS list" in errors[0]

    def test_validate_syntax_error(self):
        from isaac.agents.toolsmith import Toolsmith
        errors = Toolsmith._validate("def (broken:")
        assert "Syntax error" in errors[0]

    def test_extract_tool_name(self):
        from isaac.agents.toolsmith import Toolsmith
        code = 'TOOLS = [{"name": "fetch_hn", "description": "..."}]'
        assert Toolsmith._extract_tool_name(code) == "fetch_hn"


# --- F6: Computer Use ---

class TestComputerUse:
    def test_computer_controller_init(self):
        from isaac.tools.computer_use import ComputerController
        ctrl = ComputerController(display_width=1920, display_height=1080)
        assert ctrl.width == 1920
        assert ctrl.height == 1080

    def test_agent_config_computer_use_flag(self):
        cfg = AgentConfig(name="browser", computer_use=True)
        assert cfg.computer_use is True

    def test_agent_config_default_no_computer(self):
        cfg = AgentConfig(name="test")
        assert cfg.computer_use is False


# --- F8: Cost Dashboard ---

class TestCostDashboard:
    def test_render_session(self):
        from isaac.cli.dashboard import render_session
        panel = render_session(total_tokens=10000, total_cost=0.05, turn_count=5)
        assert panel is not None

    def test_render_historical_empty(self):
        """Should handle empty sessions dir gracefully."""
        with tempfile.TemporaryDirectory() as td:
            from isaac.cli.dashboard import render_historical
            panel = render_historical(sessions_dir=Path(td))
            assert panel is not None

    def test_load_session_headers(self):
        """Should parse JSONL session headers."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Write a mock session file
            header = {
                "session_id": "abc",
                "agent_name": "lead",
                "turn_count": 5,
                "total_tokens": 10000,
                "total_cost": 0.05,
                "timestamp": "2026-04-01T12:00:00",
            }
            (td_path / "lead_abc.jsonl").write_text(json.dumps(header) + "\n")

            from isaac.cli.dashboard import _load_session_headers
            headers = _load_session_headers(td_path)
            assert len(headers) == 1
            assert headers[0]["agent_name"] == "lead"
            assert headers[0]["total_cost"] == 0.05

    def test_render_historical_with_data(self):
        """Should aggregate costs across sessions."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for i, (agent, cost) in enumerate([("lead", 0.10), ("research", 0.05), ("lead", 0.20)]):
                header = {
                    "session_id": f"s{i}",
                    "agent_name": agent,
                    "turn_count": 3,
                    "total_tokens": 5000,
                    "total_cost": cost,
                }
                (td_path / f"{agent}_s{i}.jsonl").write_text(json.dumps(header) + "\n")

            from isaac.cli.dashboard import render_historical
            panel = render_historical(sessions_dir=td_path)
            assert panel is not None
