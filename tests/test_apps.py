"""Tests for the AppRunner system — manifests, compute, runner."""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from isaac.core.types import AgentConfig


# --- Manifest ---

class TestManifest:
    def test_parse_manifest(self):
        from isaac.apps.manifest import _parse_manifest
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: test_app
description: "A test app"
repo: https://github.com/test/repo
gpu: true
gpu_type: A100
memory_gb: 32
timeout: 7200
setup: "pip install stuff"
mode: command
run: "python main.py --query {query}"
inputs:
  query:
    type: string
    required: true
    description: "Search query"
artifacts:
  - path: "output.txt"
    description: "Output file"
state: ephemeral
""")
            f.flush()
            m = _parse_manifest(Path(f.name))

        assert m.name == "test_app"
        assert m.gpu is True
        assert m.gpu_type == "A100"
        assert m.timeout == 7200
        assert m.mode == "command"
        assert "query" in m.inputs
        assert m.inputs["query"].required is True
        assert len(m.artifacts) == 1
        assert m.state == "ephemeral"

    def test_parse_agent_mode_manifest(self):
        from isaac.apps.manifest import _parse_manifest
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: agent_app
mode: agent
agent_soul: "You are a research agent."
agent_tools: [bash, file_read, file_write, memory_write]
gpu: true
gpu_type: H100
""")
            f.flush()
            m = _parse_manifest(Path(f.name))

        assert m.mode == "agent"
        assert "research agent" in m.agent_soul
        assert "memory_write" in m.agent_tools
        assert m.gpu_type == "H100"

    def test_list_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "app1.yaml").write_text("name: app1\ndescription: First app")
            (td_path / "app2.yaml").write_text("name: app2\ndescription: Second app")

            with patch("isaac.apps.manifest.APPS_DIR", td_path):
                from isaac.apps.manifest import list_manifests
                apps = list_manifests()
                assert len(apps) == 2
                names = {a["name"] for a in apps}
                assert "app1" in names
                assert "app2" in names

    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "myapp.yaml").write_text("name: myapp\ndescription: Test\nmode: command\nrun: echo hi")

            with patch("isaac.apps.manifest.APPS_DIR", td_path):
                from isaac.apps.manifest import load_manifest
                m = load_manifest("myapp")
                assert m is not None
                assert m.name == "myapp"
                assert m.run == "echo hi"

    def test_load_manifest_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("isaac.apps.manifest.APPS_DIR", Path(td)):
                from isaac.apps.manifest import load_manifest
                assert load_manifest("nonexistent") is None


# --- Compute abstraction ---

class TestCompute:
    def test_compute_instance_defaults(self):
        from isaac.apps.compute import ComputeInstance
        inst = ComputeInstance(id="test-123", backend="e2b")
        assert inst.status == "provisioning"
        assert inst.metadata == {}

    def test_exec_result_defaults(self):
        from isaac.apps.compute import ExecResult
        r = ExecResult()
        assert r.stdout == ""
        assert r.exit_code == 0
        assert r.error is None

    def test_get_backend_raises_without_keys(self):
        from isaac.apps.compute import get_backend
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="No compute backend"):
                get_backend()

    def test_get_backend_prefers_e2b(self):
        with patch.dict("os.environ", {"E2B_API_KEY": "test"}):
            with patch("isaac.apps.backends.e2b_backend.E2BBackend") as mock:
                from isaac.apps.compute import get_backend
                # reimport to pick up patched env
                backend = get_backend("e2b")
                # Should not raise


# --- AppRunner ---

class TestAppRunner:
    @pytest.mark.asyncio
    async def test_run_unknown_app(self):
        from isaac.apps.runner import AppRunner
        from isaac.memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as td:
            with patch("isaac.apps.manifest.APPS_DIR", Path(td)):
                runner = AppRunner(memory=MemoryStore(Path(td) / "mem"))
                result = await runner.run("nonexistent", {})
                assert result.status == "error"
                assert "Unknown app" in result.error

    def test_build_agent_prompt(self):
        from isaac.apps.runner import AppRunner
        from isaac.apps.manifest import AppManifest
        runner = AppRunner()
        manifest = AppManifest(
            name="test",
            repo="https://github.com/test/repo",
            timeout=3600,
            agent_soul="You are a test agent.",
        )
        prompt = runner._build_agent_prompt(manifest, {"query": "test query"})
        assert "test agent" in prompt
        assert "test query" in prompt
        assert "3600" in prompt

    def test_build_remote_tools_includes_remote_overrides(self):
        from isaac.apps.runner import AppRunner
        from isaac.apps.compute import ComputeInstance
        from isaac.apps.manifest import AppManifest
        from isaac.core.types import ToolDef, PermissionLevel

        runner = AppRunner()
        instance = ComputeInstance(id="test", backend="mock", metadata={"workdir": "/app"})
        manifest = AppManifest(name="test")

        mock_backend = MagicMock()
        tools = runner._build_remote_tools(mock_backend, instance, manifest)

        # Remote-override tools should always be present
        assert "bash" in tools
        assert "file_read" in tools
        assert "file_write" in tools
        assert "file_list" in tools
        assert "file_search" in tools
        # All tools should be (ToolDef, handler) tuples
        for name, (tdef, handler) in tools.items():
            assert tdef.name == name
            assert callable(handler)

    def test_build_remote_tools_merges_parent_tools(self):
        from isaac.apps.runner import AppRunner
        from isaac.apps.compute import ComputeInstance
        from isaac.apps.manifest import AppManifest
        from isaac.core.types import ToolDef, PermissionLevel

        # Simulate parent tools
        async def mock_web_search(query: str): return {"results": []}
        async def mock_memory_search(query: str): return {"results": []}
        parent_tools = {
            "web_search": (ToolDef(name="web_search", description="Search web", input_schema={}, permission=PermissionLevel.AUTO), mock_web_search),
            "memory_search": (ToolDef(name="memory_search", description="Search mem", input_schema={}, permission=PermissionLevel.AUTO), mock_memory_search),
            "app_run": (ToolDef(name="app_run", description="Run app", input_schema={}, permission=PermissionLevel.ASK), None),  # should be excluded
        }

        runner = AppRunner(parent_tools=parent_tools)
        instance = ComputeInstance(id="test", backend="mock", metadata={"workdir": "/app"})
        manifest = AppManifest(name="test")

        mock_backend = MagicMock()
        tools = runner._build_remote_tools(mock_backend, instance, manifest)

        # Parent tools should pass through
        assert "web_search" in tools
        assert "memory_search" in tools
        # app_run should be excluded (prevent recursion)
        assert "app_run" not in tools
        # bash should be the REMOTE version, not parent's
        assert "bash" in tools
        tdef, _ = tools["bash"]
        assert "remote GPU VM" in tdef.description


# --- Tool registration ---

class TestAppTools:
    def test_app_list_tool_registered(self):
        from isaac.memory.store import MemoryStore
        from isaac.agents.tools import build_builtin_tools
        with tempfile.TemporaryDirectory() as td:
            store = MemoryStore(Path(td))
            registry = build_builtin_tools(store)
            assert "app_list" in registry
            assert "app_run" in registry

    @pytest.mark.asyncio
    async def test_app_list_returns_empty(self):
        from isaac.memory.store import MemoryStore
        from isaac.agents.tools import build_builtin_tools
        with tempfile.TemporaryDirectory() as td:
            with patch("isaac.apps.manifest.APPS_DIR", Path(td)):
                store = MemoryStore(Path(td) / "mem")
                registry = build_builtin_tools(store)
                _, handler = registry["app_list"]
                result = await handler()
                assert result["count"] == 0
                assert result["apps"] == []
