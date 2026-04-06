"""Tool loader — loads tools from ~/.isaac/tools/ directory, hot-reloadable.

Each .py file in the tools directory exports a TOOLS list. The ToolLoader
scans the directory, loads each file, and registers the tools. Toolsmith
writes new tool files here, and the unified MCP server picks them up.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from isaac.core.config import TOOLS_DIR

log = logging.getLogger(__name__)


class LoadedTool:
    """A single tool loaded from a tool file."""

    def __init__(self, name: str, description: str, params: dict[str, Any], handler: Any) -> None:
        self.name = name
        self.description = description
        self.params = params
        self.handler = handler

    @property
    def input_schema(self) -> dict[str, Any]:
        """Generate JSON Schema from params dict."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param_type in self.params.items():
            if param_type == str:
                properties[param_name] = {"type": "string"}
            elif param_type == int:
                properties[param_name] = {"type": "integer"}
            elif param_type == float:
                properties[param_name] = {"type": "number"}
            elif param_type == bool:
                properties[param_name] = {"type": "boolean"}
            else:
                properties[param_name] = {"type": "string"}
            required.append(param_name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


# Backward compat alias
PluginTool = LoadedTool


class ToolLoader:
    """Loads and manages tools from a directory."""

    def __init__(self, tools_dir: Path | None = None) -> None:
        self.tools_dir = tools_dir or TOOLS_DIR
        self._tools: dict[str, LoadedTool] = {}

    def scan(self) -> dict[str, LoadedTool]:
        """Scan tools directory and load all tools."""
        self._tools.clear()

        if not self.tools_dir.exists():
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            return self._tools

        for py_file in sorted(self.tools_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                self._load_tool_file(py_file)
            except Exception as e:
                log.warning(f"Failed to load tool {py_file.name}: {e}")

        return self._tools

    def _load_tool_file(self, path: Path) -> None:
        """Load a single tool file. Expects a TOOLS list."""
        module_name = f"isaac_tool_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        tools_list = getattr(module, "TOOLS", [])
        for tool_def in tools_list:
            tool = LoadedTool(
                name=tool_def["name"],
                description=tool_def["description"],
                params=tool_def.get("params", {}),
                handler=tool_def["handler"],
            )
            self._tools[tool.name] = tool
            log.info(f"Loaded tool: {tool.name}")

    def reload(self) -> dict[str, LoadedTool]:
        """Hot-reload all tools."""
        to_remove = [k for k in sys.modules if k.startswith("isaac_tool_") or k.startswith("isaac_plugin_")]
        for k in to_remove:
            del sys.modules[k]
        return self.scan()

    def get_tool(self, name: str) -> LoadedTool | None:
        return self._tools.get(name)

    @property
    def all_tools(self) -> dict[str, LoadedTool]:
        return dict(self._tools)


# Backward compat alias
PluginLoader = ToolLoader
