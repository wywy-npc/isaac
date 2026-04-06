"""Workspace management tools — flexible scope beyond a single repo.

Allows agents to navigate across projects, discover codebases,
and manage their working context dynamically.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef


def build_workspace_tools(cwd: str | None = None) -> dict[str, tuple[ToolDef, Any]]:
    """Build workspace management tools."""
    registry: dict[str, tuple[ToolDef, Any]] = {}
    _cwd = [cwd or os.getcwd()]  # mutable ref for set_workspace

    async def set_workspace(path: str) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return {"error": f"Directory not found: {target}"}
        if not target.is_dir():
            return {"error": f"Not a directory: {target}"}
        _cwd[0] = str(target)
        os.chdir(str(target))
        return {"workspace": str(target)}

    registry["set_workspace"] = (
        ToolDef(
            name="set_workspace",
            description="Change the working directory / workspace scope.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory to switch to"}},
                "required": ["path"],
            },
            permission=PermissionLevel.AUTO,
        ),
        set_workspace,
    )

    async def list_projects(root: str = "~", max_depth: int = 3) -> dict[str, Any]:
        """Find project directories (git repos, package.json, pyproject.toml, etc.)."""
        root_path = Path(root).expanduser().resolve()
        try:
            markers = [".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "Makefile"]
            projects: list[dict[str, Any]] = []

            for marker in markers:
                cmd = f"find {root_path} -maxdepth {max_depth} -name '{marker}' -type f -o -name '{marker}' -type d 2>/dev/null | head -30"
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                for line in stdout.decode().strip().split("\n"):
                    if not line:
                        continue
                    project_dir = str(Path(line).parent)
                    if not any(p["path"] == project_dir for p in projects):
                        projects.append({"path": project_dir, "marker": marker})

            return {"projects": projects[:30], "count": len(projects)}
        except asyncio.TimeoutError:
            return {"error": "Search timed out", "projects": []}
        except Exception as e:
            return {"error": str(e)}

    registry["list_projects"] = (
        ToolDef(
            name="list_projects",
            description="Discover project directories by looking for git repos, package.json, pyproject.toml, etc.",
            input_schema={
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Root directory to search from", "default": "~"},
                    "max_depth": {"type": "integer", "description": "Max search depth", "default": 3},
                },
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        list_projects,
    )

    async def recent_files(hours: int = 24, path: str = ".", extensions: str = "") -> dict[str, Any]:
        """Find recently modified files."""
        try:
            target = Path(path).expanduser().resolve()
            cmd = f"find {target} -maxdepth 4 -type f -mmin -{hours * 60}"
            if extensions:
                ext_filters = " -o ".join(f'-name "*.{ext.strip()}"' for ext in extensions.split(","))
                cmd += f" \\( {ext_filters} \\)"
            cmd += " 2>/dev/null | head -50"
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            files = [f for f in stdout.decode().strip().split("\n") if f]
            return {"files": files, "count": len(files), "hours": hours}
        except asyncio.TimeoutError:
            return {"error": "Search timed out", "files": []}
        except Exception as e:
            return {"error": str(e)}

    registry["recent_files"] = (
        ToolDef(
            name="recent_files",
            description="Find files modified in the last N hours.",
            input_schema={
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Look back N hours", "default": 24},
                    "path": {"type": "string", "description": "Directory to search", "default": "."},
                    "extensions": {"type": "string", "description": "Comma-separated file extensions (e.g. 'py,ts,md')"},
                },
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        recent_files,
    )

    async def workspace_snapshot() -> dict[str, Any]:
        """Get a quick snapshot of the current workspace."""
        try:
            cwd = _cwd[0]
            snapshot: dict[str, Any] = {"cwd": cwd}

            # Git info
            proc = await asyncio.create_subprocess_shell(
                "git rev-parse --show-toplevel 2>/dev/null && git branch --show-current 2>/dev/null && git status --short 2>/dev/null | head -20",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                lines = stdout.decode().strip().split("\n")
                snapshot["git_root"] = lines[0] if lines else ""
                snapshot["git_branch"] = lines[1] if len(lines) > 1 else ""
                snapshot["git_changes"] = lines[2:] if len(lines) > 2 else []

            # Directory tree (depth 2)
            proc = await asyncio.create_subprocess_shell(
                "find . -maxdepth 2 -type f | head -50",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
            )
            stdout, _ = await proc.communicate()
            snapshot["files"] = [f for f in stdout.decode().strip().split("\n") if f][:50]

            # Language detection
            exts: dict[str, int] = {}
            for f in snapshot.get("files", []):
                ext = Path(f).suffix
                if ext:
                    exts[ext] = exts.get(ext, 0) + 1
            snapshot["languages"] = dict(sorted(exts.items(), key=lambda x: -x[1])[:10])

            return snapshot
        except Exception as e:
            return {"error": str(e)}

    registry["workspace_snapshot"] = (
        ToolDef(
            name="workspace_snapshot",
            description="Get a quick snapshot of the current workspace: git status, file tree, language breakdown.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        workspace_snapshot,
    )

    return registry
