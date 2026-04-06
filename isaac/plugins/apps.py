"""App runner plugin — registers app_run and app_list as standard tools.

This is the plugin version of what was previously baked into agents/tools.py.
Loaded by the tool loader or explicitly by the builder.
"""
from __future__ import annotations

from typing import Any

from isaac.core.types import PermissionLevel, ToolDef


def _build_app_catalog() -> str:
    """Build a concise catalog string from all installed app manifests."""
    from isaac.apps.manifest import list_manifests
    apps = list_manifests()
    if not apps:
        return ""
    lines = []
    for a in apps:
        gpu_tag = f" [GPU: {a['gpu_type']}]" if a.get("gpu", "False") == "True" else ""
        lines.append(f"  - {a['name']}: {a['description']}{gpu_tag}")
    return "\n".join(lines)


def build_app_tools(
    memory: Any = None, parent_tools: dict | None = None,
) -> dict[str, tuple[ToolDef, Any]]:
    """Build app tools as standard registry entries."""
    registry: dict[str, tuple[ToolDef, Any]] = {}

    async def app_list() -> dict[str, Any]:
        from isaac.apps.manifest import list_manifests
        apps = list_manifests()
        return {"apps": apps, "count": len(apps)}

    registry["app_list"] = (
        ToolDef(
            name="app_list",
            description="List available external apps with full details (inputs, compute, state).",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        app_list,
    )

    async def app_run(app: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaac.apps.runner import AppRunner
        runner = AppRunner(memory=memory, parent_tools=parent_tools or {})
        result = await runner.run(app, inputs or {})
        return {
            "status": result.status,
            "summary": result.summary[:3000],
            "artifacts": result.artifacts,
            "duration": result.duration,
            "cost": result.cost,
            "error": result.error,
        }

    # Build description with embedded catalog so the model knows what's available
    catalog = _build_app_catalog()
    if catalog:
        app_run_desc = (
            "Run an external app as a tool. Provisions compute, clones repo, "
            "executes the app, and collects artifacts.\n\n"
            "Available apps:\n"
            f"{catalog}\n\n"
            "Use app_list for full input schemas. Choose the right app based on "
            "the descriptions above — do NOT use app_run for tasks that built-in "
            "tools or skills can handle."
        )
    else:
        app_run_desc = (
            "Run an external app as a tool. Provisions compute, clones repo, "
            "executes the app, and collects artifacts. "
            "No apps currently installed — add manifests to ~/.isaac/apps/."
        )

    registry["app_run"] = (
        ToolDef(
            name="app_run",
            description=app_run_desc,
            input_schema={
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "App name from the catalog above"},
                    "inputs": {
                        "type": "object",
                        "description": "App-specific inputs (use app_list for full schema)",
                        "additionalProperties": True,
                    },
                },
                "required": ["app"],
            },
            permission=PermissionLevel.AUTO,
        ),
        app_run,
    )

    return registry
