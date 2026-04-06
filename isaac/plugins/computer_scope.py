"""Computer-scope tools — aggressive VM system tools for full computer control.

These tools are designed to run INSIDE sandboxed VMs, not on the host Mac.
The sandbox bridge routes them to the VM. On the host, they're gated behind
permission (DENY by default, override with ASK).

This is what makes ISAAC a "work harness" not a "repo tool" — agents can
manage the entire machine, not just files in a directory.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef


def build_computer_scope_tools() -> dict[str, tuple[ToolDef, Any]]:
    """Build aggressive system tools. Intended for VM execution."""
    registry: dict[str, tuple[ToolDef, Any]] = {}

    # --- Clipboard ---

    async def clipboard_read() -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "xclip", "-selection", "clipboard", "-o",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                # Fallback to pbpaste (macOS)
                proc = await asyncio.create_subprocess_exec(
                    "pbpaste", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
            return {"content": stdout.decode()[:10000]}
        except Exception as e:
            return {"error": str(e)}

    registry["clipboard_read"] = (
        ToolDef(
            name="clipboard_read",
            description="Read the system clipboard contents.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        clipboard_read,
    )

    async def clipboard_write(content: str) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "xclip", "-selection", "clipboard",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            await proc.communicate(input=content.encode())
            if proc.returncode != 0:
                # Fallback to pbcopy (macOS)
                proc = await asyncio.create_subprocess_exec(
                    "pbcopy", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate(input=content.encode())
            return {"written": len(content)}
        except Exception as e:
            return {"error": str(e)}

    registry["clipboard_write"] = (
        ToolDef(
            name="clipboard_write",
            description="Write content to the system clipboard.",
            input_schema={
                "type": "object",
                "properties": {"content": {"type": "string", "description": "Content to copy"}},
                "required": ["content"],
            },
            permission=PermissionLevel.AUTO,
        ),
        clipboard_write,
    )

    # --- Notifications ---

    async def notify(title: str, message: str = "") -> dict[str, Any]:
        try:
            # Linux: notify-send
            proc = await asyncio.create_subprocess_exec(
                "notify-send", title, message,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                # macOS: osascript
                script = f'display notification "{message}" with title "{title}"'
                proc = await asyncio.create_subprocess_exec(
                    "osascript", "-e", script,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
            return {"sent": True}
        except Exception as e:
            return {"error": str(e)}

    registry["notify"] = (
        ToolDef(
            name="notify",
            description="Send a desktop notification.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Notification title"},
                    "message": {"type": "string", "description": "Notification body"},
                },
                "required": ["title"],
            },
            permission=PermissionLevel.AUTO,
        ),
        notify,
    )

    # --- Screenshot ---

    async def screenshot(output_path: str = "/tmp/screenshot.png") -> dict[str, Any]:
        try:
            # Linux: scrot or gnome-screenshot
            proc = await asyncio.create_subprocess_exec(
                "scrot", output_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                # macOS
                proc = await asyncio.create_subprocess_exec(
                    "screencapture", "-x", output_path,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
            return {"path": output_path, "exists": os.path.exists(output_path)}
        except Exception as e:
            return {"error": str(e)}

    registry["screenshot"] = (
        ToolDef(
            name="screenshot",
            description="Take a screenshot of the current display.",
            input_schema={
                "type": "object",
                "properties": {"output_path": {"type": "string", "default": "/tmp/screenshot.png"}},
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        screenshot,
    )

    # --- App launching ---

    async def open_app(name: str) -> dict[str, Any]:
        try:
            # Try xdg-open first (Linux), then open (macOS)
            proc = await asyncio.create_subprocess_exec(
                "xdg-open", name,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                proc = await asyncio.create_subprocess_exec(
                    "open", "-a", name,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
            return {"opened": name}
        except Exception as e:
            return {"error": str(e)}

    registry["open_app"] = (
        ToolDef(
            name="open_app",
            description="Open an application by name.",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Application name or path"}},
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
        ),
        open_app,
    )

    async def open_url(url: str) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "xdg-open", url,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                proc = await asyncio.create_subprocess_exec(
                    "open", url,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
            return {"opened": url}
        except Exception as e:
            return {"error": str(e)}

    registry["open_url"] = (
        ToolDef(
            name="open_url",
            description="Open a URL in the default browser.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to open"}},
                "required": ["url"],
            },
            permission=PermissionLevel.AUTO,
        ),
        open_url,
    )

    # --- Process management ---

    async def process_list(filter_name: str = "") -> dict[str, Any]:
        try:
            cmd = "ps aux"
            if filter_name:
                cmd += f" | grep -i {filter_name}"
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            lines = stdout.decode().strip().split("\n")[:50]
            return {"processes": lines, "count": len(lines)}
        except Exception as e:
            return {"error": str(e)}

    registry["process_list"] = (
        ToolDef(
            name="process_list",
            description="List running processes. Optionally filter by name.",
            input_schema={
                "type": "object",
                "properties": {
                    "filter_name": {"type": "string", "description": "Filter processes by name"},
                },
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        process_list,
    )

    async def process_kill(pid: int = 0, name: str = "", signal: str = "TERM") -> dict[str, Any]:
        try:
            if pid:
                cmd = f"kill -{signal} {pid}"
            elif name:
                cmd = f"pkill -{signal} {name}"
            else:
                return {"error": "Provide pid or name"}
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return {"exit_code": proc.returncode, "output": stdout.decode(), "error": stderr.decode()}
        except Exception as e:
            return {"error": str(e)}

    registry["process_kill"] = (
        ToolDef(
            name="process_kill",
            description="Kill a process by PID or name.",
            input_schema={
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "Process ID"},
                    "name": {"type": "string", "description": "Process name (for pkill)"},
                    "signal": {"type": "string", "description": "Signal (TERM, KILL, HUP)", "default": "TERM"},
                },
            },
            permission=PermissionLevel.AUTO,
        ),
        process_kill,
    )

    # --- Service management ---

    async def service_manage(action: str, service: str) -> dict[str, Any]:
        try:
            cmd = f"systemctl {action} {service}"
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return {"exit_code": proc.returncode, "output": stdout.decode(), "error": stderr.decode()}
        except Exception as e:
            return {"error": str(e)}

    registry["service_manage"] = (
        ToolDef(
            name="service_manage",
            description="Manage system services (start, stop, restart, status).",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "start, stop, restart, status, enable, disable"},
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["action", "service"],
            },
            permission=PermissionLevel.AUTO,
        ),
        service_manage,
    )

    # --- Package management ---

    async def package_install(packages: str, manager: str = "apt") -> dict[str, Any]:
        try:
            if manager == "apt":
                cmd = f"apt-get install -y {packages}"
            elif manager == "pip":
                cmd = f"pip install {packages}"
            elif manager == "npm":
                cmd = f"npm install -g {packages}"
            else:
                return {"error": f"Unknown package manager: {manager}"}
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return {"exit_code": proc.returncode, "stdout": stdout.decode()[:5000], "stderr": stderr.decode()[:2000]}
        except Exception as e:
            return {"error": str(e)}

    registry["package_install"] = (
        ToolDef(
            name="package_install",
            description="Install system or language packages.",
            input_schema={
                "type": "object",
                "properties": {
                    "packages": {"type": "string", "description": "Space-separated package names"},
                    "manager": {"type": "string", "description": "apt, pip, or npm", "default": "apt"},
                },
                "required": ["packages"],
            },
            permission=PermissionLevel.AUTO,
        ),
        package_install,
    )

    # --- Network info ---

    async def network_info() -> dict[str, Any]:
        try:
            results: dict[str, str] = {}
            for cmd_name, cmd in [("ip", "ip addr"), ("ports", "ss -tlnp"), ("dns", "cat /etc/resolv.conf")]:
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                results[cmd_name] = stdout.decode()[:3000]
            return results
        except Exception as e:
            return {"error": str(e)}

    registry["network_info"] = (
        ToolDef(
            name="network_info",
            description="Get network configuration: IP addresses, listening ports, DNS.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        network_info,
    )

    # --- System info ---

    async def system_info() -> dict[str, Any]:
        try:
            results: dict[str, str] = {}
            for cmd_name, cmd in [
                ("os", "uname -a"),
                ("cpu", "nproc"),
                ("memory", "free -h"),
                ("disk", "df -h"),
                ("uptime", "uptime"),
            ]:
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                results[cmd_name] = stdout.decode().strip()
            return results
        except Exception as e:
            return {"error": str(e)}

    registry["system_info"] = (
        ToolDef(
            name="system_info",
            description="Get system information: OS, CPU, memory, disk, uptime.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        system_info,
    )

    # --- Environment management ---

    async def env_manage(action: str, key: str = "", value: str = "") -> dict[str, Any]:
        if action == "list":
            env = {k: v[:100] for k, v in sorted(os.environ.items())[:50]}
            return {"env": env, "count": len(os.environ)}
        elif action == "get":
            return {"key": key, "value": os.environ.get(key, "")}
        elif action == "set":
            os.environ[key] = value
            return {"set": key, "value": value}
        elif action == "unset":
            os.environ.pop(key, None)
            return {"unset": key}
        return {"error": f"Unknown action: {action}"}

    registry["env_manage"] = (
        ToolDef(
            name="env_manage",
            description="Manage environment variables: list, get, set, unset.",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "list, get, set, unset"},
                    "key": {"type": "string", "description": "Environment variable name"},
                    "value": {"type": "string", "description": "Value (for set)"},
                },
                "required": ["action"],
            },
            permission=PermissionLevel.AUTO,
        ),
        env_manage,
    )

    # --- File search (unrestricted in VM) ---

    async def find_files(pattern: str, path: str = "/", max_depth: int = 5) -> dict[str, Any]:
        try:
            cmd = f"find {path} -maxdepth {max_depth} -name '{pattern}' -type f 2>/dev/null | head -50"
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            files = [f for f in stdout.decode().strip().split("\n") if f]
            return {"files": files, "count": len(files)}
        except asyncio.TimeoutError:
            return {"error": "Search timed out", "files": []}
        except Exception as e:
            return {"error": str(e)}

    registry["find_files"] = (
        ToolDef(
            name="find_files",
            description="Find files by name pattern anywhere on the system.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Filename pattern (glob)"},
                    "path": {"type": "string", "description": "Starting directory", "default": "/"},
                    "max_depth": {"type": "integer", "description": "Max directory depth", "default": 5},
                },
                "required": ["pattern"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        find_files,
    )

    # --- Disk usage ---

    async def disk_usage(path: str = "/") -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "du", "-sh", path, "--max-depth=1",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            lines = stdout.decode().strip().split("\n")[:30]
            return {"usage": lines}
        except Exception as e:
            return {"error": str(e)}

    registry["disk_usage"] = (
        ToolDef(
            name="disk_usage",
            description="Show disk usage for a directory.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path", "default": "/"}},
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        disk_usage,
    )

    # --- Cron management ---

    async def cron_manage(action: str, schedule: str = "", command: str = "") -> dict[str, Any]:
        try:
            if action == "list":
                proc = await asyncio.create_subprocess_shell(
                    "crontab -l 2>/dev/null || echo 'no crontab'",
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                return {"crontab": stdout.decode()}
            elif action == "add":
                # Append to existing crontab
                proc = await asyncio.create_subprocess_shell(
                    f'(crontab -l 2>/dev/null; echo "{schedule} {command}") | crontab -',
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
                return {"added": f"{schedule} {command}"}
            elif action == "clear":
                proc = await asyncio.create_subprocess_shell(
                    "crontab -r", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
                return {"cleared": True}
            return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}

    registry["cron_manage"] = (
        ToolDef(
            name="cron_manage",
            description="Manage cron jobs: list, add, clear.",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "list, add, clear"},
                    "schedule": {"type": "string", "description": "Cron schedule (for add)"},
                    "command": {"type": "string", "description": "Command to run (for add)"},
                },
                "required": ["action"],
            },
            permission=PermissionLevel.AUTO,
        ),
        cron_manage,
    )

    # --- GUI automation (xdotool) ---

    async def keystrokes(text: str = "", key: str = "") -> dict[str, Any]:
        try:
            if text:
                proc = await asyncio.create_subprocess_exec(
                    "xdotool", "type", "--clearmodifiers", text,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
                return {"typed": text}
            elif key:
                proc = await asyncio.create_subprocess_exec(
                    "xdotool", "key", key,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
                return {"pressed": key}
            return {"error": "Provide text or key"}
        except Exception as e:
            return {"error": str(e)}

    registry["keystrokes"] = (
        ToolDef(
            name="keystrokes",
            description="Type text or press keys via xdotool (GUI automation).",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                    "key": {"type": "string", "description": "Key combo (e.g. 'ctrl+c', 'Return', 'alt+F4')"},
                },
            },
            permission=PermissionLevel.AUTO,
        ),
        keystrokes,
    )

    async def mouse_click(x: int = 0, y: int = 0, button: int = 1) -> dict[str, Any]:
        try:
            if x and y:
                await asyncio.create_subprocess_exec(
                    "xdotool", "mousemove", str(x), str(y),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            proc = await asyncio.create_subprocess_exec(
                "xdotool", "click", str(button),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            await proc.communicate()
            return {"clicked": {"x": x, "y": y, "button": button}}
        except Exception as e:
            return {"error": str(e)}

    registry["mouse_click"] = (
        ToolDef(
            name="mouse_click",
            description="Move mouse and click at coordinates.",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "button": {"type": "integer", "description": "1=left, 2=middle, 3=right", "default": 1},
                },
            },
            permission=PermissionLevel.AUTO,
        ),
        mouse_click,
    )

    # --- Window management ---

    async def window_list() -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "wmctrl", "-l",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            lines = stdout.decode().strip().split("\n")[:30]
            return {"windows": lines, "count": len(lines)}
        except Exception as e:
            return {"error": str(e)}

    registry["window_list"] = (
        ToolDef(
            name="window_list",
            description="List open windows.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        window_list,
    )

    async def window_focus(title: str = "", window_id: str = "") -> dict[str, Any]:
        try:
            if title:
                proc = await asyncio.create_subprocess_exec(
                    "wmctrl", "-a", title,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
                return {"focused": title}
            elif window_id:
                proc = await asyncio.create_subprocess_exec(
                    "wmctrl", "-i", "-a", window_id,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                await proc.communicate()
                return {"focused": window_id}
            return {"error": "Provide title or window_id"}
        except Exception as e:
            return {"error": str(e)}

    registry["window_focus"] = (
        ToolDef(
            name="window_focus",
            description="Focus a window by title or ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Window title (partial match)"},
                    "window_id": {"type": "string", "description": "Window ID from window_list"},
                },
            },
            permission=PermissionLevel.AUTO,
        ),
        window_focus,
    )

    return registry
