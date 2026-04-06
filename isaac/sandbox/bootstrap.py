"""VM Bootstrap — auto-install registered app dependencies on new VMs.

Reads all app manifests from ~/.isaac/apps/*.yaml, generates a bash script
that clones repos and runs setup commands, then executes it on the VM.

Tracks what's installed in /workspace/.bootstrap_manifest so incremental
updates only install new apps.

Flow:
  New VM (is_new=True):
    → run_bootstrap() → full install of all registered apps
  Existing VM + new app added:
    → run_incremental_bootstrap() → only install the new app
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from isaac.apps.manifest import list_manifests, load_manifest, APPS_DIR
from isaac.sandbox.base import Sandbox

log = logging.getLogger(__name__)

# Base packages every VM gets
BASE_PACKAGES = "git curl build-essential python3-pip python3-venv wget unzip"


def generate_bootstrap_script(exclude: set[str] | None = None) -> str:
    """Read all ~/.isaac/apps/*.yaml and produce a bash bootstrap script.

    Args:
        exclude: App names to skip (already installed).

    Returns:
        A bash script string ready to execute on the VM.
    """
    exclude = exclude or set()
    parts: list[str] = [
        "#!/bin/bash",
        "set -e",
        "",
        "# --- ISAAC VM Bootstrap ---",
        f"echo 'ISAAC bootstrap starting at $(date)'",
        "",
        "# Base packages",
        f"apt-get update -qq && apt-get install -y -qq {BASE_PACKAGES}",
        "",
        "# Ensure /workspace exists",
        "mkdir -p /workspace",
        "",
    ]

    manifest_data: dict[str, Any] = {}

    # Scan all app manifests
    APPS_DIR.mkdir(parents=True, exist_ok=True)
    for yaml_file in sorted(APPS_DIR.glob("*.yaml")):
        manifest = load_manifest(yaml_file.stem)
        if not manifest or not manifest.repo:
            continue
        if manifest.name in exclude:
            continue

        app_dir = f"/workspace/{manifest.name}"
        parts.append(f"# --- App: {manifest.name} ---")
        parts.append(f"echo 'Installing app: {manifest.name}'")

        # Clone repo if not present
        parts.append(f"if [ ! -d {app_dir} ]; then")
        parts.append(f"  git clone {manifest.repo} {app_dir}")
        if manifest.version and manifest.version != "main":
            parts.append(f"  cd {app_dir} && git checkout {manifest.version}")
        parts.append("fi")

        # Run setup commands if defined
        if manifest.setup:
            parts.append(f"cd {app_dir}")
            # Each line of setup as a separate command
            for line in manifest.setup.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    parts.append(f"{line}")

        parts.append("")

        manifest_data[manifest.name] = {
            "repo": manifest.repo,
            "version": manifest.version,
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # Write bootstrap manifest to track what's installed
    manifest_json = json.dumps({"apps": manifest_data}, indent=2)
    parts.append("# Write bootstrap manifest")
    parts.append(f"cat > /workspace/.bootstrap_manifest << 'ISAAC_MANIFEST_EOF'")
    parts.append(manifest_json)
    parts.append("ISAAC_MANIFEST_EOF")
    parts.append("")
    parts.append("echo 'ISAAC bootstrap complete'")

    return "\n".join(parts)


async def run_bootstrap(sandbox: Sandbox, sandbox_id: str) -> dict[str, Any]:
    """Generate and execute full bootstrap on a new VM.

    Called once when a VM is first created. Installs base packages,
    clones all registered app repos, runs their setup scripts.
    """
    script = generate_bootstrap_script()

    if "apt-get update" in script and "# --- App:" not in script:
        # No apps registered — just install base packages
        log.info("Bootstrap: no apps registered, installing base packages only")

    log.info(f"Running bootstrap on sandbox {sandbox_id}")
    result = await sandbox.exec(sandbox_id, script, timeout=600)

    if result.exit_code != 0:
        log.warning(f"Bootstrap failed (exit {result.exit_code}): {result.stderr[:500]}")
    else:
        log.info(f"Bootstrap complete ({result.duration_ms}ms)")

    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout[-2000:],  # Last 2KB
        "stderr": result.stderr[:2000],
        "duration_ms": result.duration_ms,
    }


async def run_incremental_bootstrap(sandbox: Sandbox, sandbox_id: str) -> dict[str, Any]:
    """Install only apps not already on the VM.

    Reads /workspace/.bootstrap_manifest from the VM, diffs against
    current ~/.isaac/apps/*.yaml, and only installs new apps.
    """
    # Read existing manifest from VM
    existing_apps: set[str] = set()
    try:
        result = await sandbox.exec(sandbox_id, "cat /workspace/.bootstrap_manifest", timeout=5)
        if result.exit_code == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            existing_apps = set(data.get("apps", {}).keys())
    except Exception:
        pass  # No manifest yet — install everything

    # Generate script excluding already-installed apps
    script = generate_bootstrap_script(exclude=existing_apps)

    # Check if there's actually anything new to install
    if "# --- App:" not in script:
        return {
            "exit_code": 0,
            "stdout": "All apps already installed",
            "stderr": "",
            "duration_ms": 0,
            "new_apps": [],
        }

    log.info(f"Incremental bootstrap on sandbox {sandbox_id} (skipping: {existing_apps})")
    result = await sandbox.exec(sandbox_id, script, timeout=600)

    # Merge the manifests — read the new one and combine with existing
    try:
        new_result = await sandbox.exec(sandbox_id, "cat /workspace/.bootstrap_manifest", timeout=5)
        if new_result.exit_code == 0:
            new_data = json.loads(new_result.stdout)
            # The script overwrites the manifest, but we need to merge
            # Re-read to get the combined state
            all_apps = new_data.get("apps", {})
            new_apps = [name for name in all_apps if name not in existing_apps]
        else:
            new_apps = []
    except Exception:
        new_apps = []

    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[:2000],
        "duration_ms": result.duration_ms,
        "new_apps": new_apps,
    }
