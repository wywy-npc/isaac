---
name: bolt-on
description: Take a GitHub repo and turn it into an ISAAC app manifest + optional skill. The "steal this repo" workflow.
params:
  repo: "GitHub repo URL or org/name"
  name: "Short name for the app (optional, derived from repo name)"
tools_used: [bash, file_read, file_write, memory_write, app_list]
user_invocable: true
---

# Bolt-On: {{repo}}

You are integrating an external project into ISAAC as a callable app tool.

## Phase 1: Reconnaissance

1. Clone the repo to a temp directory:
   ```
   bash: git clone {{repo}} /tmp/bolt-on-inspect --depth 1
   ```

2. Read and understand:
   - README.md (what it does, how to use it)
   - requirements.txt / package.json / Cargo.toml (dependencies)
   - Main entry point (look for main.py, cli.py, index.js, etc.)
   - Any config files or examples

3. Determine:
   - **What it does** in one sentence
   - **How to run it** (command, args, env vars)
   - **What inputs it needs** (files, API keys, parameters)
   - **What outputs it produces** (files, stdout, reports)
   - **Compute needs** (CPU only? GPU? How much memory?)

## Phase 2: Write App Manifest

Create `~/.isaac/apps/{{name}}.yaml`:

```yaml
name: {{name}}
description: <one-line description from README>
repo: {{repo}}
compute:
  gpu: false  # or true if it needs GPU
  memory: "2GB"  # estimated
  timeout: 300  # seconds
setup: |
  pip install -r requirements.txt
  # or: npm install, cargo build, etc.
mode: command  # or "agent" if it needs interactive tool use
command: python main.py  # the actual run command
inputs:
  # Define what parameters the user can pass
  query:
    type: string
    description: "What to process"
outputs:
  - "output/**"  # Glob patterns for artifacts to collect
state: ephemeral  # or checkpoint/persistent
```

Adjust based on what you found in Phase 1.

## Phase 3: Verify

1. Run `app_list` to confirm the manifest loads
2. If possible, do a dry-run: `app_run("{{name}}", {"query": "test"})`
3. If it fails, read the error and fix the manifest

## Phase 4: Write a Skill (optional)

If the tool is complex enough to benefit from a workflow, create `~/.isaac/skills/{{name}}.md`:

```markdown
---
name: {{name}}
description: <what the skill does using this app>
params:
  query: What to process
tools_used: [app_run, memory_write]
user_invocable: true
---

# Run instructions here that guide the agent to use app_run effectively
```

## Phase 5: Report

Write a summary to memory at `apps/{{name}}.md` documenting:
- What the app does
- How to use it (app_run params)
- Any API keys or env vars needed
- Known limitations

## Rules
- Don't modify the original repo — ISAAC runs it as-is
- If the repo needs API keys, note them but don't hardcode them
- If the setup is complex, document it clearly in the manifest
- Clean up: `bash: rm -rf /tmp/bolt-on-inspect`
