# ISAAC — Architecture Reference

> This file is loaded into every agent's system prompt so agents understand
> the system they're running in. It lives in the static (cached) layer —
> costs nothing after turn 1.

## What ISAAC Is

A thin multi-agent harness for work across the entire computer — not just repos.
The core is a headless orchestration engine (HarnessCore) that any consumer can
embed: CLI, SDK, gateway, delegation. New capabilities are added by dropping
plugin files, not changing the orchestrator.

## 3-Layer Taxonomy

Everything in ISAAC fits into three layers:

```
Skills (HOW — prompt workflows that combine tools)
  ↓ guide the agent to use
Tools (WHAT — all actions, regardless of origin)
  ├── Built-in (memory, files, bash, web)
  ├── Plugins (~/.isaac/tools/*.py)
  ├── Apps (git repos as callable tools via app_run)
  └── From Connectors (auto-registered when connected)
  ↓ which can access
Connectors (WHERE — data sources that bring their own tools)
  └── MCP servers with pre-built tool lists
```

**Skills** are not code — they're markdown prompt templates with {{params}} that tell the agent HOW to do complex workflows. Use `use_skill(name, params)` or `/<skill-name> args` in the REPL.

**Tools** are actions. The agent doesn't know or care where a tool came from (built-in, plugin, connector, app). They all show up as the same `(ToolDef, handler)` in the registry.

**Connectors** are data sources (Google, Slack, databases) connected via MCP. Each connector brings its own tools that auto-register. Connect one → get tools. Disconnect → tools disappear.

## Harness Architecture

```
HarnessBuilder (assembly — picks LLM, tools, skills, memory, sandbox, connectors)
    ↓ builds
HarnessConfig (all deps injected — zero side effects)
    ↓ creates
HarnessCore (headless engine — zero UI imports)
    ↓ delegates to
Orchestrator (agentic loop) + ToolExecutor (concurrent dispatch) + LLMClient (provider-agnostic)
    ↓ consumed by
Terminal REPL / SDK / Gateway / Delegation (thin consumers)
```

## Directory Structure

```
~/.isaac/                    # ISAAC_HOME — all state lives here
├── agents.yaml              # Agent constellation config
├── souls/                   # Markdown personality files
├── tools/                   # Hot-reloadable tool plugins (Python files)
├── skills/                  # Skill workflows (markdown + YAML frontmatter)
│   ├── deep-research.md     # Multi-step web research
│   ├── bolt-on.md           # Turn a GitHub repo into an ISAAC app
│   └── onboard-repo.md     # Explore and document a codebase
├── memory/                  # Persistent knowledge (markdown + frontmatter)
│   └── *.md                 # Memory nodes with [[wiki-links]]
├── sessions/                # JSONL conversation transcripts
├── apps/                    # App manifests (YAML)
├── checkpoints/             # Sandbox checkpoints
└── history/                 # Per-agent input history

~/ISAAC/                     # Source code
├── isaac/
│   ├── core/                # Brain (zero UI imports)
│   │   ├── types.py         # Message, AgentConfig, SessionState, ToolDef
│   │   ├── config.py        # agents.yaml loading, env vars, ISAAC_HOME
│   │   ├── harness.py       # HarnessConfig + HarnessCore (headless engine)
│   │   ├── builder.py       # HarnessBuilder (assembly factory)
│   │   ├── orchestrator.py  # THE agentic loop (provider-agnostic)
│   │   ├── executor.py      # Concurrent tool execution (parallel safe, serial exclusive)
│   │   ├── llm.py           # LLMClient Protocol + StreamEvent/LLMResult types
│   │   ├── llm_anthropic.py # Native Anthropic client (prompt caching, betas)
│   │   ├── llm_litellm.py   # Universal adapter via litellm (100+ models)
│   │   ├── state.py         # Frozen HarnessState snapshot
│   │   ├── skills.py        # Skill loader + renderer (markdown templates)
│   │   ├── soul.py          # 5-layer personality resolution + taxonomy awareness
│   │   ├── context.py       # 3-layer prompt caching + auto-compaction
│   │   ├── permissions.py   # auto/ask/deny per tool
│   │   └── router.py        # Model selection heuristics (cost optimization)
│   ├── agents/              # Agent runtime
│   │   ├── tools.py         # Built-in tools (memory, files, bash, web, delegation)
│   │   ├── delegation.py    # Agent-to-agent delegation with streaming progress
│   │   ├── session.py       # JSONL transcript persistence
│   │   └── toolsmith.py     # Toolsmith agent config
│   ├── plugins/             # Pluggable tool modules (not in core)
│   │   ├── apps.py          # App runner tools (app_run, app_list)
│   │   ├── skills.py        # Skill tools (use_skill, list_skills)
│   │   ├── computer_scope.py # 20 aggressive VM system tools
│   │   └── workspace.py     # Scope management (set_workspace, list_projects, etc.)
│   ├── mcp/                 # Bolt-on system
│   │   ├── client.py        # MCP client (stdio + HTTP/SSE transport)
│   │   ├── unified_server.py # Single MCP server for all tools
│   │   ├── tool_loader.py   # Hot-reload plugins from tools/ dir
│   │   ├── connections.py   # Persistent connection state (connections.yaml)
│   │   └── catalog.py       # Service registry (built-in + custom)
│   ├── memory/              # Persistent knowledge
│   │   ├── store.py         # File-backed store with wiki-link graph
│   │   ├── scout.py         # Multi-step search with token budgeting
│   │   ├── linker.py        # Auto-linking + entity stub creation
│   │   └── embeddings.py    # Local vector search (fastembed, optional)
│   ├── sandbox/             # VM isolation
│   │   ├── base.py          # Sandbox ABC
│   │   ├── registry.py      # Plugin registry for sandbox backends
│   │   ├── bridge.py        # Tool routing: brain on Mac, hands in VM
│   │   ├── fly.py           # Fly.io Machines backend
│   │   ├── e2b.py           # E2B sandbox backend
│   │   └── lifecycle.py     # VM provisioning + suspend/resume
│   ├── apps/                # Cloud job runner
│   │   ├── runner.py        # App execution orchestrator
│   │   ├── manifest.py      # YAML app manifest schema
│   │   └── backends/        # Compute backend plugins (E2B, Modal)
│   ├── gateway/             # Chat provider adapters
│   │   ├── base.py          # Abstract adapter interface
│   │   ├── telegram.py      # Telegram Bot API
│   │   └── webhook.py       # HTTP webhook (WhatsApp, iMessage bridges)
│   ├── cli/                 # Interface (thin consumer of HarnessCore)
│   │   ├── main.py          # CLI commands (chat, start, stop, gateway, etc.)
│   │   └── terminal.py      # Rich REPL — event rendering only
│   └── sdk.py               # Programmatic API: run(), create_harness()
└── tests/
```

## How the Agentic Loop Works

1. Consumer calls `HarnessBuilder("agent").build()` → `HarnessCore`
2. Consumer calls `harness.run(message, state)` → yields `Event` objects
3. Orchestrator loads soul, runs memory scout, builds 3-layer cached system prompt
4. Calls LLM via `LLMClient.create_stream()` (Anthropic, OpenAI, Gemini, local, etc.)
5. If LLM returns tool_use → `ToolExecutor.execute_batch()`:
   - Read-only tools (`is_read_only=True`) run **in parallel** via `asyncio.gather`
   - Exclusive tools run **sequentially** after safe tools finish
   - Results appended → loop back to step 4
6. If LLM returns end_turn → done
7. Auto-compacts at 80% context budget using fast model summarizer
8. Memory context injected on iteration 1 only (saves tokens on tool loops)

## LLM Client Interface

The orchestrator never imports `anthropic` or `openai` directly. It uses `LLMClient`:

```python
class LLMClient(Protocol):
    async def create_stream(self, model, system, messages, tools, ...) -> AsyncIterator[StreamEvent | LLMResult]: ...
    async def create(self, model, system, messages, tools, ...) -> LLMResult: ...
```

Implementations:
- **AnthropicClient** — native, preserves prompt caching + extended thinking
- **LiteLLMClient** — universal, supports 100+ models (OpenAI, Gemini, Mistral, ollama, Bedrock, Vertex)

Builder auto-selects: `claude-*` models → AnthropicClient, everything else → LiteLLMClient.

## Dependency Injection

HarnessConfig holds **all** dependencies. Nothing is imported at runtime by the orchestrator:

```python
@dataclass
class HarnessConfig:
    agent_config: AgentConfig       # Name, model, tools, sandbox, scope
    tool_registry: dict             # name → (ToolDef, async handler)
    permission_gate: PermissionGate # auto/ask/deny rules
    llm_client: LLMClient          # Provider-agnostic LLM interface
    memory_fn: Callable | None      # async (query) → context string
    approval_fn: Callable | None    # async (ToolCall) → bool (None = auto-approve)
    soul_mode: str                  # "full" or "minimal"
    soul_override: str              # Custom soul injection
    event_handler: Callable | None  # Optional event callback
```

## Plugin Contract

To add a new tool, create a .py file in ~/.isaac/tools/:

```python
# tools/my_tool.py
TOOLS = [
    {
        "name": "my_tool_action",
        "description": "What this tool does (max 150 chars in prompt)",
        "params": {"query": str, "limit": int},
        "handler": do_action,
    }
]

async def do_action(query: str, limit: int = 10) -> dict:
    return {"result": "done"}
```

Hot-reloads on `/reload` in the REPL. Handler must be async. Return a dict or string.

## Skill Contract

To add a skill, create a .md file in ~/.isaac/skills/ with YAML frontmatter:

```markdown
---
name: my-skill
description: What this workflow does
params:
  topic: The subject to work on
  depth: How deep (shallow/medium/deep)
tools_used: [web_search, memory_write, file_write]
user_invocable: true
---

# My Skill: {{topic}}

Instructions for the agent to follow...
Use {{topic}} and {{depth}} as template variables.
```

**Agent-invoked**: `use_skill("my-skill", {"topic": "AI agents", "depth": "deep"})`
**User-invoked**: `/my-skill AI agents` (first param gets the argument)

Skills are NOT code — they're prompt recipes. The agent follows the instructions
using its available tools. Skills hot-reload from disk on every `list_skills` call.

Bundled skills: `deep-research`, `bolt-on`, `onboard-repo`.

## Connector Contract

Connectors are MCP servers that provide data + tools. Managed via connections.yaml:

```yaml
services:
  google:
    type: mcp
    command: npx
    args: ["@anthropic-ai/google-mcp-server"]
    env:
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
    enabled: true
```

When connected, the connector's tools auto-register in the agent's tool registry.
The agent sees `gmail_send`, `calendar_create`, etc. as native tools — no special handling.

Agents can discover and connect new services via `catalog_search` + `connect_service`.

## agents.yaml Format

```yaml
agents:
  lead:
    soul: default            # References ~/.isaac/souls/default.md
    model: claude-sonnet-4-6 # Any model: claude-*, gpt-4o, gemini-*, ollama/*
    tools: ["*"]             # All tools, or explicit list
    mcp_servers: []          # External MCP servers to connect
    max_iterations: 25
    context_budget: 180000   # Token budget before compaction
    cwd: ~/projects/myapp    # Working directory (optional)
    auto_start: true
    expose_as_tool: false    # If true, other agents can call this one
    tool_description: ""     # Description when exposed as a tool
    sandbox: fly             # "fly" (default), "e2b", or "" to opt-out
    sandbox_size: ""         # Machine size for sandbox
    scope: cwd               # "cwd" | "home" | "system" — controls tool reach
    computer_scope: false    # Enable aggressive VM tools (clipboard, process, gui, etc.)
```

## Computer-Scope Tools (VM-targeted)

When `computer_scope: true`, agent gets 20+ system tools designed for sandboxed VMs:

| Category | Tools |
|----------|-------|
| Clipboard | clipboard_read, clipboard_write |
| Notifications | notify |
| Screenshot | screenshot |
| App control | open_app, open_url |
| Process mgmt | process_list, process_kill |
| Services | service_manage (systemctl) |
| Packages | package_install (apt/pip/npm) |
| Network | network_info |
| System | system_info, env_manage |
| Files | find_files, disk_usage |
| Scheduling | cron_manage |
| GUI automation | keystrokes, mouse_click, window_list, window_focus |

On host Mac: these tools default to DENY. In sandbox VMs: AUTO (unrestricted).

## Workspace Tools (always available)

| Tool | Description |
|------|-------------|
| set_workspace | Change working directory / scope |
| list_projects | Discover git repos, package.json, etc. |
| recent_files | Find files modified in last N hours |
| workspace_snapshot | Git status + file tree + language breakdown |

## Memory System

File-backed markdown with YAML frontmatter in ~/.isaac/memory/.
Nodes support [[wiki-links]] for graph traversal.
Scout searches: path match → embeddings (optional) → full-text → link traversal.
High-importance nodes get full content, low-importance get summary only.
Budget: 2000 tokens max in system prompt.
Auto-linker creates entity stubs for proper nouns on every write.

## Token Efficiency

- 3-layer prompt caching (soul+tools cached, memory fresh)
- Memory injected on iteration 1 only
- Auto-compaction at 80% budget using fast model
- Tool results overflow at 2000 tokens with retrieval marker
- Tool descriptions capped at 150 chars in prompt
- Proactive compaction on session resume
- Cache reads cost 90% less, writes 25% more
- Concurrent tool execution reduces wall-clock time

## Sandbox Bridge (Brain/Hands Split)

When an agent has `sandbox: fly` (or `e2b`), the SessionBridge routes tools:

| Routes to VM | Stays on Mac |
|-------------|-------------|
| bash | memory_* |
| file_read/write/list/search | mcp__* (credentials stay local) |
| computer_scope tools | web_search |
| | delegate_* |
| | local_read/local_write (whitelisted) |

## SDK Usage

```python
from isaac.sdk import run, create_harness

# One-shot
result = await run("default", "What files are in this directory?")

# Session-based
harness = await create_harness("default")
async for event in harness.run("Search my memory for projects", state):
    if isinstance(event, TextEvent):
        print(event.text)
```

## Built-in Tools

| Tool | Read-Only | Description |
|------|-----------|-------------|
| memory_search | yes | Search memory nodes |
| memory_read | yes | Read a specific memory node |
| memory_write | no | Write/update a memory node (auto-links) |
| file_read | yes | Read a file |
| file_write | no | Write a file |
| file_list | yes | List directory contents |
| file_search | yes | Grep file contents |
| bash | no | Execute shell command |
| web_search | yes | Search the web |
| delegate_agent | no | Delegate task to another agent |
| write_continuation | no | Structured handoff for heartbeats |
| app_run | no | Run cloud app (plugin) |
| app_list | yes | List available apps (plugin) |
| connect_service | no | Add MCP service connection |
| disconnect_service | no | Remove MCP service |
| catalog_search | yes | Search connectable services |
| catalog_setup | yes | Get service setup instructions |

Read-only tools run **in parallel** when called together. Non-read-only tools run sequentially.

## CLI Commands

```
isaac                    # Start default agent REPL
isaac chat [agent]       # Start specific agent
isaac start              # All agents in tmux tabs
isaac stop               # Graceful shutdown
isaac status             # Running agents
isaac gateway telegram   # Start Telegram gateway
isaac gateway webhook    # Start HTTP webhook gateway
isaac memory [query]     # Search/list memory
isaac init               # Initialize ~/.isaac
```

## Design Principles

1. **99% harness** — orchestration, routing, config. Not an app.
2. **3-layer taxonomy** — Skills (HOW) → Tools (WHAT) → Connectors (WHERE + bundled tools)
3. **Every feature is a tool** — new capability = plugin file, not orchestrator change
4. **Memory is the moat** — agents share knowledge via wiki-linked graph
5. **Bolt-on, not built-in** — connectors, sandbox backends, LLM providers all pluggable
6. **Dependency injection** — HarnessConfig holds everything, nothing imported at runtime
7. **Provider-agnostic** — LLMClient protocol, not `import anthropic`
8. **Computer-wide, not repo-scoped** — workspace tools + scope config + VM system tools
9. **Concurrent by default** — safe tools run in parallel
10. **Headless core** — CLI is just one consumer. SDK, gateway, delegation all use HarnessBuilder.
