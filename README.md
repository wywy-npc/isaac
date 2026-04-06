# ISAAC — Intelligent Self-Assembling Agent Constellation

A headless multi-agent harness that turns your terminal into a fleet of autonomous AI agents. Each agent runs in its own tmux tab, shares knowledge through a wiki-linked memory graph, and can delegate tasks to other agents. ISAAC is a harness, not an application — new capabilities are added by dropping files, not changing the orchestrator.

## What Makes This Different

Most agent frameworks are tightly coupled to a single LLM provider, a single repo, or a chat UI. ISAAC is none of those:

- **Computer-wide, not repo-scoped** — agents can work across your entire filesystem, manage services, install packages, and automate GUI interactions (in sandboxed VMs)
- **Provider-agnostic** — the orchestrator never imports `anthropic` or `openai` directly. A protocol-based `LLMClient` interface supports Claude, GPT, Gemini, Mistral, ollama, and 100+ models via LiteLLM
- **Headless core** — the CLI is just one consumer. The same `HarnessCore` engine powers the SDK, Telegram/webhook gateways, and agent-to-agent delegation
- **Plugin-first** — tools, skills, connectors, sandbox backends, and LLM providers are all pluggable. Drop a `.py` file in `~/.isaac/tools/` and it hot-reloads

## Architecture

### The 3-Layer Taxonomy

Everything in ISAAC fits into three layers:

```
Skills (HOW — prompt workflows that combine tools)
  ↓ guide the agent to use
Tools (WHAT — all actions, regardless of origin)
  ├── Built-in (memory, files, bash, web search)
  ├── Plugins (~/.isaac/tools/*.py — hot-reloadable)
  ├── Apps (git repos as callable tools)
  └── From Connectors (auto-registered when connected)
  ↓ which can access
Connectors (WHERE — data sources that bring their own tools)
  └── MCP servers (Google, Slack, databases, etc.)
```

**Skills** are markdown prompt templates, not code. They tell the agent how to do complex multi-step workflows using its available tools.

**Tools** are actions. The agent doesn't know or care where a tool came from — built-in, plugin, connector, or app. They all appear identically in the tool registry.

**Connectors** are MCP servers that provide data + tools. Connect Google → get `gmail_send`, `calendar_create` as native tools. Disconnect → tools disappear.

### The Harness

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

The `HarnessConfig` holds every dependency via injection. The orchestrator imports no UI libraries, no LLM provider SDKs, and no tool implementations directly. This means the same agent loop runs identically whether invoked from a terminal, a Python script, a Telegram bot, or another agent.

### The Agentic Loop

1. Consumer calls `HarnessBuilder("agent").build()` → `HarnessCore`
2. Consumer calls `harness.run(message, state)` → yields `Event` objects
3. Orchestrator loads soul, runs memory scout, builds 3-layer cached system prompt
4. Calls LLM via `LLMClient.create_stream()` (provider-agnostic)
5. If LLM returns `tool_use` → `ToolExecutor.execute_batch()`:
   - Read-only tools run **in parallel** via `asyncio.gather`
   - Exclusive tools run **sequentially** after safe tools finish
   - Results appended → loop back to step 4
6. If LLM returns `end_turn` → done
7. Auto-compacts at 80% context budget using a fast model summarizer
8. Memory context injected on iteration 1 only (saves tokens on tool loops)

## Key Systems

### Memory (Wiki-Linked Knowledge Graph)

Agents share knowledge through a file-backed memory system using markdown files with YAML frontmatter. Nodes support `[[wiki-links]]` for graph traversal, and the auto-linker creates entity stubs for proper nouns on every write.

The **Memory Scout** performs multi-step search: direct path match → vector similarity (optional, via local `fastembed`) → full-text search → link traversal from top results. Results are token-budgeted (2000 tokens max) so memory never blows up the context window.

### Soul System (5-Layer Personality)

Each agent's personality is assembled from five layers:

1. **Platform soul** — base identity ("You are ISAAC, an autonomous agent")
2. **Architecture reference** — `ARCHITECTURE.md` (cached permanently, costs nothing after turn 1)
3. **Runtime context** — live system state (OS, loaded plugins, other agents, workspace)
4. **Soul file** — role-specific personality from `~/.isaac/souls/`
5. **Agent overlay** — per-agent overrides

### Model Routing (Cost Optimization)

A router selects the cheapest model that can handle each turn:

| Tier | Model | When |
|------|-------|------|
| Haiku | claude-haiku-4-5 | Simple ops: summarization, yes/no answers |
| Sonnet | claude-sonnet-4-6 | Standard work: tool-use loops, coding (the workhorse) |
| Opus | claude-opus-4-6 | Heavy reasoning: architecture, complex debugging |

A typical 10-iteration session drops from ~$0.45 (all Opus) to ~$0.08 (Opus first turn, Sonnet for loops).

### Sandbox (Brain/Hands Split)

Agents that need compute get their own VM (Fly.io Machines or E2B). The core insight is a **brain/hands split**:

| Your Mac (brain) | VM (hands) |
|---|---|
| Memory, sessions, souls, credentials | Workspace files, packages, services |
| API keys, MCP auth, LLM calls | Git repos, builds, experiments |
| ISAAC process, orchestration | Disposable, rebuildable |

Credentials **never** reach the VM — they're proxied through the MCP bridge. The `SessionBridge` routes each tool call to the right side: `bash` and `file_*` go to the VM, `memory_*` and `mcp__*` stay on your Mac.

VMs suspend on idle (disk preserved, cost drops to ~$0.01/day) and wake in 2-3 seconds on resume.

### Delegation (Agent-as-a-Tool)

Agents can delegate tasks to other agents via the `delegate_agent` tool. The child agent:
- Is spawned via `HarnessBuilder` (no recursion — children can't delegate further)
- Gets auto-approved permissions (no approval prompts mid-delegation)
- Streams progress back to the parent via `DelegationEvent`
- Shares the same memory store (agents collaborate via wiki-links)

### AppRunner (Bolt Any Repo as a Tool)

Any git repo can become a callable tool via a YAML manifest:

```yaml
name: autoresearch
repo: https://github.com/karpathy/autoresearch
compute: modal
gpu: H100
mode: agent  # Spawns a child ISAAC agent inside the VM
state: checkpoint  # Tar workdir between runs
```

`app_run("autoresearch", {"topic": "scaling laws"})` provisions compute, clones the repo, runs setup, executes, collects output, and tears down. Backends: E2B (cloud sandbox) and Modal (serverless GPU).

### Token Efficiency

The context engine uses several strategies borrowed from Anthropic's self-chat patterns:

- **3-layer prompt caching** — soul + tools cached across turns (90% cheaper reads), memory fresh per turn
- **Memory injected once** — only on iteration 1, not on every tool loop iteration
- **Auto-compaction** — at 80% budget, older turns get summarized by a fast model (Haiku)
- **Tool result overflow** — results exceeding 2000 tokens get a retrieval marker instead of truncation
- **Concurrent tool execution** — read-only tools run in parallel, reducing wall-clock time

## Project Structure

```
isaac/
├── core/                # Brain (zero UI imports)
│   ├── harness.py       # HarnessConfig + HarnessCore (headless engine)
│   ├── builder.py       # HarnessBuilder (assembly factory)
│   ├── orchestrator.py  # The agentic loop
│   ├── executor.py      # Concurrent tool dispatch
│   ├── llm.py           # LLMClient protocol
│   ├── llm_anthropic.py # Native Anthropic client
│   ├── llm_litellm.py   # Universal adapter (100+ models)
│   ├── soul.py          # 5-layer personality resolution
│   ├── context.py       # 3-layer prompt caching + compaction
│   ├── router.py        # Model selection heuristics
│   ├── skills.py        # Skill loader + renderer
│   └── permissions.py   # auto/ask/deny per tool
├── agents/              # Agent runtime
│   ├── tools.py         # 17 built-in tools
│   ├── delegation.py    # Agent-to-agent delegation
│   ├── session.py       # JSONL transcript persistence
│   └── toolsmith.py     # Auto-plugin generation from natural language
├── plugins/             # Pluggable tool modules
│   ├── apps.py          # AppRunner tools
│   ├── skills.py        # Skill invocation tools
│   ├── computer_scope.py # 20+ VM system tools
│   └── workspace.py     # Workspace management
├── mcp/                 # Connector system
│   ├── client.py        # MCP client (stdio + HTTP/SSE)
│   ├── tool_loader.py   # Hot-reload plugins from tools/ dir
│   ├── connections.py   # Persistent connection state
│   └── catalog.py       # Service registry
├── memory/              # Persistent knowledge
│   ├── store.py         # File-backed store with wiki-link graph
│   ├── scout.py         # Multi-step search with token budgeting
│   ├── linker.py        # Auto-linking + entity stubs
│   └── embeddings.py    # Local vector search (fastembed)
├── sandbox/             # VM isolation
│   ├── bridge.py        # Tool routing (brain on Mac, hands in VM)
│   ├── fly.py           # Fly.io Machines backend
│   ├── e2b.py           # E2B sandbox backend
│   └── lifecycle.py     # VM provisioning + suspend/resume
├── apps/                # Cloud job runner
│   ├── runner.py        # Full lifecycle orchestrator
│   ├── manifest.py      # YAML app manifest schema
│   └── backends/        # E2B, Modal compute backends
├── gateway/             # Chat provider adapters
│   ├── telegram.py      # Telegram Bot API
│   └── webhook.py       # HTTP webhook (WhatsApp, iMessage)
├── cli/                 # Interface (thin consumer)
│   ├── main.py          # CLI commands
│   └── terminal.py      # Rich REPL with event streaming
└── sdk.py               # Programmatic API
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/wyattwilson/isaac.git
cd isaac
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Initialize
isaac init                  # Creates ~/.isaac/ with default config

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Start chatting
isaac                       # Default agent REPL
isaac chat research         # Start a specific agent
isaac start                 # All agents in tmux tabs
```

### Optional Extras

```bash
pip install -e ".[sandbox]"   # E2B cloud sandbox support
pip install -e ".[computer]"  # GUI automation (pyautogui)
pip install -e ".[gateway]"   # Telegram + webhook gateways
pip install -e ".[wiki]"      # Web scraping + PDF parsing
```

## Extending ISAAC

### Add a Tool (Plugin)

Drop a Python file in `~/.isaac/tools/`:

```python
# ~/.isaac/tools/my_tool.py
TOOLS = [
    {
        "name": "search_jira",
        "description": "Search Jira issues by query",
        "params": {"query": str, "max_results": int},
        "handler": search_jira,
    }
]

async def search_jira(query: str, max_results: int = 10) -> dict:
    # Your implementation here
    return {"issues": [...]}
```

Hot-reloads on `/reload` in the REPL. No orchestrator changes needed.

### Add a Skill (Prompt Workflow)

Drop a markdown file in `~/.isaac/skills/`:

```markdown
---
name: code-review
description: Deep code review with security analysis
params:
  repo: Repository path
  focus: Area of focus (security/performance/correctness)
tools_used: [file_read, file_search, bash, memory_write]
user_invocable: true
---

# Code Review: {{repo}}

1. Use file_list to understand the project structure
2. Use file_search to find patterns related to {{focus}}
3. Review each finding in detail with file_read
4. Write a summary to memory with memory_write
```

Invoke with `/code-review ~/projects/myapp security` or programmatically with `use_skill("code-review", {...})`.

### Add a Connector (MCP Service)

Edit `~/.isaac/connections.yaml`:

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

Or let the agent do it: "Connect my Google account" triggers `catalog_search` → `connect_service`.

### Configure Agents

Edit `~/.isaac/agents.yaml`:

```yaml
agents:
  lead:
    soul: default
    model: claude-sonnet-4-6
    tools: ["*"]
    max_iterations: 25
    context_budget: 180000
    scope: home

  research:
    soul: research
    model: claude-sonnet-4-6
    tools: ["memory_*", "web_search", "file_*"]
    expose_as_tool: true
    tool_description: "Deep research agent — give it a topic, get back a report"

  ops:
    soul: ops
    model: claude-sonnet-4-6
    sandbox: fly
    computer_scope: true
    scope: system
```

## SDK Usage

```python
from isaac.sdk import run, create_harness

# One-shot
result = await run("default", "What files are in this directory?")

# Session-based with event streaming
harness = await create_harness("research")
async for event in harness.run("Research quantum computing startups", state):
    if isinstance(event, TextEvent):
        print(event.text, end="")
    elif isinstance(event, CostEvent):
        print(f"\n[Cost: ${event.total_cost:.4f}]")
```

## Design Principles

1. **99% harness** — orchestration, routing, config. Not an app.
2. **3-layer taxonomy** — Skills (HOW) → Tools (WHAT) → Connectors (WHERE)
3. **Every feature is a tool** — new capability = plugin file, not orchestrator change
4. **Memory is the moat** — agents share knowledge via wiki-linked graph
5. **Bolt-on, not built-in** — connectors, sandbox backends, LLM providers all pluggable
6. **Dependency injection** — `HarnessConfig` holds everything, nothing imported at runtime
7. **Provider-agnostic** — `LLMClient` protocol, not `import anthropic`
8. **Graceful degradation** — missing dependency? Feature silently disabled. Missing API key? Skip that capability. The agent always runs.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| LLM (native) | Anthropic Claude API (prompt caching, extended thinking) |
| LLM (universal) | LiteLLM (100+ models) |
| Embeddings | fastembed (bge-small-en-v1.5, local, no API key) |
| Web Search | DuckDuckGo (free) / Brave (optional upgrade) |
| Terminal | Rich + prompt-toolkit |
| MCP | Official MCP SDK (stdio + HTTP/SSE) |
| Sandbox | Fly.io Machines / E2B |
| GPU Compute | Modal (serverless) |

## License

MIT
