"""
Service Catalog — known MCP servers and how to connect them.

This is the agent's knowledge of what services exist and how to wire them up.
When a user says "connect my Gmail," the agent looks up the catalog entry,
knows the command, required env vars, and auth method, and walks the user
through setup.

The catalog is append-only. New entries come from:
1. This built-in catalog (common services)
2. The Toolsmith agent discovers and adds entries via `catalog_add`
3. Users can add custom entries via `isaac catalog add`

Each entry contains everything needed to connect:
- The MCP server package/command
- Required environment variables
- Auth method (api_key, oauth, none)
- Setup instructions for the user
- What tools the service provides (for the agent to know what it gains)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- Built-in catalog of known MCP servers ---

@dataclass
class CatalogEntry:
    """A known service that can be connected."""
    name: str
    description: str
    command: str
    args: list[str] = field(default_factory=list)
    transport: str = "stdio"
    url: str = ""

    # Auth requirements
    auth_type: str = "api_key"  # api_key | oauth | none
    env_vars: dict[str, str] = field(default_factory=dict)
    # env_vars: {"GOOGLE_API_KEY": "Get from https://console.cloud.google.com/..."}

    # What the user gets
    provides: list[str] = field(default_factory=list)
    setup_url: str = ""
    setup_instructions: str = ""

    # Metadata
    category: str = "general"
    npm_package: str = ""


BUILTIN_CATALOG: dict[str, CatalogEntry] = {
    # --- Google ---
    "google": CatalogEntry(
        name="google",
        description="Gmail, Calendar, Drive, Docs, Sheets, Slides via Google Workspace",
        command="npx",
        args=["-y", "google-workspace-mcp", "serve"],
        auth_type="oauth",
        env_vars={
            "GOOGLE_CREDENTIALS_PATH": "Path to credentials.json downloaded from Google Cloud Console",
        },
        provides=["gmail_search", "gmail_send", "calendar_list", "calendar_create", "drive_search", "docs_read", "sheets_read"],
        setup_url="https://console.cloud.google.com/apis/credentials",
        setup_instructions=(
            "1. Go to Google Cloud Console → APIs & Services → Credentials\n"
            "2. Create an OAuth 2.0 Client ID (Desktop app)\n"
            "3. Enable: Gmail API, Calendar API, Drive API, Docs API, Sheets API\n"
            "4. Download credentials.json from the Cloud Console\n"
            "5. Run: npx google-workspace-mcp accounts add wyatt --credentials /path/to/credentials.json --open\n"
            "6. Complete the browser OAuth flow — tokens are stored locally"
        ),
        category="communication",
        npm_package="google-workspace-mcp",
    ),

    # --- Slack ---
    "slack": CatalogEntry(
        name="slack",
        description="Slack messaging — read channels, send messages, search",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        auth_type="api_key",
        env_vars={
            "SLACK_BOT_TOKEN": "Bot token from https://api.slack.com/apps → OAuth & Permissions",
        },
        provides=["slack_send", "slack_search", "slack_channels", "slack_history"],
        setup_url="https://api.slack.com/apps",
        setup_instructions=(
            "1. Go to api.slack.com/apps → Create New App\n"
            "2. Add Bot Token Scopes: channels:read, chat:write, search:read\n"
            "3. Install to Workspace\n"
            "4. Copy the Bot User OAuth Token (xoxb-...)"
        ),
        category="communication",
        npm_package="@modelcontextprotocol/server-slack",
    ),

    # --- GitHub ---
    "github": CatalogEntry(
        name="github",
        description="GitHub repos, issues, PRs, actions",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        auth_type="api_key",
        env_vars={
            "GITHUB_PERSONAL_ACCESS_TOKEN": "Personal access token from https://github.com/settings/tokens",
        },
        provides=["github_repos", "github_issues", "github_prs", "github_search", "github_create_issue"],
        setup_url="https://github.com/settings/tokens",
        setup_instructions=(
            "1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens\n"
            "2. Generate new token (classic) with repo, read:org scopes\n"
            "3. Copy the token"
        ),
        category="development",
        npm_package="@modelcontextprotocol/server-github",
    ),

    # --- Filesystem ---
    "filesystem": CatalogEntry(
        name="filesystem",
        description="Read/write files on the local system (sandboxed to allowed dirs)",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
        auth_type="none",
        env_vars={},
        provides=["read_file", "write_file", "list_directory", "search_files"],
        category="system",
        npm_package="@modelcontextprotocol/server-filesystem",
    ),

    # --- Brave Search ---
    "brave-search": CatalogEntry(
        name="brave-search",
        description="Web search via Brave Search API",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        auth_type="api_key",
        env_vars={
            "BRAVE_API_KEY": "API key from https://brave.com/search/api/",
        },
        provides=["web_search", "news_search"],
        setup_url="https://brave.com/search/api/",
        category="search",
        npm_package="@modelcontextprotocol/server-brave-search",
    ),

    # --- Puppeteer (browser automation) ---
    "puppeteer": CatalogEntry(
        name="puppeteer",
        description="Browser automation — navigate, click, screenshot, scrape",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        auth_type="none",
        env_vars={},
        provides=["browser_navigate", "browser_click", "browser_screenshot", "browser_type"],
        category="automation",
        npm_package="@modelcontextprotocol/server-puppeteer",
    ),

    # --- PostgreSQL ---
    "postgres": CatalogEntry(
        name="postgres",
        description="Query PostgreSQL databases",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        auth_type="api_key",
        env_vars={
            "POSTGRES_CONNECTION_STRING": "postgresql://user:pass@host:5432/db",
        },
        provides=["query", "list_tables", "describe_table"],
        category="database",
        npm_package="@modelcontextprotocol/server-postgres",
    ),

    # --- Notion ---
    "notion": CatalogEntry(
        name="notion",
        description="Notion pages, databases, search",
        command="npx",
        args=["-y", "notion-mcp-server"],
        auth_type="api_key",
        env_vars={
            "NOTION_API_KEY": "Integration token from https://www.notion.so/my-integrations",
        },
        provides=["notion_search", "notion_read_page", "notion_create_page", "notion_query_database"],
        setup_url="https://www.notion.so/my-integrations",
        category="productivity",
        npm_package="notion-mcp-server",
    ),

    # --- Linear ---
    "linear": CatalogEntry(
        name="linear",
        description="Linear issue tracking — create, update, search issues",
        command="npx",
        args=["-y", "linear-mcp-server"],
        auth_type="api_key",
        env_vars={
            "LINEAR_API_KEY": "API key from https://linear.app/settings/api",
        },
        provides=["linear_issues", "linear_create", "linear_update", "linear_search"],
        setup_url="https://linear.app/settings/api",
        category="development",
        npm_package="linear-mcp-server",
    ),

    # --- Granola (meeting notes) ---
    "granola": CatalogEntry(
        name="granola",
        description="Granola meeting notes — search, read transcripts",
        command="npx",
        args=["-y", "granola-mcp-server"],
        auth_type="api_key",
        env_vars={
            "GRANOLA_API_KEY": "API key from Granola settings",
        },
        provides=["meeting_search", "meeting_transcript", "meeting_notes"],
        category="productivity",
        npm_package="granola-mcp-server",
    ),
}


# --- Custom catalog (loaded from disk, added by Toolsmith or user) ---

def _custom_catalog_path():
    from isaac.core.config import ISAAC_HOME
    return ISAAC_HOME / "catalog.yaml"


def load_custom_catalog() -> dict[str, CatalogEntry]:
    """Load custom catalog entries from ~/.isaac/catalog.yaml."""
    import yaml
    path = _custom_catalog_path()
    if not path.exists():
        return {}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    entries: dict[str, CatalogEntry] = {}
    for name, cfg in raw.get("services", {}).items():
        entries[name] = CatalogEntry(
            name=name,
            description=cfg.get("description", ""),
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            transport=cfg.get("transport", "stdio"),
            url=cfg.get("url", ""),
            auth_type=cfg.get("auth_type", "none"),
            env_vars=cfg.get("env_vars", {}),
            provides=cfg.get("provides", []),
            setup_url=cfg.get("setup_url", ""),
            setup_instructions=cfg.get("setup_instructions", ""),
            category=cfg.get("category", "custom"),
            npm_package=cfg.get("npm_package", ""),
        )
    return entries


def save_custom_catalog_entry(entry: CatalogEntry) -> None:
    """Add/update a custom catalog entry (persists to disk)."""
    import yaml
    path = _custom_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    services = data.setdefault("services", {})
    services[entry.name] = {
        "description": entry.description,
        "command": entry.command,
        "args": entry.args,
        "transport": entry.transport,
        "url": entry.url,
        "auth_type": entry.auth_type,
        "env_vars": entry.env_vars,
        "provides": entry.provides,
        "setup_url": entry.setup_url,
        "setup_instructions": entry.setup_instructions,
        "category": entry.category,
        "npm_package": entry.npm_package,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_full_catalog() -> dict[str, CatalogEntry]:
    """Get the combined catalog (built-in + custom). Custom overrides built-in."""
    catalog = dict(BUILTIN_CATALOG)
    catalog.update(load_custom_catalog())
    return catalog


def search_catalog(query: str) -> list[CatalogEntry]:
    """Search the catalog by name, description, or category."""
    query_lower = query.lower()
    catalog = get_full_catalog()
    results: list[CatalogEntry] = []

    for entry in catalog.values():
        score = 0
        if query_lower in entry.name.lower():
            score += 3
        if query_lower in entry.description.lower():
            score += 2
        if query_lower in entry.category.lower():
            score += 1
        for tool in entry.provides:
            if query_lower in tool.lower():
                score += 1

        if score > 0:
            results.append(entry)

    return results
