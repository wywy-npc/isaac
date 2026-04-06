---
name: onboard-repo
description: Explore a codebase and write a structured summary to memory for future context
params:
  path: "Path to the repo (default: current directory)"
tools_used: [file_read, file_list, file_search, bash, memory_write]
user_invocable: true
---

# Onboard Repo: {{path}}

You are exploring and documenting a codebase so that future conversations have full context.

## Phase 1: Overview

1. Read the top-level structure:
   - `file_list` on {{path}} (depth 1)
   - Read README.md if it exists
   - Read any CLAUDE.md, ARCHITECTURE.md, CONTRIBUTING.md

2. Identify the tech stack:
   - Check for: package.json, pyproject.toml, Cargo.toml, go.mod, Gemfile, pom.xml
   - Read the dependency file to understand frameworks and libraries

3. Check git context:
   - `bash: git log --oneline -10` (recent activity)
   - `bash: git branch -a` (branch structure)
   - `bash: git remote -v` (where it lives)

## Phase 2: Architecture

4. Map the directory structure:
   - `file_list` on key directories (src/, lib/, api/, components/, etc.)
   - Identify the pattern: monorepo? MVC? microservices? flat?

5. Find entry points:
   - Main file (main.py, index.ts, App.tsx, cmd/main.go)
   - Config files (.env.example, config/, settings/)
   - Build/deploy (Dockerfile, CI config, Makefile)

6. Understand data flow:
   - How does data get in? (API routes, CLI args, UI forms)
   - Where is it stored? (database, files, memory)
   - How does it get out? (API responses, rendered pages, reports)

## Phase 3: Write to Memory

7. Write a structured summary to `repos/{{path}}.md`:

```markdown
# Repo: {{path}}

## Overview
One paragraph: what this project does and who it's for.

## Tech Stack
- Language: ...
- Framework: ...
- Database: ...
- Key dependencies: ...

## Structure
- `src/` — ...
- `lib/` — ...
- (key directories and their purpose)

## Entry Points
- Main: ...
- Config: ...
- Tests: ...

## Key Patterns
- How state is managed
- How routing works
- How auth works (if applicable)

## Development
- How to run locally
- How to test
- How to deploy

## Notes
Anything surprising, non-obvious, or worth remembering.
```

## Rules
- Read before guessing — always check the actual files
- Focus on what's non-obvious — don't document what the README already says
- Keep the memory node under 3000 chars — summary, not encyclopedia
- Link to related memory nodes if they exist (e.g. [[people/author.md]])
