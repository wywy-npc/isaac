"""Default schema template for new wikis.

The schema tells the LLM how the wiki is structured, what the conventions are,
and what workflows to follow. User and LLM co-evolve this over time.
"""


def default_schema(name: str, description: str = "") -> str:
    desc_line = f"\n{description}\n" if description else ""
    return f"""# {name} Wiki Schema
{desc_line}
## Structure
- `raw/` — immutable source documents. The LLM reads from these but never modifies them. This is the source of truth.
- `pages/` — LLM-generated wiki pages. The LLM owns this layer entirely. It creates pages, updates them when new sources arrive, maintains cross-references, and keeps everything consistent.
- `index.md` — content catalog: every page listed with a one-line summary, organized by category. Updated on every ingest.
- `log.md` — append-only chronological record of operations. Format: `## [YYYY-MM-DD] operation | description`
- `schema.md` — this file. Conventions and workflows.

## Page Types
- **summary** — Summary of a single source document. One per raw source.
- **concept** — Core concept explanation (what it is, why it matters, how it works).
- **entity** — Person, company, tool, project, or other named entity.
- **comparison** — Side-by-side analysis of two or more concepts/entities.
- **synthesis** — Cross-source analysis, evolving thesis, or meta-observation.

## Conventions
- Every page starts with `# Title`
- Use `[[page-name]]` for cross-references between wiki pages
- Include `## Sources` section at the bottom of each page, citing raw/ files or URLs
- Cross-references should be bidirectional when meaningful
- Page filenames: lowercase, hyphens, no spaces (e.g. `tool-use-patterns.md`)

## Ingest Workflow
When processing a new source:
1. Read the source document
2. Discuss key takeaways with the user
3. Write a summary page in `pages/`
4. Identify related existing pages and update them with new information
5. Create new concept/entity pages if the source introduces them
6. Update cross-references (`[[links]]`) across touched pages
7. Regenerate `index.md` with updated page list
8. Append an entry to `log.md`

A single source typically touches 10-15 wiki pages.

## Query Workflow
When answering a question:
1. Read `index.md` to find relevant pages
2. Read those pages for context
3. Synthesize an answer with citations to wiki pages
4. If the answer is valuable, file it back as a new wiki page
5. Append a query entry to `log.md`

## Lint Checks
Periodic health checks to maintain wiki quality:
- Contradictions between pages
- Stale claims that newer sources have superseded
- Orphan pages with no inbound links
- Important concepts mentioned but lacking their own page
- Missing cross-references between related pages
- Data gaps that could be filled with a web search
"""
