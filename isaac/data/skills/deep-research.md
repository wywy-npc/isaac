---
name: deep-research
description: Deep research on a topic with web search, source synthesis, and memory persistence
params:
  topic: "The subject to research"
  depth: "How deep — shallow (3 searches), medium (6), deep (10+). Default medium."
tools_used: [web_search, memory_write, file_write]
user_invocable: true
---

# Deep Research: {{topic}}

You are conducting deep research on: **{{topic}}**

Depth level: {{depth}} (default: medium)

## Strategy

### Phase 1: Broad Survey
1. Run 3 diverse web searches on different angles of {{topic}}
   - One on the core concept/definition
   - One on recent developments or news
   - One on expert opinions or analysis
2. For each result, extract: key claims, data points, dates, and source URLs

### Phase 2: Deep Dive (medium/deep only)
3. Based on Phase 1 findings, identify the 2-3 most interesting threads
4. Run targeted searches on each thread
5. Look for primary sources, research papers, official documentation

### Phase 3: Synthesis
6. Cross-reference findings — identify consensus, contradictions, and gaps
7. Write a structured research memo to memory at `research/{{topic}}.md`

## Output Format

Write your findings to memory as a structured document:

```markdown
# Research: {{topic}}

## Executive Summary
3-sentence overview of what you found.

## Key Findings
- Finding 1 (source: URL)
- Finding 2 (source: URL)
- ...

## Analysis
What the findings mean. Where experts agree/disagree.

## Sources
1. [Title](URL) — brief description
2. ...

## Open Questions
- What you couldn't find or needs follow-up
```

## Rules
- Always cite sources with URLs
- Flag contradictory information explicitly
- Don't hallucinate — if you can't find it, say so
- Write to memory so the research persists across sessions
