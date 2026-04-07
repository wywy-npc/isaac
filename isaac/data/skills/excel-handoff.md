---
name: excel-handoff
description: Collaborative Excel workflow — agent and human take turns editing a spreadsheet
params:
  file: "Path to the spreadsheet file"
  task: "What to do with the spreadsheet (analyze, build, update, review)"
tools_used: [spreadsheet_read, spreadsheet_write, spreadsheet_edit, spreadsheet_diff, spreadsheet_inspect, spreadsheet_convert, memory_write]
user_invocable: true
---

# Excel Handoff: {{task}}

You are working on a spreadsheet collaboration with the user.

File: **{{file}}**
Task: **{{task}}**

## The Handoff Protocol

This skill implements the **AI-Human-AI handoff loop** for spreadsheets:

```
Agent creates/edits → User opens in Excel → User edits → Agent diffs & continues
```

### Phase 1: Understand the File

1. Run `spreadsheet_inspect` on {{file}} to understand its structure
2. Run `spreadsheet_read` on the relevant sheets (use `include_formulas: true`)
3. Note the **fingerprint** — this is your baseline for detecting user changes later
4. Save the fingerprint to memory at `handoffs/{{file}}.md` so you can track the round-trip

### Phase 2: Do Your Work (based on {{task}})

**If "analyze":**
- Read all sheets, understand the data model
- Identify key metrics, trends, anomalies
- Write a summary to the user — do NOT modify the file unless asked

**If "build":**
- Create the spreadsheet from scratch with `spreadsheet_write`
- Use proper headers, data types, and structure
- Add formulas where appropriate using `spreadsheet_edit` (formulas like `=SUM()`, `=VLOOKUP()`, etc.)
- Bold headers, sensible column widths

**If "update":**
- Read the current state
- Apply changes with `spreadsheet_edit` for surgical updates (preserves formatting)
- Or `spreadsheet_write` for a full sheet rewrite

**If "review":**
- Read the file and check for: data consistency, formula errors, missing values, structural issues
- Report findings without modifying the file

### Phase 3: Handoff to User

After you finish your work:

1. Tell the user what you did and what they should review
2. Say: *"The file is ready for you to open in Excel. When you're done editing, let me know and I'll pick up where you left off."*
3. Save the current fingerprint to memory for later comparison

### Phase 4: Receive Back from User

When the user says they're done editing:

1. Re-read the file with `spreadsheet_read`
2. Compare the new fingerprint against the saved one — if unchanged, nothing to do
3. If changed, run `spreadsheet_diff` between the original version and the new one
4. Report what the user changed: added rows, modified values, new formulas, etc.
5. Ask if the user wants you to continue working based on their changes

## Rules

- **XLSX vs CSV**: They are NOT interchangeable. XLSX preserves formulas, formatting,
  multiple sheets, and cell types. CSV is flat text. Always prefer XLSX for collaborative
  work. Only use CSV for data import/export where simplicity matters.
- **Never silently drop data**: If a conversion would lose formulas or formatting, warn first.
- **Fingerprints are your memory**: Always save them so you can detect what changed.
- **Surgical edits over full rewrites**: Use `spreadsheet_edit` when changing a few cells.
  Use `spreadsheet_write` only when creating from scratch or rewriting a whole sheet.
- **Formulas are first-class**: Read them, write them, preserve them. Financial users
  depend on formula integrity.
- **Explain your formulas**: When you add formulas, briefly explain what they compute.

## Model Capabilities & Limitations

Current LLMs handle tabular data well when:
- Data is presented with clear headers and consistent types
- Tables have fewer than ~200 columns (context window limits)
- Formulas use standard Excel functions (SUM, VLOOKUP, IF, INDEX/MATCH, etc.)

Watch out for:
- Very wide tables (50+ columns) — summarize or focus on relevant columns
- Complex nested formulas — break them into named helper cells
- Circular references — flag them, don't try to resolve automatically
- Macro-enabled workbooks (.xlsm) — read the data, but don't touch the VBA
