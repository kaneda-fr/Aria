# Code intelligence workflow

The requirements below preserve the repository's existing GitNexus policy. The old
indexed symbol counts were a snapshot, not durable project context.

## Serena

Activate Aria and read Serena's initial instructions before code work. Prefer symbol
overviews, definitions, references, and semantic edits over whole-file search/editing.
Memory files may be reviewed under `.serena/memories/`. Do not treat them as the
canonical product specification.

On 2026-10-18 the empty `language_servers` list was repaired in
[project configuration](../../.serena/project.yml): Python, TypeScript (including JS),
and C/C++. The active MCP session was updated through its local dashboard API;
Python/JS/C symbol lookups succeeded and `serena project index .` indexed 184 files.
Clangd reported unresolved include paths: firmware-wide references still need a
complete ESP-IDF compilation database. Earlier sessions used GitNexus fallback.
Check availability in each new session; do not silently pretend a tool succeeded.

## GitNexus requirements

- Before editing any function, class, or method, run upstream impact analysis and
  report direct callers, affected processes, and risk to the user.
- Warn before proceeding when impact is HIGH or CRITICAL.
- Use query for unfamiliar execution flows and context for callers/callees.
- Use graph-aware rename; never rename symbols through blind find-and-replace.
- Before committing, run change detection and verify only expected scope is affected.
- When the index is stale, refresh it before trusting graph results.

MCP operations are `gitnexus_query`, `gitnexus_context`, `gitnexus_impact`,
`gitnexus_rename`, and `gitnexus_detect_changes`. Resources include
`gitnexus://repo/Aria/context`, `/clusters`, `/processes`, and `/process/{name}`.

When MCP is unavailable, inspect `npx gitnexus --help` for the equivalent installed
CLI. This review used GitNexus 1.6.12, including `query`, `context`, and `analyze`.
Refresh with `npx gitnexus analyze --index-only` to preserve curated instruction files.
Do not drop the impact requirement because a tool is missing; report the limitation
and obtain a working analysis path before editing existing symbols.

## Existing local skill references

These pre-existing local skills provide detailed workflows when present:

- `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-guide/SKILL.md`
- `.claude/skills/gitnexus/gitnexus-cli/SKILL.md`

They were untracked at the initial review. Do not assume a fresh clone includes them.
