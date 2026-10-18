# Maintaining LLM context

## Loading model

`AGENTS.md` is the short, shared entry point. `CLAUDE.md` imports it with `@AGENTS.md`;
scoped Claude files import their neighboring AGENTS files. Explicit root instructions
tell the agent to read project/status plus relevant scoped guidance, because a normal
Markdown link is not a promise that every harness will load the target automatically.

Load architecture, workflow, decisions, and legacy specifications only when relevant.
Do not import the whole documentation tree at startup. Keep root instructions under
50 lines and scoped instructions under 60; move growing explanations into linked docs.

This follows [OpenAI's instruction-discovery explanation](https://openai.com/index/unrolling-the-codex-agent-loop/)
and Anthropic's [memory/import documentation](https://code.claude.com/docs/en/memory)
and [concise-instruction guidance](https://code.claude.com/docs/en/best-practices).
Sources checked 2026-10-17. Harness behavior can change; verify actual loading in a
fresh session, rather than assuming a link or filename guarantees it.

## Source of truth

For intent: current explicit maintainer decisions, then accepted entries in
[decisions](decisions.md), then the [objective](project.md). For implemented behavior:
current source and reproducible tests, then [status](status.md), then descriptive docs.
Contradictions are evidence to resolve; code does not silently redefine the objective.

Older specs, `.copilot-instructions.md`, `IMPLEMENTATION_ROADMAP.md`, summaries, and
test reports are historical material. Their “authoritative”/“production” headings do
not override newer decisions or prove implementation. Read with the status audit.
Keep original documents as evidence rather than erasing their history.

## Automatic maintenance loop

At the end of every task, the working agent reviews context affected by its changes:

| Change | Review/update |
| --- | --- |
| User goal, priority, architecture choice | project.md and decisions.md |
| Runtime flow, interface, package/module layout | architecture.md, status.md, scoped AGENTS.md |
| Commands, prerequisites, test behavior | workflow.md and tests/AGENTS.md |
| Fixed issue or newly reproduced limitation | status.md; include evidence and remove obsolete current claims |
| Instruction/checker changes | this file, source-map.json, and applicable entry points |

Run `python3 scripts/check_context.py`. For review against a branch or commit, run
`python3 scripts/check_context.py --base <base-ref>`. The checker validates maintained
local links, import wrappers, context size, and required files. With a base, it also
requires context review for mapped source changes. It uses the working tree relative
to the merge base, including relevant untracked files; it does not stage or write files.

The GitHub workflow runs these checks on pull requests and pushes. It becomes active
when the workflow is committed/pushed. Requiring the check for merges is a separate
repository setting, not configured by this change. It has read-only permissions and
does not invoke an LLM, contact devices, or rewrite product decisions.

If source changes have no durable context effect, append a concise entry to
[review log](reviews.md): date, affected area, why no context update is needed, and
evidence. The checker accepts this explicit review record. Editing a document is only
a review signal, not machine proof that its meaning is correct; reviewers must inspect it.

## Avoid memory drift

- Store durable rationale once, with links to evidence. Keep status bounded; use git
  history for past snapshots rather than growing an endless session transcript.
- Promote memory findings only after project/content/date verification. Assistant
  summaries are not user decisions; archives can have misleading project labels.
- Do not auto-bump review dates or let a generated summary choose between open strategies.
- Revisit context when behavior changes, tests contradict a claim, a user changes direction,
  or an agent repeatedly makes the same mistake. Prune rules that no longer help.
- Keep graph indexes and private tool memories derived/disposable. Refresh GitNexus with
  `npx gitnexus analyze --index-only` to avoid regenerating duplicate startup instructions.

## Fresh-session smoke check

Start a new session at the repository root and ask for the objective, current control
path, open architecture decision, and appropriate tests. It should identify D-001 as
OPEN and distinguish voice from text control. Repeat from a scoped directory and
confirm both root and local guidance. Claude's `/context` can help inspect loading.
This smoke check has not been executed merely by validating files locally.
