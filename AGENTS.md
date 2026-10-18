# Aria — shared LLM seed context

Build a responsive **local-only** French/English voice assistant with reliable home
control and truthful spoken feedback. **The future control architecture is OPEN**;
current LLM tool calling and the Option A parser/orchestrator remain alternatives.
**Core hardware constraint: the existing old Xeon server and ESP32 devices.**
Design for limited compute, memory, and power; do not assume a hardware upgrade.
**Conversational latency is core:** minimize speech-end-to-useful-audio delay alongside
correct control; measure full turns on Apollo, including queueing and tool round trips.
**Current focus: Apollo development plus a simple ESP32-S3 microphone client.**
Keep the Mac development client; measure before moving features onto ESP32 (D-009).

## Every session

1. Read [objective and direction](docs/agents/project.md) and [current evidence](docs/agents/status.md).
2. Inspect branch/HEAD and working-tree changes; preserve inherited work.
3. Read applicable scoped AGENTS.md files before editing that area, even if your
   harness did not load them automatically. Use the [architecture map](docs/agents/architecture.md).
4. Follow [development workflow](docs/agents/workflow.md); verify relevant claims in source/tests.

## Non-negotiable context

- Local-only operation is mandatory: processing and data stay on owned devices/LAN.
  No cloud APIs, remote inference/fallback, external telemetry, or runtime Internet dependency.
  Startup may download missing models; once operational, no external calls are allowed.
  Preserve bounded audio ingestion and the working audio/echo path.
- Judge models, dependencies, and server/device work placement against the Xeon/ESP32
  resource envelope. Validate on target hardware; desktop benchmarks are insufficient.
- Voice and text share bounded request admission and processing. Preserve parity;
  MQTT publish acceptance is not physical completion. Historical benchmarks do not
  prove end-to-end reliability.
- Use Serena for semantic analysis/editing. Follow [code intelligence](docs/agents/code-intelligence.md).
- Run GitNexus upstream impact before editing existing symbols; report callers,
  processes, risk, and warn on HIGH/CRITICAL. Use graph-aware rename.
- Run GitNexus change detection before committing; verify affected scope.
- Respect current maintainer decisions. Old specs and memories are evidence, not
  automatic authority; see [decisions](docs/agents/decisions.md).

## Keep context current

Update affected context in the same task, following [maintenance](docs/agents/maintenance.md).
Run `python3 scripts/check_context.py`; use `--base <base-ref>` for source-change coverage.
If no durable update is needed, record why in [reviews](docs/agents/reviews.md).
Keep startup instructions small; link detail instead of duplicating it.
