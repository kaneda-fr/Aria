# Server context

Read [architecture](../docs/agents/architecture.md) and [status](../docs/agents/status.md).
Root guidance applies. Read deeper scoped guidance before touching routing or plugins.

- Preserve the shared [wake flow](../docs/agents/wake-flow-2026-10-18.md): gate before
  model/tools, retain raw transcripts, clear context on close/expiry, and use voice
  onset separately from queue age. Preserve [confidence-aware self rejection](../docs/agents/spoken-wake-2026-10-18.md)
  and Echo Guard; nearest self identity alone must not discard human input.
- Preserve bounded audio ingestion and process only FINAL transcripts through the LLM.
- Client feature work must preserve the existing `/ws/asr` protocol without server
  updates. Request completion is not playback completion or pending-audio cancellation.
- Keep `process_text` and voice admission on the shared `_process_request` worker.
  Preserve bounded FIFO admission, explicit rejection/expiry, request IDs, and tool parity.
- Preserve [grounded control](../docs/agents/grounded-control-2026-10-18.md): assistant
  actions require server-selected, single-use authorization; render actual outcomes
  without claiming physical completion from recording or publish acceptance.
- Preserve current-request language through speech scheduling and every TTS chunk;
  never reuse a previous request/session's language for short numeric answers.
- Keep local clock/capability replies grounded in runtime state, with shared history
  and speech; never use the model to estimate the current time or describe missing tools.
- History prefill is speculative cache work: discard all output, preserve exact
  history/tools, wait for audio preparation, and cancel it before new processing.
  Measure queue-inclusive latency and cancellation cost on Apollo.
- Keep optional providers/LLM/TTS failures isolated from unrelated functionality.
- Keep all processing/data local to owned devices/LAN and inference asynchronous
  with respect to audio ingestion. No cloud APIs, remote fallback, external telemetry,
  or external calls once operational. Startup may download missing models before
  readiness; report missing local assets during operation without fetching them.
- Fit the existing old Xeon: measure combined ASR/router/LLM/TTS CPU and peak RAM,
  bound concurrency, and do not assume GPU acceleration or modern CPU instructions.
- Preserve Echo Guard v2 and in-memory speaker analysis unless the task changes that policy.
- Update architecture/status for flow changes and ENVIRONMENT.md for configuration changes.
- Follow root GitNexus impact requirements before editing symbols.
