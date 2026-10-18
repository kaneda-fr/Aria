# Plugin and device context

Read [architecture](../../docs/agents/architecture.md), [decisions](../../docs/agents/decisions.md),
and the relevant [provider specification](../../aria_full_spec_with_observability.md).
The specification describes target capabilities as well as implemented ones.

- Inventory supplies device/command identity and capabilities; do not invent IDs.
- Keep explicit room/action constraints strict, exclude disabled devices, and retain
  all matches for ambiguity checks; a display limit must not imply unique selection.
- Every provider must work entirely on owned devices/LAN. No Internet APIs, cloud
  fallback, or external telemetry, including optional integrations.
- Keep tool schemas centralized in `app/llm/tools.py`, not per-device/provider prompts.
- Validate subtype, range, payload, and failure behavior before executing side effects.
- Preserve Jeedom MQTT2 payload semantics; do not copy generic example payloads blindly.
- Distinguish publish acceptance from observed physical completion.
- Do not assume expectation tracking or shared MQTT connections are implemented because a spec asks for them.
- Keep control-strategy D-001 open. Add tests with fake providers before live-device checks.
- Update architecture/status and relevant protocol docs when contracts change.
