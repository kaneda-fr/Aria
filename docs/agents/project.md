# Aria: objective and direction

Reviewed: 2026-10-17. Owner of product and architecture decisions: project maintainer.

## Objective

Build a responsive, local-only voice assistant that understands French and English,
converses naturally, and reliably controls home devices through extensible providers.
The user should be able to speak a request, get the intended action or a useful
clarification, and hear a truthful, concise response. Preserve the working audio path.
The system must run within the limited compute, memory, and power envelope of the
maintainer's existing old Xeon server and ESP32 devices. This is a core design
constraint, not a later optimization or an optional deployment profile.
The maintainer explicitly requires engagement to feel like a conversation: minimize
perceived and actual response latency as part of correctness work, not afterward.
Measure from the end of user speech to the first useful audible response, and
separately to the requested outcome/final confirmation. An acknowledgment does not
make a slow action complete. See [D-007](decisions.md).

The product is **speech-only**, using microphones and speakers in the house; the Mac
is only a development client. Voice interruption is deferred. Keep latency work
independent of barge-in and do not depend on a keyboard/button interface; see
[D-008](decisions.md). Say “Aria” once, then allow follow-ups until inactivity or
an explicit goodbye ([D-010](decisions.md)). The server implements a configurable
30-second window; [wake-flow evidence](wake-flow-2026-10-18.md) separates conversation
state correctness from spoken-name recognition.

This is a synthesis of the [user guide](../../README.md),
[audio requirements](../../.copilot-instructions.md), and
[provider specification](../../aria_full_spec_with_observability.md), not a claim that
the complete experience is implemented or validated.

## Stable constraints

- **Existing hardware first:** architecture, models, dependencies, concurrency, and
  server/device work placement must fit the old Xeon and ESP32 devices already used
  in the project. Do not make new hardware or a GPU a prerequisite. Cloud offload is prohibited.
- **Local-only is mandatory:** ASR, routing, LLM, TTS, speaker recognition, device
  control, state, logs, and any learning must remain on owned devices/local network.
  No cloud APIs, external telemetry, remote fallback, or runtime Internet dependency.
  Local HTTP, WebSocket, MQTT, and Sonos LAN traffic are compatible with this rule.
- Startup may download missing models and their required model assets. Complete
  downloads before the affected feature becomes operational; distinguish startup
  preparation from ready operation. Once operational, no external calls are allowed,
  including lazy downloads, update checks, telemetry, and remote fallback. If required
  local assets are unavailable during operation, report a local error or disable the
  affected optional feature rather than fetching them in the request path.
- Streaming audio must remain bounded and must not wait on LLM or playback latency.
- Preserve the existing macOS client's native-rate capture and resampling to 16 kHz mono PCM16LE,
  and Sonos HTTP pull playback of generated WAV clips.
- Device capabilities come from inventory. Generated prose is not evidence of an
  action, and MQTT publish acceptance is not evidence of physical completion.
- Keep providers optional and decoupled from the core voice pipeline; Jeedom MQTT
  is the first device integration, not the only possible integration.
- Retain Echo Guard v2 while evaluating speaker identification; do not remove a
  working feedback guard merely because a replacement appears in a plan.

## Strategy remains open

On 2026-10-17 the maintainer explicitly asked to keep the control architecture open
and revisit development, issues, and memories before deciding. This supersedes the
earlier same-session preference for LLM tool calling as the target.

The shared voice/text processor uses LLM tool calling when providers are available.
The maintainer clarified the immediate priority on 2026-10-17: Aria was working
with the Mac client sending audio to Apollo; the issue to address is tool calling
in the current development. Diagnose and harden that path first. This is a work
priority, not a decision to settle the future architecture or redesign capture.
The deterministic parser/orchestrator in
[Option A](../../NEXT_STEPS_ACTION_PLAN.md) is an alternative proposal. Neither
architecture is the accepted future target. See [decisions](decisions.md).
The [architecture review](architecture-options.md) recommends compact local
interpretation with server-owned deterministic execution, subject to comparison on
the actual hardware. It includes the proposed ESP32/server split and startup boundary.
Both must satisfy the hardware constraint. A larger model or additional model call
is not an acceptable reliability fix without measuring the resource/latency cost.

## Recommended next milestone, not an approved implementation plan

On 2026-10-18 the maintainer expanded the Apollo/Mac development scope to start
an ESP32-S3 microphone-only client, with speaker playback handled by Apollo and
settings hardcoded initially. The maintainer subsequently requested VAD/automatic
gain and microphone-feature parity with the Mac ([D-012](decisions.md)); the client
now includes those features. Further processing placement depends on measured results.
Both clients must remain compatible with the existing Apollo audio protocol;
client improvements must not require a server update. The Mac
remains a development client. This supersedes D-005's ESP32 deferral without
settling the control architecture; see [D-009](decisions.md) and the
[ESP32 client](../../client/esp32/README.md).

Establish a reproducible end-to-end control baseline, then compare approaches on the
same held-out French/English requests, devices, model configuration, and hardware.
Keep paired voice and text tests for the shared request path; a transport refactor
does not establish model control accuracy or physical-device completion.

| Outcome to measure | Evidence required |
| --- | --- |
| Conversation does not actuate devices | Execution trace with zero side effects, not just a plausible answer |
| Control reaches the correct device | Resolved inventory command, validated payload, exact execution count |
| Ambiguity is handled honestly | Clarification before any ungrounded action |
| Confirmation matches reality | Distinguish requested, publish accepted, observed completion, failure, timeout |
| Interaction feels conversational | Speech-end-to-first-useful-audio and speech-end-to-outcome/final-confirmation p50/p95 on Apollo; separate cold/warm runs, failures, and queue load; account for endpointing, ASR, queueing, every model/tool round trip, TTS, and playback |
| Fits existing hardware | Peak/steady RAM, CPU load, model size, cold start, and idle/active power on Apollo; measure ESP32 heap, buffering, transport, and speech-end-to-audio latency against the Mac baseline |
| Operates entirely locally | Test startup downloads separately, then block Internet egress with LAN available once ready; exercise required flows and inspect attempted external calls, including background work |
| Changes preserve speech usability | Audio continuity, interruption behavior, feedback suppression checks |

Do not turn historical latency targets or synthetic accuracy into a production SLO.
Choose numerical acceptance thresholds with the maintainer after measuring the baseline.
Optimize useful work per model call, compact grounded context, and short complete
responses. Evaluate removing redundant model turns and generating simple truthful
command confirmations directly from structured outcomes. These are experiments for
the current loop, not acceptance of Option A. Keep models ready within measured RAM
and power limits; bound CPU competition with audio. Do not lower token limits blindly,
clip utterances, skip execution checks, or speak speculative success to gain speed.
For entity matching, the maintainer prefers model-free approaches; additional models
are the least preferred option because of compute and latency (2026-10-18). Evaluate
normalization, declared aliases and local lexical resources before model inference.
Integration expansion and self-learning should follow reliable execution and evaluation.

Apollo is verified as a four-core/thread Xeon E3-1225 v5 with nominal 64 GB RAM and
AVX2; see [deployment evidence](deployment.md). Measure other server workloads and
power, and establish ESP32 variants/PSRAM, firmware, and roles before setting numerical
budgets. These remaining details are unverified. Resource limits must never be met by
weakening execution correctness or inventing confirmations.
