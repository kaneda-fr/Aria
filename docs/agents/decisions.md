# Decisions and open questions

## D-001 — Control architecture: OPEN

Updated: 2026-10-17. Decision owner: project maintainer.

The latest instruction is to keep the choice open, review development and issues,
and determine how to meet the objective. Neither the earlier preference for tool
calling nor Option A's historical “single source of truth” heading settles this.

| Candidate | Potential benefit | Evidence needed / risk |
| --- | --- | --- |
| Harden the existing LLM tool loop | Reuse current providers and natural multi-step interaction | Small-model tool selection, skipped execution, repeated actions, truthful confirmations |
| LLM as parser + deterministic orchestration (Option A) | Explicit workflow and validation | Parsing quality, ambiguity handling, action vocabulary limits, migration and parity cost |
| Bounded tool calling with deterministic execution checks | Retain flexible language while checking side effects | Define authority boundaries; test whether added complexity earns its cost |

The third row is a review candidate, not an existing implementation or an accepted
design. [Option A](../../NEXT_STEPS_ACTION_PLAN.md) remains a proposal to evaluate.

The [2026-10-17 architecture review](architecture-options.md) recommends a shared
server-owned control service with compact local interpretation and deterministic
execution. Compare revised Option A with a bounded tool proposal under that common
boundary. This is a recommendation pending evaluation, not acceptance of D-001.

Before deciding, pin hardware/model/configuration, fix the test oracle, compare the
same held-out cases, and measure false actuation, correct execution, clarification,
confirmation truth, tool repetitions, and latency. Include conversation, status
questions, ambiguity, follow-ups, ASR noise, provider failure, and bilingual cases.
Use a fake executor for broad evaluation and a separately authorized live-device run.
All candidates must fit the existing Xeon/ESP32 hardware constraint in D-003; compare
compute, memory, power, and concurrent audio load as well as standalone latency.

## D-002 — Context ownership: adopted for this context setup

One shared instruction source (`AGENTS.md`) with thin `CLAUDE.md` imports; small
mandatory startup context and task-specific linked detail. Version-controlled docs
hold portable decisions. Private conversation archives and Serena memories are
discovery aids, not competing specifications. Automation checks consistency and
review coverage; it cannot decide product strategy or certify prose as true.

## D-003 — Existing low-capacity hardware: ACCEPTED

Date: 2026-10-17. Owner/source: explicit maintainer instruction in this session.

Aria must run on the existing old Xeon server and ESP32 devices, within their limited
compute, memory, and power budgets. This guides architecture and component selection
from the start; replacement hardware is not the baseline. Cloud inference is
prohibited by D-004.

Consequences:

- Measure candidate ASR, routing, LLM, and TTS combinations together under realistic
  load, including multiple device connections and existing server workloads.
- Prefer bounded queues, controlled concurrency, small working sets, and avoiding
  unnecessary inference calls. Compare compact/quantized models where supported;
  accept them only when correctness and target-hardware measurements justify them.
- Keep ESP32 responsibilities within verified firmware/device capabilities. Do not
  assume desktop Python, large on-device models, PSRAM, or a particular chip variant.
- Verify CPU instruction support before selecting optimized kernels. Historical
  Xeon/AVX512-VNNI claims in benchmark docs are not hardware capability evidence.
- Document budgets and measurements before accepting increased resource demand.

Read-only inspection subsequently verified Apollo's Xeon E3-1225 v5, four cores/threads,
nominal 64 GB RAM, and AVX2 without AVX512; see [deployment evidence](deployment.md).
The maintainer subsequently identified a Seeed XIAO ESP32-S3 camera/Sense listing;
exact Sense/Plus SKU and firmware/roles still require verification. Power budget and future server/device partition remain
to be established. This constraint does not select a winner for D-001.

## D-004 — Local-only operation: ACCEPTED, MANDATORY

Date: 2026-10-17. Owner/source: explicit maintainer instruction in this session.

All Aria runtime processing and data must stay on owned devices and the local
network. “Local-first” is insufficient: no cloud services, remote inference or
fallback, external telemetry, or Internet dependency during operation is permitted.
This applies to all providers and all stages, including optional features and learning.

LAN communication between the Xeon, ESP32 devices, Jeedom/MQTT, and Sonos remains
allowed. The maintainer clarified that startup may download missing models and their
required model assets. Pre-provisioning before startup is not mandatory. Finish this
preparation before declaring the affected feature ready. Once operational, no external
calls are allowed, including lazy downloads, update checks, telemetry, or remote fallback.
Missing local assets during operation must cause a clear local error or disable the
affected optional feature; request handling must not initiate external downloads.

Acceptance tests startup preparation separately from ready operation: verify missing
models can be downloaded at startup, then block Internet egress while preserving LAN
and exercise relevant flows. Inspect attempted external calls, including background
dependency behavior; blocked calls are not compliance. Also test cached-model startup
offline. This has not yet been tested end to end. The constraint applies equally to
every candidate in D-001.

## D-005 — Server development with Mac client first: ACCEPTED

Date: 2026-10-17. Owner/source: explicit maintainer instruction in this session.

Focus current development on the Aria server on Apollo. Keep the macOS microphone
client as the audio frontend because it makes development easier to iterate.
Defer ESP32 firmware, microphone-client replacement, and device integration work.
The earlier recommendation to replace the Mac client as the first milestone is
superseded. Preserve the audio protocol for eventual ESP32 integration; D-003 and
D-004 remain mandatory. This sequencing decision does not settle D-001.

Later in the same session, the maintainer requested immediate Mac audio playback
instead of Sonos, with ASR, model inference, and TTS still running on Apollo to
measure target-server latency. The Mac is the capture/playback endpoint only.
The [live adapter](../../client/macos/README.md#live-apollo-inference-with-mac-playback)
provides this temporary development mode; it does not replace the retained Sonos
implementation or claim the pending control fixes are complete.

## D-006 — Current priority: tool calling in the working Mac/Apollo setup: ACCEPTED

Date: 2026-10-17. Owner/source: explicit maintainer clarification in this session.

The maintainer says Aria was working with the current development, using the Mac
client to send audio to the server, and identifies tool calling as the issue.
Prioritize actual command selection, grounded arguments, execution count, and
truthful feedback in the existing tool loop. Preserve the working audio setup;
do not treat acoustic diagnosis or narrative truncation as the next milestone.
Conversation-only baselines with providers disabled do not measure tool accuracy.
Use real local inference and fixture devices/captured execution before live actions.
This sequencing decision leaves D-001 open and preserves D-003/D-004/D-005.

## D-007 — Conversational latency is a core requirement: ACCEPTED

Date: 2026-10-17. Owner/source: explicit maintainer instruction: keep latency minimal
and make Aria's engagement feel like a conversation.

Treat latency and correct tool execution as joint acceptance criteria for D-006.
Measure speech-end-to-first-useful-audio and speech-end-to-requested-outcome/final
confirmation separately, including endpointing, ASR, queueing, model/tool turns,
TTS, and playback. Report cold/warm behavior and tail latency on Apollo under named
load; server inference or HTTP audio retrieval alone does not measure audible delay.
Early acknowledgments must be truthful and cannot substitute for completing work.

Favor fewer necessary model round trips, concise complete replies, and bounded local
work. Any extra model call, larger model, concurrency increase, streaming change,
or keep-warm policy needs measured latency/resource benefit on existing hardware.
Preserve audio continuity, turn integrity, validated actions, and truthful outcomes.
No numerical latency SLO is yet agreed; do not present current measurements or an
assistant-proposed target as an accepted threshold. D-001 remains open; local-only
operation and the Xeon/ESP32 constraint remain mandatory.

## D-008 — Speech-only interaction; interruption deferred: ACCEPTED

Date: 2026-10-18. Owner/source: explicit maintainer clarification in this session.

The product interface is speech-only through microphone/speaker devices in the house.
The Mac is a development client, not the intended user interface. Do not make keyboard,
button, or screen interaction a requirement or an intermediate product solution.

Park voice interruption/barge-in for now, including the acoustic echo cancellation
work proposed to support it. Continue latency and tool-correctness work independently.
Earlier sentence playback remains a proposal; if implemented, preserve microphone
suppression for the whole reply, including gaps between queued sentences, and retain
echo guards. Resume listening after playback and its echo tail. This does not authorize
an interruption implementation or expand the deferred ESP32 integration scope.

## D-009 — Start a simple ESP32 microphone client: ACCEPTED

Date: 2026-10-18. Owner/source: explicit maintainer request and follow-up.

Start firmware for the supplied Seeed Studio XIAO ESP32S3 Sense/Plus seller listing.
The ESP32 is microphone-only; Apollo handles speaker playback. Hardcode connection
and other settings for now; runtime configuration comes later. The listing does
not resolve the physical Sense/Plus revision, so verify the microphone profile on
hardware and do not assume PSRAM or camera use.

Start with a simple capture/stream design. How many features should move onto ESP32
to reduce latency remains OPEN: improve the partition based on measured results.
No local VAD, wake model, or inference approach is selected by this request.
Keep the Mac development path and existing server/echo/control behavior. This
supersedes D-005's ESP32 deferral and D-008's reference to that deferred scope;
voice interruption remains deferred and D-001 remains open. Implementation and
validation limits: [ESP32 client](../../client/esp32/README.md).

## D-010 — Wake once, continue until inactivity or goodbye: ACCEPTED

Date: 2026-10-18. Owner/source: explicit maintainer request in this session.

Say Aria once to start a conversation. Follow-ups in that conversation do not need
the name; inactivity or an explicit goodbye returns to waiting for the name.
Validate French and English on Apollo, with tools log-only and no Mac audio.

The initial implementation uses a configurable 30-second window after server-side
playback completion, scoped to a WebSocket or explicit HTTP session ID. The numeric
window is an implementation choice, not a maintainer-selected SLO. Activation is
post-ASR, with no extra inference/model on ESP32. This does not settle D-001 or add
barge-in. See [wake flow and evidence](wake-flow-2026-10-18.md), including ASR limits.

## D-011 — Repair spoken activation and confidence-aware self rejection: ACCEPTED

Date: 2026-10-18. Owner/source: explicit maintainer request in this session.

Fix spoken wake-name recognition and low-confidence self-speaker rejection. Rerun
French/English on Apollo, with tools log-only and no Mac audio during automated
tests; check latency and tool calling. Then re-enable both the Mac microphone and
response playback, as explicitly confirmed by the maintainer. Apollo still owns
inference, and tool actions remain log-only.

Implementation retains nearest-human attribution and raw confidence logs, but an
uncertain self candidate must not reject a human request. Confident-self rejection,
Echo Guard, and playback microphone suppression remain. Bounded voice vocative
normalization is an implementation choice, not a decision to accept arbitrary
phonetic matches. Bare-name acoustic robustness remains open. See
[implementation and evidence](spoken-wake-2026-10-18.md).

## D-012 — Mac/ESP32 microphone feature parity: ACCEPTED

Date: 2026-10-18. Owner/source: maintainer asked for VAD plus automatic gain,
then feature parity between the clients except static ESP32 settings.

Add local VAD, gain control, pre-roll, optional noise gates, bounded recovery, and
diagnostics to the ESP32. Preserve its confirmed
microphone-only role; Apollo handles speaker playback. The Mac remains the development
client. This extends D-009's streaming baseline; future hardware partition changes
still depend on measured results. D-001 remains open and interruption stays deferred.

Implementation choices, not maintainer-selected tuning: portable WebRTC VAD, 200 ms
pre-roll, 800 ms endpoint, and speech-gated AGC capped at 8x. Identical feature
availability does not establish identical acoustics or default gain policies.

Compatibility clarification from the maintainer: client features must work with the
existing Apollo protocol. The attempted mandatory playback/cancel extension was
rejected and removed, including its server changes. Local VAD/gain require no Apollo
update. Playback-aware ESP32 muting remains a gap because the existing protocol has
no playback state; do not invent it from request completion. Existing client-VAD
disconnect/stop can finalize partial speech and cannot promise cancellation.
See the [parity matrix and validation limits](../../client/esp32/README.md).

## D-013 — Named client identity and confirmed input mute: ACCEPTED

Date: 2026-10-18. Source: maintainer requests.
The client ID is its unique, stable name. Reconnects should recover the appropriate
conversation under the normal expiry rules. Show named client status/filtering and
speaker confidence in the local monitor. The initially read-only monitor may now
mute/unmute client input directly, automatically unmuting after five minutes.
Spoken mute applies to the **current client**, asks for confirmation, and uses either
the default or requested duration. Apollo ignores audio; the client keeps operating.
See [protocol](client-protocol.md) and
[implementation/evidence](client-mute-2026-10-18.md). No authentication was requested;
client names and speaker attribution are not credentials. Current tests are Apollo
only with saved/synthetic inputs, tools log-only, no physical microphones/speakers.

## Questions to settle through evidence

- What latency and correct-action thresholds define the first useful deployment?
- Which devices/actions are initially in scope, and which need explicit confirmation?
- How should read-only device-state questions be distinguished from actuation?
- What evidence qualifies as completion when an integration lacks a state observer?
- Given Apollo's verified hardware, what are the available resource/power budgets,
  ESP32 specifications/roles, and local model candidates for comparisons?
- What user feedback is sufficient for a learning label? Silence alone is not a
  verified label; the historical learning rules require review before implementation.

Record new decisions here with date, status (proposed/accepted/superseded), owner,
evidence, alternatives, and consequences. Do not silently rewrite an open question
as a requirement because a later implementation happens to choose one branch.
