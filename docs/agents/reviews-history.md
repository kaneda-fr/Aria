# Earlier context reviews — 2026-10-17

## 2026-10-17 — Initial context review

Reviewed repository history, remote branch heads, current source paths, historical
reports, and available project memories. Kept the control strategy open per the
maintainer's latest instruction. Findings and provenance are in [status](status.md).
No application behavior was changed and no live-device tests were run.

Validation: seven standard-library context-checker tests passed, including rejection
of missing links, oversized seeds, duplicate Claude policy, and unreviewed source
changes. Structural and HEAD-relative context checks, actionlint, and diff whitespace
checks passed. Existing application tests could not run: the local virtual environment
has no pytest installed. No model benchmark or fresh-agent loading test was run.

## 2026-10-17 — Hardware constraint clarified by maintainer

Promoted the existing old Xeon server and ESP32 devices to a core, accepted design
constraint (D-003), including model selection, memory/compute/power budgets, and
server/device work placement. Updated startup and scoped guidance plus evaluation
criteria. Exact specifications remain unverified; D-001 stays open. Documentation
changes only; no hardware measurements or application changes.

## 2026-10-17 — Local-only requirement made explicit

Recorded the maintainer's mandatory local-only boundary as D-004 and replaced
local-first wording in the startup seed, objective, and README. Propagated the rule
to architecture, provider/routing guidance, and evaluation: no runtime Internet
dependency, remote fallback, or external telemetry. The initial prohibition on
startup downloads was too strict and is superseded by the clarification below.
LAN communication remains allowed. Offline compliance is explicitly unverified;
this update changes documentation, not runtime behavior.

## 2026-10-17 — Startup downloads allowed; operation stays local-only

Corrected D-004 and all affected guidance per the maintainer: missing models may
download at startup. Once operational, no external calls are allowed. Validation now
separates startup preparation from ready operation and checks both request-time and
background calls. No runtime implementation or connectivity tests were performed.

## 2026-10-17 — Architecture options reviewed against operating constraints

Compared model-controlled tools, constrained interpretation, bounded tool proposals,
and fixed commands against current source/report failures, the old Xeon/ESP32
baseline, and startup-download/operational-offline requirements. Added the
[architecture recommendation](architecture-options.md), with staged delivery and
validation gates. D-001 remains open. Verified additional queue and CPU-threading
findings in source; refreshed GitNexus using `--index-only`. Serena semantic overview
was unavailable because no language server was active; used the graph and targeted
source reads. Consulted official llama.cpp, Hugging Face, and Espressif documentation.
No application behavior, deployment settings, or models were changed; no hardware
benchmarks or connectivity tests were run.

## 2026-10-17 — Recovered actual Mac/Apollo launch setup

Inspected local scripts and queried Apollo read-only over SSH: deployed launch
script, relevant shell-history entries, CPU/memory details, listening ports, and
installed Ollama model list. Added [deployment evidence](deployment.md), preserving
the difference between deployed Ollama and local llama.cpp configurations. Corrected
the previously unknown Xeon hardware details. No startup, restart, inference, device
action, or audio test was performed. Historical launch commands do not establish
that today's complete deployment works without additional checks.

## 2026-10-17 — XIAO ESP32-S3 hardware identified

Recorded the maintainer's Seeed XIAO ESP32S3 Sense/Plus camera listing in
[deployment context](deployment.md) and checked Seeed's official specifications and
microphone documentation. Distinguished standard Sense specifications from Plus;
the seller title alone does not settle the physical SKU. Recommended microphone
satellite work placement remains a proposal, since firmware and actual roles have
not been supplied. No firmware, device, or server settings were changed.

## 2026-10-17 — Server-first iteration scope confirmed

Recorded D-005 from the maintainer: develop on Apollo with the existing Mac
microphone client; defer ESP32 integration. Updated the seed, milestone, architecture
recommendation, deployment notes, and client guidance. This supersedes the proposed
ESP32-first milestone without changing hardware/local-only constraints or selecting
a control architecture. Documentation only.

## 2026-10-17 — Repeatable Mac/Apollo E2E workflow established

Added an isolated runner, capture-only Sonos adapter, real-client fixture replay,
microphone authorization probe, and network/completion regression tests. Used a
separate Apollo test dependency directory and unique source snapshots; retained
deployed configuration and existing model assets. Runtime network isolation uses
unprivileged namespaces, with SSH Unix-socket forwarding and traced inference.

Run `20261017T093232Z-f894aa` passed text and English/French audio pipelines through
HTTP retrieval of Piper audio, plus 23 existing Apollo tests; six harness tests and
seven context tests passed locally. No external/DNS destinations were observed by
the trace audit. Verified no test units remained active after cleanup. Live Mac
microphone capture remains pending normal macOS consent (`not determined`); no
physical Sonos playback or home-device actions were performed. See [workflow](e2e-testing.md)
and [current evidence](status.md) for exact coverage and limits.

## 2026-10-17 — Live Mac microphone verified after consent

The maintainer granted microphone access; the read-only authorization probe returned
`authorized`. Run `20261017T094050Z-34e0aa` passed the real microphone acoustic test,
three automated pipeline scenarios, and 23 existing Apollo tests. Capture ran at
96 kHz through the unmodified Mac client; it exited cleanly with no reported queue
or WebSocket drops. The trace audit found 15 loopback destinations and no rejected
external/DNS attempts. Verified cleanup removed all test units. Updated the earlier
permission limitation in current status; physical Sonos and home actions remain
outside this test profile. No runtime code or deployment configuration changed.

## 2026-10-17 — Bilingual E2E baseline and server diagnosis

Added the 18-case French/English characterization suite and deterministic handler
probes. Run `20261017T095752Z-0c1757` completed all pipelines, but a tightened review
flags nine quality cases. Preserved the original report and separate rescoring
artifact: four narrative cutoffs, two wrong voices, a split French WAV turn, and
both microphone interactions. Extra microphone transcripts remain acoustically
unattributed. The English question did not reach the LLM despite a plausible answer.

Recorded timings with their actual boundary, memory samples, model/source hashes,
and coverage exclusions in [the baseline report](baseline-2026-10-17.md). Probes
confirm voice/text tool divergence, silent queue displacement, and the broken forced
CHAT API branch. Prioritized shared request handling/admission, bilingual behavior,
and validated outcomes; D-001 remains open. No production code or deployed runtime
configuration changed and no device actions executed. Runtime isolation audit found
24 loopback destinations with no external/DNS attempts; cleanup was verified.

Validation: 23 Apollo tests and diagnostic assertions; 12 local harness tests,
including lost-request/false-answer and wrong-voice regressions; seven context tests.
GitNexus impact checks scoped edits to the runner/metric tests; Serena semantic
retrieval remained unavailable because no language server was active.


## 2026-10-17 — Shared voice/text path and bounded admission implemented

Per maintainer authorization, replaced duplicate voice inference with a shared
processor and one bounded FIFO worker for voice/HTTP. Added explicit busy rejection,
request IDs, terminal outcomes, pre-execution expiry, pending cancellation, and
shutdown handling. Fixed the shared CHAT fallback/speech behavior while preserving
the existing tool-loop strategy; D-001 remains open. Updated environment, startup
seed, architecture, status, testing guide, and [implementation report](shared-request-path.md).

Final isolated Apollo run `20261017T102130Z-22e4e8`: 39 unit/handler tests passed;
18 bilingual pipeline cases produced audio; lifecycle audit found 24 accepted and
24 completed requests with no missing/duplicate terminal outcomes. Ten quality
cases remain flagged for turns, voices, and truncation. Runtime audit observed 26
loopback destinations and no external/DNS attempts. Verified cleanup and exact
local/Apollo application source hashes. No live device actions or persistent
deployment changes. Current diagnostic assertions passed separately.

Fifteen local harness tests, six queue tests, seven context tests, source-review
checks, and whitespace checks passed. GitNexus aggregate change detection reports
high integration risk across expected startup/voice/LLM flows; individual impacts
were checked before edits and transport/lifecycle regressions exercised on Apollo.
Preserved inherited changes; no commit or bulk staging.
