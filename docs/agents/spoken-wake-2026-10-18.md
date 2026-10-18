# Spoken activation and self-speaker rejection — 2026-10-18

[D-011](decisions.md) authorizes fixing spoken activation and low-confidence
self-speaker rejection, validating on Apollo with no Mac audio, then restoring
both the Mac microphone and playback. Tools stay log-only throughout.

## Implemented behavior

The nearest speaker candidate and its confidence are different facts. The old
voice-finalization branch rejected any base user `aria`, bypassing the existing
self-confidence threshold. Synthetic English inputs with scores around 0.13–0.25
were therefore discarded despite `known=false`.

`self_speech_decision` now requires both `known` and a finite score at or above
`ARIA_SPK_SELF_MIN_SCORE` (default 0.75) to suppress self speech. This applies to
base user `aria` and configured self names. An uncertain self candidate proceeds
as `unknown`; nearest-human attribution is unchanged. Raw candidate, user, score,
and known flag remain logged, alongside the effective suppression decision.
Echo Guard still runs; the Mac still mutes capture during playback and its tail.
This is not an identity verification or enrollment change.

`normalize_spoken_vocative` only repairs the name in voice address position:
Ari, Arya, Ariya, and Ariah become Aria. The ambiguous English word `area` requires
an explicit greeting, and an inline request requires punctuation after that name.
Thus `Hello area.` works, but `area of the kitchen` and `Hello area code 123` do
not wake Aria. A trailing alias is repaired only after an otherwise complete
supported goodbye. The request body is unchanged. HTTP remains exact-name only.
Raw input is preserved in logs/responses; `spoken_name_normalized` records the
correction. No additional model, inference call, wait, or runtime dependency was
introduced. ASR, model, prompt, tool execution, queue, and TTS are unchanged.

This deliberately bounded name grammar does not solve arbitrary recognition
errors or expand the control grammar. Bare synthetic `Aria` still sometimes
becomes `Ohio` or `À l'eau`; those unrelated words are not activation aliases.
Use `Bonjour Aria` / `Hello Aria` for current live testing.

## Verification and provenance

- 202 targeted tests pass on Apollo, including the real WebSocket finalization
  path with controlled speaker/ASR results, confident-self and echo rejection,
  raw transcript preservation, API/voice separation, negative near-name phrases,
  human attribution, shared admission, queue, history, tools, and TTS language.
  Stage: `20261018T093550Z-wake`; four existing dependency warnings remain.
- Runtime: `20261018T093650Z-spoken-wake-6845cc`, isolated loopback namespace on
  the four-core Apollo Xeon, cached Qwen2.5 3B and Ollama 0.14.2, Parakeet, Piper,
  12 history messages, tool-capable warm-up, history prefill, actual speaker model.
  No microphone or physical playback during automated checks; all actions captured.
- Private artifacts live under `~/Library/Logs/Aria/e2e/<run>/` on the Mac and
  `/data/aria/e2e/reports/<run>/` on Apollo. They retain exact replies, inventory
  commands, speaker scores, WAV references/hashes, event timings, and source hashes.

The 32 wake text cases and both initial overlap/overload scenarios pass. A further
26-case FR/EN conversation regression passes, including eight exact log-only
commands, ambiguous/missing devices, memory, arithmetic, facts, capabilities, and
clock answers. Every non-clock reply matches the history-prefill reference
verbatim; clocks match the actual current request interval. No unintended command
or physical action occurs. These checks do not retest all scientific knowledge or
remove the [existing model-quality limitations](science-2026-10-18.md).

| Measurement | Earlier reference | Current run |
| --- | ---: | ---: |
| French follow-up after local answers, accepted to first WAV | 2.239 s | 1.678 s |
| English follow-up after local answers, accepted to first WAV | 1.979 s | 1.596 s |
| Wake-suite controls, four-case median | 0.721 s | 0.715 s |
| Four matched queued replies, median | 4.257 s | 4.003 s |
| Greeting wake, PCM end to first WAV, four voices | Previously missed/suppressed | 0.554–0.606 s |

Queue trials are FR 4.162/4.200 s and EN 3.821/3.844 s, including actual wait.
All four assert correct first/queued answers, explicit overload rejection, and
no tool calls. The reference is `20261018T083203Z-wake-voice2-658bb9`; conversation
reference is `20261018T064141Z-prefill-logged-5fc113`. Inspect `paired-queue.json`,
`regression.json`, and `regression-comparison.json` for individual results.
The broader suite reuses on/off WAVs from the wake suite; its 0.03–0.05 s on/off
figures are cache hits and are not claimed as a control-execution speedup. A few
short turns vary by tens of milliseconds (French clock +0.10 s); these small
sequential samples show no material regression, not a statistical latency SLO.
Audio readiness excludes microphone endpointing, LAN delivery, and physical
Mac playback. Failed farewell turns are not included as successful latency data.

## Acoustic results and remaining failures

Both original greeting recordings activate successfully in each of two voice sets;
follow-up arithmetic immediately after activation returns `4` in all four cases.
The previously rejected weak self candidates now reach the gate as unknown.
No confidence threshold was lowered.

The original audio suites remain **strict failures**, and their files/flags are
retained (`wake.json`, `wake-v2.json`). Name-spelling transcript mismatches remain
visible even where the canonicalized address produces correct behavior. One
French question also has a lexical ASR error (`de` instead of `deux`).

Two farewell recordings still fail: original English `Goodbye Aria` becomes
`You would buy area`; the second French voice's `Au revoir Aria` becomes
`Au revoir à rien`. These do not close the session. Subsequent input therefore
remains active, and in the French run the resulting model history also produces
irrelevant control prose for arithmetic. No device command executes, but this is
a real quality failure, not a successful conversation. The other two original
farewell recordings close correctly, including canonicalized `Goodbye area`.
Do not promote these fixtures into arbitrary sentence-specific ASR repairs.

An additional manifest reuses the second voice set and substitutes only plain
`Au revoir.` / `Goodbye.` recordings. All 12 behavioral checks pass: asleep input
is ignored, the greeting wakes, a nameless follow-up works, goodbye closes,
subsequent input is ignored, and inline reactivation answers correctly in both
languages. `wake-plain-goodbye.json` still has three literal name-spelling
transcript flags and therefore remains `passed=false` under the strict combined
oracle. The original manifests/results were not replaced or their flags erased.
This is narrower evidence for the plain-goodbye flow, not a claim all audio passes.

Final server audit: 118 accepted requests, 118 completed exactly once, six explicit
overload rejections; 40 loopback destinations and no observed external/DNS attempt.
The benchmark service was stopped before live startup. The source archive matches
the tested files. During testing, concurrent opt-in ESP32 signaling changes landed
in the current `app/main.py`; they are preserved and recorded in the private
`concurrent-source-change.diff`. Those changes are outside these measurements.
The Mac live launch uses `live-source/`, extracted from this run's exact tested app
archive, plus the existing Mac client. `live-snapshot-files.json` records its hashes.
Do not infer that this report validates the later concurrent ESP32 integration.

## Live handoff

Run `20261018T095025Z-live-449f47` is running with Apollo inference, Mac microphone
and Mac playback, speaker attribution, exact conversation logs, and log-only tools.
The readiness WAV played successfully and the client reported READY after receiving
microphone frames at the native 96 kHz capture rate. This confirms startup/capture,
not the content or acoustic success of a subsequent human conversation. The isolated
Apollo service has the launcher's three-hour lifetime limit.

This session uses the exact tested server snapshot described above. The concurrent
Mac client gained a new server playback handshake; its default handshake timed out
against the older tested snapshot on the first launch (`20261018T094907Z-live-4d5280`,
stopped). The successful launch sets `ARIA_CLIENT_PLAYBACK_GUARD=0` only for that
process to select the legacy protocol. `live_mac.py` still mutes microphone input
during actual Mac playback and its 0.3-second tail; server Echo Guard/self rejection
remain enabled. Do not copy this compatibility setting to a standalone satellite
client without its own playback suppression. No workspace client/server changes
were reverted. Live provenance is recorded in `snapshot-provenance.json` beside
`live.json`, `config.json`, conversation logs, and playback timings.

## Rejected experiments

Adding 400/800 ms of silence to short-name recordings did not reliably repair
recognition. A separately prepared Sherpa ONNX small keyword-spotting candidate
also missed the French and bare-name fixtures. It added roughly 30–160 ms in the
probe without resolving the problem, so neither experiment is in the runtime.
Candidate assets/dependencies remain isolated under the private E2E directory;
the application environment and dependency declarations are unchanged. A future
acoustic detector needs held-out human voices, false-wake measurements, and
Xeon/ESP32 resource evidence before selection.

## Scope review

Production changes in this task are limited to `app/main.py` voice finalization,
`app/speaker_recognition/recognizer.py`'s standalone decision helper, and
`app/llm/conversation.py` voice vocative normalization. New/extended tests are
`tests/test_spoken_wake.py` and `tests/test_speaker_identity.py`. The working tree
already contained substantial other changes; none were staged or committed.

Serena symbol retrieval was unavailable (no active language servers). GitNexus
and targeted source inspection supplied impact checks: voice finalization's
caller is `ws_asr`, affecting audio/status/reset/buffer flows; the conversation
decorator feeds the shared processor. The graph reports LOW for these functions,
MEDIUM for the recognizer class. Refreshed aggregate change detection reports
CRITICAL across the inherited working tree, not a three-file task diff; a task
snapshot comparison was also reviewed. The graph reports truncated process
coverage, so absence of a graph edge is not proof of no caller.
