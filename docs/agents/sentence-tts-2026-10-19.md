# Sentence-level voice routing

2026-10-19. The maintainer approved adapting French/English TTS at sentence
boundaries, with no perceptible added latency. This does not authorize the LLM to
ignore the conversation language or introduce another generation call.

## Implementation

`SpeechOutput.speak` preserves the current request hint. `speech_chunks` classifies
each sentence with the existing local lexical `speech_language` evidence and keeps
the last clear language within this reply for ambiguous text (initially the request
hint). No voice carries over from another request. Adjacent sentences with the same
language remain grouped within the existing size chunks. Monolingual inputs retain
the same text, voices and cache keys. The exact model reply and history are unchanged.
No Lingua/CLD2 dependency, model, generation retry or response replacement is added.

The sentence splitter preserves closing quotes, decimal points and common titles/
initials. It is a punctuation heuristic, not a general multilingual sentence parser.
Language boundaries still apply with `ARIA_TTS_CHUNKING=0`; that option disables
size-based grouping limits, not voice routing. Configure both local Piper voices.

For replies needing different voices, one next WAV is prepared while the current
clip plays. Only one lookahead task exists; the sink still receives clips sequentially.
Echo records are created at delivery, not speculative synthesis. Input remains
blocked across clips; the 300 ms tail starts after the final clip. `audio_ready`
resolves only after delivery of the last clip; failures/cancellation resolve false.
Mute revisions discard a prepared clip even if mute was subsequently lifted.
Cancellation cannot stop an already-running Piper subprocess inside a thread, but
its result is never delivered and existing Piper timeouts still bound that process.

The development Mac player polls every 10 ms for at most 250 ms after a clip finishes,
then returns to its 100 ms idle interval. This removes its fixed 100 ms transition
sleep without increasing idle polling permanently. Real playback was never started
by these tests.

## Validation and limits

Evidence is retained under private run `20261019T103333Z-sentence-tts`.
The first evaluation staged/tested the implementation without changing live Apollo.
All model/TTS processing is on Apollo; synthetic text and captured WAVs only.
Tests use mocked or timer-acknowledged players, never real microphones/speakers.
Tools remained log-only; that evaluation did not replace the monitor pointer.

The existing detector is deliberately unchanged: ambiguous short text and language
switches inside one sentence remain limitations. This implements per-sentence voice
routing, not a reliable mixed-language rejection guard. Neither fast timing nor a
matching voice establishes factual answer quality or subjective audibility.


## Results on Apollo

Xeon E3-1225 v5, existing Python 3.11/Piper/voices. No dependency upgrades.

- **211 distinct regression tests pass** on Apollo: sentence routes, cached voices,
  one-chunk lookahead, player acknowledgment, mute revisions, cancellation during
  preparation, preparation failure, shared requests, bounded queue, grounded commands,
  echo/speaker/wake and history. Final output: `final-unit-tests.log`.
- Existing corpus: **286/344** monolingual/neutral cases correct before and after,
  **zero newly failing cases**. The other 58 remain known detector errors; unsupported
  and mixed labels are excluded from this voice-consistency comparison. Twelve new
  routing examples cover both directions, short greetings, borrowed terms, quotes,
  abbreviations and numeric replies. This is not household accuracy.
- **160 captured speech runs**: baseline and candidate, six scenarios, six repetitions
  per cache state, plus eight real-duration timer deliveries per profile. Real Piper
  and ffmpeg generate valid stereo 44.1 kHz WAVs; no audio device is opened. Mixed
  replies use the expected French/English models at the correct text boundaries.
  Baseline uses one voice and therefore cannot match both languages in those replies.
- Full pipeline: **12/12 scenarios pass in each profile**, with exactly identical
  replies and identical per-case voice/text lengths. Four exact on/off commands per
  profile are log-only; recall and arithmetic pass. Both FR/EN queue/overload probes
  pass in each profile; no new model calls or tool actions.

| Measurement | Baseline | Sentence routing |
| --- | ---: | ---: |
| Short-text planning p95 | 0.0093 ms | 0.0124 ms |
| 40-sentence stress planning p95 | 0.113 ms | 1.533 ms |
| Cached French first WAV, median | 1.07 ms | 1.08 ms |
| Cached English first WAV, median | 0.97 ms | 1.05 ms |
| Uncached French first WAV, median | 563 ms | 523 ms |
| Uncached English first WAV, median | 720 ms | 536 ms |
| FR queued admission → first WAV | 4.853 s | 4.822 s |
| EN queued admission → first WAV | 4.427 s | 4.354 s |

Short planning has 27,520 timings/profile; the long stress check has 160/profile.
First-WAV rows have only six timings/cache state/scenario: they are observed values,
not stable population tails. Processes run baseline/candidate/candidate/baseline;
CPU load of other host services is uncontrolled, and a short unit suite overlapped
the start of the first baseline batch. Do not attribute the lower uncached medians
to routing: monolingual synthesis receives the same text/model and does the same work.
Piper's uncached WAV bytes varied even within a profile; byte-identical generated
speech is not claimed. Cache keys/input boundaries are preserved.

In the eight candidate mixed-voice timer deliveries, the next clip became available
**0.65–3.58 ms** after prior completion. This measures server preparation/delivery,
not Mac audio-device startup or LAN delay. The output adapter's bounded 10 ms poll
window is tested with fake HTTP/player/clock; no real speaker was used.

Full conversation admission-to-first-WAV medians were **FR 2.491→2.667 s** and
**EN 1.711→1.815 s**. Individual differences range -831 to +313 ms; model timing
accounts for much of the variation in generated replies. Unchanged deterministic
control replies also show Piper/host variation. Thus the new routing work is tiny,
and no queue slowdown was observed, but this small sequential run does **not** prove
an imperceptible total latency difference or acoustic continuity on the Mac/ESP32.
A listening test remains necessary for that subjective claim. The feature adds no
LLM pass, preserves monolingual grouping and hides the measured next-voice synthesis
inside playback; those are the mechanisms limiting added latency.

## Reproduction, integrity and operational scope

Run `scripts/e2e/sentence_tts_bench.py --output <private-json> --rounds 3 --playback`
inside an empty Apollo network namespace, with prepared local dependencies. The
`--playback` flag uses timers only; it never invokes a sound player. Stage separate
baseline/candidate sources and alternate process order. Keep pipeline checks separate from detector-only timing and retain the frozen baseline.

The private directory contains baseline archive/hash, task-only source diffs,
per-profile JSON/traces/logs, full-pipeline cases/events, planning/corpus results and
source manifests. Seven collected network traces (four TTS, two accepted pipeline
runs and the excluded first baseline harness attempt) contain no non-loopback
connection/send destination or DNS attempt. Each server runs in its own network
namespace; local Unix sockets/loopback are allowed. No live read bridge is configured.

An initial pipeline attempt omitted the wake name, so its 12 ignored turns were
invalid evidence. That attempt is retained, excluded, and followed by fresh paired
runs with explicit activation. Its queue probes passed separately. No application
change was made to bypass the wake requirement. All owned test services are stopped;
Apollo's active log-only monitor pointer was unchanged throughout this task.

GitNexus: `_run_speak` and sentence-splitter upstream impact LOW; `speak` has unresolved
dynamic callers, verified manually at `app.main._speak_reply`. Mac `listen` impact LOW,
called by its CLI main. No inference/tool path was edited. Context checks pass.

Final GitNexus all-working-tree detection reports CRITICAL across 36 already-modified
files, 122 symbols and 44 flows, including inherited client/UI/LLM work. This is not
an isolated change-scope result. Task-only baseline diffs separately identify the
three edited runtime/adapter files; new runner/tests and context are listed in the
manifest. No bulk staging, commit or deployment was performed during that evaluation.


## Deployment after maintainer approval

At 12:12 UTC on October 19, staged `20261019T121200Z-sentence-live` from the latest
validated lexical/Jeedom source `20261019T102932Z-lexical-implementation-candidate-final`.
The prior monitor target had expired; only the monitor was running. Applied only
`speech_output.py`, `text_chunker.py` and `mac_playback.py` from the tested candidate;
all three SHA-256 hashes match. Other runtime app files are identical to that later
base, preserving its read-intent/context and lexical-selection fixes, which are not
all present in this worktree. Do not redeploy the entire stale checkout.

The assembled source passes **169 compatibility tests**, then **12/12 FR/EN full
conversation/control scenarios** and both queue/overload checks. Four commands are
recorded in log-only mode. Admission-to-first-WAV medians across the six scenarios
per language are FR **1.955 s**, EN **1.746 s**; these are deployment smoke measurements,
not a new paired speedup claim. The detailed latency comparison remains above.

The backend and read-only Jeedom bridge run with the existing three-hour development
bound, ending **15:12 UTC / 17:12 Zurich**. No autostart or automation was created.
The inference namespace has loopback networking only; its observed IPv4 destination
is 127.0.0.1. The bridge is separately connected to the existing owned-LAN broker.
Warm-up, six speaker profiles, six history messages and conversation logs are ready.
The monitor pointer was changed only after the checks; its LAN WebSocket confirms
this run and `log_only`. Previous pointer is saved in the new run for rollback.

ESP32 ingress and all Mac capture/playback remain stopped. No player or microphone
client is connected; there is no pending playback or input block. The updated Mac
adapter is staged but not started. Deployment does not establish acoustic continuity
or perceptual latency. Private evidence: `deployment-tests.log`, `deployment-smoke.json`,
`deployment-ready.json`, `monitor-check.json`, source overlay hashes and network trace
under the deployment run; a copy is kept in the Mac's dated Aria log directory.


## Scope of the main-branch integration

The scoped merge includes sentence routing, speech scheduling and mute-revision
invalidation, the optional playback broker, live adapter endpoints and Mac player.
These delivery prerequisites were previously only in the development deployment.
It does not merge unrelated inventory/Jeedom, history, monitor UI, or microphone
admission changes. SpeechOutput exposes its input-block state; main-branch ASR
admission does not yet consume it. The deployed instance above includes those
independent changes, so its full-pipeline evidence is not proof of their presence
in this merge. Exact branch tests are recorded in the pull request.

Exact merge-branch validation on Apollo: **167 distinct tests pass** (165 in the
speech/shared-request/queue/control/live/history/echo suite, then two adapter cases
with six logging cases repeated). Seven context-check unit tests and source-review
coverage pass. The adapter test exercises registration, conflicting registration,
start/finish ordering, completion, unregister, no-player fallback and input-state
reporting without requiring the unmerged main-module helper. No real audio was used.


PR review found three lifecycle gaps in the merge candidate: cancellation before
broker-task startup stranded its slot; muted scheduling was ignored by two callers;
and explicit interruption could restart already-waiting replies. Four reproduction
cases failed before fixes and passed after. Broker completion now settles/releases
its underlying future even before startup. Both shared/direct conversation callers
propagate rejected scheduling and do not attach the previous playback to that turn.
An interruption generation invalidates waiters, including requests arriving while
sink stop is pending; new requests after interruption remain usable. Automatic
voice interruption remains deferred. The expanded Apollo suite passes **235 tests**,
including a fifth stop-in-progress race case and wake/conversation regressions.
These review fixes are validated in the merge candidate; the earlier deployed run
has not been replaced by this PR task.


The second review confirmed those fixes and exposed two additional delivery cases.
The Mac now validates the existing `started` acknowledgment immediately before
launching the audio process, and ignores canceled-ticket conflicts at start/finish
while continuing to later clips. This moves the existing HTTP acknowledgment ahead
of process launch; it adds no HTTP request, but the pre-start LAN round trip remains
part of unmeasured physical onset latency. `started` is permission to begin playback;
actual process-launch/finish timestamps remain in the player log. This is not remote
mid-clip interruption: voice barge-in remains deferred.
The broker waits against both clip completion and the renewable player lease; a
silent reader loss no longer blocks for a long clip's full duration. Four new tests
cover expired/renewed leases and canceled start/finish events; three failed before
these fixes. **239 tests pass** on Apollo afterward, without real audio or actions.


The third review identified a body-only lease-renewal gap during slow network reads.
The Mac player now owns one five-second heartbeat task for its registered session,
including state polling and WAV transfers. Renewal errors are surfaced before
starting audio or while awaiting process completion; exit cancels and awaits the
task before unregistering. Waiting for playback uses FIRST_COMPLETED so the heartbeat
cannot add a five-second transition delay. A delayed-transfer reproduction failed
before the fix. The expanded suite passes 240 tests; the parameterized renewal-failure
follow-up adds one distinct passing case (**241 total**), with no physical audio.
A fresh external review of this last fix is pending: the workflow's three automated
review cycles have been used. The PR is not yet merged at this checkpoint.
