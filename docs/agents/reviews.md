# Context review log

[Language evaluation](language-context-evaluation-2026-10-19.md): six tests pass; no runtime change.

## 2026-10-18 — Local answers and history-prefill regression

Clock/capability answers use runtime facts with shared history and speech. Bounded
prefill after audio preparation restores following FR/EN turns to 2.24/1.98 s on
Apollo, with identical replies and steady queue latency. Quality stays 40/42;
140 targeted tests and seven context tests pass. Tools remain log-only, Mac audio
off. Reviewed [local-answer evidence](local-answers-2026-10-18.md) and
[final measurements, impact and limits](history-prefill-2026-10-18.md).

## 2026-10-18 — Completed warm-up comparison without Mac audio

Maintainer resumed implementation, then required no Mac audio/client activity.
Changed comparison playback to opt-in (LOW impact: case → main); Mac microphone
never opens in this runner. Eight fresh isolated Apollo runs completed in alternating
condition order. Mean request-to-audio-ready improved 17.13→6.39 s FR and 16.55→5.74 s
EN; reply text/token counts matched. Warm-up did not dispatch tools or add history.
All eight network audits passed. Twenty-four targeted Apollo tests had passed for
the implementation; final runner syntax/context checks passed. No new runtime/model
behavior changed during the comparison. See [evidence](warmup-2026-10-18.md).
Headless Apollo run `20261017T223829Z-headless-live-788648` verified warm-up readiness,
six speaker profiles, bounded history, and log-only tools; startup network audit
passed. Comparison services cleaned up; Mac client/audio remains off.

## 2026-10-18 — Warm-up implemented; comparison stopped for the night

Added isolated tool-capable startup warm-up and direct Piper readiness, with 24
targeted Apollo tests passing. Began matched first-response/Mac-playback comparisons;
only one control completed before the maintainer requested stopping for the night.
Stopped comparison/live processes, playback/tunnels, and owned Apollo services;
verified none active. Saved [implementation and partial evidence](warmup-2026-10-18.md).
No completed comparison or latency benefit claimed; no automatic restart authorized.

## 2026-10-18 — Speech-only interface; interruption parked

Recorded the maintainer's clarification as D-008: household microphone/speaker devices,
Mac for development only, no keyboard/button interaction dependency, and voice
interruption deferred. Updated objective/status and qualified the latency review's
future barge-in discussion. Earlier sentence playback remains a proposal with
whole-reply microphone suppression. No runtime or live-service changes.

## 2026-10-18 — Reviewed live recognition, tool choices, and latency

Copied a private snapshot of live Apollo logs and correlated three transcribed voice
turns with Mac playback timing. Separated the readiness/API action check from user
conversation. Attribution worked as requested; two irrelevant searches ran, no voice
action call. Prompt/generation cost dominates; existing clock shortcut is disabled
through router gating. Documented measurement gaps and prioritized repairs in the
[conversation review](conversation-review-2026-10-18.md). No code, configuration,
threshold, or live service changes; no new capture or model benchmark. Network audit
passed for the snapshot. Source inspection and log arithmetic support the findings;
no tests needed for this documentation-only review.

## 2026-10-18 — Restored attribution; live tools capture commands only

Restored deployed nearest-person attribution by explicit maintainer instruction;
threshold/score/runner-up diagnostics are retained. Added development-only cached
inventory and an action provider that validates and logs proposed payloads without
constructing an MQTT client. Unknown IDs are rejected at this adapter boundary.
The existing model loop selects tools; model decisions, calls, and proposed payloads
are private request-correlated events. No production tool-selection fixes claimed.
Live startup requires the isolated network namespace and a nonempty inventory.
GitNexus: LOW Mac-adapter impact (main caller); speaker/live wrapper dynamic callers
unresolved, reviewed directly. Serena has no active language server; used fallback.
Twenty targeted Apollo tests passed (speaker attribution/confidence, shared request
path, log-only validation/no real execution, exact transcript logging). Python/bash
syntax passed. Updated live guide, architecture, and status; restart authorized.
Run `20261017T220249Z-live-b90112` is active on Apollo with Mac playback. Verified
ASR/LLM/TTS, six speaker profiles, history=12, two installed providers, and
`tools_mode=log_only`. A live voice turn retained the nearest user below threshold
with confidence logged. A direct installed-provider check returned success=true,
dry_run=true, executed=false and logged the proposed payload. This checks the action
boundary, not LLM command-selection accuracy. The first voice model turn offered
tools and chose none; processing took about 15.6 s, so latency remains unresolved.

## 2026-10-17 — Stopped live test; reviewed person recognition regression

Stopped the Mac live process/tunnel and owned Apollo service at the maintainer's
request; verified no live units remain. Compared deployed/current recognizers and
enrollment source, inspected readiness/profile counts and live identity events,
and replayed saved EN/FR audio through both recognizers on Apollo with networking
disabled. Scores agree; the task-introduced below-threshold `unknown` policy changes
the identity sent to the LLM. No new runtime changes or threshold/profile edits in
this review. [Evidence, limitations, and next steps](speaker-review-2026-10-17.md).
Tool detection/calls remain pending and authorized only for logging, not actuation.

## 2026-10-17 — Live history, complete transcript logging, and speaker recognition

Enabled six recent exchanges per live session and added private exact request/reply
JSONL logs on Apollo, mirrored on the Mac. Added model timings/history count/stop
reason, speaker status, and request IDs for playback correlation; ambiguous client
turn timing associations are left unavailable. Smoke/baseline settings are unchanged.
Restored speaker recognition after the maintainer questioned its omission. Prepared
the cached model before operational startup and verified it loaded with networking
blocked; existing private enrollment profiles were retained.

Fixed `SpeakerRecognizer.identify` returning the best candidate's user below the
similarity threshold. Such results now stay unknown while preserving diagnostic
name/score. GitNexus reports LOW impact for live adapter edits and UNKNOWN for the
dynamically called recognizer; inspected the voice processing caller and self-voice
filter directly. Three exact-log/context tests, 12 shared-handler/history tests,
and two speaker-threshold tests passed on Apollo. No raw microphone recording added.
Restarted live run `20261017T214203Z-live-f569ac`; subsequently stopped at the
maintainer's request (see review above).
Confirmed speaker model readiness, six loaded profiles, exact local transcript
mirroring, correlated Mac playback, and actual history growth across voice turns.
A separate two-turn French-to-English real-model recall check passed. Live identity
scores remained below threshold at inspection; enabled does not mean recognized.

## 2026-10-17 — Live Apollo inference with Mac playback

Added `scripts/e2e/live_mac.py`, a foreground development adapter using the existing
isolated Apollo ASR/Qwen/Piper profile. Mac captures at native rate with the existing
client, fetches generated WAVs through SSH, and plays them with afplay. Playback
temporarily mutes transmitted microphone frames, with a 300 ms tail; voice barge-in
is not supported by this adapter. No existing runtime/client symbols were edited.
The original Sonos path is retained and not invoked in this mode.

Live run `20261017T213110Z-live-33a0e1` started in a visible Mac Terminal, verified
Apollo readiness, successful Mac playback, real microphone capture, and subsequent
voice replies. This run was subsequently stopped before the history-enabled run.
Ctrl+C cleans up microphone, tunnel, and owned unit (three-hour upper bound). Latency
logs distinguish last client VAD speech from turn end and Mac player launch; actual
hardware audible onset is not instrumented. Conversation mode excludes home actions,
router, speaker identity, and history; pending tool fixes remain pending. Updated
the client guide, E2E workflow, and D-005. Python syntax and context checks passed.

## 2026-10-17 — Priority corrected; current tool loop diagnosed

Recorded the maintainer's clarification: normal Mac-client/Apollo operation worked;
tool calling is the current issue. Added D-006 without settling D-001. Corrected
status/project priority and separated ordinary-use clarification from the unresolved
earlier unsaved acoustic content. Condensed older infrastructure smoke history to
keep the current status within its context budget.

Added a test-only real-model diagnostic with synthetic inventory and captured MQTT
publication; no production runtime or deployed configuration changes. Findings and
reproduction limits are in the [tool report](tool-calling-2026-10-17.md). Serena's
symbol/content tools still fail with no language servers, so GitNexus plus targeted
source inspection and fallback edits were used. No existing code symbols were
edited in this task. Aggregate GitNexus change detection reports CRITICAL risk across
20 flows from inherited runtime/client changes; new diagnostic files were reviewed
separately because untracked files are not included by that change detector.
Completed 14 real-model cases plus five deterministic tool probes on Apollo;
captured execution exposed wrong device/action, skipped execution, and unknown-ID
fallback. All 14 request lifecycles completed once, which did not establish correct
control. Verified isolated-service cleanup/network audit, seven context tests,
Python/shell syntax, whitespace, and context source-review coverage. No commits.

## 2026-10-17 — Controlled capture traced and reference contract corrected

Verified both submitted recording hashes and listener notes; user confirmed saying
the answers instead of the displayed questions. Direct isolated Apollo ASR repeated
the confirmed statements, and full native-client replay matched PCM hashes, raw ASR,
and model input. Corrected the test harness's unconfirmed displayed-prompt assumption:
manual references now require explicit user confirmation tied to the audio hash.
Added saved-capture replay and stage checks that flag extra words in merged input.
Updated testing/workflow guidance, status, historical follow-up links, and the
[controlled-capture report](controlled-capture.md). Older unsaved microphone results
remain unattributed; no ASR/echo/queue implementation change was justified by these clips.

Run `20261017T210457Z-c14268`: 50 Apollo tests, 18/18 pipelines, 18 accepted/completed,
zero wrong voices or capture trace mismatches. Four unchanged story token cutoffs.
Network audit: 26 loopback destinations, no external/DNS attempts. Verified cleanup
and unchanged application/client fingerprints. Thirty-five local tests, source/context
checks, and whitespace checks passed. GitNexus impact checks preceded harness edits;
aggregate dirty-checkout risk remains critical across expected earlier refactor flows.
Serena semantic retrieval was unavailable. No persistent deployment change or commit.

## 2026-10-17 — Manual capture prepared at maintainer request

Added a standalone user-controlled capture/replay utility and two mocked tests.
It preserves native audio privately, signals readiness from actual input frames,
closes the microphone before replay, and collects listener notes. No existing
runtime symbols changed and no microphone capture was initiated. Documented the
new optional recording retention separately from the automatic baseline. The
extra-content diagnosis and Apollo rerun await the user's controlled samples;
do not infer a root cause or claim a fix from tool preparation.

Use only for meaningful context reviews or source changes needing no prose update.
Each entry needs a date, affected area, rationale, and evidence. Do not append routine
session chatter or declare an architectural decision here.

## 2026-10-17 — Turn boundaries and request-specific TTS language

Raised Mac client silence endpointing to 800 ms and carried current-request FR/EN
through shared processing and each TTS chunk. Removed global short-answer language
reuse. Preserved model, prompt, 80-token budget, queue policy, and open D-001.
Updated architecture, environment/client guidance, test workflow, current status,
and [the measured report](turns-and-language.md). The local wrapper now defaults to
800 ms; no persistent Apollo launch/deployment changes.

Run `20261017T110759Z-d028e0`: 50 Apollo tests passed; 18/18 pipelines completed,
19 accepted/19 completed requests, no wrong voices, all eight WAV cases one request.
Five automatic quality flags remain; manual review additionally rejects the French
microphone's appended answer-like content (six distinct cases needing attention).
Both microphone inputs retain the question, but unexpected answer-like content
remains unexplained. Four narratives still truncate. Runtime network trace found
25 loopback destinations and no external/DNS attempts. Verified cleanup and exact
server/Mac sender source hashes. No physical Sonos or live device actions.

Five local endpoint tests, 15 harness tests, six queue tests, seven context tests,
and current fake-handler diagnostics passed. GitNexus impacts preceded symbol
edits; aggregate detection reports critical risk across accumulated expected flows.
Serena semantic retrieval remained unavailable (no active language server). Preserved
inherited files and historical results; no commit/bulk staging. The report explicitly
qualifies timing contention, acoustic uncertainty, and automatic-oracle limitations.


Earlier setup, hardware, baseline, and shared-path reviews are preserved in
[review history](reviews-history.md).

## 2026-10-18 — autonomous Apollo conversation benchmark

Added `scripts/e2e/conversation_bench.py` and three oracle regression tests. Completed
28 FR/EN baseline requests plus six supplementary natural/voice turns on the isolated
Apollo runtime, with no Mac audio or physical actions. Updated status and
[benchmark report](conversation-bench-2026-10-18.md); private artifacts retain exact
transcripts, inventory, command traces, manual review, timings, and source hashes.
Serena had no language server; targeted AST inspection was used. GitNexus upstream
impact for first-audio instrumentation was LOW, limited to the benchmark main flow.
No production control fix is claimed; inherited working-tree changes were preserved.

## 2026-10-18 — grounded selection and truthful control replies

Implemented the maintainer-authorized explicit FR/EN light/shutter path with strict
inventory selection, single-use execution authority, and outcome-rendered replies.
Updated architecture, status, and [implementation/evidence](grounded-control-2026-10-18.md).
105 targeted Apollo tests passed after the negation follow-up; the baseline improved exact controls
from 1/8 to 8/8 and truthful complete controls from 0/8 to 8/8. Median control audio-ready
latency fell from 13.21 s to 0.71 s. Utility/model-quality gaps remain explicit.
No Mac audio or physical actions; inherited changes preserved. Future architecture
remains open and the low-level administrative action API is outside this request guard.

Final source run `20261017T231411Z-control-final-8894d5`: 42 turns, 8/8 exact and truthful
controls, eight challenge cases passed, median control audio-ready 0.70 s. Six utility/
factual failures remain; 22 loopback destinations and no external/DNS attempts.
