# Shared voice/text processing — 2026-10-17

Subsequent [turn-boundary and request-language fixes](turns-and-language.md) improve
the WAV baseline; this report preserves the earlier implementation/run evidence.

The maintainer authorized fixing shared request handling and the queue, then rerunning
the French/English baseline on Apollo. This implements that scope using the existing
tool loop. It does not choose the future control architecture (D-001 remains open).

## Implemented behavior

- Both finalized voice requests and `POST /api/llm/process` enter one bounded FIFO
  and one worker, then the same `_process_request` function. Routing, tool execution,
  history, and requested speech are shared. The existing five-iteration tool loop
  remains; tools are advertised only when requested and inventory is available.
- Default capacity is one running request plus one pending request. Full queues
  reject new work instead of displacing accepted work. HTTP returns 429 with a
  request ID; voice receives a rejection status frame. Accepted pending work older
  than 30 seconds expires before execution; HTTP returns 504. Limits are bounded
  configuration values documented in [ENVIRONMENT](../ENVIRONMENT.md).
- Every accepted request has request/session IDs and lifecycle events. WebSocket
  admission/terminal frames contain status only; the current Mac client drains them
  without displaying busy feedback. Notification delivery is bounded and best-effort
  after disconnect; structured server logs retain the outcome.
- Cancellation of an HTTP handler skips pending work. Running work is allowed to
  finish; no automatic action retry is introduced. Shutdown resolves interrupted
  and pending requests before stopping providers. An interrupted action may already
  have had side effects. Accepted voice requests survive socket disconnect.
- Router CHAT results use canned responses only for explicit greetings/time queries.
  Other CHAT inputs reach the model. Shortcuts honor requested speech.
- Worker/model failures remain visible as failed outcomes and HTTP 500 responses;
  the worker stays available for the next request.

This is one inference slot per server process, not a global cluster scheduler or
per-session fair queue. Keep one server process on Apollo. `completed` means request
processing finished and optional speech was scheduled; it does not certify audible
playback or observed device completion. Direct `/api/actions/execute` remains a
separate provider endpoint; this change unifies the conversational text/voice paths.

## Validation and provenance

Final source is staged in isolated run `20261017T102130Z-22e4e8`; raw artifacts are
under `~/Library/Logs/Aria/e2e/20261017T102130Z-22e4e8/`. The runner records the exact
source archive hash and unchanged Git HEAD. No deployment script or shared Ollama
service is replaced. No physical Sonos or home-device action is executed.

The 39 Apollo tests comprise 23 existing plugin/Sonos tests, six queue lifecycle
tests, and ten shared-handler cases. They cover EN/FR fake-tool parity, exact fake
execution sequence, mixed-transport overload, FIFO preservation, expiry without
execution, cancelled HTTP waiters, shutdown, failure recovery, session history,
optional speech/providers, generic CHAT handling, and actual WebSocket handler
status messages with fake ASR. These do not measure real model tool accuracy.

Local tests cover the E2E quality oracle, network audit, target-output correlation,
and accepted-request accounting. The latter is reproducible with:

```sh
.venv/bin/python scripts/e2e/audit_requests.py /absolute/path/to/report.json
```

## Final Apollo bilingual rerun

Run `20261017T102130Z-22e4e8` completed **18/18 pipeline cases**: eight text, eight
synthetic WAV through the real Mac sender, and two physical Mac microphone cases.
The lifecycle audit found **24 accepted, 24 running, 24 completed**, with no missing
or duplicate terminal outcomes. No rejection/expiry occurred in this sequential
corpus; overload and expiry are covered by the deterministic transport tests above.

The runtime network trace recorded **26 loopback destinations and zero external/DNS
attempts**. The namespace had no external route. Cleanup left no test units or test
listener; the existing shared Ollama service remained. Apollo's tested `app/main.py`
and `request_queue.py` SHA-256 values match the final local runtime files. Archive:
`7fe672761599779a9667be9f9ed1e3dbdfc93e4c083dc29bbb5b84274e732479`.

**Quality remains imperfect: 10/18 cases carry quality flags**, versus nine in the
original reviewed run. This small repeated corpus is not evidence of an acoustic
error-rate change or model improvement. Four story outputs still hit the unchanged
80-token cap. Three cases choose the wrong TTS language for short output. Four audio
cases generate multiple replies (overlapping one of the voice-language cases):
French arithmetic, French capital, and both microphone cases.

French arithmetic is mathematically correct but fails the requested number-only
format: the question and format instruction became separate requests. Both physical
microphone questions reach the LLM in this run; English also produces two extra
`Yeah.` transcripts. Their acoustic source remains unresolved, as in the original
baseline. Do not promote these microphone checks to a clean quality pass.

| Synthetic WAV metric | Original EN / FR | Final EN / FR |
| --- | --- | --- |
| First processing/LLM start to first fetched audio, p50 | 2.35 / 2.28 s | 2.32 / 2.39 s |
| Same metric, nearest-rank p95 (maximum of four per language) | 7.64 / 7.82 s | 7.58 / 7.71 s |

These timings exclude capture/ASR and physical playback, include cached TTS where
available, and measure the first reply in split cases. They do not establish a
performance improvement. The refactor's verified gain is shared behavior and explicit
request accounting. Next address turn boundaries and request-specific TTS language;
then completion/token budgets and truthful device outcomes.

Final verification: **39 Apollo tests**, updated standalone diagnostics, **15 local
E2E harness tests**, six queue tests locally, seven context tests, structural/source
context checks, and diff whitespace checks passed. Queue tests are also included in
the 39 Apollo tests, not additional independent evidence. No application code changed
after the final run's source snapshot; later additions were the offline report auditor
and documentation.

An initial test-only run (`20261017T101331Z-7f1958`) exposed a notification/cancellation
race before services started. It was stopped and the race repaired with bounded
notification timeouts that preserve cancellation. A 120-second remote test limit
now prevents indefinite setup waits. An intermediate full run
(`20261017T101545Z-73c05b`) completed 18 cases with 23 accepted/23 completed requests,
but predates the final HTTP failure-status correction; it is not the final evidence.

GitNexus reports high aggregate integration risk across the expected startup,
WebSocket, and LLM flows. Individual impact queries were run before edits; dynamic
FastAPI wiring required direct inspection and transport tests. The change-detection
view includes inherited tracked documentation edits and omits untracked files;
new queue/test/harness files were reviewed separately. No commit was created.

## Deliberately remaining work

Turn segmentation, short-answer voice selection, output token budgeting, physical
device outcome tracking, per-session overload fairness, and Mac busy-status UX are
not fixed by this refactor. Preserving pending requests can expose multiple replies
when client VAD splits one intended turn; dropping a fragment is not a valid remedy.
Compare the raw transcript/response events, not only a plausible final answer.

The deployed profile still differs from the isolated baseline (48-token cap,
plugins/speaker recognition, prompt). Actual control accuracy, full-feature offline
operation, long-run load, and power remain separate acceptance work.
