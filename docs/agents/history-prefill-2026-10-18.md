# History prefill latency — Apollo, 2026-10-18

## Cause and implementation

The local clock/capability fixes preserved exact history but stopped running the
model on those turns. The following generated reply therefore had to prefill both
new exchanges. Prompt token counts alone hid the difference: the uncached portion
increased, while generation and speech preparation stayed nearly unchanged.

`app/llm/history_prefill.py` prepares the exact bounded session history, original
system prompt and unchanged tool definitions after the last TTS chunk is ready.
A temporary empty user message and one discarded token prime the prefix for the
next real user message. Neither enters stored history, speech or tool dispatch.
There is one task, a four-second inference deadline, a 60-second recent-model
eligibility window and no periodic idle activity. Only local clock/capability
replies with scheduled speech and same-session recent Ollama use qualify.
Pending work skips prefill; any new processing and worker shutdown cancel it.
The audio-ready future resolves false on failed/cancelled speech. Other providers
and text-only requests retain their existing behavior. `ARIA_HISTORY_PREFILL=0`
is the comparison/rollback switch.

Prefill shifts compute into playback; it does not remove the work. A new request
can arrive while a backend batch is running, so cancellation must be measured on
the actual Xeon, including any residual backend work. No history pruning, model
change, removed tool schema, or synthetic conversation reply is used to improve
these measurements.

## Preliminary direct cache probe

With the same saved exact history/tools and two repetitions per condition/language,
French prompt evaluation fell from 4.00–4.01 s to 1.196–1.199 s and English from
3.456–3.488 s to 0.955–0.962 s. All eight real follow-up replies were exactly `5`.
The explicit combined-history prefill itself cost 3.38–3.41 s FR / 3.07–3.08 s EN;
these isolated calls did not test playback overlap or queueing. Private probe
artifacts are under Apollo run `20261018T062743Z-prefill-probe-2d99ff`.

## Full comparison

The paired Apollo runs used the Xeon E3-1225 v5 (4 cores/threads), CPU-only
Qwen2.5:3b-instruct Q4_K_M, context 2048, temperature 0, prediction limit 80,
12 retained history messages, unchanged tools/system prompt, Parakeet and Piper.
Tools were log-only; all audio went to the private WAV capture sink. No Mac client,
microphone or playback process ran. The isolated server had loopback-only networking.

- Disabled baseline: `20261018T063330Z-prefill-off-90db89`.
- Enabled final run: `20261018T064141Z-prefill-logged-5fc113`.
- Private artifacts: `~/Library/Logs/Aria/e2e/<run>/` on the Mac and
  `/data/aria/e2e/reports/<run>/` on Apollo. `prefill-comparison.json` in the final
  local directory contains per-case metrics, quality comparisons and audits.
- All runtime files match except the prefill logger namespace: the enabled final
  run routes cancellation events through Aria's collector. Disabled mode never
  starts/logs prefill. The valid queue probes use the same corrected runner;
  its saved baseline copy is `queue_bench.py` beside the baseline report.
- An initial queue oracle expected 503; the existing busy API correctly returns
  429. Corrected the oracle and reran all four baseline queue cases. No queue
  behavior was changed. An intermediate enabled run was restarted to correct
  missing cancellation-event logging; it is corroborating, not the final run.

All times below are seconds from admission acceptance to first captured WAV,
including queue wait. They do not claim physical speaker audibility.

| Measurement | Disabled | Enabled |
| --- | ---: | ---: |
| French follow-up after local answers | 4.962 | 2.239 |
| English follow-up after local answers | 4.506 | 1.979 |
| Follow-up prompt evaluation, FR / EN | 3.977 / 3.497 | 1.197 / 1.009 |
| Eight exact controls, median | 0.732 | 0.714 |
| Local clocks, median | 0.549 | 0.539 |
| Capability names, median | 0.634 | 0.624 |
| Four natural-language answers, median | 5.829 | 5.836 |
| All 42 cases, median / sample p95 | 0.752 / 5.626 | 0.724 / 5.615 |
| Four deliberately queued replies, median | 4.048 | 3.990 |

The normal suite's queue waits were tiny; the separate overlap probe creates a
running request, one pending request and a rejected third request. It sends the
first overlapping request without speech, the pending request with captured speech,
and checks exact `84` / `5` replies, one queued audio output, no tools, and HTTP
429/busy. Both languages run twice. Queued waits were 2.495–2.754 s disabled and
2.460–2.664 s enabled. Total queued audio readiness was 3.821–4.572 s disabled and
3.779–4.172 s enabled. Every paired queued case was slightly faster in this sample;
this is evidence of no observed cancellation penalty, not a throughput guarantee.

Eight prefills were observed in the enabled measurement windows: two completed
and six were cancelled by new processing. All began after the associated audio
was ready; client cancellation completed 1.21–1.87 ms after the next
`Live.Request`. Actual next-response and queue timings include any residual
backend batch work. Four cancellations were forced about 65 ms after prefill
started in the overlap probe. There were no timeout/failure outcomes, tool
dispatches, spoken prefill outputs or synthetic history exchanges. `Live.Prefill`
is separate from the 18 real `Live.Model` calls in the 42-case suite.

`latency.prefill_s` in a case report sums completion events observed in that
collection window; a cancellation can belong to the preceding request. Join its
request ID for attribution and deduplicate event indices for aggregate costs.
It is not additional blocking time for the current request.

## Quality, voice and limits

All 42 reply texts matched across conditions except the actual clock minutes.
All command IDs, payloads, counts, dry-run outcomes and request-local TTS languages
passed. Eight primary controls plus two polite challenge commands were recorded;
none was physically executed. Manual quality remains **40/42**: the same French
sky/sunset explanations are wrong. This change does not fix factual reasoning.

Saved-PCM-end to first audio includes ASR and admission: fresh EN 3.088→3.112 s,
fresh FR 3.390→3.339 s, repeated EN 2.546→2.546 s and repeated FR 2.755→2.821 s.
These are the original user-confirmed recordings of answers, not new live capture.

One final-run English fact turn rose 1.508→1.718 s, mainly prompt evaluation;
the intermediate enabled run measured 1.519 s on that same case. No prefill was
active during that turn. Other increases were below 0.15 s. This limited sample
supports removal of the specific history regression and no measured queue/quality
regression; it does not prove every future turn is faster or establish statistical
latency bounds. Long rolling histories, cold model residency, simultaneous voice
ASR during speculative work, physical playback and household concurrency remain
outside this test's guarantee.

The optimization shifts work: foreground model wall time summed to 45.962 s
disabled versus 41.043 s enabled across 42 cases. The four associated background
prefills consumed 7.310 s of client wall time including cancellations, overlapped
with playback. Thus this is a latency/compute tradeoff, not a compute saving.
The one-task/token/deadline/recent-session limits prevent an unbounded idle loop.
No new model, larger context or additional inference worker was introduced.
CPU energy, peak RSS and watts were not newly measured.

## Validation and deployment state

- 140 targeted Apollo tests passed: history snapshot/isolation, cancellation,
  failed audio, shared voice/text requests, bounded queue, truthful control,
  request-local language, recognition, logging, warm-up and benchmark oracles.
- Seven local context-checker tests passed; source/context coverage checked against HEAD.
- All 54 accepted requests represented in each combined report had exactly one
  completed terminal event, plus four explicit overload rejections. Four extra
  setup turns per queue run precede the measured windows and are not in that count.
- Network audits observed 23/30 loopback destinations (disabled/enabled), zero
  external or DNS attempts. Cached assets were used throughout.
- Existing-symbol upstream impact was LOW for the shared handler/worker and
  measurement path; unresolved dynamic speech/hook callers were manually checked.
  After indexing tests, the new helper had HIGH reach including six test callers.
  Whole-dirty-tree change detection is CRITICAL because it includes inherited
  voice/control/plugin changes; nothing was staged or committed as part of this task.
- Final isolated Apollo unit: `aria-live-20261018t064141z-prefill-logged-5fc113`,
  with the existing three-hour runtime cap, log-only tools and capture-only audio.
  This does not replace a production service. Current server source hashes match
  the final local runtime files; report paths contain private transcripts.
