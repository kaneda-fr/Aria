# Model residency across idle periods — 2026-10-18

The maintainer requested fixing the model going cold after idle, followed by
French/English Apollo checks after more than five minutes idle, preserving quality
and tool calling with actions log-only. The [live review](live-wake-review-2026-10-18.md)
found a model reload and 20.87 seconds of prompt processing after the default
five-minute idle window; startup warm-up alone did not cover real usage.

## Implementation

`app/llm/ollama_client.py` now sends top-level `keep_alive` on every Ollama chat
request. `OllamaConfig.keep_alive_s` defaults to `-1`; the environment setting
`ARIA_LLM_KEEP_ALIVE_S` accepts `-1` or nonnegative integer seconds. Warm-up and
prefill use dataclass replacement and retain the policy. Streaming and all tool
rounds use the same request construction. Sampling, model, prompt, history, control
authorization, ASR, speaker decisions, TTS, and queue behavior are unchanged.
llama.cpp receives no Ollama-specific field.

This uses [Ollama's documented residency semantics](https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)
and [chat API parameter](https://docs.ollama.com/api/chat), checked October 18.
No periodic heartbeat, dummy generation, model download, or background inference
was added. The selected model/cache consumes RAM while idle. Eviction by another
model/client or an Ollama restart can still invalidate the cache. Stopping Aria
does not independently unload a shared Ollama model; the isolated E2E service owns
and stops its own Ollama process. See [configuration](../ENVIRONMENT.md).

## Validation design

- 17 local residency/warm-up tests passed; the combined Apollo suite passed 217
  tests with four existing dependency warnings. Apollo test stage:
  `20261018T103452Z-idle`. Checks cover numeric overrides/invalid settings, both
  transport modes, tool calls, warm-up output discard, prefill policy retention,
  llama.cpp separation, and the existing conversation/control/audio guards.
- Reference: `20261018T103200Z-idle-reference-6a97f4`, explicit 300-second policy.
  Candidate: `20261018T103318Z-idle-resident-d6634d`, default retained model.
  Both use identical current app source, Qwen2.5 3B, Ollama 0.14.2, four CPU threads,
  tool-capable warm-up, 12 history messages, prefill, cached inventory, actual ASR
  and speaker inference, captured Piper WAVs, and isolated loopback networking.
- `scripts/e2e/idle_bench.py` idles both profiles together for 310 seconds before
  each language. No model call during idle is allowed. Inference measurements are
  sequential on Apollo. The first request in each profile/language is streamed
  prerecorded PCM with inline wake and arithmetic; then all 13 established text
  conversation/control cases for that language run. Raw ASR discrepancies remain
  recorded separately from functional response/state/tool flags.
- Mac capture/playback was paused during measurements. Runtime source archives,
  hashes, private exact replies, fixture hashes, commands, model timings, and
  per-process resource samples are stored in the candidate run's E2E directory.

## Idle and conversation results

Both French and English idle intervals lasted 310.025 seconds with unchanged
model HTTP and inference-event counts. The reference runner unloaded in both; the
candidate retained PID 3750955 throughout, using 2,118,588 / 2,133,916 KiB RSS
and 0.02 CPU-seconds per idle interval. These are process
measurements, not whole-machine power measurements; Apollo exposes no usable power
counter for this run.

The first French and English saved-PCM requests answered `84` correctly in both profiles:

| Speech-end → first WAV ready | Reference | Candidate |
| --- | ---: | ---: |
| French | 19.817 s | 3.160 s |
| English | 19.094 s | 2.849 s |

French model load fell from 2.166 to 0.143 s, prompt processing from 15.544 to
0.935 s. English load fell from 1.846 to 0.147 s and prompt work from 15.272 to
0.730 s. Generation stayed near 0.18–0.20 s. Audio is captured server-side; these timings
include actual ASR/speaker processing but exclude Mac transport and physical
playback. This is one first-voice measurement per language/profile, not a tail-latency
or acoustic robustness guarantee. The retained profile waits through the reference
suite too, so its actual no-inference period is longer than the common 310 seconds.

All 28 functional checks per profile passed (2 saved voice + 26 established text
cases). All non-clock replies were byte-for-byte identical across profiles; clocks
matched actual Zurich request times. All eight exact command IDs/actions/values/
payloads per profile matched the fixture oracle and reported dry-run/non-executed.
Missing-room and ambiguous requests produced clarification without commands. Warm
text first-WAV median was 0.734 s reference versus 0.725 s candidate. Raw ASR
`transcript_mismatch` flags remain on all four saved-voice rows (the same wake-name
spellings in each profile); `strict_passed` is therefore false while
`functional_passed` is true. No new ASR correction is claimed.

## Tool lookup, queueing, and audit

Four queued trials per profile (two per language) passed, including admission,
order, expected rejection, and replies. Median accepted-to-first-WAV was 3.984 s
reference versus 3.978 s candidate. These are warm queued comparisons, in addition
to the first post-idle voice requests above.

Two additional free-form inventory lookup prompts per profile exposed **unchanged
failures**, outside the passing 28-case set:

- French made the correct `find_devices` call with exact name/room and returned
  the two expected command candidates. However, the final reply was in English
  and incorrectly described the two commands as two devices. Both policies produced
  the same reply. First WAV: 13.690 s reference / 14.170 s candidate (0.480 s slower
  in this single sample); model time 12.619 / 13.039 s. This does not establish a
  new latency regression, but this slow two-round path remains a problem.
- English produced an empty reply without invoking a tool in both profiles. The
  request failed, and no audio or command was produced. Model time: 3.739 / 3.793 s.
- No command was executed in these lookup probes. The fixed residency policy
  preserves observed behavior; it does **not** fix free-form tool invocation,
  response language, candidate/device interpretation, two-channel clarification,
  or the other live-review issues. These should remain visible in the next task.

Each full server log contains 46 accepted requests with exactly 46 terminal
outcomes: 45 completed and the expected English lookup failure; four queue-test
rejections are separate. Network audits saw only loopback destinations (33
reference / 23 candidate), with no forbidden outbound calls. Runtime source hashes
match the tested snapshot. Both benchmark services were stopped after collecting
evidence. Local context validation and all seven context tests pass; `git diff
--check` is clean.

Private evidence is under `~/Library/Logs/Aria/e2e/<run-id>/`: candidate `idle.json`
and `idle-summary.json`; each profile's `tool-lookup.json`, `paired-queue.json`,
`audit.json`, source manifest, and `server-evidence.tar`. Exact utterances, inventory,
speaker labels, and WAVs remain private. The maintained idle harness can reproduce
the paired comparison; supplementary lookup/queue harness copies are in the
corresponding Apollo report directory.

## Live restoration

Mac mic and playback are ready on Apollo run `20261018T105214Z-live-92c831`, unit
`aria-live-20261018t105214z-live-92c831`, with history (six exchanges), speaker
recognition, tool warm-up, exact conversation logging, and tools log-only verified
by the live client. Startup speech played and microphone frames reached READY.
The live staged `ollama_client.py` hash matches the tested local file:
`d733b1473877907ded2681f1f7c4318790f92faf2bbc8fbd3dbb1a284cc02349`.
Source archive: `5dfee350a8172bd932ec709d50a548716cbbc884f4770b4b92462b7c7d3903c3`.
Private live logs are in `~/Library/Logs/Aria/e2e/20261018T105214Z-live-92c831/`.
The isolated service retains its existing three-hour TTL; the Mac launcher owns
cleanup. No Sonos playback or physical device actions were enabled.

## Scope and impact

The production change is confined to the Ollama configuration and request payload.
Serena was unavailable (no language servers); GitNexus and targeted source reads
were used. Upstream impact is HIGH for `load_ollama_config` and
`generate_ollama_response`: direct callers are the shared loader, model dispatcher,
and tool probe; affected flows include voice/text processing, live startup, and
history prefill. The config type is MEDIUM. No client/protocol changes were made.
Final GitNexus change detection sees the entire inherited working tree as CRITICAL
(12 tracked files, 35 symbols, 27 flows). This task's production scope remains the
five config/request symbols in `ollama_client.py`; the affected shared paths were
covered by the targeted suite and paired runtime checks. Inherited uncommitted
work remains preserved; nothing was staged or committed.
