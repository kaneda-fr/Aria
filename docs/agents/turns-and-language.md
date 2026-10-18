# Turn boundaries and request language — 2026-10-17

Follow-up: [controlled capture and reference verification](controlled-capture.md)
confirmed the user spoke answer statements in the later manual recordings and
verified their exact client/ASR/request trace. It does not establish the source of
extra content in this earlier run, whose raw microphone audio was not retained.

The maintainer authorized this follow-up to the [shared request path](shared-request-path.md).
The Mac client now preserves sentence pauses better, and TTS uses the current
request's language. The Apollo rerun improves the deterministic audio cases; the
physical microphone still admits unexplained answer-like content. Do not call this
a clean microphone-quality pass or a universal endpointing solution.

## Changes and tradeoffs

- Mac client VAD previously ended a turn after 300 ms of audio silence. Recorded
  French arithmetic/capital cases restarted only 80–125 ms after that boundary,
  splitting the question from its answer-format instruction. The default is now
  800 ms, clamped to 200..3000 ms. `client_start.sh` also defaults to 800 instead of
  its previous 400 ms override and permits an environment override.
- Endpointing remains based on audio duration, with bounded queues/backpressure.
  Capture/resampling and server protocol are unchanged. With client VAD enabled,
  explicit client boundaries are authoritative; changing server VAD silence alone
  would not fix this path. Without client VAD, existing server endpointing remains.
- The tradeoff is roughly 500 ms more endpoint delay than the old 300 ms default,
  for otherwise identical VAD decisions. Pauses longer than the threshold still
  split; separate requests with short gaps can merge. The server's existing
  10-second ring buffer is unchanged. Long utterances/overflow remain unvalidated.
- `_process_request` detects FR/EN from the current input once, logs the language,
  and passes it to `SpeechOutput.speak` for model replies and shortcuts. The same
  language applies to every TTS chunk, including numeric answers. Removed global
  previous-language reuse and the `ARIA_TTS_LANG_SHORT_REUSE_CHARS` behavior.
- Direct speech callers without a hint detect from their own full reply. Ambiguous
  text defaults to English; the lightweight heuristic is not a general language
  classifier or an interpreter of requests to answer in another language. Optional
  installed langid retains its existing local fallback. No extra model call,
  download, or cloud dependency was added.

No model, prompt, token budget, queue policy, or control architecture changed.
The local launch wrapper's other noise-gating settings were preserved; the baseline
uses the standard client VAD profile, not that wrapper's complete configuration.

## Reproducible run

Command: `.venv/bin/python scripts/e2e/run.py --host apollo --suite baseline --mic`.

- Run: `20261017T110759Z-d028e0`, Apollo Xeon E3-1225 v5, four cores/threads,
  nominal 64 GB RAM, existing CPU-only local models.
- Artifacts: `~/Library/Logs/Aria/e2e/20261017T110759Z-d028e0/` on the test Mac;
  `report.json`, `baseline-progress.json`, microphone text logs, synthetic WAVs,
  runtime/network logs, `diagnostic-probes.log`, `request-audit.json`, and a separate
  `review.json` retaining manual microphone findings. No raw microphone audio saved.
- Remote source: `/data/aria/e2e/runs/20261017T110759Z-d028e0/src`.
- Source archive SHA-256:
  `882bd3a9874c33eaf01cced513ef6363d11be54d453d777e2f7bb873181e9b51`.
  It now includes the Mac sender as well as server/harness/tests. Compared local
  and staged file hashes after execution; all three changed runtime files matched.
- `app/main.py`: `9050d1474409e4d077a50d5a58268afc89b01fd4e002da9d4047f780eb9be5e4`.
- `app/speech_output/speech_output.py`: `7ed73d4fbdc10a6b398023e29be4e3c94b6258c5003a0f30f71ec373a2e1cc62`.
- `client/macos/mic_stream.py`: `a8d6296e17a5bb09c7bdd0b1603bd0b1f14449b667053561f1c03634b36c0391`.
- Qwen2.5 3B Q4_K_M, Ollama 0.14.2, temperature 0, context 2048,
  `num_predict=80`, history/streaming off. Same models/prompt as the preceding run.
  Baseline pins client silence to 800 ms, uses two seconds of trailing replay silence,
  and gracefully finishes the replay sender. See [test workflow](e2e-testing.md).
- Isolated loopback-only Aria/Ollama runtime; real ASR/Piper/normalization/HTTP WAV
  retrieval. Physical Sonos, live control, router, and speaker recognition excluded.
  Mac capture uses its native 96 kHz rate and actual client resampling/WebRTC VAD.

## Results versus the preceding shared-path run

| Measure | Before: `102130Z-22e4e8` | After: `110759Z-d028e0` |
| --- | --- | --- |
| Completed pipelines | 18/18 | 18/18 |
| Accepted / completed requests | 24 / 24 | 19 / 19 |
| Missing/duplicate terminal outcomes | 0 | 0 |
| Automatic quality-flagged cases | 10 | 5 |
| Cases with wrong TTS voice | 3 | 0 |
| WAV cases with multiple model requests | 2 | 0 |
| Story cases cut off at 80 tokens | 4 | 4 |
| Microphone model requests, EN / FR | 4 / 2 | 2 / 1 |

All eight WAV cases have one recognized turn/model request and the intended TTS
voice. Both arithmetic cases answer only `4`; both capital cases answer Paris.
French recognition still contains minor word errors such as “Répose,” but the
question and instruction reach the same request. Four stories still end at the
80-token limit. The unavailable-control cases state inability; the French response
still begins with an awkward imperative. No real device action was attempted.

**Microphone evidence remains contaminated in both languages.** English preserved
the entire question/instruction, then admitted an extra “The capital of France is
Paris” segment and generated another reply. French preserved the entire question
but appended “La capitale de la France est Paris” to that same transcript. The
French case escapes the current automatic count/keyword checks; manual review
therefore identifies **six distinct cases needing attention**, not just the five
automatic flags. A plausible answer and one request do not prove clean capture.

The harness plays only question WAVs. Its output adapter retrieves generated audio
without playing it. The French extra content was transcribed before that request's
response was generated, so it cannot be feedback from that response. Room audio,
another playback source, and ASR behavior remain unresolved; raw microphone PCM was
not retained. Do not label these transcripts proven echo or hallucination. Both
microphone clients exited cleanly with no reported queue/WebSocket drops. The extra
English segment and the French segment finalized on client `stop`, so this run also
does not establish natural silence endpointing for those microphone segments.

Synthetic WAV first-LLM-start → first HTTP audio retrieval: p50 2.35 s EN / 2.37 s FR;
nearest-rank p95 (maximum of four samples) 7.58 / 10.41 s. These exclude capture,
endpoint wait, ASR, and physical playback, and reuse cached TTS where available.
A separate fake-handler diagnostic ran during the audio phase; its CPU load and
other host workloads were not controlled. These are descriptive timings, not a
performance comparison or full-system capacity/power measurement.

Runtime trace: 25 loopback destinations, zero external/DNS attempts. Dedicated
test units/tunnel stopped; ports 8000/8001/11435 absent, existing shared Ollama on
11434 retained. No persistent Apollo deployment changes. This audit covers the
isolated profile, not certification of every deployed provider or LAN interaction.

## Validation and next work

- 50 Apollo tests passed: plugin/Sonos mocks, queue and shared transport handlers,
  alternating numeric voices/cache separation, per-response chunk language,
  stateless fallback, invalid hints, and local input-language detection.
- Five Mac endpoint tests passed: 440 ms sentence pause retained, one-second pause
  separated, old 300 ms split reproduced, sender stop flush once, silence ignored.
  They mock VAD/resampling decisions; the real pipeline is covered by the rerun.
- Fifteen local harness tests, six queue tests, seven context tests, source/context
  checks, shell syntax, and whitespace checks passed. Current handler diagnostic
  assertions also passed on Apollo without live inference/device actions.
- Serena still had no active language server. GitNexus impacts preceded existing
  symbol edits. Aggregate change detection flags critical integration risk across
  the accumulated dirty checkout, including the previous shared-path change;
  expected startup/voice/worker/speech flows were exercised. No commit or bulk stage.

Next: isolate the microphone's extra content using a controlled acoustic capture
and explicit playback timing before changing echo or ASR filters. Do not discard
valid speech to fit this corpus. Then handle generation truncation explicitly and
evaluate complete concise replies within the same hardware budget. Keep D-001 open;
these fixes do not establish model control reliability or device completion.
