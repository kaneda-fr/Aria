# Controlled capture and spoken-reference correction — 2026-10-17

The maintainer supplied manual English/French recordings, marked both replays
“clean,” then explicitly confirmed speaking the **answers**, not the displayed
questions. Direct ASR and complete client/server replay reproduce those spoken
answers exactly. No content insertion by ASR, the queue, or the LLM request path
was reproduced with these recordings.

The confirmed defect was in evaluation: the capture manifest stored the displayed
question as `expected_text` without confirming what the user actually said. A
“clean” listening note describes the recording, not its exact words. Comparing
against that unconfirmed question would misdiagnose correct transcription.

This resolves the content of the supplied recordings. It does **not** prove the
origin of answer-like content in the earlier automatic microphone runs: their raw
audio was not retained and participation in those tests has not been confirmed.
Preserve that distinction; do not retroactively label earlier content proven echo,
hallucination, or user speech.

## Evidence chain

Source capture: `20261017T205712Z-236a1d`, stored privately under
`~/Library/Logs/Aria/captures/` on the test Mac. Native 96 kHz mono float samples:
English 6.28 seconds, French 5.04 seconds. Both reported no capture errors and
successful local replay. All original file hashes were verified. Separate
`reference.json` files record the user's chat confirmation, bound to each native
file hash; original recordings/manifests were preserved.

Confirmed spoken text:

- EN: “The capital of France is Paris.”
- FR: “La capitale de la France est Paris.”

1. Direct Apollo Parakeet transcription of the 16 kHz analysis WAVs, twice per
   language, returned exactly those statements. No LLM, TTS, or request queue was
   involved. The process ran in an isolated network namespace with local assets.
2. Original native floats were passed through the real Mac client sender's soxr,
   WebRTC VAD, framing, and silence endpointing. English yielded one speech segment
   plus a quiet trailing candidate; French yielded one speech segment.
3. The existing server audio gate rejected the quiet English candidate before ASR:
   1.04 seconds, RMS 0.000214, speech ratio 0. Speech segments transcribed unchanged.
4. In the full baseline, client PCM hashes matched the server's actual ASR inputs;
   raw ASR text equalled the text forwarded to the model and the confirmed reference.
   Both languages produced one model request and the corresponding TTS voice.

English speech PCM SHA-256:
`28948f6bfd8c6bed1e5267d98157f39903d9908c657a6bab92a4828c1915952b`.
French speech PCM SHA-256:
`5f477c1a05b785f9b9ea5482e38a2f2c46dd7011c4d26295de85431718860e1f`.

Direct diagnostic logs are retained beside the captures as `apollo-direct-asr.log`
and `apollo-wire-asr.log`. Transmitted PCM segments and raw stage events are retained
in the full-run artifacts. Native recordings remain outside the repository.

## Fix and reproducible workflow

- Manual capture now asks whether the displayed sentence was spoken exactly. If
  not, the user types the actual wording. Unconfirmed recordings cannot be used
  as scored saved-input cases.
- `reference.json` binds user-confirmed words to the original native audio hash.
  The loader rejects missing confirmation, mismatched hashes, or invalid audio.
- `--capture-dir` replays saved native floats through the actual Mac sender without
  opening the microphone or playing local audio. It runs the existing 16 text/WAV
  cases followed by two confirmed-recording cases.
- Test-only ASR instrumentation records input hashes and raw output. Saved-input
  checks compare transmitted PCM → ASR input → raw transcript → model input →
  confirmed wording. Extra words in a merged transcript are now flagged even when
  a plausible answer and a single request would have passed the earlier checks.
- The guide explicitly distinguishes automatic `--mic` tests (remain silent while
  the fixture plays) from manual human capture. See [E2E workflow](e2e-testing.md).

Command shape (supply the saved capture directory):

```sh
PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH .venv/bin/python scripts/e2e/run.py --host apollo --suite baseline --capture-dir /absolute/path/to/capture-run
```

The PATH override selects the working installed Command Line Tools Git; the default
Xcode shim was blocked by its license prompt. No license was accepted or system
setting changed. Future machines should use their own functioning Git installation.

## Full Apollo rerun

Run `20261017T210457Z-c14268`, artifacts under `~/Library/Logs/Aria/e2e/`.
Source archive SHA-256:
`2d3099fe5804e9b7e541d2a5a1db3134b9bbe109bd15a43fbcadff3cc0ab1f55`.

Same Xeon E3-1225 v5, CPU-only local ASR/Qwen2.5 3B/Piper, 80-token limit,
temperature 0, context 2048, history/streaming off, 800 ms client silence endpoint.
Plugins, physical Sonos, router, and speaker recognition remain excluded.

- **18/18 pipelines completed**: original 16 text/WAV cases plus two confirmed saved
  recordings. This is not a fresh live capture or a direct comparison of identical
  microphone questions with the preceding run; the user recorded answer statements.
- **18 accepted / 18 completed requests**, no missing/duplicate terminal outcomes.
- **No wrong TTS voices or saved-input trace/reference mismatches.** All eight
  synthetic WAV cases and both saved speech cases produced one model request each.
- **Four automatic quality flags remain**, all story truncation at 80 tokens.
  Unavailable-control replies still need prose-quality review; no live control was tested.
- **26 loopback destinations, zero external/DNS attempts** in the isolated runtime
  trace. Owned test services/tunnel were stopped; shared Ollama on 11434 retained.
- Synthetic WAV first-LLM-start → first HTTP audio p50: 2.28 s EN / 2.37 s FR;
  nearest-rank p95 (maximum of four samples): 7.64 / 7.80 s. These exclude capture,
  endpoint waiting, ASR, and physical playback; caching and host workloads affect them.
  This is not a capacity or power benchmark.

Validation: 50 Apollo unit/handler tests; 35 local tests (six capture/reference,
15 harness, two manual capture, five endpoint, seven context). Python compilation,
context/source-review checks, whitespace checks, and request audit passed. Serena
still had no language server. GitNexus impacts for edited existing harness symbols
were low/unknown (the ASR-start route is called through HTTP); aggregate detection
remains critical across the accumulated earlier runtime refactors and expected flows.

No production ASR, filtering, queue, LLM, or TTS behavior changed in this task.
Application/client fingerprints match the preceding run; no persistent Apollo
deployment change, new model, or live device action. Next server priority remains
complete concise responses within the hardware budget, with explicit truncation
handling. Historical microphone attribution remains limited by missing raw audio.
