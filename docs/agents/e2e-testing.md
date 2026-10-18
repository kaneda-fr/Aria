# Mac + Apollo end-to-end test workflow

The [runner](../../scripts/e2e/run.py) stages isolated source; it does not use deployed
`start.sh` or `sync.sh`. See the [live warm-up comparison](warmup-2026-10-18.md).

For control failures, use the separate [tool-calling diagnostic](tool-calling-2026-10-17.md).
It exercises real local inference, inventory matching, and payload validation with
captured publication. Conversation runs below keep device providers disabled and
must not be used as evidence of tool-calling accuracy.

For interactive use with **all inference on Apollo and audio output on the Mac**,
run `.venv/bin/python scripts/e2e/live_mac.py --host apollo`. See the
[Mac playback guide](../../client/macos/README.md#live-apollo-inference-with-mac-playback).
This uses a dedicated live profile (12 history messages, local speaker recognition,
cached inventory, and log-only tool actions)
and adds local WAV playback, temporary
microphone muting during replies, and same-clock client latency observations. It
does not load models on the Mac. Ctrl+C cleans up the owned server/tunnel; a three-hour
service limit bounds abandoned sessions. No home actions or Sonos are enabled.
The live profile logs exact recognized requests/replies privately on Apollo and the
Mac with request IDs; the independent smoke/baseline profiles still disable history
and speaker recognition. Prepare `/data/aria/models/speaker-ecapa` before live startup:
download the official SpeechBrain ECAPA model assets if absent and resolve its YAML
`pretrained_path` to that local directory. All inference then runs offline, including
speaker recognition. Keep enrollment profiles private and never copy them into source.

## Run

From the repository root on the Mac:

```sh
.venv/bin/python scripts/e2e/run.py --host apollo
```

Include actual microphone capture, client resampling/VAD, and an acoustic fixture:

```sh
.venv/bin/python scripts/e2e/run.py --host apollo --mic
```

`--mic` plays a synthetic English sentence through the Mac's current audio output
and opens the MacBook Pro microphone briefly. Use speakers within microphone range;
headphones, mute, denied microphone permission, and ambient noise can make this
physical test fail. The runner stores diagnostic text locally, not raw microphone
audio. Synthetic WAV fixtures are stored for reproducibility. Do not leave an
interactive microphone client running alongside the automated test.

During `--mic`, remain silent: the played question is the test input, not an invitation
to answer it. Spoken answers become additional valid microphone input and invalidate
the fixture-only comparison. For human speech, use the manual capture workflow and
confirm what was actually spoken before replaying it on Apollo.

The SSH alias `apollo` was verified in this environment. Other machines can supply
`--host USER@HOST` or set `ARIA_E2E_SSH`. No credentials are stored in the runner.

## Prerequisites and preparation

### User-controlled acoustic diagnosis

When the maintainer wants to control capture, use:

```sh
.venv/bin/python scripts/e2e/manual_capture.py
```

This standalone local tool prompts for English then French. Enter arms the microphone;
`READY` appears only after actual input frames arrive. Speak the displayed sentence,
leave two seconds of silence, then press Enter to stop (20-second automatic limit).
The microphone closes before `afplay` replays the recording. The user can enter
listening notes, then must confirm whether they said the displayed sentence or type
what they actually said. This produces a hash-bound `reference.json`; the displayed
prompt and a “clean” note alone are not transcription ground truth. It does not start Apollo or transmit
audio, and never plays a generated question or answer during capture.

Unlike the automatic baseline, this explicitly requested diagnostic saves raw
native-rate float samples, native PCM WAV, a 16 kHz analysis WAV, hashes, capture
status, and listening notes under `~/Library/Logs/Aria/captures/<run-id>/`, with
private file permissions. Preserve the native samples for later replay through
the actual client; the analysis WAV uses offline soxr and is not claimed byte-identical
to streaming resampling. Keep recordings out of the repository. Do not initiate
these manual captures on the user's behalf when they have asked to control them.

Readiness/close ordering and saved-audio integrity have mocked unit coverage in
`tests/test_manual_capture.py`. For confirmed saved input, run:

```sh
.venv/bin/python scripts/e2e/run.py --host apollo --suite baseline --capture-dir /absolute/path/to/capture-run
```

This runs the original 16 text/WAV cases plus two saved-input cases. It never opens
the microphone or plays local audio. It verifies the capture/reference hashes,
feeds the original native floats through the real Mac sender, and records transmitted
16 kHz segments. Test-only Apollo instrumentation hashes ASR input and records raw
ASR text; the report checks those bytes against client output, raw ASR against LLM
input, and normalized words against the user-confirmed reference. Extra words in a
merged transcript now fail saved-input evaluation even if the answer looks correct.
Quiet VAD candidates can be rejected by the existing server audio gate before ASR.

`--capture-dir` requires both EN/FR recordings and confirmed references, and cannot
be combined with `--mic`. An older capture can receive a separate `reference.json`
only after explicit user confirmation; bind it to the native file SHA-256 without
changing original audio or manifests. Saved-input replay proves reproducibility
through the client/server path, not a new live acoustic capture. If the user spoke
different words than an old automatic fixture, do not present the cases as equivalent.

Six regressions in `tests/test_capture_replay.py` cover reference confirmation,
hash binding, extra/changed transcript content, and exact transmitted PCM retention.

### Runtime prerequisites

- Mac: existing `.venv` with `httpx`, `websockets` (the tested environment supports
  `proxy=None`), and the [client dependencies](../../client/macos/requirements.txt).
- Apollo: passwordless SSH access, user systemd, unprivileged user/network namespaces,
  `ip`, `strace`, `curl`, and the existing `/data/aria/dev/.venv` Python 3.11 runtime.
- Existing local ASR assets in `/data/aria/models`, Piper at `/opt/piper-tts/piper`,
  English Amy low and French Siwis low assets in `/data/aria/voices`.
- Ollama executable `/usr/local/bin/ollama` and readable existing model store
  `/usr/share/ollama/.ollama/models`, containing `qwen2.5:3b-instruct`.
- Test dependencies installed separately at `/data/aria/e2e/deps`; this session
  installed the versions in [requirements-apollo.txt](../../scripts/e2e/requirements-apollo.txt).

For a new host, prepare dependencies/assets before running tests. Do not perform
downloads in request handlers. The runner intentionally fails on missing runtime
assets rather than fetching them. It reuses the installed application dependencies
without upgrading the deployed virtual environment. Record/version those dependencies
when comparing results across hosts; this is not a hermetic package installation.

## What happens

1. Stage only `app`, `client/macos/mic_stream.py`, the E2E scripts, and selected unit tests in a unique directory
   under `/data/aria/e2e/runs/`. Record the source archive SHA-256 and local Git HEAD;
   the archive includes uncommitted application and Mac sender changes. Mac capture
   still executes the local client; avoid editing it during a run and verify its hash
   against the staged copy when recording results.
2. Run plugin infrastructure, mocked Sonos, shared-request handler, queue, and TTS-language tests
   on Apollo, with a 120-second bound on this test phase. New handler tests use fake
   inference/execution and exercise HTTP plus finalized-voice admission and WebSocket status.
3. Start a dedicated user-systemd unit with a 15-minute lifetime limit. A private
   network namespace contains only loopback and no external route. Start a separate
   Ollama instance using existing weights, then Aria on a Unix socket.
4. Forward a random Mac loopback port to that socket over SSH. No shared Aria TCP
   port is taken and the existing Ollama service is not restarted.
5. Assert component readiness, text inference with a known answer, and English/French
   synthetic audio through the actual Mac client's sender and `/ws/asr` endpoint.
   Feed 96 kHz float audio into the client's real resampler, VAD, and framing code.
   Assert recognized keywords, then LLM completion and fetched audio occurring after
   that target request. An earlier greeting's audio cannot satisfy the request.
   Verify Piper output and successful HTTP fetching of a valid WAV.
6. With `--mic`, also run the actual Mac client against the same instance and assert
   the acoustic fixture's keyword reaches the LLM after ASR.
7. Stop owned services/tunnel, collect logs, and inspect the runtime network trace.
   The audit fails on observed external destinations or DNS attempts, even failed
   attempts. It also fails when network evidence is missing.

## Coverage and intentional boundaries

The test adapter [server.py](../../scripts/e2e/server.py) replaces Sonos discovery
and the physical playback sink only. ASR, LLM, Piper, normalization, the WAV route,
and echo-guard wiring remain real. The capture sink fetches actual generated audio
through the application's HTTP route. It does not prove Sonos hardware playback.

Plugins/actions, router, and speaker recognition are disabled in this baseline.
No home commands are executed. Synthetic WAV cases exercise real client resampling
and VAD from injected native-rate frames; only the optional microphone case tests
physical capture. These checks establish conversation-pipeline connectivity and
basic behavior, not control correctness, routing accuracy, echo suppression quality,
speaker identity, sustained load, or production offline certification.

The private namespace enforces isolation for the tested Aria/Ollama/Piper processes.
Its trace includes startup and request activity. Existing host services and the Mac
OS are outside that isolation/audit scope. The trace checker covers recorded
IPv4/IPv6 socket destinations and DNS calls; retain the raw trace for inspection.

## Evidence and cleanup

Each run prints its artifact path under `~/Library/Logs/Aria/e2e/<run-id>/`:
`report.json`, test output, server/Ollama logs, network trace, route evidence, model
listing, and synthetic input WAVs. These private artifacts stay outside the repository.
Reports include elapsed scenario times, not a statistically meaningful latency SLO.
The dedicated remote directory is retained for debugging and can be removed when
its evidence is no longer needed. Never remove `/data/aria/dev` or the model store.

Normal success, failure, and Ctrl+C trigger cleanup. If the Mac process is forcibly
killed, the remote service expires after 15 minutes. The run's `run.json` records the
exact unit name for early cleanup:

```sh
ssh apollo systemctl --user stop aria-e2e-RUN_ID
```

Inspect `server.log`, `ollama.log`, `journal.log`, and `tunnel.log` on failure. Missing
ASR/LLM/TTS readiness, missing events, incorrect transcript keywords, failed synthesis,
or attempted external calls must remain failures rather than being labelled passes.

Check microphone authorization without opening or recording audio:

```sh
.venv/bin/python scripts/e2e/mac_audio.py
```

`--mic` requires `authorized`. A `not determined`, `denied`, or `restricted` result
needs resolution through macOS's normal microphone consent/settings flow. The agent
does not change the privacy database. The default synthetic run remains available.

Verified on 2026-10-17 after the maintainer granted consent: run
`20261017T094050Z-34e0aa` passed the physical Mac microphone scenario and the three
automated scenarios, with clean client exit and no reported queue/WebSocket drops.
The network audit passed and no test units remained active after cleanup. See
[current evidence](status.md) for coverage limits.

## Bilingual characterization suite

```sh
.venv/bin/python scripts/e2e/run.py --host apollo --suite baseline --mic
```

The baseline suite runs four matched English/French prompts through text and WAV
input (16 cases), then a factual question through the physical microphone in each
language (two more). Categories: arithmetic, factual answer, short narrative, and
an unavailable device command. The last category checks the response when control
is disabled; it does not measure the real control system's accuracy.

The baseline pins `ARIA_CLIENT_VAD_SILENCE_MS=800` for replay and physical capture,
records it in the profile, and supplies two seconds of trailing silence to replay.
Replay shuts down the sender gracefully to exercise end flushing. Local endpoint
regressions use `.venv/bin/python -m unittest discover -s tests -p test_client_turns.py -v`;
these mock VAD/resampling decisions, while the Apollo baseline uses real WebRTC,
soxr, and microphone capture. Compare endpoint latency separately from model latency.

[baseline_server.py](../../scripts/e2e/baseline_server.py) adds synthetic fixture
creation and a response hook on the real local Ollama client. It records model
content, token counts, stop reasons, and timings without changing inference outputs.
[baseline.py](../../scripts/e2e/baseline.py) preserves per-case events, transcripts,
voice selection, generated WAV retrieval, and separate pipeline/quality results.
`passed` is the runner's execution/network result; it is **not** a quality score.
Inspect `quality_failures`, stop reasons, and raw replies. Narrative correctness,
and truthful inability require explicit review beyond the numeric/Paris checks.
Compare microphone transcripts with the question actually played. An appended
answer can evade request-count/keyword checks when merged into one turn; report
manual findings separately from automatic `quality_failures` (see
[turn/language results](turns-and-language.md)). No raw microphone audio is saved,
so these logs cannot prove the acoustic origin of unexpected recognized content.
The suite also flags multiple model calls, missing expected factual-question input,
and wrong TTS language. These flags identify interactions to investigate, not proven
server root causes. Accent/punctuation-normalized WER still counts
spoken-number-to-digit formatting differences; it is not semantic error rate.

Per-case scenario duration includes input transport, generated speech duration,
and a settling interval. `first_llm_start_to_first_audio_ms` excludes input capture
and ASR. It is not speech-end-to-audible-response latency. TTS caching, first model
load, and split utterances affect comparisons; retain the raw events. This small
corpus does not establish statistical capacity.

The optional [diagnostic probes](../../scripts/e2e/diagnose_server.py) assert
preserved pending requests, voice/text tool parity, and the router's CHAT response branch
using real handlers with fake inference/forced classification. Run in a separate
Python process with Apollo dependencies; never import into a serving app. They do
not test model routing accuracy or physical devices. See the dated
[baseline report](baseline-2026-10-17.md) for results and fix priorities.

For runs using shared admission, also audit the request lifecycle:

```sh
.venv/bin/python scripts/e2e/audit_requests.py /absolute/path/to/report.json
```

The audit requires exactly one terminal event for each accepted request, deduplicates
overlapping report windows by server event index, and fails on missing/duplicate
outcomes. A terminal failure/expiry is accounted for, not a successful interaction;
read the outcome counts alongside pipeline and quality results. Reports from before
shared admission have no status events and cannot satisfy this audit.
