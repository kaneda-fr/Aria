# Architecture and task map

Reviewed: 2026-10-18 against `feature/plugin-system` at `905fd38` plus working-tree
changes. Source presence is not deployment evidence; see [status](status.md).

For future choices, see the [options review](architecture-options.md).
This page describes implementation; D-001 remains open.

## Hardware envelope

**Mandatory boundary:** all runtime computation, storage, and communication remain
on owned devices/local network; no Internet-dependent provider, remote fallback,
or external telemetry once operational. See [D-004](decisions.md). Startup may download
missing models before readiness. Audit loaders and dependencies so lazy downloads,
update checks, and other external calls cannot occur during operation. The current
implementation has not been certified to meet this boundary.

The required deployment baseline is the existing old Xeon server plus the project's
ESP32 devices, confirmed by the maintainer on 2026-10-17. See
[D-003](decisions.md) for architectural consequences. The macOS client below is the
inspected development/capture path, not a replacement for the ESP32 deployment goal.

Apollo has a Xeon E3-1225 v5 (four cores/threads), nominal 64 GB RAM, and AVX2
without AVX512; see [deployment evidence](deployment.md). D-009 assigns the Seeed
XIAO ESP32-S3 a microphone-only role, with Apollo managing playback. The
[firmware](../../client/esp32/README.md) uses Sense PDM without PSRAM/camera.
Bound endpoint buffers/traffic and size server stages against measured capacity.

## Current runtime paths

```text
macOS microphone -> WebSocket /ws/asr -> VAD -> Parakeet ASR final transcript
  -> echo/speaker filtering -> voice admission --+
POST /api/llm/process -> text admission ---------+-> bounded FIFO -> one worker
  -> conversation gate -> _process_request -> local facts / explicit shortcut / grounded control / model loop
  -> grounded control: strict inventory selection -> single-use execution authorization
  -> execute_command -> actions -> Jeedom MQTT (or log-only test provider)
  -> conversation model: read-only lookup -> grounded FR/EN reply; ungrounded generated actions rejected
  -> final response -> optional speech
```

`app/llm/conversation.py` gates the shared processor after admission. Waiting sessions
ignore requests without a leading “Aria” (optional FR/EN greeting), with no model,
tool, or audio. Name-only activation gets a short local acknowledgment; inline
requests have the vocative stripped only from processor input, preserving raw logs.
Follow-ups retain history and grounded device references. Explicit complete goodbye
utterances or expiry clear both, and cancel matching speculative prefill. State is
bounded to 256 endpoint sessions; eviction also clears context. Unnamed sockets start
waiting. Named sockets use `client:<client_id>` and preserve context over reconnects
until normal expiry/goodbye/eviction. The ID is the unique client name; duplicate
live claims are rejected. Audio buffers are never resumed. See the
[client protocol](client-protocol.md). HTTP callers must reuse a nonempty session ID (maximum 256 characters);
stateless requests must include the name each time. Person recognition does not
serve as the conversation-session key.

Voice input alone canonicalizes bounded vocative spellings (Ari, Arya, Ariya,
Ariah; greeting-qualified area). The request body and raw transcript are preserved;
HTTP still requires the exact name. This repairs observed ASR name spellings, not
arbitrary recognition errors. See [spoken-wake follow-up](spoken-wake-2026-10-18.md).

The default 30-second idle window begins after `SpeechOutput.playback_done`, which
estimates playback from audio duration and does not acknowledge physical audibility.
Voice onset is captured by the server before ASR and carried separately from queue
age. Speech beginning inside the window remains eligible after ASR/queue delay.
Expired state is cleaned lazily at the next request or capacity eviction. No timer
loop, wake model, or extra LLM call was added. ASR, speaker, and echo filtering still
run before the gate; this is not a low-power acoustic wake detector. See
[wake-flow evidence](wake-flow-2026-10-18.md) and [settings](../ENVIRONMENT.md).

Both transports now share tool handling, session history, routing, and requested
speech. Tools are offered only when requested and an inventory provider exists.
Unknown CHAT phrases use the LLM; known shortcuts honor requested speech. This fixes
the earlier wiring divergence, but fake-provider parity tests do not prove live
voice-to-device control or model interpretation accuracy.

`app/llm/local_answers.py` answers explicit FR/EN clock/capability questions using
local runtime facts, retaining shared history and request language. Unsupported
phrasing reaches the model. `app/llm/history_prefill.py` prepares the exact history/tool
prefix during playback, only after recent same-session model use. It bounds discarded
speculative work to four seconds/one token; the worker cancels it before new work.
No prefill output enters history/tools. `SpeechOutput.audio_ready` is per-call and
resolves false on failure/cancellation. See [history-prefill evidence](history-prefill-2026-10-18.md),
[science/model rejections](science-2026-10-18.md), and [settings](../ENVIRONMENT.md).

The [grounded-control change](grounded-control-2026-10-18.md) adds a narrow FR/EN
imperative path for light on/off, shutter open/close, and whole-number percentages.
It resolves a unique device/room/capability, revalidates inventory and payload at tool
dispatch, executes once, and renders the actual outcome without a model confirmation.
Ambiguity produces clarification with no action. Session-and-speaker device references
expire after 120 seconds and are bounded to 128 entries. This is not general language
understanding or authentication, and D-001 remains open. The administrative action
API remains a separate lower-level interface outside this assistant-request guard.

Mac client VAD owns `utterance_start`/`utterance_end`, with 800 ms silence
(configurable 200..3000 ms). Server VAD applies otherwise. The 10-second ring buffer
and backpressure remain bounded. Acoustic pauses can still split or merge turns.
TTS uses the request language for ambiguous text, adapting voices at sentence
boundaries with bounded lookahead and optional player acknowledgments. See
[sentence routing, input-state limits and measurements](sentence-tts-2026-10-19.md).

[RequestQueue](../../app/llm/request_queue.py) permits one running request and one
pending request by default. It preserves FIFO accepted work and rejects new work
when full (HTTP 429 / WebSocket `request_status`). Pending work older than 30 seconds
expires before execution (HTTP 504); limits are configurable and bounded. Structured
`ARIA.Request.Status` events carry request/session IDs and accepted, running, completed,
failed, rejected, expired, or cancelled states. WebSocket clients receive admission
and terminal status frames without transcripts or model content; the existing Mac
client and ESP32 log these frames as diagnostic status.

Cancellation of an HTTP handler cancels pending work only. Already running work can
finish; shutdown reports interrupted work and drains pending work before stopping
providers. No action is automatically retried. `completed` means processing finished
and speech was scheduled if requested, not physical playback or device completion.
Disconnect does not revoke accepted voice requests. Queue fairness is FIFO, not
per-session quotas; these and device-outcome tracking remain separate enhancements.

## Ownership and entry points

The ESP32 uses local portable WebRTC VAD, 200 ms pre-roll, an 800 ms endpoint,
speech-gated AGC/fixed gain, and optional raw RMS/SNR gates. It sends 16 kHz mono
PCM16 in 20 ms frames with explicit turn boundaries. Capture drains independently
of transport, with an eight-frame queue, finite writes and fresh sessions after loss.
See [D-012](decisions.md) and the [parity matrix](../../client/esp32/README.md).

Both clients use `/ws/asr`: start, client-VAD boundaries, PCM, stop; no new mandatory
acknowledgment. VAD/gain run locally. Status is diagnostic, not playback completion.
The Mac adapter suppresses capture during playback; ESP32 playback-aware muting is
still a gap. Retain echo/speaker filters. Disconnect/stop can finalize partial speech
and does not cancel accepted work. See [client protocol](client-protocol.md).

| Area | Read first | What belongs here |
| --- | --- | --- |
| Server integration | [app guidance](../../app/AGENTS.md), [main.py](../../app/main.py) | FastAPI lifespan, endpoints, queues, history, feature wiring |
| Audio input and ASR | [VAD](../../app/audio/vad.py), [Parakeet](../../app/asr/parakeet_onnx.py) | Segmentation, final/partial processing, model loading |
| Intent routing | [routing guidance](../../app/routing/AGENTS.md), [router](../../app/routing/router.py) | CHAT/CONTROL/UNKNOWN, language models, centroid decisions |
| Language models | [LLM client](../../app/llm/ollama_client.py), [tools](../../app/llm/tools.py) | Ollama/llama.cpp transport, stable tool schema and dispatch |
| Device integration | [plugin guidance](../../app/plugins/AGENTS.md), [registry](../../app/plugins/registry.py) | Lifecycle, inventory, candidate ranking, MQTT execution |
| Speech output | [speech orchestration](../../app/speech_output/speech_output.py), [Sonos sink](../../app/audio_sink/sonos_http_sink.py) | Chunked Piper synthesis, normalization, playback, echo events |
| Speaker identity | [recognizer](../../app/speaker_recognition/recognizer.py) | In-memory finalized segments; enrollment remains client tooling |
| Client | [client guidance](../../client/macos/AGENTS.md), [client README](../../client/macos/README.md) | PortAudio capture, soxr resampling, streaming, enrollment |
| ESP32 client | [firmware guidance](../../client/esp32/AGENTS.md), [firmware README](../../client/esp32/README.md) | PDM capture, bounded LAN streaming, hardcoded settings, no local inference/playback |
| Live monitor | [monitor guide](../../monitor/README.md), [client protocol](client-protocol.md) | Read-only log WebSocket, client status/filtering, correlated speaker evidence; separate LAN audio ingress |
| Evaluation | [test guidance](../../tests/AGENTS.md), [routing benchmarks](../../tests/routing/README.md) | Offline regression versus model benchmarks versus live integration |

## Implementation boundaries that matter

- Speaker recognition returns the nearest candidate's base user independently of
  the `known` threshold flag, restoring deployed behavior by maintainer instruction
  on 2026-10-18. Confidence remains logged; this is not a verified identity guarantee.
  Self-speech rejection separately requires `known` and a finite score at or above
  `ARIA_SPK_SELF_MIN_SCORE`. An uncertain self candidate becomes `unknown` for the
  request, with raw candidate/score retained. Echo Guard compares captured onset to
  recent TTS; unknown speakers additionally use a bounded opening match. See
  [feedback rejection](feedback-prefix-2026-10-18.md).
- The development-only live adapter enables 12 session history messages and exact
  request/reply logging, with request IDs propagated to Mac audio playback. Its
  speaker model is prepared locally before offline startup; production modules
  do not import this adapter. See the [live guide](../../client/macos/README.md).
- Live startup primes the system prompt plus `TOOLS` with one token before readiness;
  output is discarded without history, tool dispatch or speech. A fixed Piper message
  preserves that prefix. See [warm-up evidence](warmup-2026-10-18.md).
- Every Ollama request sends `keep_alive_s` (default `-1`, configurable through
  `ARIA_LLM_KEEP_ALIVE_S`). This retains model/cache without periodic inference,
  not arbitrary history or immunity to eviction. See [idle evidence](idle-residency-2026-10-18.md).
- Live-only tools use a private copy of cached Apollo inventory with no discovery
  sources and `LogOnlyActions`, which validates/logs payloads without MQTT. The
  shared voice/text processor selects supported explicit commands in server code;
  the conversation model can propose tools but cannot authorize an action. Model/tool/command
  events are logged; normal runtime providers are unchanged. No physical outcome
  can be inferred from this simulated execution.

- `app/main.py:lifespan` creates optional providers, LLM resources, speech output, and
  router. Disabled features and initialization failures must remain understandable.
- `app/main.py:_process_request` supplies central `TOOLS` to the model and dispatches its
  requests through `app/llm/tools.py`. Explicit grounded controls bypass generation;
  generated execute calls without internal request authorization are rejected.
  This does not enforce all Option A invariants or enable broader control grammar.
- `InventoryProvider` normalizes devices/commands, indexes candidates, caches data,
  and ingests state updates. It does not establish action completion by itself.
- `ActionsProvider.execute_command` validates when given an execution spec, builds
  the MQTT payload, and reports publish result. An expectation tracker is specified
  in the design docs but is not part of this inspected implementation.
- `SelfLearningRouter.update_centroids` exists; the refreshed graph shows no caller
  implementing a verified-outcome learning loop. Treat self-learning as incomplete.
- Tests on the control benchmark branch use separate benchmark classifiers; their
  scores do not establish the behavior of `app/routing/router.py`.

See [settings](../ENVIRONMENT.md); verify source readers. The explicit package list
in [pyproject.toml](../../pyproject.toml) includes routing and speaker recognition.
The merge check built a wheel and imported the packaged app outside the checkout.

Ollama startup prepares a local alias when the installed model template renders
`.Function` as a Go struct: it renders the whole tool as JSON and quotes the
function-name example. Existing weights/settings are reused; no pull or request-time
repair is added. Lookup results group commands by device, declare current state
unknown, and carry the request language. Paired evaluation remains necessary.

Read-only `find_devices` batches now finish through `app/llm/lookup_reply.py` without
a second generation pass. It renders inventory capabilities in the request language,
keeps same-named device identities separate, exposes truncation/failures and never
claims current state or completed control. Model selection and first-call prompt/tool
schema remain unchanged; this is not expanded control grammar or a decision on
Option A. See [lookup latency evaluation](lookup-latency-2026-10-18.md).

`monitor.server` reads bounded local logs and polls `/api/clients` via Unix socket;
it never enters the inference queue. `monitor.ingress` separately forwards LAN
microphones to the log-only development server (currently disabled for audio-free tests).
The monitor can mute/unmute client input; monotonic expiry defaults to five minutes.
The shared processor handles a spoken mute request with duration, then requires a
same-session/speaker confirmation. No model/device tool authorizes mute. Buffered
speech is discarded across mute changes, while accepted jobs finish. See
[client mute](client-mute-2026-10-18.md) for scope, grammar and verification limits.
`ARIA.SPK.Request` joins similarity to its exact request, avoiding a global
latest-speaker guess across simultaneous clients.
The monitor's Clear offline control hides disconnected registry rows until their
next claim. Mute deadlines, session keys, pending requests and logs are preserved;
it does not disconnect clients or clear conversation history. The layout prioritizes
the live log viewport, with compact status/metrics and expandable help.
