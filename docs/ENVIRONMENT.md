# Aria — Environment Variables Reference

This document lists **all supported environment variables** for the Aria server and client,
their purpose, defaults, and expected values.

Unset variables use the documented defaults.

---

## Server Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_HOST | 0.0.0.0 | Bind host for the API server |
| ARIA_PORT | 8000 | Bind port for the API server |
| ARIA_WORKERS | 1 | Number of server workers (processes) |
| ARIA_LOG_LEVEL | info | Log level for the server (uvicorn) |
| ARIA_REQUEST_QUEUE_CAPACITY | 1 | Pending voice/text requests behind one shared inference worker; integer 1–16. Full queues reject new work, never evict accepted work |
| ARIA_REQUEST_MAX_WAIT_S | 30 | Maximum queue age before execution; greater than 0 and at most 300 seconds. Expired work is not executed |

The shared queue is per server process. Use one server worker on Apollo; increasing
`ARIA_WORKERS` multiplies model/queue resources and does not provide global admission.
Invalid queue settings leave the optional LLM unavailable. HTTP overload returns
429, expiry returns 504, processing failures return 500; responses include
`request_id` and `status`. Voice clients
receive `request_status` frames for admission and terminal outcomes. Accepted voice
work survives a socket disconnect; cancelling an HTTP handler skips its queued work
but does not retry or revoke a running action. Shutdown-interrupted work may have
already produced side effects and must not be blindly retried. Request completion
does not mean physical device completion or completed audio playback.

## Conversation activation

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_WAKE_ENABLED | 1 | Gate both voice/text processing; `0`, `false`, or `off` explicitly selects legacy ungated behavior |
| ARIA_CONVERSATION_IDLE_S | 30 | Finite 1–600 seconds of inactivity after server-estimated playback ends; voice uses server-observed speech onset before ASR |
| ARIA_WAKE_DEFAULT_LANGUAGE | fr | `fr` or `en` for a bare name without a language-bearing greeting |

With activation enabled, say “Aria” or “Bonjour/Hello Aria” once; include the request
in the same utterance or speak it next. Follow-ups need no name. “Au revoir” /
“Goodbye” closes the conversation. Full supported endings and recognition limits:
[wake flow](agents/wake-flow-2026-10-18.md). Prefer “Bonjour Aria” / “Hello Aria”
for the current ASR: bare-name clips remain unreliable. Voice-only bounded name
canonicalization and remaining errors are documented in the
[spoken-wake follow-up](agents/spoken-wake-2026-10-18.md).

Each WebSocket has its own conversation. HTTP callers must reuse `session_id`
(nonempty, at most 256 characters) for follow-ups; stateless requests need the name
every time. Successful ignored requests return an empty response, `ignored: true`,
`conversation_action: ignored`, and no speech/tool execution. Completion is not an
action confirmation. The gate retains at most 256 sessions and clears history/device
references on expiry, explicit closure, or eviction. Invalid activation settings fail
startup; use one server worker to keep session state consistent.

## Models & Paths

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_MODELS_DIR | (required) | Directory containing ASR models (e.g., Parakeet ONNX) |

---

## Binary Dependencies

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_PIPER_BIN | piper | Path to Piper TTS binary |
| ARIA_FFMPEG_BIN | ffmpeg | Path to ffmpeg binary for audio normalization |

---

## Voice Activity Detection (VAD)

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_VAD_AGGR | 2 | WebRTC VAD aggressiveness (0-3) |
| ARIA_VAD_START_FRAMES | 3 | Consecutive speech frames to start utterance |
| ARIA_VAD_MIN_RMS | 0.0 | RMS threshold (0 = disabled) |
| ARIA_VAD_SNR_DB | 0.0 | SNR threshold in dB (0 = disabled) |
| ARIA_VAD_NOISE_ALPHA | 0.05 | EMA coefficient for adaptive noise floor |
| ARIA_VAD_SILENCE_MS | 700 | Silence duration to finalize utterance |

---

## Automatic Speech Recognition (ASR)

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_ASR_MIN_SECONDS | 0.30 | Minimum audio duration to run ASR |
| ARIA_ASR_MIN_RMS | 0.004 | Minimum RMS to run ASR |
| ARIA_ASR_MIN_SPEECH_RATIO | 0.18 | Minimum speech ratio to run ASR |
| ARIA_ASR_DROP_FILLERS | 1 | Drop filler words (yeah, ok, um, etc.) |
| ARIA_ASR_FILLER_MAX_SECONDS | 0.9 | Max duration to consider as filler |
| ARIA_ASR_FILLER_MIN_SPEECH_RATIO | 0.35 | Min speech ratio for filler detection |

---

## Partial Transcripts (Server-side)

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_PARTIAL_EVERY_MS | 0 | Print partial transcripts every N ms (0 = disabled) |
| ARIA_PARTIAL_WINDOW_MS | 1500 | Audio window for partial transcripts |

---

## Speaker Recognition

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_SPK_ENABLED | 1 | Enable speaker recognition |
| ARIA_SPK_PROFILES_DIR | /var/aria/profiles | Directory containing speaker profile JSON files |
| ARIA_SPK_MODEL | speechbrain/spkrec-ecapa-voxceleb | SpeechBrain embedding model |
| ARIA_SPK_DEVICE | cpu | Compute device: cpu or cuda |
| ARIA_SPK_NUM_THREADS | 4 | Number of worker threads |
| ARIA_SPK_MIN_SECONDS | 1.0 | Minimum audio duration for recognition |
| ARIA_SPK_THRESHOLD_DEFAULT | 0.65 | Similarity threshold for known speaker |
| ARIA_SPK_SELF_MIN_SCORE | 0.75 | Self-speech rejection requires `known` and a finite score at least this high, including base user `aria`; uncertain self candidates proceed as `unknown` through Echo Guard |
| ARIA_SPK_SELF_NAMES | aria_fr,aria_en | Comma-separated self identity names |
| ARIA_SPK_TIMEOUT_SEC | 1.0 | Recognition timeout |
| ARIA_SPK_HOT_RELOAD | 0 | Hot reload profiles on change |
| ARIA_SPK_DEBUG | 0 | Enable speaker recognition debug logging |
| ARIA_SPK_DEBUG_DUMP_WAV | 0 | Dump audio segments for debugging |

---

## Echo Guard v2 (Text-based Feedback Suppression)

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_ECHO_V2_ENABLED | 1 | Enable echo guard v2 |
| ARIA_ECHO_V2_TAIL_SEC | 2.0 | Whole-text similarity tail after estimated playback, evaluated at speech onset |
| ARIA_ECHO_V2_UNKNOWN_PREFIX_TAIL_SEC | 8.0 | Unknown-speaker opening-match tail; clamped to 0–30 seconds; accommodates output transport/player delay |
| ARIA_ECHO_V2_MIN_WORDS | 5 | Minimum words in transcript to check |
| ARIA_ECHO_V2_MIN_CHARS | 20 | Minimum characters in transcript to check |
| ARIA_ECHO_V2_CONTAINMENT_THRESHOLD | 0.65 | Token containment threshold (0-1) |
| ARIA_ECHO_V2_FUZZY_THRESHOLD | 0.75 | Fuzzy similarity threshold (0-1) |
| ARIA_ECHO_V2_ALLOWLIST_REGEX | (empty) | Regex for interrupt/bypass words |

The opening-match path requires effective speaker `unknown`, recent TTS and three
matching normalized opening words. A short numeric reply can match two opening words
only with a numeric continuation (digits or supported FR/EN numeral words). It runs
before whole-text length limits; it does not change person attribution. Both paths
use server-captured speech onset, with at most 60 seconds of ASR processing delay,
and each candidate's own playback interval. TTS duration comes from the generated
WAV, with a word estimate only if WAV metadata cannot be read. These are estimated
playback windows, not acoustic cancellation or physical playback acknowledgments.

---

## Echo Guard (Legacy v1)

Aria still supports a legacy text-level echo suppression mechanism (v1).
Prefer **Echo Guard v2** unless you specifically need v1 tuning.

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_ECHO_GUARD_ENABLED | 1 | Enable legacy echo guard |
| ARIA_ECHO_GUARD_WINDOW_SEC | 15.0 | Time window (seconds) to compare against recent TTS |
| ARIA_ECHO_GUARD_MIN_CHARS | 25 | Minimum characters in transcript to evaluate |
| ARIA_ECHO_GUARD_MIN_WORDS | 6 | Minimum words in transcript to evaluate |
| ARIA_ECHO_GUARD_SEQ_THRESHOLD | 0.85 | Sequence similarity threshold (0–1) |
| ARIA_ECHO_GUARD_JACC_THRESHOLD | 0.70 | Jaccard similarity threshold (0–1) |
| ARIA_ECHO_GUARD_PARTIAL_THRESHOLD | 0.78 | Partial match threshold (0–1) |
| ARIA_ECHO_GUARD_OVERLAP_THRESHOLD | 0.60 | Word overlap threshold (0–1) |
| ARIA_ECHO_GUARD_SPEAKING_GRACE_SEC | 6.0 | Grace period while speaking (seconds) |
| ARIA_ECHO_GUARD_STRICT_WHILE_SPEAKING | 1 | Strict suppression during speaking |

---

## Plugin System

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_PLUGINS_ENABLED | 0 | Enable plugin system (required for tool calling) |
| ARIA_INVENTORY_ENABLED | 0 | Enable inventory provider (device discovery) |
| ARIA_ACTIONS_ENABLED | 0 | Enable actions provider (command execution) |

---

## Router (CHAT/CONTROL Classification)

The router classifies STT transcripts as CHAT (conversation) or CONTROL (device action) before LLM processing.
Uses language-specific embedding models for improved accuracy.

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_ROUTER_ENABLED | 0 | Enable router classification |
| ARIA_ROUTER_MODEL_EN | BAAI/bge-small-en-v1.5 | Embedding model for English |
| ARIA_ROUTER_MODEL_FR | intfloat/multilingual-e5-small | Embedding model for French |
| ARIA_ROUTER_CONTROL_THRESHOLD | 0.70 | Confidence threshold for CONTROL classification (0-1) |
| ARIA_ROUTER_CHAT_THRESHOLD | 0.70 | Confidence threshold for CHAT classification (0-1) |
| ARIA_ROUTER_UNKNOWN_THRESHOLD | 0.50 | Distance threshold for UNKNOWN classification |
| ARIA_ROUTER_USE_SEED_DATA | 1 | Initialize centroids from built-in seed examples |
| ARIA_ROUTER_LANG_DETECT | heuristic | Language detection method: heuristic or langid |

**How it works:**
1. Detects language (FR/EN) from STT transcript
2. Selects language-specific embedding model
3. Compares to pre-computed CHAT/CONTROL centroids
4. CHAT → fast response only for explicit known greetings/time questions; other conversation uses the shared LLM path
5. CONTROL/UNKNOWN → proceeds to LLM with tool calling

---

## Inventory Provider

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_INVENTORY_CACHE_PATH | ./inventory.json | Path to cache device inventory JSON |
| ARIA_INVENTORY_REFRESH_INTERVAL_SEC | 300 | Seconds between inventory refreshes (0 = only on startup) |

---

## Jeedom MQTT Integration

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_JEEDOM_MQTT_HOST | 127.0.0.1 | Jeedom MQTT broker host |
| ARIA_JEEDOM_MQTT_PORT | 1883 | Jeedom MQTT broker port |
| ARIA_JEEDOM_MQTT_ROOT_TOPIC | jeedom | Root MQTT topic for Jeedom |
| ARIA_MQTT_QOS_SUBSCRIBE | 1 | QoS level for MQTT subscriptions (0-2) |
| ARIA_MQTT_QOS_PUBLISH | 1 | QoS level for MQTT publishes (0-2) |

---

## LLM Integration

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_LLM_PROVIDER | (none) | LLM provider: ollama or llamacpp |
| ARIA_LLM_URL | http://localhost:11434 | LLM endpoint URL (Ollama) or http://127.0.0.1:8001/v1 (llama.cpp) |
| ARIA_LLM_MODEL | qwen2.5:3b-instruct | Model name (Ollama) or local (llama.cpp). Select using Apollo quality, tool and latency measurements; a larger model is not automatically better. |
| ARIA_LLM_TIMEOUT_S | 12 | Request timeout in seconds |
| ARIA_LLM_KEEP_ALIVE_S | -1 | Ollama only: integer seconds of model residency after each request; `-1` retains the model, `0` unloads immediately, `300` restores the old five-minute policy. Applied to warm-up, ordinary/tool calls, streaming, and history prefill. |
| ARIA_LLM_NUM_PREDICT | (none) | Max generated tokens. Apollo conversation tests use 80; check truncation and tool JSON completeness when changing this bound. |
| ARIA_LLM_TEMPERATURE | (none) | Sampling temperature (recommend 0.1 for tool calling) |
| ARIA_LLM_TOP_P | (none) | Top-p sampling (recommend 0.85 for tool calling) |
| ARIA_LLM_STREAM | 0 | Enable streaming responses |
| ARIA_LLM_SYSTEM_PROMPT | You are ARIA... | System prompt for LLM (must include tool calling rules) |
| ARIA_LLM_HISTORY_MAX_MESSAGES | 8 | Max retained messages, counting user and assistant separately |
| ARIA_LLM_DEBUG | 0 | Enable debug logging for LLM |
| ARIA_LLM_DEBUG_MAX_CHARS | 800 | Max characters to log in debug mode |

---

## Text-to-Speech (TTS)

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_TTS_ENABLED | 0 | Enable TTS |
| ARIA_TTS_ENGINE | piper | TTS engine (currently only piper supported) |
| ARIA_TTS_VOICE | (empty) | Default voice model path for Piper |
| ARIA_TTS_RATE | 1.0 | Speech rate (1.0 = normal, >1 faster, <1 slower) |
| ARIA_TTS_CACHE_DIR | /tmp/aria_tts_cache | Cache directory for TTS audio files |
| ARIA_TTS_TIMEOUT_SEC | 10 | TTS generation timeout |
| ARIA_TTS_PIPER_SPEAKER | (empty) | Piper speaker ID (for multi-speaker models) |
| ARIA_TTS_VOLUME_DEFAULT | 0.3 | Default volume gain (0.0-2.0) |
| ARIA_TTS_CHUNKING | 1 | Enable text chunking for lower latency |
| ARIA_TTS_MAX_CHARS_PER_CHUNK | 220 | Maximum characters per TTS chunk |
| ARIA_TTS_MIN_CHUNK_CHARS | 60 | Minimum characters per chunk |

---

## Multi-language TTS (Auto Voice Selection)

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_TTS_VOICE_MAP | (empty) | Voice mapping: "en:/path/en.onnx,fr:/path/fr.onnx" |
| ARIA_TTS_LANG_DETECT | heuristic | Local FR/EN detection from each request: heuristic or optional installed langid; unavailable langid falls back to heuristic |
| ARIA_TTS_LANG_SHORT_REUSE_CHARS | (removed) | Ignored; language is no longer inherited from another response |

The shared voice/text processor detects language from the current input, then passes
it to TTS for both model replies and shortcuts. Every response chunk uses that
language, including numeric replies. Direct `SpeechOutput.speak` callers can supply
`language="en"` or `"fr"`; otherwise detection uses the full current reply. Ambiguous
text defaults to English; this lightweight detector is not a general multilingual
classifier or an interpreter for instructions such as “answer in another language.”

---

## Sonos Output

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_TTS_SINK | sonos_http | Audio sink type (currently only sonos_http) |
| ARIA_SONOS_IP | (optional) | Override Sonos speaker IP (discovery used when unset) |
| ARIA_SONOS_NAME | (empty) | Friendly Sonos name to target during discovery (e.g., "Living Room") |
| ARIA_SONOS_VOLUME | 30 | Sonos speaker volume (0-100), set before each TTS playback |
| ARIA_SONOS_DISCOVERY_TIMEOUT | 5.0 | Seconds to wait while discovering speakers via SoCo |
| ARIA_HTTP_BASE_URL | (required) | Base URL for HTTP audio serving (e.g., http://192.0.2.10:8000) |

---

## Server Clock

Server local clock answers use `ARIA_TIMEZONE` (an IANA timezone such as
`Europe/Zurich`). When unset, the host timezone applies. An explicit Zurich or UTC
request overrides it; an invalid configured zone produces an unavailable reply,
never an estimated time from the model. These answers do not require router enablement.

## Ollama model residency

Aria explicitly sends `keep_alive` on every Ollama request. The default `-1` keeps
the small model loaded across idle periods, preserving the opportunity to reuse
its prompt cache without periodic dummy inference. This overrides Ollama's daemon
default for Aria's requests only; llama.cpp is unchanged. Invalid values (including
duration strings such as `5m`) fail configuration loading; use integer seconds.

Residency consumes model/cache RAM until Ollama unloads or evicts the model or its
process stops. Another client/model can still replace it; this is not a memory
reservation or a guarantee that every history prefix stays cached. Stopping Aria
alone does not unload a model from a separately managed/shared Ollama process.
Use a bounded positive value when that tradeoff is intended. Do not add periodic
generation to keep it warm. See the [Apollo idle comparison](agents/idle-residency-2026-10-18.md).

## History prefill

`ARIA_HISTORY_PREFILL=0` disables opportunistic Ollama history prefill (default on).
After a local clock/capability reply has prepared all audio, Aria may prefill the
exact bounded history and tools while the last audio chunk plays. This requires a
model turn in the same session within 60 seconds. One discarded token, one task,
and a four-second inference timeout bound the work; new processing and shutdown
cancel it. Pending work, missing/failed speech, and other providers skip it.
It neither dispatches tools nor changes history. This shifts compute away from the
next response; it does not eliminate inference cost or guarantee cache residency.

## Client-side Environment Variables

These variables are used by the macOS client (`client/macos/mic_stream.py`).

### Client Debug & Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_CLIENT_DEBUG | 0 | Enable client debug statistics output |

ESP32 settings are compile-time constants in
[settings.h](../client/esp32/main/settings.h); the environment variables below
apply only to the Mac. See the [feature parity matrix](../client/esp32/README.md)
for different default gain policies and playback timing limits.

### Client-side VAD

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_CLIENT_VAD | 0 | Enable client-side VAD (reduces bandwidth) |
| ARIA_CLIENT_VAD_AGGR | 2 | Client VAD aggressiveness (0-3) |
| ARIA_CLIENT_VAD_SILENCE_MS | 800 | Audio silence to end a turn; clamped to 200..3000 ms. Only applies when client VAD is enabled |
| ARIA_CLIENT_VAD_PRE_MS | 200 | Pre-roll audio before speech start |
| ARIA_CLIENT_VAD_START_FRAMES | 3 | Consecutive frames to start utterance |
| ARIA_CLIENT_VAD_MIN_RMS | 0.0 | RMS threshold to ignore noise |
| ARIA_CLIENT_VAD_SNR_DB | 0.0 | SNR threshold vs adaptive noise floor |
| ARIA_CLIENT_VAD_NOISE_ALPHA | 0.05 | EMA coefficient for noise estimation |

### Client Audio Gain

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_CLIENT_GAIN | (none) | Fixed gain multiplier (disables AGC) |
| ARIA_CLIENT_AGC | varies | Enable AGC (default: 1 if VAD off, 0 if VAD on) |
| ARIA_CLIENT_TARGET_RMS | 0.05 | Target RMS for AGC |
| ARIA_CLIENT_MAX_GAIN | 50 | Maximum AGC gain multiplier |

## Named clients and local monitoring

- `ARIA_CLIENT_ID`: Mac microphone's unique, stable name; default
  `aria-mac-<lowercase-hostname>`. One to 64 lowercase letters/digits, dots,
  underscores or hyphens, starting with a letter/digit. Names are not authentication.
- ESP32 uses its DHCP hostname (prefix plus full Wi-Fi MAC hex) as `client_id`.
- `ARIA_MONITOR_ACTIVE`: local active-run JSON pointer, default
  `/data/aria/monitor/active.json`.
- `ARIA_MONITOR_RUN_ROOT`: allowed parent of run directories, default
  `/data/aria/e2e/runs`.
- Ollama startup may create a local `aria-tools-<hash>:latest` template alias with
  existing weights; the model store must allow local alias creation. No model pull
  occurs here. Missing local models/template preparation errors disable the optional
  LLM through its existing initialization failure path. The isolated launcher uses
  a writable per-run store linked to cached blobs/manifests.

See [named-client protocol](agents/client-protocol.md) and [monitor operations](../monitor/README.md).


Sentence voice routing uses the existing `ARIA_TTS_VOICE_MAP` French/English models.
`ARIA_TTS_CHUNKING=0` disables size grouping, but retains sentence boundaries when
the voice changes. No detector dependency or additional model is required.

## Optional local Jeedom reads (development adapter)

`ARIA_JEEDOM_READ_SOCKET` defaults to unset. In `scripts.e2e.live_server`, set it to
an absolute Unix-domain socket path to select `ReadOnlyInventory` instead of the
cached-inventory provider. The bridge must already be running and accessible to
the server process; an unreachable socket fails adapter startup. It exposes
`GET /catalog` and `POST /read` for enabled info commands only, never actions.
The adapter's action provider remains log-only. The ordinary server CLI does not
consume this switch.

Prepare the bridge with a local inventory cache and an owned private-LAN MQTT
broker address, for example (substitute local paths/address):

```sh
python -m app.plugins.inventory.read_bridge --cache /path/to/inventory.json \
  --host 192.168.1.2 --socket /path/to/jeedom-read.sock
export ARIA_JEEDOM_READ_SOCKET=/path/to/jeedom-read.sock
```

The bridge uses
restrictive socket permissions (`umask 077`), so run it under the same account or
provide deliberately scoped local access. Start it before the adapter; stop the
adapter before the bridge. Runtime calls stay on owned LAN/Unix sockets; the
bridge does not provide forecast inference, a cloud fallback or action execution.
Tests must use a synthetic read fixture and captured output, not this live broker.
