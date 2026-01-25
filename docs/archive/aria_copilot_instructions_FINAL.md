# ARIA — Final Copilot Instructions (Authoritative)

This document consolidates and supersedes:
- `.copilot-instructions.md` (initial)
- `aria-copilot-instructions-updated.md` (updated)
- `aria_copilot_instructions_part4.md` (conversational AI + tools)

If any older instruction conflicts with this file, **follow this file**.

## Non‑Negotiables

- **Do not change existing, working audio behavior.**
- **All processing is local-only.** No cloud APIs/services.
- The WebSocket audio ingestion loop must **never block** on LLM, tools, or TTS.
- Bounded buffers/queues only; drop data when overloaded (latest wins).

---

## Architecture (Single Server)

```
macOS mic client  ──(WS /ws/asr, 16k PCM16LE mono)──▶  ARIA Server
                                                ASR (Parakeet)
                                                VAD (WebRTC)
                                                Conversation LLM (Ollama)
                                                Tool Router (Jeedom / HTTP)
                                                TTS (Piper) → normalize (ffmpeg)
                                                Sonos playback (SoCo + HTTP pull)
```

Mental model:
- Mic audio is **pushed** to ARIA via WebSocket.
- TTS audio is **pulled** by Sonos from ARIA via HTTP.

---

# PART 1 — macOS Microphone Streaming Client

Target file:
- `client/macos/mic_stream.py`

### Important context
- `ffmpeg` + AVFoundation caused timing issues (accelerated audio). **Do not use ffmpeg/avfoundation** for capture.
- Use **Python + PortAudio** (`sounddevice`) for capture.

### Capture
- Use `sounddevice`.
- Select the **default system microphone** automatically (no hardcoded device IDs).
- Capture at the device’s **native sample rate** (query it; do not force 16k at capture).
- Use `float32` internally.

### Resampling
- Use `soxr` for high-quality resampling.
- Resample from device rate → **16000 Hz**.
- Convert to PCM signed 16-bit little-endian (`pcm_s16le`) before sending.

### Streaming contract
- `asyncio` + `websockets`
- Connect to `ws://HOST:PORT/ws/asr`
- On connect send:

```json
{
  "type": "start",
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le",
  "chunk_ms": 40
}
```

- Then stream binary PCM frames.
- Target ~40 ms chunks at 16 kHz; ensure chunk boundaries are correct **after resampling**.
- Handle backpressure by **dropping frames** (no unbounded buffering).
- On Ctrl+C: send `{"type":"stop"}` and close cleanly.

### Client output (IMPORTANT)
- The client **must not rely on receiving transcripts**.
- The server is the source of truth for transcript + LLM logs.
- Client may print capture diagnostics (device name, sample rate, dropped chunk counters).

### Dependencies
Create/maintain:
- `client/macos/requirements.txt`

Must include:
- `sounddevice`, `soxr`, `numpy`, `websockets`

---

# PART 2 — Server-Side Processing of FINAL Transcribed Text

Server responsibilities:
- Accept audio on `/ws/asr`.
- Use VAD to segment utterances.
- Run ASR on utterance end.
- Print final transcript to server console.

### Transcript processing rules
- Only **FINAL** transcripts are processed by the LLM (not partials).
- Printing format (server console):

```
ARIA.TRANSCRIPT: <text>
ARIA.LLM: <llm_response>
```

### LLM architecture
- Local LLM served over HTTP (default: **Ollama**).
- ARIA does **not** embed model weights.
- LLM call must be off the WebSocket audio path (background worker).

Env:
```
ARIA_LLM_PROVIDER=ollama
ARIA_LLM_URL=http://localhost:11434
ARIA_LLM_MODEL=qwen2.5:3b-instruct
# Optional:
ARIA_LLM_TIMEOUT_S=12
```

LLM prompting:
- System: `You are ARIA, a local assistant running on a private server.`
- User: final transcript text
- Responses should be concise (1–3 sentences).

---

# PART 3 — Server-Side TTS Output to Sonos (Local, Low Latency)

## Do not use AirPlay/RAOP
AirPlay buffering adds too much latency. Use Sonos native HTTP playback.

### Objective
- Convert LLM response text into speech.
- Produce Sonos-friendly WAV, serve it over HTTP from the ARIA server.
- Command Sonos to play the URL via **SoCo**.
- Entirely local.

### Audio format requirements (Sonos-friendly)
Audio served to Sonos must be:
- WAV PCM `s16le`
- **stereo**
- **44100 Hz**
- Optional gain scaling via a `volume` parameter (default `0.3`).

Use `ffmpeg` to normalize Piper output into this format.

### TTS env
```
ARIA_TTS_ENABLED=1
ARIA_TTS_ENGINE=piper
ARIA_TTS_VOICE=<voice_id_or_path_or_model_path>
ARIA_TTS_RATE=1.0

ARIA_TTS_CHUNKING=1
ARIA_TTS_MAX_CHARS_PER_CHUNK=220
ARIA_TTS_MIN_CHUNK_CHARS=60

ARIA_TTS_SINK=sonos_http
ARIA_SONOS_IP=<sonos_speaker_ip>

ARIA_HTTP_BASE_URL=http://<aria_server_ip>:8000

ARIA_TTS_CACHE_DIR=/tmp/aria_tts_cache
ARIA_TTS_TIMEOUT_SEC=10
ARIA_TTS_VOLUME_DEFAULT=0.3
```

### Modules (TTS + HTTP + Sonos control)
Create/update:
- `app/tts/tts_engine.py`
  - `synthesize_to_wav(text: str, *, voice: str, rate: float) -> Path`
- `app/tts/piper_engine.py`
  - Piper via subprocess, cached by hash(text+voice+rate), timeouts, graceful failure
- `app/tts/audio_normalizer.py`
  - `normalize_for_sonos(in_path: Path, out_path: Path, volume: float) -> Path`
  - uses `ffmpeg` (`-ac 2 -ar 44100 -sample_fmt s16`, `-filter:a volume=<volume>`)
- `app/tts/text_chunker.py`
  - sentence-boundary chunking; summarize code blocks; avoid long URLs
- `app/http_audio/server.py`
  - serves `GET /tts/<key>.wav` as `audio/wav`
  - do not delete files while Sonos may fetch them
- `app/audio_sink/base.py`
  - `play_url(url: str) -> Handle`
  - `stop(handle)`
  - `stop_all()`
- `app/audio_sink/sonos_http_sink.py`
  - SoCo wrapper (`play_uri`, `stop`), async-friendly
- `app/speech_output/speech_output.py`
  - orchestrates chunking → synth → normalize → cache → URL build → sequential playback
  - API: `speak(text: str, volume: float | None = None)`, `interrupt()`

### Barge-in
On user speech start (VAD):
- Immediately stop Sonos playback.
- Clear queued chunks.
- No attempt to resume previous Sonos playback.

---

# PART 4 — Conversational AI + Tool Execution (Local, Xeon)

## Objective
- Maintain per-session conversation memory.
- Allow tool execution (Jeedom/APIs) only when needed.
- Keep LLM + tools off the WebSocket audio thread.

## LLM output contract (STRICT)
The LLM must output exactly one of:

A) Assistant message (spoken):
- Plain text only.

B) Tool call JSON (no markdown, no extra text):
```json
{
  "type": "tool_call",
  "name": "jeedom.execute",
  "arguments": {
    "cmd_id": 1234,
    "value": null
  }
}
```

If parameters are missing, the assistant asks a clarifying question instead of calling tools.

## Conversation state
Create:
- `app/conversation/state.py`
- `app/conversation/manager.py`

Env:
```
ARIA_CONVO_MAX_TURNS=20
ARIA_CONVO_MAX_CHARS=12000
ARIA_CONVO_SUMMARY_ENABLED=0
```

## Tools framework
Create:
- `app/tools/types.py`
- `app/tools/registry.py`
- `app/tools/jeedom.py`
- `app/tools/http_tool.py`

Rules:
- allowlisted tools only
- async execution
- failures are safe and logged; never crash server

Jeedom env:
```
ARIA_TOOLS_ENABLED=1
ARIA_JEEDOM_URL=http://jeedom.local
ARIA_JEEDOM_API_KEY=xxxx
ARIA_JEEDOM_TIMEOUT_S=8
```

## Device mapping (optional)
- `config/devices.json`
- load into `app.state.device_map`

## LLM client upgrade
Update:
- `app/llm/ollama_client.py`

Add:
- `generate_chat(messages: list[dict], config, client=None) -> str`

Use `/api/chat`.
System prompt must include:
- ARIA identity
- strict tool-call rules
- concise style
- available tools + device aliases

## Queue payload change
Change final transcript enqueue to include `session_id`:
- from `_enqueue_final_llm(app, transcript)`
- to `_enqueue_final_llm(app, session_id, transcript)`

Queue item:
```python
{"session_id": session_id, "transcript": text}
```

Queue stays bounded; latest transcript wins.

## LLM worker pipeline
In `_llm_worker(app)`:
1) Get session state
2) Add user transcript
3) Build chat context
4) Call LLM
5) Parse response

If normal message:
- add assistant message
- log `ARIA.LLM: ...`
- if `ARIA_TTS_ENABLED=1`, speak via `speech_output.speak()`

If tool_call:
- log `ARIA.TOOL.CALL: ...`
- validate tool name + arguments
- execute tool
- log `ARIA.TOOL.RESULT: ...`
- add tool result to conversation
- call LLM again for a final spoken confirmation

Logging format:
```
ARIA.TRANSCRIPT: ...
ARIA.LLM: ...
ARIA.TOOL.CALL: ...
ARIA.TOOL.RESULT: ...
ARIA.LLM: ...
```

---

## Acceptance tests (behavioral)
1) Memory: “My name is Sebastien” → “What’s my name?”
2) Tool execution: “Turn on living room light” → Jeedom call + spoken confirmation
3) Clarification: “Set thermostat” → asks temperature
4) Queue stability: rapid speech does not crash
5) Barge-in preserved: speech interrupts TTS immediately

```
