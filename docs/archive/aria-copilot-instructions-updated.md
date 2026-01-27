> **ARCHIVED / OBSOLETE**
>
> This file is kept for historical reference only and may conflict with current project behavior.
> Use **[.copilot-instructions.md](../../.copilot-instructions.md)** as the single authoritative specification.

# ARIA Copilot Instructions — ASR, LLM, and Local TTS (Sonos)

Use the instructions defined in `$HOME/dev/aria/.copilot-instructions.md` as the authoritative project specification.

This task has **THREE parts**:

1. macOS microphone streaming client  
2. Server-side processing of FINAL transcribed text using a local LLM  
3. Server-side TTS output to Sonos (local-only, low latency)

> **Important**
> - Do NOT change existing, working audio behavior.
> - All processing must remain local (no cloud or external services).

---

## PART 1 — macOS MICROPHONE STREAMING CLIENT

Create the macOS microphone streaming client at:

```
client/macos/mic_stream.py
```

### IMPORTANT CONTEXT (DO NOT IGNORE)
- `ffmpeg` + AVFoundation caused audio timing issues (accelerated speech)
- SoX `rec` worked correctly
- Therefore: **DO NOT use ffmpeg or avfoundation**
- Use **Python + PortAudio** instead

### REQUIREMENTS

#### Audio capture
- Use `sounddevice` (PortAudio backend)
- Select the default system microphone automatically
- Capture at the device’s **native sample rate** (query it, do NOT force 16k at capture)
- Use `float32` internally

#### Resampling
- Use `soxr` for high-quality resampling
- Resample audio from device rate → **16000 Hz**
- Convert to PCM signed 16-bit little-endian before sending

#### Streaming
- Use `asyncio` + `websockets`
- Connect to: `ws://HOST:PORT/ws/asr`
- On connect, send JSON:

```json
{
  "type": "start",
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le",
  "chunk_ms": 40
}
```

- Then stream binary PCM frames
- Drop audio frames if websocket backpressure occurs (**do NOT buffer unbounded**)
- On Ctrl+C, send `{"type":"stop"}` and close cleanly

#### Chunking
- Target ~40 ms chunks at 16 kHz
- Ensure chunk boundaries are correct after resampling

#### Client Output
- Print partial transcripts as they arrive
- Print final transcripts on completion
- Prefix output with **`ARIA.TRANSCRIPT`**

#### Structure
- No GUI
- CLI usage:
  ```
  python mic_stream.py ws://localhost:8000/ws/asr
  ```
- Clean, readable, production-quality code
- Add comments explaining:
  - why capture is at native rate
  - why resampling is done client-side
  - why ffmpeg is not used

#### Dependencies
- sounddevice
- soxr
- numpy
- websockets

Also create:
```
client/macos/requirements.txt
```

#### Do NOT
- use ffmpeg
- hardcode device IDs
- hardcode sample rates except 16000 as output
- block the event loop

---

## PART 2 — SERVER-SIDE PROCESSING OF TRANSCRIBED TEXT

### Behavior
- ONLY FINAL transcripts (not partials) are processed
- Final transcript text is sent to a local LLM
- The LLM response is printed to the server console
- No TTS or audio playback yet (console output only)

### LLM ARCHITECTURE REQUIREMENTS
- Use a local LLM served over HTTP (default: Ollama)
- Do NOT embed model weights in the ARIA process
- Communication via HTTP POST only
- The LLM call must be **non-blocking** with respect to audio streaming

### LLM CONFIGURATION (ENV)
```
ARIA_LLM_PROVIDER=ollama
ARIA_LLM_URL=http://localhost:11434
ARIA_LLM_MODEL=qwen2.5:3b-instruct
```

### SERVER IMPLEMENTATION

Create:
```
app/llm/ollama_client.py
```

Responsibilities:
- send a prompt to the local LLM
- return generated text
- handle timeouts and HTTP errors gracefully

Prompting rules:
- System: *"You are ARIA, a local assistant running on a private server."*
- User: final transcribed text
- Keep responses concise (1–3 sentences max)

### Integration point
When ASR emits a FINAL transcript:

1. Print transcript  
2. Call LLM  
3. Print LLM output  

Console format:
```
ARIA.TRANSCRIPT: <text>
ARIA.LLM: <llm_response>
```

### Non-functional
- LLM processing must NOT block audio ingestion
- LLM errors must NOT crash the server
- LLM must be easy to disable via env var

---

## PART 3 — SERVER-SIDE TTS OUTPUT TO SONOS (LOCAL ONLY, LOW LATENCY, HTTP)

### Architecture choice: **Option 1 — Single Server**
The **same Aria server process** that receives microphone audio and runs ASR/LLM must also:
- host a small **HTTP audio endpoint** that Sonos can fetch from (pull model)
- control Sonos playback via **SoCo** (`play_uri`, `stop`)

**Do not** use AirPlay/RAOP for Aria speech output (too much buffering / latency).

**Mental model**
- Mic audio is **pushed** to Aria (client → server)
- TTS audio is **pulled** by Sonos (Sonos → Aria HTTP)

---

### Objective
- Convert the LLM response text into speech
- Output speech to a Sonos speaker using **Sonos native HTTP playback**:
  1) Aria generates a Sonos-friendly WAV file (cached)
  2) Aria serves it over HTTP
  3) Aria tells Sonos to play the URL via SoCo `play_uri(url)`
- Entirely local (no cloud services)
- Target low perceived latency using **chunked synthesis**
- Overriding Sonos playback is acceptable (no resume required)

---

### TTS ENGINE REQUIREMENTS
- Default engine: **Piper** (local, CPU-based)
- No cloud TTS services
- Engine invoked via subprocess (non-blocking / async-friendly)
- Output WAV PCM 16-bit (preferred)

### AUDIO FORMAT REQUIREMENTS (Sonos-friendly)
Ensure the audio served to Sonos is:
- WAV PCM **s16le**
- **stereo**
- **44100 Hz**
- Optional gain scaling via `volume` parameter (default 0.3)

Use `ffmpeg` to normalize Piper output into the required format.

---

### TTS CONFIGURATION (ENV)
```
ARIA_TTS_ENABLED=1
ARIA_TTS_ENGINE=piper
ARIA_TTS_VOICE=<voice_id_or_path_or_model_path>
ARIA_TTS_RATE=1.0

# Chunking to reduce perceived latency
ARIA_TTS_CHUNKING=1
ARIA_TTS_MAX_CHARS_PER_CHUNK=220
ARIA_TTS_MIN_CHUNK_CHARS=60

# Output sink
ARIA_TTS_SINK=sonos_http

# Sonos control
ARIA_SONOS_IP=<sonos_speaker_ip>                 # e.g. 192.168.100.222

# HTTP server for Sonos to fetch audio (must be reachable by Sonos)
ARIA_HTTP_BASE_URL=http://<aria_server_ip>:8000  # e.g. http://192.168.100.100:8000

# Caching
ARIA_TTS_CACHE_DIR=/tmp/aria_tts_cache
ARIA_TTS_TIMEOUT_SEC=10

# Default volume scaling applied at normalization time (0.0–2.0)
ARIA_TTS_VOLUME_DEFAULT=0.3
```

---

### MODULES TO CREATE / UPDATE

#### 1) `app/tts/tts_engine.py`
- Interface:
  - `synthesize_to_wav(text: str, *, voice: str, rate: float) -> Path`
- Responsible only for generating **raw** WAV via the selected engine.

#### 2) `app/tts/piper_engine.py`
- Piper-based implementation
- Cache raw outputs by hash(text + voice + rate)
- Enforce timeouts
- Fail gracefully with actionable logs

#### 3) `app/tts/audio_normalizer.py` (NEW)
- Convert raw WAV → Sonos-friendly WAV
- Uses `ffmpeg` subprocess
- Interface:
  - `normalize_for_sonos(in_path: Path, out_path: Path, volume: float) -> Path`
- Required normalization:
  - `-ac 2 -ar 44100 -sample_fmt s16`
  - volume filter: `-filter:a volume=<volume>`

#### 4) `app/tts/text_chunker.py`
- Split text into speakable chunks
- Prefer sentence boundaries
- Replace code blocks with short spoken summaries
- Avoid speaking long URLs

#### 5) `app/http_audio/server.py` (NEW) — HTTP audio endpoint (same Aria server)
- Expose cached/normalized audio for Sonos to fetch
- Must serve stable URLs, e.g.:
  - `GET /tts/<key>.wav` → returns normalized WAV
- Must not delete files while Sonos is fetching them
- Set `Content-Type: audio/wav`
- Bind to the main API host/port (single-server)

> Implementation hint: FastAPI + Uvicorn, or Starlette static file serving.
> This HTTP server runs **in the same process / deployment** as the Aria server (same port).

#### 6) `app/audio_sink/base.py`
- Sink interface:
  - `play_url(url: str) -> Handle`
  - `stop(handle)`
  - `stop_all()`

#### 7) `app/audio_sink/sonos_http_sink.py` (NEW)
- Use **SoCo**
- Minimal required methods:
  - `play_url(url)` → `SoCo(ARIA_SONOS_IP).play_uri(url)`
  - `stop_all()` → `SoCo(ARIA_SONOS_IP).stop()`
- Async-friendly: run network calls in a thread executor if needed.

#### 8) `app/speech_output/speech_output.py`
- Orchestrates:
  - chunking
  - TTS generation (raw)
  - normalization (Sonos-ready)
  - caching (normalized files)
  - URL construction (`ARIA_HTTP_BASE_URL + /tts/<key>.wav`)
  - sequential playback via `sonos_http_sink`
  - barge-in interrupt
- Public API:
  - `speak(text: str, volume: float | None = None)`
  - `interrupt()`

---

### Integration Flow (single server)

On FINAL transcript:
1. Print transcript
2. Call LLM
3. Print LLM output
4. If `ARIA_TTS_ENABLED=1`:
   - `speech_output.speak(llm_response)`

Notes:
- `speech_output.speak()` must ensure the normalized audio exists in cache **before** returning the URL to Sonos.
- Prefer sentence-level chunking so playback can start quickly while later chunks are still being synthesized.

Optional logs:
```
ARIA.TTS: speaking <n> chunks
ARIA.TTS: url=<http_url>
ARIA.TTS.ERROR: <error>
```

---

### BARGE-IN REQUIREMENT
- On user speech start (VAD):
  - Stop current Sonos playback immediately (`sonos.stop()`)
  - Clear queued chunks
- No attempt to restore previous Sonos playback

---

### Non-functional
- Entirely local
- Errors must not crash the server
- Easy to disable via env var
- Low latency via:
  - chunking
  - caching
  - pipelined synth/normalize/playback
- Avoid constant live streams; prefer **on-demand cached clips** + sequential play.

````
