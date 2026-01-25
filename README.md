# ARIA (server)

This repo contains a FastAPI WebSocket server that accepts 16kHz mono PCM16LE audio and prints transcripts to the server console.

## Quickstart (dev: server + macOS client)

### 1) Create a venv

```zsh
cd /Users/kaneda/dev/aria
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install server

```zsh
python -m pip install -e .
```

### 3) Install macOS client deps

```zsh
python -m pip install -r client/macos/requirements.txt
```

Note: `sounddevice` uses PortAudio. If installation fails, you may need PortAudio installed on macOS (e.g. via Homebrew).

### 4) Configure models

Model files are not bundled. Set `ARIA_MODELS_DIR` to a directory containing either:
- `parakeet-tdt-0.6b-v3/` (single `.onnx` waveform model), or
- `parakeet-tdt-0.6b-v3-onnx/` (onnx-asr directory layout)

```zsh
export ARIA_MODELS_DIR=/opt/aria/models
```

### 5) Run the server

Option A (recommended for local dev):

```zsh
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Option B (installed script):

```zsh
aria-server --host 0.0.0.0 --port 8000
```

### 6) Run the macOS mic streamer

```zsh
python client/macos/mic_stream.py ws://localhost:8000/ws/asr
```

The client streams audio only; transcripts and LLM output are printed by the server.

## Install (server)

Build a wheel on your dev machine:

```zsh
cd /Users/kaneda/dev/aria
/Users/kaneda/dev/aria/.venv/bin/python -m pip wheel . -w dist
```

Copy the wheel to your server and install:

```zsh
python -m pip install /path/to/aria-*.whl
```

## Models

Model files are not bundled in the wheel.

Point the server at your model directory:

- Set `ARIA_MODELS_DIR` to the directory containing:
  - `parakeet-tdt-0.6b-v3/` (single `.onnx` waveform model), or
  - `parakeet-tdt-0.6b-v3-onnx/` (the onnx-asr directory layout).

Example:

```zsh
export ARIA_MODELS_DIR=/opt/aria/models
```

## Run

```zsh
aria-server --host 0.0.0.0 --port 8000
```

## LLM (optional)

Final transcripts can be sent to a local LLM (default: Ollama) and printed server-side.

```zsh
export ARIA_LLM_PROVIDER=ollama
export ARIA_LLM_URL=http://localhost:11434
export ARIA_LLM_MODEL=qwen2.5:3b-instruct
```

Console output:
- `ARIA.TRANSCRIPT: ...`
- `ARIA.LLM: ...`

Latency knobs:
- `ARIA_VAD_AGGR` (0-3)
- `ARIA_VAD_SILENCE_MS`
- `ARIA_VAD_START_FRAMES`
- `ARIA_VAD_MIN_RMS`
- `ARIA_VAD_SNR_DB`
- `ARIA_VAD_NOISE_ALPHA`
- `ARIA_PARTIAL_EVERY_MS` (0 disables partial printing)
- `ARIA_PARTIAL_WINDOW_MS`

If you see false triggers (noise -> transcripts like "Yeah" / "Oh"), see docs/tuning.md.

## Configuration (environment variables)

This section lists environment variables that are currently used by the codebase.

### Server

**Process / Uvicorn (`aria-server`)**
- `ARIA_HOST` (default: `0.0.0.0`): bind host
- `ARIA_PORT` (default: `8000`): bind port
- `ARIA_WORKERS` (default: `1`): uvicorn workers
- `ARIA_LOG_LEVEL` (default: `info`): uvicorn log level

**Models / ONNX Runtime**
- `ARIA_MODELS_DIR`: directory containing model folders (see Models section)
- `ORT_INTRA_OP`: ONNX Runtime intra-op threads (default: CPU count)
- `ORT_INTER_OP`: ONNX Runtime inter-op threads (default: `1`)

**VAD / utterance segmentation**
- `ARIA_VAD_AGGR` (default: `2`): WebRTC VAD aggressiveness (`0..3`, higher = stricter)
- `ARIA_VAD_START_FRAMES` (default: `3`): consecutive 20ms speech frames required to start an utterance
- `ARIA_VAD_MIN_RMS` (default: `0.0`): RMS floor below which frames are forced to non-speech
- `ARIA_VAD_SNR_DB` (default: `0.0`): SNR threshold (dB) vs adaptive noise floor; `0` disables
- `ARIA_VAD_NOISE_ALPHA` (default: `0.05`): EMA rate for adaptive noise floor (0..1)
- `ARIA_VAD_SILENCE_MS` (default: `700`): how much silence ends an utterance

**ASR gating (skip Parakeet on noise)**
- `ARIA_ASR_MIN_SECONDS` (default: `0.30`): minimum utterance duration to run ASR
- `ARIA_ASR_MIN_RMS` (default: `0.004`): minimum RMS (0..1) to run ASR
- `ARIA_ASR_MIN_SPEECH_RATIO` (default: `0.18`): minimum WebRTC speech-frame ratio to run ASR
- `ARIA_ASR_DROP_FILLERS` (default: `1`): drop single-word fillers ("yeah/oh/okay") when the utterance looks like noise
- `ARIA_ASR_FILLER_MAX_SECONDS` (default: `0.9`): filler drop applies only for utterances shorter than this
- `ARIA_ASR_FILLER_MIN_SPEECH_RATIO` (default: `0.35`): filler drop applies only when speech ratio is below this

**Partial transcript printing (server-side)**
- `ARIA_PARTIAL_EVERY_MS` (default: `0`): how often to print partials during an utterance (`0` disables)
- `ARIA_PARTIAL_WINDOW_MS` (default: `1500`): tail window size (ms) used for partial ASR

**LLM (Ollama over HTTP)**
- `ARIA_LLM_PROVIDER` (default: disabled): set to `ollama` to enable
- `ARIA_LLM_URL` (default: `http://localhost:11434`): Ollama base URL (recommend `http://127.0.0.1:11434` on Linux)
- `ARIA_LLM_MODEL` (default: `qwen2.5:3b-instruct`): Ollama model name
- `ARIA_LLM_TIMEOUT_S` (default: `12`): HTTP timeout seconds
- `ARIA_LLM_NUM_PREDICT` (default: unset): cap generated tokens (lower = faster/shorter), e.g. `64`
- `ARIA_LLM_TEMPERATURE` (default: unset): sampling temperature, e.g. `0.2`
- `ARIA_LLM_TOP_P` (default: unset): nucleus sampling, e.g. `0.9`
- `ARIA_LLM_STREAM` (default: `0`): set to `1` to enable streaming mode (logs each chunk as received, reduces time-to-first-token)
- `ARIA_LLM_DEBUG` (default: off): set to `1` to log the exact system/user text sent to the LLM
- `ARIA_LLM_DEBUG_MAX_CHARS` (default: `800`): max chars of user transcript to log when `ARIA_LLM_DEBUG=1`

**LLM (llama-cpp-python server)**
- `ARIA_LLM_PROVIDER=llamacpp`: uses an OpenAI-compatible local endpoint (e.g. `llama_cpp.server`)
- `ARIA_LLM_URL` (default: `http://127.0.0.1:8001/v1`): base URL for the server (must expose `/v1/chat/completions`)
- `ARIA_LLM_MODEL` (default: `local`): model name (often ignored by llama.cpp servers, but required by the API)

Notes:
- `ARIA_LLM_TIMEOUT_S`, `ARIA_LLM_NUM_PREDICT` (mapped to `max_tokens`), `ARIA_LLM_TEMPERATURE`, `ARIA_LLM_TOP_P`, `ARIA_LLM_STREAM` apply to both `ollama` and `llamacpp`.

## TTS to Sonos (optional)

ARIA can speak the LLM response via a local TTS engine and a Sonos speaker.
This implementation uses a **single API server**: the same FastAPI app serves `/ws/asr` and `/tts/<key>.wav` on the same host/port.

Runtime prerequisites on the server:
- `piper` CLI installed and a Piper voice model available (set `ARIA_TTS_VOICE` to the model path)
- `ffmpeg` installed (used to convert to Sonos-friendly WAV)

Minimal env:

```zsh
export ARIA_TTS_ENABLED=1
export ARIA_TTS_ENGINE=piper
export ARIA_TTS_VOICE=/path/to/piper/model.onnx
export ARIA_SONOS_IP=192.168.100.222

# This must be reachable by the Sonos speaker.
# You can use the same port as the main server (recommended).
export ARIA_HTTP_BASE_URL=http://192.168.100.100:8000
```

Tuning:
- `ARIA_TTS_CHUNKING` (default `1`)
- `ARIA_TTS_MAX_CHARS_PER_CHUNK` (default `220`)
- `ARIA_TTS_MIN_CHUNK_CHARS` (default `60`)
- `ARIA_TTS_RATE` (default `1.0`)
- `ARIA_TTS_CACHE_DIR` (default `/tmp/aria_tts_cache`)
- `ARIA_TTS_TIMEOUT_SEC` (default `10`)
- `ARIA_TTS_VOLUME_DEFAULT` (default `0.3`)

When enabled, ARIA serves the generated audio at:
- `GET /tts/<key>.wav`

### macOS client

**Diagnostics**
- `ARIA_CLIENT_DEBUG` (default: off): set to `1` to print send/queue/RMS/gain stats

**Client-side VAD (optional)**
- `ARIA_CLIENT_VAD` (default: `0`): set to `1` to enable client-side WebRTC VAD (sends only speech + utterance boundaries)
- `ARIA_CLIENT_VAD_AGGR` (default: `2`): VAD aggressiveness (0-3)
- `ARIA_CLIENT_VAD_SILENCE_MS` (default: `300`): silence duration that ends an utterance
- `ARIA_CLIENT_VAD_PRE_MS` (default: `200`): pre-roll audio to include before speech start
- `ARIA_CLIENT_VAD_START_FRAMES` (default: `3`): consecutive 20ms speech frames required to start an utterance
- `ARIA_CLIENT_VAD_MIN_RMS` (default: `0.0`): RMS floor below which frames are forced to non-speech (helps ignore mic noise)
- `ARIA_CLIENT_VAD_SNR_DB` (default: `0.0`): SNR threshold (dB) vs adaptive noise floor; `0` disables
- `ARIA_CLIENT_VAD_NOISE_ALPHA` (default: `0.05`): EMA rate for adaptive noise floor (0..1)

**Gain / AGC**
- `ARIA_CLIENT_GAIN`: fixed gain multiplier (disables AGC logic)
- `ARIA_CLIENT_AGC` (default: `1`): set to `0` to disable automatic gain control (note: when `ARIA_CLIENT_VAD=1`, the client defaults AGC off)
- `ARIA_CLIENT_TARGET_RMS` (default: `0.05`): AGC target RMS level
- `ARIA_CLIENT_MAX_GAIN` (default: `50`): AGC max gain cap
