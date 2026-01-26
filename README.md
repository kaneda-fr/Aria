# ARIA — Adaptive Real‑time Intelligent Assistant

Aria is a **local-first voice assistant stack** built around a FastAPI server that:
- receives **real-time microphone audio** over **WebSocket**,
- performs **VAD + ASR** (Parakeet ONNX),
- optionally calls a **local LLM** (e.g. Ollama),
- generates **TTS** (Piper),
- and can **play TTS on Sonos** via **HTTP streaming** (SoCo / `play_uri`).

> Current repo baseline: the server accepts **16kHz mono PCM16LE** audio over WebSocket and prints transcripts server-side. citeturn7view2


## High-level architecture

```text
(macOS) mic_stream.py
    │  WS: 16kHz PCM16LE frames
    ▼
(ubuntu) aria-server (FastAPI)
    ├─ VAD → utterance segmentation
    ├─ ASR (Parakeet ONNX) → final transcript
    ├─ Echo Guard v2 (keep) → suppress obvious feedback loops
    ├─ Speaker ID (optional) → label speaker / drop self-speech
    ├─ LLM (optional, local) → response text
    ├─ TTS (Piper) → WAV
    └─ Sonos output (HTTP stream) → SoCo.play_uri("http://server/tts.wav?...")

Notes:
- Speaker ID is intended to replace echo suppression long-term, but you requested to keep
  Echo Guard v2 for now.
```

## Quickstart (dev: server + macOS client)

The existing README documents the baseline dev flow (venv → install → run server → run mic streamer). citeturn7view2

### Create venv + install

```bash
cd ~/dev/aria
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r client/macos/requirements.txt
```

### Configure models

Model files are not bundled; set `ARIA_MODELS_DIR`. citeturn7view2

```bash
export ARIA_MODELS_DIR=/opt/aria/models
```

### Run

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
python client/macos/mic_stream.py ws://<server-ip>:8000/ws/asr
```

## Sonos: HTTP TTS streaming

Sonos can play a URL. Generate audio on the server and expose it via HTTP (FastAPI route),
then ask Sonos to play it:

```python
from soco import SoCo

SONOS_IP = "10.1.1.222"
SERVER_IP = "10.1.1.100"

url = f"http://{SERVER_IP}:9000/tts.wav?text=hello%20sonos&volume=0.30"
SoCo(SONOS_IP).play_uri(url)
```

## Speaker identification (speaker ID)

Goal: detect who is speaking for each finalized VAD segment; optionally drop “self speech”.
Recommended:
- enrollment tools remain separate (sandbox),
- server loads JSON embedding profiles,
- recognition runs before calling the LLM,
- allow multiple profiles per user (e.g. `seb_fr`, `seb_en`).

Suggested env vars:
- `ARIA_SPK_PROFILES_DIR=/var/aria/profiles`
- `ARIA_SPK_THRESHOLD_DEFAULT=0.65`
- `ARIA_SPK_SELF_NAMES=aria_fr,aria_en`

## Configuration

Aria is configured entirely via environment variables.

A **complete and authoritative reference** of all supported environment variables,
their defaults, and behavior is maintained here:

➡️ **[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)**

This includes configuration for:
- audio input and VAD
- ASR
- speaker recognition
- echo suppression (Echo Guard v2)
- LLM integration
- TTS and Sonos output
- debug and development options

If a variable is not documented there, it should be considered unsupported.

## Troubleshooting

### Big delay on Sonos
- Sonos buffers; keep clips short.
- Prefer “generate then serve file” vs serving while still generating.
- Ensure you’re not double-routing audio (AirPlay + Sonos URL at the same time).

### No audio
- Check mute and volume.
- Confirm Sonos can fetch the URL from its network segment.
