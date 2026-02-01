# ARIA — Adaptive Real‑time Intelligent Assistant

Aria is a **local-first voice assistant stack** built around a FastAPI server that:
- receives **real-time microphone audio** over **WebSocket**,
- performs **VAD + ASR** (Parakeet ONNX),
- optionally calls a **local LLM** (e.g. Ollama),
- generates **TTS** (Piper),
- and can **play TTS on Sonos** via **HTTP streaming** (SoCo / `play_uri`).

> Current repo baseline: the server accepts **16kHz mono PCM16LE** audio over WebSocket and prints transcripts server-side.

> **Docs scope**: This README is the user guide for running and configuring Aria.


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

This quickstart covers the baseline dev flow (venv → install → run server → run mic streamer).

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

Model files are not bundled; set `ARIA_MODELS_DIR`.

```bash
export ARIA_MODELS_DIR=/opt/aria/models
```

### Run

```bash
# Start server (copy start.sh.example to start.sh and customize with your IPs)
cp start.sh.example start.sh
# Edit start.sh with your configuration, then:
bash start.sh

# Or run server directly with minimal config:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start client (from another terminal)
python client/macos/mic_stream.py ws://<server-ip>:8000/ws/asr
```

> **Note**: For full configuration with LLM and TTS, use the provided `.example` scripts. Copy them (e.g., `cp start.sh.example start.sh`) and customize with your real IPs. The actual scripts are gitignored to prevent committing sensitive information.

## Sonos: HTTP TTS streaming

Aria generates TTS audio on the server and serves it over HTTP for Sonos to fetch.

Example manual trigger:
```python
from soco import SoCo

SONOS_IP = "192.0.2.20"   # example Sonos IP (documentation-only)
SERVER_IP = "192.0.2.10"  # example Aria server IP (documentation-only)

# Aria serves normalized audio at /tts/<cache_key>.wav
url = f"http://{SERVER_IP}:8000/tts/<key>.wav"
SoCo(SONOS_IP).play_uri(url)
```

When `ARIA_TTS_ENABLED=1`, LLM responses are automatically spoken to Sonos.

## Speaker identification (speaker ID)

Aria includes optional speaker recognition to identify who is speaking and suppress self-speech (TTS feedback).

Features:
- Runs on in-memory finalized VAD segments (no WAV files in hot path)
- Language-specific profiles (e.g., `alice_fr`, `alice_en`, `aria_fr`, `aria_en`)
- Self-speech suppression before LLM processing
- Echo Guard v2 remains active as a safety backstop

Configuration:
```bash
export ARIA_SPK_ENABLED=1
export ARIA_SPK_PROFILES_DIR=/var/aria/profiles
export ARIA_SPK_SELF_NAMES=aria_fr,aria_en
export ARIA_SPK_THRESHOLD_DEFAULT=0.65
export ARIA_SPK_SELF_MIN_SCORE=0.75
```

Enrollment tools (recording samples and generating profiles) are kept separate.
See [`docs/install_speaker_recognition.md`](docs/install_speaker_recognition.md) for setup details.

## Configuration

Aria is configured entirely via environment variables.

A **complete and authoritative reference** of all supported environment variables,
their defaults, and behavior is maintained here:

➡️ **[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)**

This includes configuration for:
- audio input and VAD (client and server)
- ASR (quality controls, filler detection)
- speaker recognition (self-speech suppression)
- echo suppression (Echo Guard v2)
- LLM integration (Ollama and llama.cpp support)
- **plugin system (tool calling, home automation)**
- TTS (Piper, multi-language voice selection, chunking)
- Sonos output (HTTP streaming)
- debug and development options

For tuning VAD sensitivity and reducing false triggers, see:
➡️ **[`docs/tuning.md`](docs/tuning.md)**

For plugin system architecture and tool calling details, see:
➡️ **[`docs/PLUGIN_SYSTEM.md`](docs/PLUGIN_SYSTEM.md)**

If a variable is not documented in ENVIRONMENT.md, it should be considered unsupported.

## Plugin System (Tool Calling)

ARIA includes an optional **plugin system** that enables the LLM to interact with external systems through function calling.

**Current plugins:**
- **Inventory Provider** — Discovers devices and commands from Jeedom via MQTT
- **Actions Provider** — Executes commands on devices via MQTT (Jeedom MQTT2 protocol)

**Example:**
```
User: "allume la lumière du salon"
→ LLM calls find_devices(query="lumière salon")
→ LLM calls execute_command(cmd_id="5587")
→ MQTT: jeedom/cmd/set/5587 ← {}
→ Response: "J'ai allumé la lumière du salon."
```

Enable with:
```bash
export ARIA_PLUGINS_ENABLED=1
export ARIA_INVENTORY_ENABLED=1
export ARIA_ACTIONS_ENABLED=1
export ARIA_JEEDOM_MQTT_HOST=192.168.100.60
export ARIA_JEEDOM_MQTT_PORT=1883
```

See [`docs/PLUGIN_SYSTEM.md`](docs/PLUGIN_SYSTEM.md) for full architecture and implementation details.

---

## Troubleshooting

### Big delay on Sonos
- Sonos buffers; keep clips short.
- Prefer “generate then serve file” vs serving while still generating.
- Ensure you’re not double-routing audio (AirPlay + Sonos URL at the same time).

### No audio
- Check mute and volume.
- Confirm Sonos can fetch the URL from its network segment.

## Roadmap (directional)

- Graduate speaker recognition to the primary feedback-loop guard and keep Echo Guard v2 purely as a fallback.
- Improve tool calling reliability (pre-filter layer, upgrade to 7b model).
- Expand plugin system with additional integrations (Zigbee, Z-Wave, HomeAssistant).
- Add lightweight observability (latency budget per stage: VAD, ASR, LLM, TTS, Sonos fetch).

## For contributors

- Copilot build/spec instructions: [.copilot-instructions.md](.copilot-instructions.md)
