# Aria — Environment Variables Reference

This document lists **all supported environment variables** for the Aria server,
their purpose, defaults, and expected values.

Unset variables use the documented defaults.

---

## Core server

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_HOST | 0.0.0.0 | Bind address |
| ARIA_PORT | 8000 | HTTP / WebSocket port |
| ARIA_LOG_LEVEL | INFO | Logging level |

---

## Models & paths

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_MODELS_DIR | (required) | Directory containing ASR / TTS models |
| ARIA_TMP_DIR | /tmp/aria | Temp files (TTS WAV, debug dumps) |

---

## Audio input (mic / ASR)

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_AUDIO_SAMPLE_RATE | 16000 | Expected PCM sample rate |
| ARIA_AUDIO_CHANNELS | 1 | Mono audio |
| ARIA_AUDIO_FORMAT | s16le | PCM format |

---

## Voice Activity Detection (VAD)

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_VAD_ENABLED | 1 | Enable VAD |
| ARIA_VAD_MIN_SPEECH_MS | 300 | Minimum speech duration |
| ARIA_VAD_SILENCE_MS | 500 | Silence to close an utterance |

---

## Automatic Speech Recognition (ASR)

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_ASR_ENABLED | 1 | Enable ASR |
| ARIA_ASR_MODEL | parakeet | ASR backend |
| ARIA_ASR_LANGUAGE | auto | Language hint |

---

## Speaker Recognition (Speaker Verification)

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_SPK_ENABLED | 0 | Enable speaker recognition |
| ARIA_SPK_PROFILES_DIR | /var/aria/profiles | Directory of speaker profiles |
| ARIA_SPK_MODEL | speechbrain/spkrec-ecapa-voxceleb | Embedding model |
| ARIA_SPK_DEVICE | cpu | cpu / cuda |
| ARIA_SPK_NUM_THREADS | 4 | Worker threads |
| ARIA_SPK_MIN_SECONDS | 1.0 | Minimum segment length |
| ARIA_SPK_THRESHOLD_DEFAULT | 0.65 | Known-speaker threshold |
| ARIA_SPK_SELF_MIN_SCORE | 0.75 | Threshold for self-speech suppression |
| ARIA_SPK_SELF_NAMES | aria_fr,aria_en | Comma-separated self identities |
| ARIA_SPK_TIMEOUT_SEC | 1.0 | Max wait before fallback |
| ARIA_SPK_DEBUG_DUMP_WAV | 0 | Debug-only WAV dumps |

---

## Echo Guard v2 (text-based backstop)

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_ECHO_V2_ENABLED | 1 | Enable echo guard |
| ARIA_ECHO_V2_TAIL_SEC | 2.0 | Post-TTS tail window |
| ARIA_ECHO_V2_MIN_WORDS | 5 | Min transcript words |
| ARIA_ECHO_V2_CONTAINMENT_THRESHOLD | 0.65 | Token containment |
| ARIA_ECHO_V2_FUZZY_THRESHOLD | 0.75 | Fuzzy similarity |
| ARIA_ECHO_V2_ALLOWLIST_REGEX | (empty) | Interrupt words |

---

## LLM integration (optional)

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_LLM_ENABLED | 0 | Enable LLM |
| ARIA_LLM_BACKEND | ollama | LLM backend |
| ARIA_LLM_MODEL | llama3 | Model name |
| ARIA_LLM_TIMEOUT_SEC | 30 | Request timeout |

---

## Text-to-Speech (TTS)

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_TTS_ENABLED | 0 | Enable TTS |
| ARIA_TTS_ENGINE | piper | TTS backend |
| ARIA_TTS_MODEL | fr_FR-siwis-fast | Default voice |
| ARIA_TTS_VOLUME | 0.30 | Linear gain (0–1) |

---

## Sonos output

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_SONOS_ENABLED | 0 | Enable Sonos output |
| ARIA_SONOS_IP | (none) | Sonos speaker IP |
| ARIA_SONOS_TTS_PORT | 9000 | HTTP port for WAV serving |

---

## Debug / development

| Variable | Default | Description |
|--------|---------|-------------|
| ARIA_DEBUG | 0 | Enable debug mode |
| ARIA_DUMP_AUDIO | 0 | Dump audio segments |
