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
| ARIA_SPK_SELF_MIN_SCORE | 0.75 | Threshold to suppress self-speech |
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
| ARIA_ECHO_V2_TAIL_SEC | 2.0 | Post-TTS tail window for suppression |
| ARIA_ECHO_V2_MIN_WORDS | 5 | Minimum words in transcript to check |
| ARIA_ECHO_V2_MIN_CHARS | 20 | Minimum characters in transcript to check |
| ARIA_ECHO_V2_CONTAINMENT_THRESHOLD | 0.65 | Token containment threshold (0-1) |
| ARIA_ECHO_V2_FUZZY_THRESHOLD | 0.75 | Fuzzy similarity threshold (0-1) |
| ARIA_ECHO_V2_ALLOWLIST_REGEX | (empty) | Regex for interrupt/bypass words |

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

## LLM Integration

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_LLM_PROVIDER | (none) | LLM provider: ollama or llamacpp |
| ARIA_LLM_URL | http://localhost:11434 | LLM endpoint URL (Ollama) or http://127.0.0.1:8001/v1 (llama.cpp) |
| ARIA_LLM_MODEL | qwen2.5:3b-instruct | Model name (Ollama) or local (llama.cpp) |
| ARIA_LLM_TIMEOUT_S | 12 | Request timeout in seconds |
| ARIA_LLM_NUM_PREDICT | (none) | Max tokens to generate |
| ARIA_LLM_TEMPERATURE | (none) | Sampling temperature |
| ARIA_LLM_TOP_P | (none) | Top-p sampling |
| ARIA_LLM_STREAM | 0 | Enable streaming responses |
| ARIA_LLM_SYSTEM_PROMPT | You are ARIA... | System prompt for LLM |
| ARIA_LLM_HISTORY_MAX_MESSAGES | 8 | Max conversation history (user+assistant pairs) |
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
| ARIA_TTS_LANG_DETECT | heuristic | Language detection: heuristic or langid |
| ARIA_TTS_LANG_SHORT_REUSE_CHARS | 20 | Reuse last language for short texts (char threshold) |

---

## Sonos Output

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_TTS_SINK | sonos_http | Audio sink type (currently only sonos_http) |
| ARIA_SONOS_IP | (optional) | Override Sonos speaker IP (discovery used when unset) |
| ARIA_SONOS_NAME | (empty) | Friendly Sonos name to target during discovery (e.g., "Living Room") |
| ARIA_SONOS_DISCOVERY_TIMEOUT | 5.0 | Seconds to wait while discovering speakers via SoCo |
| ARIA_HTTP_BASE_URL | (required) | Base URL for HTTP audio serving (e.g., http://192.0.2.10:8000) |

---

## Client-side Environment Variables

These variables are used by the macOS client (`client/macos/mic_stream.py`).

### Client Debug & Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_CLIENT_DEBUG | 0 | Enable client debug statistics output |

### Client-side VAD

| Variable | Default | Description |
|----------|---------|-------------|
| ARIA_CLIENT_VAD | 0 | Enable client-side VAD (reduces bandwidth) |
| ARIA_CLIENT_VAD_AGGR | 2 | Client VAD aggressiveness (0-3) |
| ARIA_CLIENT_VAD_SILENCE_MS | 300 | Silence duration to end utterance |
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
