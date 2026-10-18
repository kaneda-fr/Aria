#!/bin/bash

# ARIA Mac Development Configuration
# Run with: bash start-osx.sh

# ============================================================================
# PLUGIN SYSTEM (Enable tool calling)
# ============================================================================
export ARIA_PLUGINS_ENABLED=1
export ARIA_INVENTORY_ENABLED=1
export ARIA_ACTIONS_ENABLED=1

# ============================================================================
# MQTT CONFIGURATION (Jeedom integration)
# ============================================================================
# Using remote MQTT broker - change to 127.0.0.1 if running local broker
export ARIA_JEEDOM_MQTT_HOST=127.0.0.1
export ARIA_JEEDOM_MQTT_PORT=1883
export ARIA_JEEDOM_MQTT_ROOT_TOPIC=jeedom

# MQTT Quality of Service
export ARIA_MQTT_QOS_SUBSCRIBE=1
export ARIA_MQTT_QOS_PUBLISH=1

# ============================================================================
# INVENTORY CACHE
# ============================================================================
# Mac-friendly cache path
export ARIA_INVENTORY_CACHE_PATH=./inventory.json

# ============================================================================
# SELF-LEARNING ROUTER (Embedding-based CHAT vs CONTROL classification)
# ============================================================================
export ARIA_ROUTER_ENABLED=1
export ARIA_ROUTER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export ARIA_ROUTER_CONTROL_THRESHOLD=0.70
export ARIA_ROUTER_CHAT_THRESHOLD=0.70
export ARIA_ROUTER_UNKNOWN_THRESHOLD=0.50
export ARIA_ROUTER_USE_SEED_DATA=1

# Learning configuration (for future)
export ARIA_ROUTER_LEARNING_ENABLED=1
export ARIA_ROUTER_LEARNING_DB=./router_learning.db
export ARIA_ROUTER_MIN_CONFIDENCE_TO_LEARN=0.60

# ============================================================================
# LLM CONFIGURATION (Ollama with tool calling)
# ============================================================================
export ARIA_LLM_PROVIDER=ollama
export ARIA_LLM_URL=http://localhost:11434
export ARIA_LLM_MODEL=qwen2.5:3b-instruct
export ARIA_LLM_NUM_PREDICT=1024
export ARIA_LLM_TEMPERATURE=0.1
export ARIA_LLM_TOP_P=0.85
export ARIA_LLM_DEBUG=1
export ARIA_LLM_HISTORY_MAX_MESSAGES=8
export ARIA_LLM_TIMEOUT_S=60

# System prompt for tool calling - instructs LLM to use find_devices and execute_command
export ARIA_LLM_SYSTEM_PROMPT="You are ARIA, a home automation assistant.

CRITICAL RULES:
1. EVERY device control request MUST use find_devices + execute_command tools
2. NEVER claim to have controlled a device without calling execute_command
3. Even if you remember device IDs from earlier in the conversation, you MUST still call find_devices and execute_command for EVERY new request
4. Session history is for context only - you MUST use tools for every action
5. If you answer without calling tools, you are lying to the user

WORKFLOW FOR EVERY REQUEST:
User: 'turn on the light' → find_devices() → execute_command() → confirm
User: 'turn it off' → find_devices() → execute_command() → confirm
User: 'set to 50%' → find_devices() → execute_command() → confirm

No shortcuts. No assumptions. Always call the tools."

# ============================================================================
# ASR (Speech-to-Text) CONFIGURATION
# ============================================================================
export ARIA_MODELS_DIR=./models
export ARIA_VAD_SILENCE_MS=300
export ARIA_PARTIAL_EVERY_MS=400

# ============================================================================
# TTS (Text-to-Speech) CONFIGURATION
# ============================================================================
# Disabled on Mac by default - uncomment if you have Piper installed
export ARIA_TTS_ENABLED=0
# export ARIA_TTS_ENGINE=piper
# export ARIA_TTS_VOICE=./voices/en_US-amy-low.onnx
# export ARIA_TTS_VOICE_MAP=en:./voices/en_US-amy-low.onnx,fr:./voices/fr_FR-siwis-low.onnx
# export ARIA_PIPER_BIN=/opt/piper-tts/piper

# ============================================================================
# SONOS CONFIGURATION
# ============================================================================
# Disabled on Mac by default - uncomment if needed
# export ARIA_SONOS_NAME="Living Room"

# ============================================================================
# ECHO CANCELLATION V2
# ============================================================================
export ARIA_ECHO_V2_ENABLED=1
export ARIA_ECHO_V2_TAIL_SEC=2.0
export ARIA_ECHO_V2_MIN_WORDS=5
export ARIA_ECHO_V2_MIN_CHARS=20
export ARIA_ECHO_V2_CONTAINMENT_THRESHOLD=0.65
export ARIA_ECHO_V2_FUZZY_THRESHOLD=0.75
export ARIA_ECHO_V2_ALLOWLIST_REGEX="\b(stop|cancel|aria|tais[- ]toi|stoppe)\b"

# ============================================================================
# SPEAKER RECOGNITION
# ============================================================================
export ARIA_SPK_ENABLED=0
# Disabled on Mac by default - uncomment if needed
# export ARIA_SPK_PROFILES_DIR=./profiles
# export ARIA_SPK_MODEL=speechbrain/spkrec-ecapa-voxceleb
# export ARIA_SPK_DEVICE=cpu
# export ARIA_SPK_NUM_THREADS=4
# export ARIA_SPK_MIN_SECONDS=1.0
# export ARIA_SPK_THRESHOLD_DEFAULT=0.65
# export ARIA_SPK_SELF_MIN_SCORE=0.75
# export ARIA_SPK_TIMEOUT_SEC=1.0
# export ARIA_SPK_HOT_RELOAD=0
# export ARIA_SPK_DEBUG=1
# export ARIA_SPK_DEBUG_DUMP_WAV=0
# export ARIA_SPK_SELF_NAMES=aria_fr,aria_en

# ============================================================================
# HTTP SERVER
# ============================================================================
export ARIA_HTTP_BASE_URL=http://127.0.0.1:8000

# ============================================================================
# LOGGING
# ============================================================================
export ARIA_LOG_LEVEL=debug
export ARIA_DEBUG=1

# ============================================================================
# START SERVER
# ============================================================================
echo "================================================"
echo "ARIA Mac Development Server"
echo "================================================"
echo "LLM: ${ARIA_LLM_MODEL} @ ${ARIA_LLM_URL}"
echo "MQTT: ${ARIA_JEEDOM_MQTT_HOST}:${ARIA_JEEDOM_MQTT_PORT}"
echo "Plugins: Enabled (inventory + actions)"
echo "STT: Enabled (ASR)"
echo "TTS: ${ARIA_TTS_ENABLED:-0} (disabled on Mac)"
echo "================================================"
echo ""
echo "API will be available at: http://127.0.0.1:8000"
echo "Docs: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

# Run with virtual environment
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
