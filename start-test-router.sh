#!/bin/bash

# ARIA Router Test Configuration
# Quick test setup with router enabled, plugins enabled, minimal config

echo "=================================="
echo "ARIA Router Test Configuration"
echo "=================================="
echo ""

# ============================================================================
# ROUTER CONFIGURATION (ENABLED)
# ============================================================================
export ARIA_ROUTER_ENABLED=1
export ARIA_ROUTER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export ARIA_ROUTER_CONTROL_THRESHOLD=0.70
export ARIA_ROUTER_CHAT_THRESHOLD=0.70
export ARIA_ROUTER_UNKNOWN_THRESHOLD=0.50
export ARIA_ROUTER_USE_SEED_DATA=1
export ARIA_ROUTER_LEARNING_ENABLED=0  # Disabled for testing

# ============================================================================
# PLUGIN SYSTEM (ENABLED)
# ============================================================================
export ARIA_PLUGINS_ENABLED=1
export ARIA_INVENTORY_ENABLED=1
export ARIA_ACTIONS_ENABLED=1

# ============================================================================
# MQTT CONFIGURATION
# ============================================================================
export ARIA_JEEDOM_MQTT_HOST=127.0.0.1
export ARIA_JEEDOM_MQTT_PORT=1883
export ARIA_JEEDOM_MQTT_ROOT_TOPIC=jeedom
export ARIA_MQTT_QOS_SUBSCRIBE=1
export ARIA_MQTT_QOS_PUBLISH=1
export ARIA_INVENTORY_CACHE_PATH=./inventory.json

# ============================================================================
# LLM CONFIGURATION (3b model for low latency)
# ============================================================================
export ARIA_LLM_PROVIDER=ollama
export ARIA_LLM_URL=http://localhost:11434
export ARIA_LLM_MODEL=qwen2.5:3b-instruct
export ARIA_LLM_NUM_PREDICT=1024
export ARIA_LLM_TEMPERATURE=0.1
export ARIA_LLM_TOP_P=0.85
export ARIA_LLM_DEBUG=0
export ARIA_LLM_HISTORY_MAX_MESSAGES=8
export ARIA_LLM_TIMEOUT_S=60

export ARIA_LLM_SYSTEM_PROMPT="You are ARIA, a home automation assistant.

CRITICAL RULES:
1. EVERY device control request MUST use find_devices + execute_command tools
2. NEVER claim to have controlled a device without calling execute_command
3. Even if you remember device IDs from earlier, you MUST still call find_devices and execute_command for EVERY new request
4. Session history is for context only - you MUST use tools for every action

WORKFLOW FOR EVERY REQUEST:
User: 'turn on the light' → find_devices() → execute_command() → confirm
User: 'turn it off' → find_devices() → execute_command() → confirm
User: 'set to 50%' → find_devices() → execute_command() → confirm"

# ============================================================================
# DISABLE FEATURES NOT NEEDED FOR TESTING
# ============================================================================
export ARIA_TTS_ENABLED=0
export ARIA_SONOS_ENABLED=0
export ARIA_SPK_ENABLED=0
export ARIA_ECHO_V2_ENABLED=0

# ============================================================================
# ASR (Minimal config)
# ============================================================================
export ARIA_MODELS_DIR=./models
export ARIA_VAD_SILENCE_MS=300

# ============================================================================
# START SERVER
# ============================================================================
echo "Router Configuration:"
echo "  Enabled: $ARIA_ROUTER_ENABLED"
echo "  Model: $ARIA_ROUTER_MODEL"
echo "  Control Threshold: $ARIA_ROUTER_CONTROL_THRESHOLD"
echo "  Chat Threshold: $ARIA_ROUTER_CHAT_THRESHOLD"
echo ""
echo "Starting ARIA server..."
echo "Press Ctrl+C to stop"
echo "=================================="
echo ""

# Activate virtual environment and start server
source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
