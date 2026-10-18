#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${ARIA_E2E_RUN:?Set the dedicated run directory}"
cd "$ARIA_E2E_RUN/src"
mkdir -p "$ARIA_E2E_RUN/home" "$ARIA_E2E_RUN/tts"

# This script is launched INSIDE unshare --user --map-root-user --net.
ip link set lo up
test "$(ip -o link show | wc -l)" -eq 1
ip route > "$ARIA_E2E_RUN/routes.txt"
test ! -s "$ARIA_E2E_RUN/routes.txt"
export HOME="$ARIA_E2E_RUN/home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/data/aria/dev/.cache/huggingface
export DO_NOT_TRACK=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export ORT_INTRA_OP=2 ORT_INTER_OP=1
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NO_CLOUD=1
# Isolate startup template aliases from the deployed model store. Existing blobs
# and manifests are read through symlinks; no download or weight duplication.
export ARIA_E2E_MODEL_SOURCE="${ARIA_E2E_MODEL_DIR:-/usr/share/ollama/.ollama/models}"
export OLLAMA_MODELS="$ARIA_E2E_RUN/models"
/data/aria/dev/.venv/bin/python - <<'PY'
import os
from pathlib import Path
source = Path(os.environ['ARIA_E2E_MODEL_SOURCE'])
target = Path(os.environ['OLLAMA_MODELS'])
for path in source.rglob('*'):
    local = target / path.relative_to(source)
    if path.is_dir():
        local.mkdir(parents=True, exist_ok=True)
    elif not local.exists():
        local.symlink_to(path)
PY
export OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=2048
export ARIA_MODELS_DIR=/data/aria/models
export ARIA_LLM_PROVIDER=ollama ARIA_LLM_URL=http://127.0.0.1:11435
export ARIA_LLM_MODEL="${ARIA_E2E_MODEL:-qwen2.5:3b-instruct}" ARIA_LLM_TIMEOUT_S=120
export ARIA_LLM_NUM_PREDICT=80 ARIA_LLM_TEMPERATURE=0 ARIA_LLM_DEBUG=1
export ARIA_LLM_HISTORY_MAX_MESSAGES=0 ARIA_LLM_STREAM=0
export ARIA_LLM_SYSTEM_PROMPT='You are Aria. Answer briefly in the language of the user. Never claim to control a device.'
export ARIA_PLUGINS_ENABLED=0 ARIA_INVENTORY_ENABLED=0 ARIA_ACTIONS_ENABLED=0
export ARIA_ROUTER_ENABLED=0 ARIA_SPK_ENABLED=0 ARIA_ECHO_V2_ENABLED=1
export ARIA_PARTIAL_EVERY_MS=0 ARIA_VAD_SILENCE_MS=400
export ARIA_TTS_ENABLED=1 ARIA_TTS_ENGINE=piper ARIA_PIPER_BIN=/opt/piper-tts/piper
export ARIA_TTS_TIMEOUT_SEC=60 ARIA_TTS_CACHE_DIR="$ARIA_E2E_RUN/tts"
export ARIA_TTS_VOICE=/data/aria/voices/en_US-amy-low.onnx
export ARIA_TTS_VOICE_MAP=en:/data/aria/voices/en_US-amy-low.onnx,fr:/data/aria/voices/fr_FR-siwis-low.onnx
export ARIA_HTTP_BASE_URL=http://127.0.0.1 ARIA_SONOS_IP=127.0.0.1
export ARIA_TTS_CHUNKING=0 PYTHONUNBUFFERED=1
export PYTHONPATH="$ARIA_E2E_RUN/src:/data/aria/e2e/deps"

# Interactive conversation keeps bounded session context and the deployed speaker
# recognizer. Benchmark suites retain their original independent-request profile.
if [[ "${ARIA_E2E_SUITE:-smoke}" == live ]]; then
    export ARIA_LLM_SYSTEM_PROMPT='You are Aria, a local French/English voice assistant. Answer briefly in the language of the user. The speaker tag identifies the closest enrolled person. For ordinary conversation answer directly without tools. For device requests use find_devices to find the exact device and action, then execute_command with its cmd_id. Never invent IDs. Ask for clarification if ambiguous. All device commands are LOG ONLY in this test: nothing is physically executed. Say a command was recorded for testing, never that the device changed. If no command was recorded, do not claim it was.'
    export ARIA_LLM_HISTORY_MAX_MESSAGES=12 ARIA_SPK_ENABLED=1
    export ARIA_SPK_PROFILES_DIR=/data/aria/profiles ARIA_SPK_DEVICE=cpu
    export ARIA_SPK_NUM_THREADS=2 ARIA_SPK_MIN_SECONDS=1.0 ARIA_SPK_TIMEOUT_SEC=1.0
    export ARIA_SPK_THRESHOLD_DEFAULT=0.65 ARIA_SPK_SELF_MIN_SCORE=0.75
    export ARIA_SPK_HOT_RELOAD=0 ARIA_SPK_DEBUG=1 ARIA_SPK_DEBUG_DUMP_WAV=0
    export ARIA_SPK_SELF_NAMES=aria_fr,aria_en
    export ARIA_SPK_MODEL="${ARIA_E2E_SPK_MODEL:-/data/aria/models/speaker-ecapa}"
fi

# Paired model/prompt experiments use cached assets inside the same offline
# namespace. Defaults retain the measured baseline; this does not pull models.
if [[ -n "${ARIA_E2E_SYSTEM_PROMPT:-}" ]]; then
    export ARIA_LLM_SYSTEM_PROMPT="$ARIA_E2E_SYSTEM_PROMPT"
fi

"${ARIA_E2E_OLLAMA_BIN:-/usr/local/bin/ollama}" serve > "$ARIA_E2E_RUN/ollama.log" 2>&1 &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT
for attempt in $(seq 1 60); do
    if curl --silent --fail http://127.0.0.1:11435/api/tags > "$ARIA_E2E_RUN/models.json"; then
        break
    fi
    sleep 1
done
server_module=scripts.e2e.server
if [[ "${ARIA_E2E_SUITE:-smoke}" == baseline ]]; then
    server_module=scripts.e2e.baseline_server
elif [[ "${ARIA_E2E_SUITE:-smoke}" == live ]]; then
    server_module=scripts.e2e.live_server
fi
/data/aria/dev/.venv/bin/python -m "$server_module" > "$ARIA_E2E_RUN/server.log" 2>&1
