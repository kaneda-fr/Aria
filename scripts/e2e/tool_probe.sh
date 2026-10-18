#!/usr/bin/env bash
# Test-only: invoke through unshare --user --map-root-user --net on Apollo.
set -euo pipefail
umask 077
: "${ARIA_E2E_RUN:?Set a dedicated, staged run directory}"
cd "$ARIA_E2E_RUN/src"
ip link set lo up
test "$(ip -o link show | wc -l)" -eq 1
ip route > "$ARIA_E2E_RUN/routes.txt"
test ! -s "$ARIA_E2E_RUN/routes.txt"
mkdir -p "$ARIA_E2E_RUN/home"
export HOME="$ARIA_E2E_RUN/home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NO_CLOUD=1
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
export OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=2048
export ARIA_LLM_PROVIDER=ollama ARIA_LLM_URL=http://127.0.0.1:11435
export ARIA_LLM_MODEL=qwen2.5:3b-instruct ARIA_LLM_TIMEOUT_S=120
export ARIA_LLM_NUM_PREDICT=48 ARIA_LLM_TEMPERATURE=0.1 ARIA_LLM_TOP_P=0.85
export ARIA_LLM_HISTORY_MAX_MESSAGES=0 ARIA_LLM_STREAM=0
unset ARIA_LLM_SYSTEM_PROMPT
export ARIA_PLUGINS_ENABLED=0 ARIA_INVENTORY_ENABLED=0 ARIA_ACTIONS_ENABLED=0
export ARIA_ROUTER_ENABLED=0 ARIA_SPK_ENABLED=0 ARIA_TTS_ENABLED=0
export PYTHONPATH="$ARIA_E2E_RUN/src:/data/aria/e2e/deps" PYTHONUNBUFFERED=1
/usr/local/bin/ollama serve > "$ARIA_E2E_RUN/ollama.log" 2>&1 &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT
for attempt in $(seq 1 60); do
    if curl --silent --fail http://127.0.0.1:11435/api/tags > "$ARIA_E2E_RUN/models.json"; then
        break
    fi
    sleep 1
done
/data/aria/dev/.venv/bin/python -m scripts.e2e.tool_probe > "$ARIA_E2E_RUN/probe.log" 2>&1
