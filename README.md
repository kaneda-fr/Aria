# ARIA (server)

This repo contains a FastAPI WebSocket server that accepts 16kHz mono PCM16LE audio and prints transcripts to the server console.

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
- `ARIA_PARTIAL_EVERY_MS` (0 disables partial printing)
- `ARIA_PARTIAL_WINDOW_MS`
