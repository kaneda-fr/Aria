# ARIA macOS client (Python)

Streams the default macOS microphone to an ARIA server over WebSocket.

## Install

```zsh
cd /Users/kaneda/dev/aria
/Users/kaneda/dev/aria/.venv/bin/python -m pip install -r client/macos/requirements.txt
```

## Run

Start the server (in a separate terminal):

```zsh
cd /Users/kaneda/dev/aria
ARIA_VAD_SILENCE_MS=300 ARIA_PARTIAL_EVERY_MS=400 /Users/kaneda/dev/aria/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Latency knobs (server):
- `ARIA_VAD_AGGR` (0-3): WebRTC VAD aggressiveness (default `2`)
- `ARIA_VAD_SILENCE_MS`: how much silence ends an utterance (default `700`)
- `ARIA_PARTIAL_EVERY_MS`: print partials during an utterance (default `0` = off)
- `ARIA_PARTIAL_WINDOW_MS`: audio window used for partials (default `1500`)

Then run the client:

```zsh
cd /Users/kaneda/dev/aria
/Users/kaneda/dev/aria/.venv/bin/python client/macos/mic_stream.py ws://localhost:8000/ws/asr
```

Optional: select an input device (PortAudio index or name substring):

```zsh
/Users/kaneda/dev/aria/.venv/bin/python client/macos/mic_stream.py ws://localhost:8000/ws/asr --device "MacBook Pro Microphone"
```

Optional: print capture stats (RMS/gain):

```zsh
ARIA_CLIENT_DEBUG=1 /Users/kaneda/dev/aria/.venv/bin/python client/macos/mic_stream.py ws://localhost:8000/ws/asr
```

Notes:
- Audio contract is fixed to 16kHz mono PCM16LE.
- Stop with Ctrl+C.

If your mic level is very low, the client enables a conservative AGC by default.
You can override it with:
- `ARIA_CLIENT_GAIN=10` (fixed gain)
- `ARIA_CLIENT_AGC=0` (disable AGC)

Server prints transcripts; the client does not receive them.
