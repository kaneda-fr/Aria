# ARIA macOS client (Python)

Streams the default macOS microphone to an ARIA server over WebSocket.

## Live Apollo inference with Mac playback

From the repository root, using the prepared Apollo environment:

```sh
.venv/bin/python scripts/e2e/live_mac.py --host apollo
```

This starts a dedicated Apollo ASR/Qwen/Piper instance and an SSH tunnel. The Mac
only captures audio and plays Apollo's returned WAV through its default output;
no inference runs on the Mac and Sonos is not used. After the spoken readiness
check, wait for `READY` and speak French or English. Ctrl+C stops the microphone,
playback, tunnel, and owned Apollo service. The service has a three-hour limit.

Startup silently primes the normal system prompt and tool schemas before readiness.
Warm-up generates at most one token, discards all output/calls, and bypasses action
dispatch, speech, and conversation history. The spoken readiness message uses Piper
directly so a tool-free LLM request cannot replace the warmed prompt. Failure to warm
up prevents readiness. See the [Apollo comparison](../../docs/agents/warmup-2026-10-18.md).

This is the live conversation profile: the router is disabled, with
a 3B model, 80-token reply budget, and the last 12 history messages (six exchanges)
per session. History is in memory; a new microphone connection/session or restart
starts fresh. Apollo speaker recognition is enabled with existing profiles and a
locally prepared ECAPA model. Per maintainer instruction, the nearest person's name
is passed to conversation even below threshold; score, `known`, and runner-up score
remain in the logs. This is attribution, not a confidence guarantee.
Tool calling uses Apollo's cached inventory and a **log-only** action provider.
The provider validates known commands and records payloads without creating an MQTT
client; the server's network namespace has no LAN/Internet route. The cached snapshot
may be stale. Tool choices, arguments, results, and proposed commands appear in the
private logs and terminal. Real control correctness remains unproven.
Microphone frames are replaced with
silence during playback plus a 300 ms tail; this avoids self-capture but means
voice interruption during a reply is unavailable in this adapter.

Private logs are under `~/Library/Logs/Aria/e2e/<run-id>/`. `timings.jsonl` records
last client VAD speech, turn end, Mac player launch, and playback completion on one
monotonic clock. Last-VAD-speech-to-player-launch includes the 800 ms endpoint wait,
transport, Apollo processing, WAV transfer, and up to 100 ms polling. It is an
audible-onset proxy: microphone buffering and OS output-device latency are not
directly measured. The console's turn-end metric excludes endpoint silence.
`conversation.jsonl` records complete recognized requests, exact returned replies,
request/session IDs, speaker status, generation stop reason, and model timing.
Apollo writes the authoritative log; the Mac mirrors events while connected and
copies `conversation-apollo.jsonl` on shutdown. The terminal displays both sides.
Generated speech URLs and Mac playback timings share request IDs. Timing association
is marked unavailable when multiple unmatched turns make it ambiguous.
No raw microphone recordings are saved. These private logs contain conversation text.

## Playback only, with an ESP32 microphone

Attach to an already running isolated Apollo instance without starting Mac capture:

```sh
.venv/bin/python scripts/e2e/mac_playback.py --host apollo \
  --run /data/aria/e2e/runs/<active-run> \
  --artifacts "$HOME/Library/Logs/Aria/playback-only/<active-run>"
```

This utility opens only an SSH tunnel and the Mac output player. It checks readiness
and log-only tools, skips historical audio, then plays newly generated replies.
It never imports microphone capture or launches/restarts Apollo. Ctrl+C closes its
tunnel/player; Apollo remains running. `ready.json`, `console.log` (when redirected)
and `playback.jsonl` identify readiness and player outcomes. Playback is for this
run only; reattach after replacing the active server run. ESP32 needs the separate
LAN ingress described in the [monitor guide](../../monitor/README.md).
Do not run audible synthetic benchmarks while this listener is active. ESP32
playback-aware suppression remains unimplemented; server echo/speaker filters remain.

## Install

The microphone client uses the existing Apollo audio protocol; no new server
acknowledgment or capability is required. Request statuses are printed as diagnostics
and never used as playback-completion signals or reasons to retry actions.
After any ended WebSocket session, including a clean close or rejected session,
the client waits five seconds before reconnecting. Stopping interrupts that wait.
Playback-aware muting belongs to the live adapter above, which observes the local
Mac player. The existing protocol exposes no equivalent playback signal to the
ESP32. The [ESP32 parity matrix](../esp32/README.md) records that gap and differences
in default gain policy/hardware roles. The removed `ARIA_CLIENT_PLAYBACK_GUARD`
experiment is not a current setting.

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

Optional: enable client-side VAD (reduces bandwidth; server bypasses its own VAD):

```zsh
ARIA_CLIENT_VAD=1 /Users/kaneda/dev/aria/.venv/bin/python client/macos/mic_stream.py ws://localhost:8000/ws/asr
```

Client VAD tuning:
- `ARIA_CLIENT_VAD_AGGR` (0-3): aggressiveness (default `2`)
- `ARIA_CLIENT_VAD_SILENCE_MS`: audio silence that ends a turn (default `800`, clamped to `200..3000` ms). Shorter thresholds split sentence pauses; longer ones add response delay and can merge separate requests. With client VAD enabled, the server honors these explicit boundaries; `ARIA_VAD_SILENCE_MS` does not override them.
- `ARIA_CLIENT_VAD_PRE_MS`: pre-roll before speech start (default `200`)
- `ARIA_CLIENT_VAD_START_FRAMES`: consecutive 20ms speech frames required to start (default `3`)
- `ARIA_CLIENT_VAD_MIN_RMS`: RMS floor to ignore low-level noise (default `0.0`)
- `ARIA_CLIENT_VAD_SNR_DB`: SNR threshold (dB) vs adaptive noise floor (default `0.0` = off)
- `ARIA_CLIENT_VAD_NOISE_ALPHA`: EMA rate for adaptive noise floor (default `0.05`)

Notes:
- Audio contract is fixed to 16kHz mono PCM16LE.
- Stop with Ctrl+C.

If your mic level is very low, the client enables a conservative AGC by default (when client VAD is off).
When `ARIA_CLIENT_VAD=1`, AGC defaults off to avoid boosting background noise and triggering VAD.
You can override it with:
- `ARIA_CLIENT_GAIN=10` (fixed gain)
- `ARIA_CLIENT_AGC=0` (disable AGC)

Server prints transcripts; the client does not receive them.

## Stable client name

The start handshake sends `client_id`, the client's unique name. Set
`ARIA_CLIENT_ID=office-mac` explicitly; otherwise it uses
`aria-mac-<lowercase-hostname>`. Reconnects reuse that name. The server preserves
unexpired conversation context and rejects duplicate live claims. Older servers
ignore the extra field; audio never waits for an acknowledgment. See the
[protocol](../../docs/agents/client-protocol.md) and [monitor](../../monitor/README.md).
