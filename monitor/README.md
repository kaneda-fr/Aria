# Aria live monitor

A FastAPI/WebSocket log reader with per-client input mute, separate from inference. Static HTML/CSS/JS
has no CDN, remote fonts, analytics or runtime Internet dependency. The visual design
uses the local UI UX Pro Max skill and [design notes](../design-system/aria-monitor/MASTER.md).

The first tab shows exact input/replies, model steps and timing, tool commands,
request IDs, queue state and speaker evidence. Client chips show connected, audio
transmission, processing/queued, offline or unavailable states; selecting one filters
the event stream and latest exchange. Category/search filters and pause affect the
view only. Speaker color indicates above-threshold recognition, below-threshold
nearest attribution, or unknown. Similarity is not an identity probability.

No authentication is configured, as requested. Serve only on the owned LAN. Private
transcripts, inventory and speaker names are visible to anyone who can reach it.
Per-client input mute/unmute lasts five minutes, with a live countdown
and automatic expiry. Muting discards incoming PCM on the server; it does not switch
off microphone hardware. Accepted requests finish normally. The control uses a strict
boolean POST, same-origin header checks and an active-run guard. No inference,
restart or home-device action is exposed. Spoken mute adds explicit confirmation
and a requested duration; see [mute behavior](../docs/agents/client-mute-2026-10-18.md).

**Clear offline** removes disconnected clients from the displayed list until they
reconnect. Apollo owns this visibility state, so it applies to every viewer. It
preserves mute deadlines, conversation state, accepted work and logs; connected
clients remain visible. The control uses the same origin and active-run checks as
mute. A cleared client filter returns to All clients. This is display cleanup,
not a disconnect, device reset or log deletion.

The compact layout uses one header/status row, bounded scrolling client chips and
a viewport-filling log area. Mute help and reading guidance are expandable. Empty
client lists collapse; small/short screens use normal page scrolling as needed.

## Run and select a source

```sh
ARIA_MONITOR_ACTIVE=/data/aria/monitor/active.json \
  python -m uvicorn monitor.server:app --host <apollo-lan-ip> --port 8088 --no-access-log
```

The local pointer contains `{"run_dir":"/data/aria/e2e/runs/<run>","tools_mode":"log_only"}`.
`ARIA_MONITOR_RUN_ROOT` defaults to `/data/aria/e2e/runs`; only direct children are
accepted. Replace the pointer atomically to follow a new run. The Mac live launcher
does this when `/data/aria/monitor` exists. Historical runs are not proof of server
health: the page reports source availability and event age separately from connection
to the feed. Client status becomes unavailable if the local server cannot be queried.

The deployed user service is `aria-monitor.service`. Use
`systemctl --user status aria-monitor` or `journalctl --user -u aria-monitor -f` on
Apollo. Browser URL is `http://<apollo-lan-ip>:8088`.

Log reads are capped per poll/line, retained events at 1,000, browser windows at 200,
and monitor viewers at 16. Slow/gapped viewers get a fresh snapshot. The UI renders
log values as text. CSP allows only local assets/connections. This is a live bounded
view, not a persistent archive; original private JSONL/log files remain the archive.

## Separate audio ingress for development

`monitor.ingress:app` exposes only `/ws/asr` on LAN port 8000 and forwards bounded
frames to the active run's Unix socket. Each connection checks actual live config:
the instance must be ready with tools in `log_only` mode. It cannot expose the
administrative HTTP APIs. The monitor on 8088 only adds its narrowly scoped mute route. The ingress user
service is `aria-ingress.service`; it accepts at most 32 connections, 64 KiB frames
and eight queued upstream messages. No inference runs in the proxy.

Use the Mac live launcher with `--audio-url ws://<apollo-lan-ip>:8000/ws/asr` when
testing the same ingress as ESP32. Playback and private log collection retain the
SSH tunnel. Restarting the ingress disconnects microphones without restarting the
inference server, allowing a real reconnect/context-continuity test. This deliberate
test interrupts audio and must not be presented as cancellation of accepted work.

Protocol details: [named clients](../docs/agents/client-protocol.md). Backend tests:
`python -m pytest tests/test_monitor.py tests/test_client_identity.py`.
