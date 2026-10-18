# Named clients and reconnects

The maintainer requested client identification, reconnect continuity and a live
client view on 2026-10-18. **The client ID is its unique, stable name**; there is no
separate opaque client UUID or display name.

## Additive handshake

Send the existing `/ws/asr` start frame with an optional `client_id`:

```json
{"type":"start","sample_rate":16000,"channels":1,"format":"pcm_s16le","chunk_ms":20,"client_vad":true,"client_id":"kitchen-mic"}
```

Names are 1–64 lowercase ASCII letters/digits, dots, underscores or hyphens, starting
with a letter/digit. Invalid names are rejected; no implicit case normalization or
truncation merges identities. Use a different name for every microphone endpoint.

A capable server returns `session_started` with `client_id`, `session_id` and a new
`connection_id`. Named request-status frames also carry client and connection IDs.
Clients do not wait for this acknowledgment to stream: older servers ignore the
extra start field and continue the existing audio contract. Unnamed clients retain
random per-connection sessions and receive no new mandatory frames.

The session key is `client:<client_id>`; the transport connection ID changes on each
reconnect. Duplicate live claims receive `session_error: client_id_in_use` and close
code 1008. A rejected duplicate cannot release the original connection's claim.
Claims are released on stop, disconnect or handler failure. A lost transport may
need its normal WebSocket timeout before the name becomes available again.

## Context and audio lifetime

Named clients preserve their existing conversation/history/device reference on a
transport reconnect. The normal wake gate still clears context on goodbye, expiry
or bounded-session eviction. Its default inactivity window is 30 seconds; reconnect
does not extend it. A server restart loses in-memory context even though the same
name reconstructs the same session key. Person recognition is separate from client
identity: the room microphone and the person speaking are different concepts.

Audio buffers are always new. No captured audio, accepted request, tool command or
reply is automatically replayed. Existing client-VAD disconnect behavior can finalize
partial speech; this change does not promise cancellation. Accepted work can finish
after disconnect. It does not implement per-room speaker routing or reply redelivery.

IDs are assertions on the trusted LAN, not authentication. The current HTTP text
API can supply session IDs too; these names are not an authorization boundary.

## Client defaults and status

- Mac: `ARIA_CLIENT_ID`, otherwise `aria-mac-<lowercase-hostname>`. Set the override
  explicitly if machines share a hostname. Changing the name starts another session.
- ESP32: DHCP hostname and `client_id` are both
  `<ARIA_HOSTNAME_PREFIX>-<12 lowercase Wi-Fi MAC hex digits>`. The full hardware
  address provides stable unique names across firmware restart and Wi-Fi reconnect.
- `GET /api/clients` reports up to 256 known named clients from this server lifetime:
  `connected`, `transmitting`, `processing`, `queued`, `muted`, `mute_remaining_s`,
  session ID and last-seen time.
  Transmission means PCM arrived within one second; quiet VAD remains connected.
  Processing/queue state comes from actual request lifecycle events. Disconnected
  entries are retained until bounded eviction; no client discovery scan is performed.

The [monitor](../../monitor/README.md) polls this read-only endpoint over the local
Unix socket and broadcasts status to viewers. It labels/filter events by client,
joins model/tool events by request ID, and distinguishes unknown status from offline.
Speaker badges use `ARIA.SPK.Request` correlated by request ID; similarity is not a
calibrated probability. Nearest-person attribution below threshold stays visible.

## Input mute

`POST /api/clients/clear-offline` hides all currently disconnected names from the
status snapshot until their next successful `start`/claim. It returns `cleared`
names and the remaining visible `clients`. Connected clients, accepted requests,
conversation state, logs and mute deadlines are unchanged. Reconnecting restores
visibility and the same session key without extending a mute timer. The monitor
proxies this exact route with same-origin and active-run guards.

Mute is enforced on Apollo. The client remains operating, connected and free to
capture/transmit as usual; no client-stop or hardware-mute command is sent. Apollo
continues tracking connection/transmission status while discarding muted audio
before speech processing. Muted and offline are independent states.

`POST /api/clients/<name>/mute` accepts `{"muted":true}` or `{"muted":false}`.
True mutes input for five minutes; retries do not extend an active timer. Mute state
survives transport reconnect, expires against the monotonic server clock, and is lost
on process restart. Muted clients are not evicted. No new client/firmware messages
are required. Buffered speech is discarded across mute transitions, including an ASR
result finishing after mute; accepted jobs still finish normally. Text API requests
are administrative inputs and remain usable. See [spoken confirmation](client-mute-2026-10-18.md).

## Verification

Run `tests/test_client_identity.py`, `tests/test_client_signals.py`, shared-request,
queue, wake, speaker and monitor tests. Mac turn tests run on the Mac; Apollo need
not install microphone drivers. Run the ESP32 host tests and pinned cross-build.
Live validation additionally requires both physical clients, transport loss/reconnect,
duplicate-name rejection, status transitions and actual browser filtering. Unit or
cross-build success alone does not establish those outcomes.
