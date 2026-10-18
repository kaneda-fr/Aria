# Stop state and unfinished work — 2026-10-18

**Historical snapshot:** the maintainer subsequently authorized Apollo restart and
resumed work, while keeping Mac audio and physical microphones/speakers off.
See [current status](status.md) and [mute implementation](client-mute-2026-10-18.md).
The stop restriction below is superseded for Apollo server/monitor and isolated tests.

The maintainer instructed: stop Aria on Apollo until instructed otherwise. Do not
restart a server, monitor, ingress, benchmark service, microphone or speaker based
on earlier restart/test authorization. Local code work and offline checks remain
possible. The preceding request for hours of Apollo tests is paused by this stop.

Verified: no user/system `aria*` units running; no Aria listener on ports 8000/8088.
`aria-monitor.service` was stopped and disabled to prevent automatic restart.
No live Mac capture/playback or benchmark driver was running. The shared standalone
Ollama service was left alone; stopping Aria does not shut down the host or unrelated
shared services. The monitor URL will be unavailable while stopped.

## Local changes awaiting completion

- Named clients, stable reconnect keys, status/filter UI and request-specific speaker
  score display exist locally. The final server/UI integration is not deployed.
  Prior server regression suite: 247 passed; later local identity/monitor checks: 16
  passed. Neither proves live Mac/ESP32 reconnects or the new mute behavior.
- Per-client input mute is a local draft: manual mute/unmute, monotonic five-minute
  expiry, reconnect retention and a same-origin monitor endpoint. Unfinished audio
  is invalidated; accepted work completes. Add dedicated expiry/reconnect/in-flight
  ASR tests and browser checks before calling this complete. No firmware mute signal
  is sent: server-side mute discards received audio, it does not stop microphone capture.
- New request: a spoken command to mute Aria for a default or supplied duration,
  ask for confirmation and execute only after confirmation. This is **not implemented**.
  Establish scope (current client versus all Aria), confirmation session/person rules,
  cancellation/expiry, and duration parsing; test FR/EN paraphrases without real audio.
- Free-form tool lookup candidate fixes local Ollama template serialization and
  quoted tool-name examples, plus grouped/language-grounded tool results. Six lookup
  probes and 26 conversation/control cases passed functional checks; one French lookup
  took 33.75 s versus 14.12 s in the faulty reference. No no-regression acceptance yet.
  Paired extra science runs completed, but per-case quality/latency review is pending.
  Private evidence is under the existing local Aria E2E log directory; final candidate
  label `20261018T123256Z-lookup-final-25f242`, extra reference
  `20261018T125210Z-lookup-reference-extra-865eb8`. Do not treat these as running services.

## Serena repaired

The project had `language_servers: []`. Python, TypeScript and C/C++ were configured,
indexed (143/2/39 files), and enabled in the existing MCP session through its local
dashboard. Real symbol lookups succeeded in all three languages. Clangd's unresolved
include diagnostics limit firmware reference completeness. No Codex restart was needed.
