# Combined integration check — 2026-10-18

The maintainer requested merging the accumulated current-branch work: shared
voice/text admission and grounded controls, wake/session/echo behavior, named
clients and mute/monitor tools, Mac capture and live development adapters, and the
ESP32 microphone firmware with VAD/gain, bounded streaming, LED and retry backoff.
Existing model-quality, acoustic and physical-action limits remain in the dated
reports; integration does not turn those limits into accepted fixes.

Two independent reviews covered backend/control/session/monitor paths and client/
firmware/package compatibility. The merge preparation corrected the wheel package
list and Mac clean-close reconnect throttling. Mocked reconnect regressions failed
before the correction and passed afterward. The broad combined test run also
exposed live-adapter import hooks leaking into later tests; the live fixture now
restores production hooks, routes and logging handlers immediately after import.

Validation on the final combined source:

- `.venv/bin/python -m pytest tests/test_*.py tests/plugins -q`: 376 passed.
- ESP32 host contract/frontend checks: three passed with sanitizer coverage.
- Pinned ESP32 cross-build: passed; 916,777 bytes of application flash and 53,388
  bytes of static RAM. These exclude dynamic allocations.
- Offline wheel build without dependency downloads: passed. Required routing and
  speaker modules are present; packaged `app.main` imports from an extracted wheel
  outside the checkout.
- Context/source coverage and whitespace checks pass; GitNexus refresh/change
  detection and final staged-file inspection are part of the integration gate.

Git excludes private local settings, compiled firmware, personal recordings,
runtime logs/inventory snapshots, editor settings and generated memory. Previously
tracked runtime snapshots/backups are removed from the index while their local
copies remain. This does not rewrite their earlier branch history. No Wi-Fi
password was inspected. Public documentation uses a credential placeholder.

This merge does not deploy/restart Apollo or perform home actions. Prior antenna-
enabled board evidence showed 75 seconds without transport drops and four accepted/
completed requests; exact speech quality, physical completion and long-duration
latency remain separate acceptance checks.
