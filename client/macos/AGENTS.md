# macOS audio client context

Read [client README](README.md) and [architecture](../../docs/agents/architecture.md).

- Keep this microphone frontend for Apollo server development (D-005).
  A separate ESP32 microphone client is now authorized (D-009); preserve this path.
  The eventual deployment baseline also includes ESP32
  devices; do not assume this Python/audio stack can run on them or proves their compatibility.
- Capture through sounddevice/PortAudio at the microphone's native rate, then resample
  with soxr to 16 kHz mono PCM16LE for `/ws/asr`.
- Preserve the start/stop protocol and bounded backpressure behavior.
- Preserve compatibility with the existing Apollo protocol. Client-side features
  must not require a server update or new acknowledgment/playback/cancel frames.
- Client VAD owns explicit turn boundaries: default silence is 800 ms. Protect
  sentence-pause continuity, separate-turn endpointing, and stop flushing in tests;
  changing server VAD silence alone does not affect this client mode.
- Do not replace capture with ffmpeg/AVFoundation; historical capture timing was incorrect.
  Server-side ffmpeg normalization for Sonos is a different, intentional use.
- Enrollment and recording are client tools, not imports in the server hot path.
- Keep recordings, identities, and deployment addresses out of shared context.
- Verify timing/audio continuity on hardware for capture changes; unit tests alone are insufficient.
