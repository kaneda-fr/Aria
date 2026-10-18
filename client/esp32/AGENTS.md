# ESP32 microphone client

Read [README](README.md) and [D-009](../../docs/agents/decisions.md).

- XIAO ESP32-S3 Sense PDM capture profile; the seller title mixes Sense and Plus.
  Verify the physical board; do not assume PSRAM or initialize the camera.
- D-012 adds local WebRTC VAD, bounded pre-roll, gain control, and optional RMS/SNR
  gates. Apollo owns ASR, inference, echo filtering, TTS, and speaker playback.
- Future on-device work placement is open and must be driven by measured Apollo
  and board latency/resource results. There is no local wake/inference model.
- All settings remain in code; private deployment overrides/build artifacts stay ignored.
- Preserve bounded capture, independent networking, fresh sessions after audio loss,
  and no audio/action replay. Request completion is not playback completion.
- Wi-Fi/Apollo retries independently double 3 to 60 seconds and reset on success.
  LED stays on during connection attempts and blinks rapidly throughout retry
  waits, with an extra short pulse every two beats for Apollo failures. Connected
  idle uses a slow heartbeat, with VAD/transmit activity taking over when active.
- Compatibility with the existing Apollo `/ws/asr` protocol is mandatory. Do not
  require new acknowledgment, playback, or cancel frames for client-side features.
- ESP32 playback-aware muting is a gap: the existing protocol has no playback state.
  Existing client-VAD disconnect can finalize partial speech; never claim cancellation.
- Runtime traffic is LAN-only. No runtime dependency downloads, telemetry, or cloud APIs.
- Validate with the host tests and ESP32 cross-build; separately report hardware
  capture/echo/latency evidence. Follow the root semantic/impact workflow.
