# Microphone parity test strategy

Criticality: MEDIUM-HIGH (speech boundaries affect assistant requests).
Gate 0 OFF: behavior changes. Gate 1 ON: VAD/AGC/state logic. Gate 2 ON:
real WebRTC plus frontend plus message emitter; real WebSocket handler with fake
ASR/admission. Gate 3 OFF: no UI. Gate 4 ON: Mac/ESP32 deploy independently of
Apollo. Gate 5 OFF: no board deployment pipeline. Gate 6 ON: stable numeric
gain/PCM bounds, exercised with deterministic generated frames.

Selected types (rationale, type, size, framework, dependencies, gate):

- Frame math and turn transitions: unit, small, C assertions, none, Gate 1.
- Actual detector/segmentation/emitter: integration, medium, unittest plus C
  executable, vendored WebRTC and temporary compiler outputs, Gate 2.
- Existing audio exchange without acknowledgment: contract, medium, unittest,
  actual unchanged server handler with fake ASR/WebSocket, Gate 4.
- Gain remains finite/bounded and PCM never wraps: property-based, small,
  deterministic C generated inputs, none, Gate 6.

Rejected: UI component/E2E (Gate 3 OFF); post-deployment smoke (Gate 5 OFF).
Deliberately skipped: physical acoustic/E2E acceptance without a connected board;
record this limit rather than substituting synthetic results. Existing endpoint,
queue, wake-gate and speech regressions cover shared server behavior; no duplicate
model benchmark for an audio protocol change. No cloud or physical actions.

Acceptance cases:

- Silence sends no turn; 200 ms pre-roll is sent once in order at speech start.
- Three speech frames trigger; 39/40/41 silent frames test the 800 ms boundary;
  a shorter sentence pause retains one turn, longer gaps produce separate turns.
- AGC raises quiet detected speech, does not learn gain from silence/noise, and
  limits peaks for full-scale/INT16_MIN/impulsive inputs. Fixed gain and disabled
  VAD/AGC remain available as compile-time comparison profiles.
- RMS/SNR gates use raw audio so gain cannot manufacture speech activity.
- No new start fields, capability/acknowledgment, cancel, or playback frames.
  Capture/streaming can proceed while the server sends no messages at all.
- Completed turns finalize exactly once; existing stop/disconnect finalization
  of partial client-VAD audio is retained and explicitly tested/documented.
- No old audio is replayed. Fresh frontend state is allocated after reconnect.
- Retry waits double from 3 seconds to a 60-second cap, remain capped under repeated
  failure, reset after success, and use independent caller-owned Wi-Fi/Apollo state.
  ESP32 playback-aware muting is unavailable in the existing protocol; do not
  claim parity with the Mac live player's completion/suppression signal.
- Real PCM fixture exercises the actual detector; host wall time is diagnostic,
  with ESP32 processing time/heap/stack measurements required on hardware.
