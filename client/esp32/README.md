# Aria ESP32-S3 microphone client

Microphone-only firmware for the Seeed Studio XIAO ESP32-S3 **Sense** expansion
board. The ESP32 performs capture, WebRTC VAD, pre-roll, automatic/fixed gain,
optional RMS/SNR gates, and bounded streaming. Apollo performs ASR, speaker/echo
filtering, conversation gating, inference, TTS, and speaker playback.

The maintainer requested microphone-feature parity with the Mac client after the
initial streaming baseline. Settings remain in code. There is no local wake model,
camera, SD recording, speaker output, or provisioning UI. Further processing changes
still require results on the actual board/Apollo; see [D-012](../../docs/agents/decisions.md).

## Microphone feature parity

| Feature | Mac client | ESP32 client |
| --- | --- | --- |
| Capture to 16 kHz mono PCM16 | Native capture + soxr | Hardware PDM conversion |
| WebRTC VAD + turn boundaries | Available; enabled in live profile | Enabled by default |
| Speech pre-roll / silence endpoint | 200 ms / 800 ms defaults | 200 ms / 800 ms defaults |
| Automatic/fixed gain | Available; live profile AGC defaults off | AGC on; fixed gain optional |
| RMS/SNR background gates | Available | Available; zero disables each gate |
| Playback-aware microphone suppression | Live adapter observes its local player | Unavailable through existing Apollo protocol; server echo filters remain |
| Bounded capture/network + reconnect | Yes | Yes; capture never waits on network |
| Request status / audio diagnostics | Console; live adapter adds playback timing | USB serial; adds heap/stack/frontend timing |
| Configuration | CLI/environment | Compile-time settings |

Feature availability does not imply identical settings or microphone response.
ESP32 AGC learns from detected speech on raw audio, freezes gain adaptation on
non-speech, smooths changes, and limits output peaks. Default gain is capped at 8x;
the Mac's optional AGC has a different policy and defaults to a 50x cap. Use matched
settings and recordings for comparisons; gate choices and default gains are not
acoustic acceptance results. Full playback-muting parity is not claimed: the
microphone-only ESP32 has no local player or existing remote playback signal.

## Hardware profile

The manufacturer's PDM microphone profile uses **GPIO42 clock, GPIO41 data**,
I2S0, hardware PDM-to-PCM conversion, 16 kHz mono signed 16-bit little-endian audio.
The seller's combined “Sense Plus … OV2640 OV3660” title does not distinguish the
physical variants. Confirm the Sense microphone expansion is present; a different
microphone/codec requires a different driver, not just a board name change.
No PSRAM is required and the camera is never initialized.
The build explicitly selects an 8 MB flash header for the Sense profile instead
of ESP-IDF's 2 MB default. This is a build setting, not hardware detection; verify
the physical board's flash before uploading. The initial application partition
remains 1 MB and no extra flash is used for recordings or models.

References: [Seeed microphone wiring](https://wiki.seeedstudio.com/xiao_esp32s3_sense_mic/),
[Sense/Plus variants](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/),
[ESP-IDF 5.5 PDM driver](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/api-reference/peripherals/i2s.html).

## Hardcoded settings and build

Use Python 3.11+ and PlatformIO Core 6.1.18. The build pins the Espressif platform
to 6.12.0 (ESP-IDF 5.5.0) and the WebSocket component to 1.5.0. Version 1.6.0
requires a redirect API absent from this SDK package. Tool/dependency
downloads happen on the build machine; firmware has no runtime download path.

From this directory:

```sh
python3 -m venv .pio/venv
.pio/venv/bin/pip install platformio==6.1.18
```

Set values in [main/settings.h](main/settings.h), or keep the same compile-time
definitions private in ignored `main/settings.local.h`:

```c
#define ARIA_WIFI_SSID "your-2.4GHz-LAN"
#define ARIA_WIFI_PASSWORD "your-password"
#define ARIA_APOLLO_IPV4 "192.168.1.10"  // Replace with Apollo's LAN IPv4.
#define ARIA_APOLLO_PORT 8000
```

The checked-in defaults are empty and boot without starting Wi-Fi or the mic.
Apollo must have a reachable LAN listener for `/ws/asr`; a Mac SSH tunnel's
localhost address cannot be reached from the board. Only numeric RFC1918 IPv4
destinations are accepted. No DNS, NTP, cloud fallback, or discovery is used.
Wi-Fi credentials remain in the compiled binary; keep local headers/build artifacts
out of commits. There are no runtime configuration options yet.

```sh
.pio/venv/bin/pio run
.pio/venv/bin/pio device list
# Substitute the board's USB port; flashing replaces its current firmware.
.pio/venv/bin/pio run --target upload --upload-port /dev/cu.usbmodemYOUR_DEVICE
.pio/venv/bin/pio device monitor --port /dev/cu.usbmodemYOUR_DEVICE --baud 115200
```

USB Serial/JTAG logging is enabled; no button interaction is required in operation.
The Wi-Fi interface sets its DHCP hostname before connecting: `aria-wisp-` followed
by its full twelve-digit station MAC. This identifies each board
in a router's DHCP client list without reusing a name across boards. The prefix is
hardcoded in `ARIA_HOSTNAME_PREFIX`; use lowercase letters, digits, and internal
hyphens if changing it. A DHCP hostname does not enable mDNS or guarantee that the
router provides DNS resolution for it. See [ESP-NETIF hostname API](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/api-reference/network/esp_netif_programming.html#_CPPv422esp_netif_set_hostnameP11esp_netif_tPKc).
The hostname update was built and flashed on the connected board; USB serial
confirmed the configured name followed by `Wi-Fi ready`. The maintainer confirmed
the DHCP name appears; DNS resolution has not been inspected. The three host checks
still pass.
If a board with existing firmware does not enter download mode automatically,
hold BOOT while connecting USB, release it, and retry the upload. Reset after flashing.
No board is flashed automatically as part of host verification.

## Activity LED and microphone diagnostics

The onboard user LED uses GPIO21, active low ([Seeed pin reference](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/)).
Camera and SD stay disabled; GPIO21 is also the Sense SD chip-select pin.

| Pattern | Meaning |
| --- | --- |
| One short flash every five seconds | Waiting for Wi-Fi |
| Two short flashes every five seconds | Wi-Fi ready, waiting for Apollo streaming |
| Tiny flash every two seconds | Connected and idle |
| Steady light for 250 ms | VAD opened a speech turn |
| Rapid flicker, about 10 Hz | Audio frames successfully written to WebSocket |
| Steady light after the onset indication | VAD turn open, no recent successful audio write |
| Equal on/off flashes, about 5 Hz | Audio capture/queue fault |

Connection and fault patterns take precedence over speech/transmit indications.
Disconnected patterns keep repeating during the entire retry wait, independently
of how long that wait has grown. The orange user LED is firmware-controlled; the
red charge indicator is controlled by the charging circuit. Seeed documents the
red light coming on with USB and going off after about 30 seconds without a battery.
VAD is evaluated in the streaming path; it is not active while waiting for Apollo.
Transmit activity indicates a socket write, not ASR acceptance or completed playback.
The LED/status task never delays capture or network writes.

USB logs also include one-second `capture_frames`, `read_errors`, `raw_rms`,
`raw_peak`, `dc`, `ac_rms` and `nonzero` counters even without Apollo. RMS/peak are
measured before gain control on a 0–1 scale; `dc` is the signed sample mean and
`ac_rms` removes that mean to distinguish sound energy from a DC offset.
Samples/nonzero and levels describe the latest window;
frame/error counts are cumulative since boot. A one-entry overwrite queue sends
these summaries to the status task, so USB logging never blocks microphone capture.
Nonzero varying PCM establishes capture activity, not intelligible speech; compare
quiet and spoken windows, then check Apollo's transcript against a known utterance.
The connected board's sound check on 2026-10-18 measured about 16 ksample/s with
zero read errors. Centered RMS rose from approximately 0.00035 to 0.00371 during
the maintainer-assisted check and returned near baseline, establishing sound pickup.
Raw DC was approximately 0.041; this is measured, not removed from streamed audio.
No PCM recording or Apollo transcript was obtained, so intelligibility remains
unverified. Activity firmware flashed; LED appearance was not observed.

## Protocol and limits

Each connection sends one text `start` with `sample_rate:16000`, `channels:1`,
`format:"pcm_s16le"`, `chunk_ms:20`, and `client_vad:true`.
No server acknowledgment or new feature flag is required. Local VAD sends `utterance_start`,
**640-byte binary frames** (including pre-roll and trailing silence), then
`utterance_end`. Silence between turns sends no audio. Active PCM payload is
**32,000 bytes/s**, before Wi-Fi/TCP/WS overhead; this is a format calculation.
The detector needs three consecutive voiced 20 ms frames to start and 800 ms of
detector-negative audio to end. WebRTC's internal hangover also affects acoustic
endpoint latency; this is not semantic completion.

Edit `ARIA_CLIENT_*` constants in [settings.h](main/settings.h) for VAD mode, silence,
pre-roll (bounded to 500 ms), raw RMS/SNR gates, AGC target/cap, or fixed gain.
`ARIA_CLIENT_FIXED_GAIN > 0` takes precedence over AGC. `ARIA_CLIENT_VAD=0` retains
the continuous server-VAD comparison profile; `ARIA_CLIENT_AGC=0` disables automatic
gain adaptation. The default target RMS is 0.05 and the peak ceiling is about 0.98.

Wi-Fi power saving is disabled for the initial latency baseline. Measure current
draw as well as response time on the board before choosing a power/latency tradeoff.

Capture drains independently of networking. The queue holds eight audio frames
(160 ms; 5,120 PCM bytes plus metadata), with four I2S DMA buffers. There is no
growing recording buffer. Writes have a 100 ms timeout, queued audio older than
200 ms is rejected, and dropped/partial capture or a sequence gap abandons the
connection. Wi-Fi and Apollo use independent exponential retry delays of 3, 6, 12,
24, 48, then 60 seconds indefinitely. Wi-Fi backoff resets once an IP is acquired;
Apollo backoff resets after a session successfully starts streaming. Apollo delays
are after the failed attempt/cleanup, so connection timeout adds to the interval
between attempt starts. Every new WebSocket session sends a fresh `start`.
See [board retry evidence and limits](tests/hardware-2026-10-18.md).
It never replays an old utterance. The existing Apollo client-VAD handler can
finalize an unfinished utterance on disconnect or stop. There is no supported
cancel message, so a dropped connection cannot guarantee that partial speech is
discarded or that accepted work is cancelled. The client sends no invented end
boundary after a transport failure; this does not alter the server's disconnect
behavior. Test interrupted speech with log-only actions before real-device use.

Incoming text assembly is bounded to 1,023 bytes and handles fragmented messages;
request status is logged on USB serial. `completed` means server processing ended,
not that speech finished or a physical home action succeeded. No automatic action
retry is performed. Firmware does not receive transcripts or generated audio.

**No Apollo update is required.** Both clients use the existing `/ws/asr` protocol.
The attempted mandatory playback/cancel extension and its server changes were
removed after the maintainer clarified compatibility. VAD, AGC, pre-roll, and
noise gates all run locally and do not depend on new server behavior.

The existing protocol has no playback-active signal. ESP32 playback-aware muting
is therefore a documented gap; `request_status:completed` is not an adequate
substitute. Preserve Apollo's existing echo/speaker filters and validate acoustic
feedback on hardware. The Mac live adapter retains its local-player muting path.
Apollo's existing Echo Guard v2 compares finalized STT text with its two most recent
TTS chunks before LLM admission. Its default minimum length and estimated playback
window can let short or late echoes through; it does not avoid the STT processing
cost. The first ESP32 test retains this behavior without a server update.
Voice interruption stays deferred. Apollo's wake-once conversation gate applies
after ASR; the existing ten-second utterance ring remains.

## Verification and first hardware run

Host-only checks (C compiler and Python; no model or network):

```sh
python3 -m unittest discover -s client/esp32/tests -v  # from repository root
```

These compile the real firmware core with address/undefined-behavior sanitizers,
and the pinned WebRTC detector with AddressSanitizer. They exercise pre-roll/order,
pause/endpoint boundaries, gain/peak/noise properties, a public detector fixture,
stale/gapped/wrapped sequences, bounded fragmented status,
and the emitted `start` against Aria's actual validator. Server/Mac protocol tests:
`.venv/bin/python -m unittest discover -s tests -p 'test_client*.py' -v`.
The cross-build verifies ESP32 APIs/linking. Neither establishes microphone quality,
heap stability on-device, Wi-Fi continuity, echo rejection, or Apollo turn latency.

With the supplied antenna attached, a 75-second stream had no drops and Apollo
accepted/completed requests. Speech quality, echo and latency remain unverified.
See [test strategy](tests/strategy.md) and [board evidence](tests/hardware-2026-10-18.md).

For the first board run, use an Apollo instance with log-only/disabled home actions
and the intended Apollo-managed speaker sink. Verify:

1. USB startup, Wi-Fi association, and `existing Apollo protocol` in the stream log.
2. Short known French/English utterances produce the intended transcripts. Listen
   for reply feedback, missing speech onset, clipping, and sentence-pause splits.
3. Record speech-end-to-first-useful-audio plus ASR, queue, generation, TTS/playback
   timings on Apollo. One-second serial diagnostics include RMS/gain/noise, frame/
   turn counts, free heap, stack high-water mark, and peak frontend
   wall time excluding network writes (still includes task scheduling).
4. Interrupt Wi-Fi/Apollo during speech and restore it: a fresh session starts,
   old speech is not replayed, no growing queue, and a new utterance works.
5. Run with Internet blocked and LAN available; record attempts as well as successful
   traffic. Repeat with the expected board count before adding on-device features.

Record physical SKU, firmware revision, Apollo configuration, metrics, and failures
alongside the next experiment. Further detector/wake or other partition changes
need a measured latency/resource improvement without losing speech or correctness.


## Named-client protocol

The DHCP hostname and start-frame `client_id` are the same name:
`<ARIA_HOSTNAME_PREFIX>-<12 lowercase Wi-Fi MAC hex digits>`. It is stable across
reboots/reconnects and unique to the board. A capable server reuses the conversation
session until normal inactivity/goodbye; audio is never replayed. Legacy servers
ignore the extra field. See the [protocol](../../docs/agents/client-protocol.md) and
[monitor](../../monitor/README.md). Firmware identity changes require flashing; source
presence alone does not identify the running image.
