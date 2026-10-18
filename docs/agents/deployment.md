# Recovered Mac microphone / Apollo deployment

Verified read-only on 2026-10-17 using the local launch/sync scripts, SSH inspection
of Apollo's files and shell history, CPU/memory information, and listening ports.
This is configuration and historical-command evidence, not an end-to-end audio test.
Keep LAN addresses in the deployment scripts rather than duplicating them here.

For repeatable development tests, use the separate [E2E workflow](e2e-testing.md).
The configuration below records the recovered deployment, not the test harness.

## Topology and launch sequence

```text
Mac microphone -> client/macos/mic_stream.py -> LAN WebSocket :8000/ws/asr
  -> Apollo: Aria / ASR / speaker and echo guards / LLM / Piper
  -> LAN HTTP audio -> Sonos “Living Room”
Apollo inventory/actions -> LAN Jeedom MQTT broker :1883
```

The Mac [client_start.sh](../../client_start.sh) enables client VAD, disables AGC,
and points to Apollo's LAN address. The deployed copy has the same configuration:
VAD aggressiveness 3, three start frames, SNR threshold 8 dB, minimum RMS 0.005,
and silence threshold 400 ms. See [client documentation](../../client/macos/README.md)
for capture/resampling behavior. The exact microphone selected during past runs was
not recovered; the script does not specify a device override.

Apollo's shell history explicitly records:

```sh
cd /data/aria/dev/
. .venv/bin/activate
sh start.sh
```

The corresponding Mac invocation, reconstructed from the existing client script:

```sh
cd /Users/kaneda/dev/Aria
. .venv/bin/activate
bash client_start.sh
```

[sync.sh](../../sync.sh) records deployment by rsync from the Mac checkout to
`/data/aria/dev/` on Apollo, excluding `.git/`, `.venv/`, and Python cache files.
It includes `--delete`; it was inspected, not executed during this review. The
checkout and deployed configuration differ, so syncing would not preserve the
deployed script automatically. [start-osx.sh](../../start-osx.sh) starts a separate
Mac development server and is not the microphone-only launch path.

## Deployed configuration differs from the checkout

| Setting | Apollo `/data/aria/dev/start.sh` inspected remotely | Local [start.sh](../../start.sh) |
| --- | --- | --- |
| LLM provider | Ollama | llama.cpp-compatible |
| LLM endpoint | Loopback port 11434 | Loopback port 8001, `/v1` |
| Configured model name | `qwen2.5:3b-instruct` | `qwen2.5:3b-instruct` |
| Generation limit | 48 | 1024 |
| Router | No explicit router settings in script | Enabled; separate English/French models |
| Prompt | Explicit find/execute tool instructions | Short conversational responses |

The model-name field in the local script does not prove which GGUF a separately
launched llama.cpp server loads. Apollo's shell history contains manual
`python -m llama_cpp.server` commands on port 8001 for Qwen2.5 7B Q4_K_M and
Hermes-2-Pro-Llama-3 8B Q4_K_M, using context 4096, all detected CPU threads, and
experiments with `chatml` / `chatml-function-calling`. Command history establishes
experimentation, not successful startup, chronology relative to every script edit,
or the final preferred model.

The deployed Aria script enables plugins, inventory/actions, Piper TTS, Echo Guard
v2, and CPU speaker recognition with four threads. It uses `/data/aria/models`,
`/data/aria/voices`, `/data/aria/profiles`, `/data/aria/inventory.json`, and Piper at
`/opt/piper-tts/piper`. Voices are English Amy low and French Siwis low. Audio is
served from Aria's LAN HTTP address for the Living Room Sonos to fetch.

## Hardware and observed service state

- CPU: Intel Xeon E3-1225 v5, 3.30 GHz; four cores, four threads, one socket.
- Reported RAM: 65,783,072 KiB total (about 62.7 GiB usable; nominal 64 GB).
- CPU flags include AVX, AVX2, FMA; no AVX512F or AVX512-VNNI reported.
- Ollama was running and listening on loopback port 11434 during inspection.
- No listener was present on Aria port 8000 or llama.cpp port 8001 at that moment.
- `ollama list` included Qwen2.5 3B/7B/14B Instruct, Gemma 2B Instruct, and Phi3 mini.
  Installed model size/availability does not prove latency, quality, or residency.

No services were started or restarted, no models were loaded for inference, and no
device commands or microphone recordings were initiated. Exact ESP32 roles and
physical SKU, power consumption, and sustained concurrent capacity remain unverified.
The deployment now has a concrete CPU baseline: compute contention on four cores
needs particular attention; memory capacity is less restrictive than previously
unknown, but available memory and co-hosted workloads still require budgeting.

## ESP32 hardware identified by maintainer

On 2026-10-17 the maintainer supplied a product listing titled “Seeed Studio XIAO
ESP32S3 Sense Plus ESP32-S3 … OV2640 OV3660 …”. Treat this as identification of the
XIAO ESP32-S3 family with a camera/Sense configuration, not proof of the exact SKU,
camera revision, number of devices, flashed firmware, or deployed role.

Seeed documents Sense and Plus as separate variants. The standard Sense has a
dual-core ESP32-S3 up to 240 MHz, 8 MB PSRAM, 8 MB flash, and a digital microphone
on its expansion board; Plus has 16 MB flash. Do not apply the Plus flash size to
the Sense based on a combined seller title. Sources:
[Seeed specifications](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/),
[Sense microphone documentation](https://wiki.seeedstudio.com/xiao_esp32s3_sense_mic/).

On 2026-10-18 the maintainer confirmed microphone-only ESP32 use, with speaker
playback handled by Apollo, and authorized a simple client with hardcoded settings
([D-009](decisions.md)). [Initial firmware](../../client/esp32/README.md) now captures
PDM and streams bounded PCM to `/ws/asr`; the Mac remains available for development.
The subsequent parity request adds local WebRTC VAD, gain, pre-roll, optional noise
gates (D-012), using the existing Apollo protocol with no server update. Playback-aware
ESP32 muting remains a protocol gap. Physical echo behavior,
runtime heap, and end-to-end latency remain unvalidated. Further wake/processing
features will be chosen from measured results.
The camera remains inactive. This is source/build work, not deployed-device evidence.
Verify PSRAM/flash at boot and microphone capture before
assigning memory budgets; nominal PSRAM is not all free application memory.

At the existing 16 kHz mono PCM16 format, payload is 32,000 bytes/second per active
stream, before transport overhead; a 500 ms pre-roll needs 16,000 bytes. These are
format calculations, not measured throughput or a validated buffer configuration.
Validate the implemented VAD gating against speech onsets, noise, and reconnects;
local wake detection remains a future candidate to reduce server work.
