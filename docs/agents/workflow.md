# Development and validation

## Start a session

1. Read the root instructions, [project objective](project.md), and [status](status.md).
2. Inspect branch/HEAD and `git status --short`; distinguish inherited work from yours.
3. Read the applicable scoped `AGENTS.md` and use the [architecture map](architecture.md).
4. Verify relevant claims in code/tests. Follow [code intelligence](code-intelligence.md).
5. For historical decisions, search memories and validate project/date/provenance.

## Commands and prerequisites

For the ESP32 microphone client, use the pinned PlatformIO build and host checks in
[its README](../../client/esp32/README.md). Run
`python3 -m unittest discover -s client/esp32/tests -v` and
`pio run -d client/esp32`. These do not validate microphone capture or end-to-end
latency. Flashing and Apollo acoustic/transport measurements are separate steps;
keep actions disabled/log-only for baseline tests and record physical board identity.
For parity/compatibility changes, also run the `test_client*.py` suites and the shared
request/queue, conversation-gate, speech-language, and live/replay regressions.
Client features must work with the existing Apollo protocol without a server update.

For factual/model changes, add `python -m scripts.e2e.science_bench --server-run
<isolated-run> --output <private-json>` to the French/English conversation and
queue suites. Its criteria are evaluation-only; review meaning manually, including
false mechanisms and missing causes, rather than treating empty flags as accuracy.
`python -m scripts.e2e.phrase_surface --output <private-json>` audits synthetic
paraphrase routing without inference or actions; see the
[interaction-surface audit](interaction-surface-2026-10-18.md).
The isolated launcher accepts `ARIA_E2E_MODEL`, `ARIA_E2E_MODEL_DIR` and
`ARIA_E2E_SYSTEM_PROMPT` for paired experiments. `ARIA_E2E_OLLAMA_BIN` can select
a separately prepared local runtime without replacing the deployed binary. Prepare missing weights before
starting the loopback-only runtime; these overrides never trigger downloads.
Keep baseline and candidate inference sequential and compare individual turns,
tools and queued replies, not just a median dominated by deterministic controls.

For history/cache changes, run the French/English conversation suite plus
`python -m scripts.e2e.queue_bench --server-run <isolated-run> --output <private-json>`
on Apollo. Pair `ARIA_HISTORY_PREFILL=0` and enabled runs with identical prompts,
tools and exact history. Inspect per-turn submitted/accepted-to-first-WAV timings,
queue wait, prefill completion/cancellation, and exact replies; processor-only
latency or aggregate medians can hide regressions. The overlap probe keeps the
first request text-only and gives the queued request a captured WAV, avoiding
replacement of the first request's audio. No Mac audio or real actions are needed.

For server iteration with the Mac client, use the [Mac + Apollo E2E workflow](e2e-testing.md).
It stages an isolated test instance and collects evidence without replacing the
deployed scripts or executing home-device actions.

Use Python 3.11 for the repository's documented baseline. Inspect
[pyproject.toml](../../pyproject.toml) and [client dependencies](../../client/macos/requirements.txt)
before installation. Use the existing virtual environment if suitable.

For turn/language changes, run the Mac endpoint regressions and Apollo baseline in
the E2E guide. Review microphone transcripts against the actual played fixture:
correct answer keywords, one model call, and zero wrong-voice flags can still hide
extra content in a merged turn. Record automatic flags separately from manual review.
For user-controlled captures, require a confirmed spoken reference tied to the audio
hash. The displayed script and a “clean” listening note do not establish what was said.

```sh
python -m pip install -e .
python -m pip install pytest
python -m pytest tests/plugins/test_plugin_system.py tests/test_sonos_http_sink.py
python3 -m unittest discover -s tests -p test_context.py -v
python3 scripts/check_context.py
```

The selected tests cover plugin infrastructure and mocked Sonos behavior, not the
whole runtime. Model benchmarks require additional models/dependencies; use the
[routing guide](../../tests/routing/README.md) after verifying its flags and paths.
Do not run root-level integration scripts indiscriminately: some publish real MQTT
commands. Inspect their effects and use mocks unless live action is in task scope.

Starting the server requires model/configuration preparation:
`python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.
See the [user guide](../../README.md) and [environment reference](../ENVIRONMENT.md).
Do not overwrite existing local launch scripts with examples. No hardware validation
is implied by import, startup, unit-test, or context-check success.

## Finish a change

- Test changed behavior with a meaningful oracle; distinguish unit, model, integration,
  and live-device evidence. Record prerequisites and anything not run.
- Update the relevant context in the same change, following [maintenance](maintenance.md).
- Do not copy private inventory, transcripts, credentials, speaker identities, or
  deployment addresses into shared context. Use documentation placeholders.
- Before committing, run GitNexus change detection and inspect the actual diff.
  Never stage inherited untracked material indiscriminately.
- Report behavior changed, validation, remaining gaps, and context updated.

## Wake-enabled conversation checks

The default server now requires name activation on both voice and text. Run the
[FR/EN wake benchmark](wake-flow-2026-10-18.md) on Apollo with cached input fixtures,
log-only tools, and no Mac audio. It covers default expiry and queue overlap. Keep
legacy model-comparison runs explicitly ungated (`ARIA_WAKE_ENABLED=0`) or activate
each session; ignored HTTP success must never count as an answer/control pass.
Run `tests/test_conversation_gate.py`, `tests/test_spoken_wake.py`, and
`tests/test_speaker_identity.py` with shared-request, queue, history, control,
speech-language, echo/person, and live-adapter checks. Evaluate short spoken-name
recognition separately from the state-machine tests; preserve raw failed transcripts.
Keep literal transcript mismatches visible even when bounded vocative correction
makes the conversation work. Test multiple voices, inactive background/near-name
phrases, and confident-self/echo rejection; see the
[spoken-wake evidence](spoken-wake-2026-10-18.md). Automated runs stay audio-free;
only start the Mac live client when the maintainer has authorized mic/playback.

Echo changes also require `tests/test_echo_feedback.py`: capture-onset timing,
unknown versus attributed speakers, short numeric replies, stale references and
admission suppression. Replaying saved log decisions is separate from acoustic
validation. Pause live playback and ingress before generated-audio benchmarks;
restore the authorized output-only configuration after verification.

## Idle model checks

Run `tests/test_model_residency.py` plus warm-up, prefill, shared-request, tool,
speaker, and speech-language regressions. On Apollo, stage two isolated log-only
live profiles with the same source/model/prompt/history: `ARIA_LLM_KEEP_ALIVE_S=300`
and `-1`. Run `scripts/e2e/idle_bench.py` against their run directories with cached
voice fixtures and an idle period strictly greater than 300 seconds. It idles both
profiles without inference, then measures each language's first streamed voice
request and the existing conversation/control suite sequentially. It records model
process identity/RSS/CPU ticks and chat-call counts; compare literal ASR flags
separately from response/tool/state checks. Read [idle results](idle-residency-2026-10-18.md)
for provenance and limits. Keep Mac live capture paused during paired measurement,
then restore the authorized live configuration after validation.

## Named clients and monitor

For read-only lookup changes, run `scripts/e2e/lookup_bench.py` plus
`scripts/e2e/lookup_transfer.py --server-run <run> --output <private-json>` on Apollo.
The transfer suite optionally accepts `--fixtures <directory>` with a saved-audio
manifest. Compare exact results, selected device/room, action traces, language and
model rounds, alongside science and queue regressions; never treat generated prose
or an empty flag list as proof of factual correctness. See [lookup reply work](lookup-latency-2026-10-18.md).

Read the [client protocol](client-protocol.md) and [monitor guide](../../monitor/README.md).
Run identity, identity-server, legacy client-signals, monitor, shared-request, queue,
wake and speaker tests on Apollo. Run Mac endpoint tests locally and the ESP32 host
suite/cross-build. The live Mac runner can use `--audio-url` for the same LAN audio
ingress as ESP32; it updates the installed monitor's active-run pointer atomically.
For client mute, run `tests/test_client_mute.py`, `tests/test_client_mute_server.py`
and `tests/test_monitor_mute.py`; offline-list cleanup also uses `tests/test_client_cleanup.py`.
Check that cleanup preserves connected clients, mute deadlines and accepted work,
and that cleared names return on reconnect. `scripts/e2e/mute_bench.py` uses named synthetic
clients and saved PCM only; its default includes a real five-minute timer. It can
skip that long timer explicitly for a narrowly scoped confirmation rerun. Never open
hardware audio when it is disabled by the maintainer. The queue bench now prefixes
its initial request with Aria; follow-ups remain unnamed to exercise the wake window.
Verify physical reconnects keep names/session keys but change connection IDs, reject
duplicates, preserve only unexpired context and never replay audio/actions. Verify
client filters and request-correlated speaker scores in the browser. Keep tools
log-only and pause microphones while measuring paired inference latency.


Sentence voice routing: run `tests/test_sentence_voice.py`, `test_sentence_player.py`,
`test_speech_delivery.py`, `test_playback_delivery.py`, `test_mac_playback_delivery.py`
and existing speech/shared-request/queue/live tests on Apollo.
`scripts/e2e/sentence_tts_bench.py --output <private-json> --rounds 3 --playback`
uses captured WAVs and timer acknowledgments, never a physical player.
See [evidence and limits](sentence-tts-2026-10-19.md).
