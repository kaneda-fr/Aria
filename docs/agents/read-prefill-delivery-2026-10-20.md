# Read clarification and background prefill delivery — 2026-10-20

The delivery branch starts from main's sentence-voice revision. It contains the
read/clarification prerequisites plus the tested background cache integration.
Unrelated monitor, audio ingress/playback, ESP32, deployment and shutter-stop work
remain in their original working trees.

## Behavior

- French/English state and capability requests resolve from inventory without
  foreground inference. Native names, explicit rooms, device types and channels
  constrain selection; synonyms use a small bundled vocabulary, not a model.
- A source question can be answered by its offered name or ordinal. The original
  measurement/channel survives clarification. Duplicate names fail closed.
  Subsequent references are scoped by session and speaker.
- The optional read provider uses a private Unix-socket bridge. That bridge accepts
  at most eight enabled info-command IDs and publishes only MQTT get requests.
  It never publishes actions; measurement timestamps remain explicitly unknown.
- Ordinary inventory providers keep their original tool schema. The isolated live
  adapter selects the read provider only with `ARIA_JEEDOM_READ_SOCKET`. Warmup uses
  the actual schema, and shutdown closes the read connection.
- After captured audio is ready, read replies may prepare the exact retained history
  in the background. A 150 ms grace period and local residency check avoid immediate
  contention/cold-model loading. Work is staged, cancellable, and capped at 20 s.
  It never dispatches tools or stores generated output. New admitted work wins.
- `ARIA_HISTORY_PREFILL=0` disables preparation. History remains three exchanges by
  default; its contents and the foreground generation configuration are unchanged.

## Target-server evidence from the integrated development snapshot

Prior Apollo measurements used cached models, synthetic state observations,
log-only actions and captured WAVs. No physical microphone/speaker or device action
was used. Private manifests/reports remain in the maintainer's read-integration
artifact bundle; this is not a new deployment of the PR branch.

| Paired time to first captured WAV | Reference | Integrated |
| --- | ---: | ---: |
| 28 read/capability cases, median | 4.453 s | 0.875 s |
| General question after clarification, French | 4.638 s | 1.814 s |
| General question after clarification, English | 4.236 s | 1.450 s |

All eight simulated action IDs/payloads/outcomes match. All 28 non-read/non-clock
replies are identical. Six queue trials pass (largest paired increase 114 ms).
Eight immediate same/other-client follow-ups have WAV-delay differences −69 to
+31 ms; prior simulated audio masks model completion, whose largest increase is
394 ms. One unchanged English capability reply has a 373 ms synthesis outlier.
These are small paired samples, not a statistical or imperceptibility guarantee.

After the grammar correction, 20/20 read/negative/capability probes and four control
probes pass. Clarification checks pass 33/34; the remaining French factual error is
pre-existing. Two primary French science answers and six short French input turns
answered in English also remain known limitations. Duplicate-name questions can
lack useful room labels even though selection fails closed. Background inference
adds CPU work; 134.3 s summed preparation wall time is not an energy measurement.

## Reproduction and boundaries

Run the safe mocked suite from a clean checkout:

```sh
python -m pytest tests/test_*.py tests/plugins/test_plugin_system.py -q
python scripts/check_context.py --base origin/main
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/aria-wheel .
```

The read/cache harnesses in `scripts/e2e/read_requests_bench.py` and
`scripts/e2e/read_cache_bench.py` require an isolated server and synthetic read fixture;
do not aim them at live home devices. They check exact selection, immediate
follow-ups, per-client references, queue wait and captured WAV delay. Additional
source-choice transfer cases are recorded in the private Apollo evidence.

GitNexus pre-port impact is HIGH for cache scheduling, command selection, tool
lookup, capability reporting and control resolution/rendering. Callers include the
shared worker, inventory provider, live adapter and evaluation entry points. Dynamic
provider/lifespan references are incompletely resolved; passing tests complement
rather than replace this graph. No live promotion is part of this commit task.

## PR review corrections and new live evidence

The first review reproduced English noun-led state requests using French replies
when an inventory label contained accents. Both read paths now use the same
request-syntax language cues; six FR/EN cases assert exact info IDs, reply language,
TTS language and saved history with an accented device name. The optional bridge
switch and startup/shutdown requirements are documented in the
[environment reference](../ENVIRONMENT.md). No model call or detector dependency
was added. Name-only contextual language remains a separate known limit.

The coordinating task also reviewed a later real conversation, separately from
this PR's offline validation. It found unsupported household-wide status, empty
catalog enumeration, general-weather routing and spelling variants of offered
sources; canonical names/ordinals resolved in offline replay, but there were no
successful live state reads. No paired baseline for those utterances exists, so
this establishes coverage gaps, not regression attribution. These broader changes
remain separate rather than being silently folded into this review correction.
The maintainer has stopped the backend, bridge and Mac audio; nothing was restarted.

Second review corrections bind source-choice history to the contiguous preceding
turns of the current speaker. The full model/conversation history is unchanged;
this is reference scoping, not speaker authentication. MQTT timeouts quarantine
the affected command: no later get is published until its outstanding non-retained
response is received and discarded. Retained values cannot clear quarantine; other
commands remain usable. The protocol has no request nonce, so a timer cannot prove
that an old response has left the wire. Quarantine lasts for the bridge process
and may remain indefinitely when no late response arrives. This protects repeated
reads after timeout; it does not establish sensor measurement time or arbitrary
transport-duplicate/restart correlation. Healthy reads add no wait or model work.
