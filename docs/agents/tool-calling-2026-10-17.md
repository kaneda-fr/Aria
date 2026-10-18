# Tool-calling diagnosis on Apollo — 2026-10-17

The maintainer clarified that the Mac client/Apollo setup was working and the issue
was tool calling. This investigation exercises that path without microphone input.
It does not select a future architecture or change production application code.

## Profile and evidence

Run `20261017T212030Z-tools-88b103`, on the existing Xeon E3-1225 v5.
[Probe](../../scripts/e2e/tool_probe.py), [isolated launcher](../../scripts/e2e/tool_probe.sh).
Local artifacts: `~/Library/Logs/Aria/e2e/20261017T212030Z-tools-88b103/`.
`tool-report.json` contains each full model request/response, tool arguments/results,
captured publish, expected command/payload, request lifecycle, source hashes, model
digest, and timing. `deterministic-probes.json` records separate provider probes.

The real shared request worker calls real Ollama Qwen2.5 3B Q4_K_M and real inventory
matching, tool dispatch, parameter validation, and payload construction. Only MQTT
publication is captured. No MQTT connection, physical device, ASR, or TTS is started.
Twelve text cases cover EN/FR on, off, dimming, missing room, ambiguity, and conversation;
two additional cases enter via the finalized-voice handler. These two validate the
handler after transcription, not microphone/WebSocket/ASR end-to-end behavior.

The prompt, 48-token generation limit, temperature 0.1, and top-p 0.85 match Apollo's
deployed launch settings. History is deliberately disabled per independent case;
context is 2048 and only one model request runs at a time. Fixture inventory consists
of one kitchen lamp and one bedroom lamp, with French labels and on/off/slider
commands. It is synthetic, not the maintainer's private inventory. Tied candidate
ordering and model sampling are not fixed; this is diagnostic evidence, not a stable
accuracy estimate or a claim that every run produces identical commands.

## Reproduced failures

All 14 requests received exactly one accepted/running/completed lifecycle. Only
2/8 explicit control cases published the exact expected command/payload/count
(English on through each handler). Overall, 6/14 matched the publication oracle;
four of those merely required no publication and are not conversation-quality
passes. Fourteen of 32 model calls hit the token limit, affecting 11 cases.
Median request duration was 14.07 s, maximum 30.87 s including cold first inference;
these omit ASR/TTS and are not full voice latency or an isolated power benchmark.
The namespace had no external route; trace audit found 12 loopback destinations
and no external/DNS attempts. All test services exited; the shared 11434 listener
remained and no 11435/8000/8001 test listener remained.

- French kitchen **off** selected kitchen **on** (`201` instead of `202`), then claimed
  the light was off. French kitchen dimming selected the **bedroom** slider (`303`
  instead of `203`) while reporting kitchen success.
- Some clear control requests stopped after `find_devices`: the response described
  what it would do, but no command was published. Several model calls hit the
  48-token limit; the runtime discards `done_reason` and treats nonempty final prose
  as a completed assistant response. Increasing the budget alone would not fix
  wrong-device/wrong-action selection or make completion claims truthful.
- English requests for a nonexistent garage light and an unspecified light each
  published the kitchen-on command. Both should have produced no action and a
  no-match response or clarification. French factual/clarification replies also
  drifted into English; a no-publication result alone does not prove language quality.
- A direct `find_devices(query="lights", room="garage")` returned kitchen/bedroom
  commands despite no garage device. The documented room filter is only a score
  bonus. Low-score unrelated candidates survive, and there is no bilingual room
  alias normalization.
- Direct accented French `Éteins la lumière de la cuisine.` returned no candidates.
  A dimming query produced eight candidates for six unique commands: overlapping
  capability prefixes can duplicate commands. The five-command tool cutoff can
  then hide relevant commands. Each command is described as a separate “device,”
  inviting false ambiguity between on/off/slider commands for the same lamp.
- Direct execution of `not-in-inventory` reached captured publish and returned
  success. Missing inventory execution metadata permits fallback publication.
  The control probe with a known slider and value 150 correctly failed range
  validation and published nothing: retain that working check.

These failures are independent of microphone capture. Earlier unsaved acoustic
content remains unattributed; ordinary use of the Mac client does not establish
whether the user spoke during those tests.

## Fix order within the existing implementation

The maintainer subsequently made conversational latency a joint requirement
([D-007](decisions.md)). Measure latency with every correctness change below.
Prioritize compact candidate/result data and removal of redundant model turns;
evaluate concise confirmations derived directly from structured execution outcomes.
The current 14.07 s median excludes speech processing and cannot establish a
conversational experience. Track speech-end-to-first-useful-audio and completion
separately, with stage timings and cold/warm/load labels. Do not trade grounded
execution for speed or mask latency with speculative success messages.

1. **Ground selection and execution in inventory.** Reject unknown/read-only IDs and
   missing executable metadata before publish. Return compact, deduplicated device
   groups with distinct action IDs, value requirements, and ranges. Make explicit
   room filtering real, normalize French accents and supported EN/FR aliases, and
   return no match/clarification for missing or ambiguous targets. Test the full
   selection-to-execution trace, including the exact opposite-action failure above.
2. **Track tool outcomes per request.** Distinguish search-only, rejected arguments,
   publish accepted, and observed completion. Prevent duplicate side effects within
   one request; do not automatically retry a possibly executed action. Preserve
   budget exhaustion and tool failure instead of accepting unfinished promises as
   control success. Model prose alone cannot establish the requested action happened.
3. **Rerun the same EN/FR cases before tuning model or prompt.** Then extend to provider
   failures, multi-turn clarification, and the actual Mac audio transport. Measure
   any budget/prompt changes on the same Xeon; no larger model or hardware upgrade
   is assumed. Live device completion remains a separate, authorized check.

`publish_match` checks only exact captured IDs, payloads, and execution count. For
zero-action cases it proves no publication, not a useful clarification or correct
answer. Review responses and tool traces separately. Publish acceptance cannot
prove physical completion, even when the command is correct. D-001 remains open.

## Reproduction boundary

Stage `app/` and `scripts/e2e/` into a fresh private Apollo directory at
`$ARIA_E2E_RUN/src`, excluding caches and bytecode. On Apollo, launch the script using
the same isolated user-service pattern as the [E2E runner](e2e-testing.md):

```sh
systemd-run --user --collect --property=RuntimeMaxSec=900 \
  env ARIA_E2E_RUN="$ARIA_E2E_RUN" \
  strace -f -e trace=network -s 256 -o "$ARIA_E2E_RUN/network.trace" \
  unshare --user --map-root-user --net \
  bash "$ARIA_E2E_RUN/src/scripts/e2e/tool_probe.sh"
```

The script requires a dedicated staged directory and cached model files. It refuses
to run without a loopback-only namespace. Collect the report/logs, audit the network
trace with `scripts/e2e/run.py:audit_network`, and verify the service exited. The
existing `run.py --suite baseline` remains the conversation suite; it does not select
this diagnostic. No deployed launch script or shared Ollama service is changed.
