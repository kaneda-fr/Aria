# Free-form lookup investigation — 2026-10-18

Status: functional lookup improvement, **not accepted as a general quality/latency
no-regression fix**. All runs use Apollo CPU, cached Qwen2.5 3B, tools log-only,
saved inputs/text, no physical microphone/speaker and no real home actions.

Follow-up: [grounded lookup replies](lookup-latency-2026-10-18.md) remove a model
round trip and technical spoken IDs without changing the first model call. That
paired comparison preserves general replies; it does not repair the science
regressions documented here.

## Cause and change

The installed Ollama 0.14.2 model template renders `.Function` as a Go struct rather
than JSON. Raw rendered-prompt probes reproduced malformed tool calls with an
unquoted function name that native chat parsing dropped. Startup now prepares a
local alias using the whole tool's JSON serialization and a quoted name example.
Weights, sampling and model residency are unchanged; no pull or request-time repair.
Alias creation uses the isolated run's model metadata directory, reusing cached
blobs. It does not overwrite the shared deployed model.

Lookup output groups commands under actual device IDs, distinguishes devices from
commands, carries request language, and explicitly marks current state unknown.
No lookup is evidence of current physical state or executed control.

## Paired evidence

Private runs: initial reference `20261018T121137Z-lookup-reference-1748d4`, final
lookup candidate `20261018T123256Z-lookup-final-25f242`. Twenty-six conversation/control
cases pass their functional checks in both. The final candidate returns six grounded
capability lookups; the reference has three empty failures plus language/count errors.
The first intermediate candidate remained faulty and was rejected.

| Lookup | Reference audio ready (s) | Candidate (s) | Later repeat (s) |
| --- | ---: | ---: | ---: |
| FR 1 | 14.12, faulty content | 33.75 | 15.17 |
| FR 2 | 27.15 | 17.97 | 13.86 |
| FR 3 | no audio | 18.28 | 14.97 |
| EN 1 | no audio | 12.62 | 12.37 |
| EN 2 | no audio | 12.70 | 11.94 |
| EN 3 | 22.83 | 12.25 | 10.46 |

The later repeat is `20261018T132742Z-mute-confirmed-3e73e1`, with client/monitor/mute
work present. All 32 functional checks pass; 26 ordinary/control cases have median
request-to-first-audio 0.76 s and six lookups 13.11 s. This is synthetic/API processing
and generated audio readiness, not audible household end-to-end latency. The worst
initial French lookup included unusually slow token generation; concurrent workload
was not recorded well enough to attribute it. Later repetition does not erase it.
Technical command IDs/names are still spoken, which is poor conversational UX.

The original paired queue comparison was approximately 4 s for both profiles
(candidate 4.39/4.30/3.88/3.89 s versus reference 4.24/4.29/3.87/3.95 s).
The resumed maintained queue harness initially omitted wake activation and waited
for nonexistent audio; its initial request now includes “Aria”. That failed harness
attempt is not evidence of a server queue regression.

## General-answer regression remains

Paired extra-quality runs used 24 science/uncertainty questions and two saved voice
fixtures. Reference extra run: `20261018T125210Z-lookup-reference-extra-865eb8`.
Both voice replies remain 84; first audio after saved PCM ends is EN 2.95→2.96 s,
FR 3.25→2.68 s. Raw wake-name transcript mismatch flags remain explicit.
Science audio-ready median rises 3.97→5.24 s. Replies change even without a tool call,
because the offered tool prompt changes. French ice and vacuum answers improve,
but the previously correct French magnet explanation becomes vague/incorrect.
Both versions incorrectly claim two north magnetic poles attract in English.
French sky, seasons, rain, metal and day explanations remain faulty; the new English
rain wording also incorrectly describes clouds becoming heavy enough to fall.
French sunset stays scientifically wrong even though its language improves.

Do not patch these questions with literal canned answers or present benchmark
completion as correctness. Preserve held-out paraphrase/science checks while reviewing
tool exposure, model/prompt choices, and reducing the second model round trip for
grounded capability results. The broader architecture decision remains open.
