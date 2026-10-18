# Grounded device control and truthful replies — 2026-10-18

The maintainer authorized fixing device/action selection and truthful replies, then
rerunning the French/English Apollo benchmark with **log-only tools and no Mac audio**.
This implements a narrow deterministic path for explicit light/shutter requests.
It does not settle D-001 or claim general natural-language control is solved.

## Implementation

[control.py](../../app/llm/control.py) recognizes bounded FR/EN imperative requests
for light on/off, shutter open/close, and whole-number percentage settings. Common
polite prefixes and suffixes are accepted. Explicit negation is acknowledged in the
request language with zero model calls and zero actions. Compound, conditional, scheduled, missing
value, ambiguous, and unsupported forms cannot obtain execution authorization.
Unrecognized phrasing can still enter the conversation model; generated action
calls on that path are rejected. Broader phrasing needs explicit evaluation.

[Inventory selection](../../app/plugins/inventory/selection.py) normalizes accents
and punctuation, matches exact names before partial name tokens, filters explicit
rooms and capabilities, excludes disabled devices, and deduplicates commands.
It removes the previous positive score for unrelated devices. Resolution sees all
matches: the tool display's top-five limit cannot manufacture a unique result.
The existing inventory API delegates to this selector; old fuzzy scoring is unused.

A unique match produces an internal, single-use authorization containing device ID,
command ID, capability, value, name/room, and a copy of the execution specification.
[Tool dispatch](../../app/llm/tools.py) requires this authorization and fresh matching
inventory before calling the action provider. Unknown IDs, opposite actions,
changed inventory, invalid values/ranges/templates, and repeated authorization are
rejected before the provider. Generated JSON cannot supply this internal object.
The existing low-level administrative `/api/actions/execute` endpoint/provider remain
separate interfaces; this request-bound guarantee applies to the assistant tool path,
not a new authentication boundary for every application endpoint.

[The shared processor](../../app/main.py) runs the same resolver and dispatcher for
voice and HTTP, preserving FIFO admission, per-request language, history, and speech.
Explicit valid controls make one tool-dispatch call and **zero model calls**.
`ARIA.Control.Resolved`, `control_status`, and `action_outcome` distinguish this
server selection from LLM-generated decisions. Existing `tool_calls_made` counts
actual tool dispatches, including this server-selected call.

A bounded memory of at most 128 `(session, attributed speaker)` entries retains the
last successfully recorded/submitted device for 120 seconds. This supports “that
same light” follow-ups without trusting model-generated prose or cross-session
history. Failed/ambiguous controls clear that reference. Speaker attribution retains
legacy behavior and is not authentication. No anonymous global device reference.

Replies come from the structured outcome in the command's language:

- Log-only success: the command was recorded for testing; no device changed.
- Provider success: the command was submitted; physical change is unconfirmed.
- Rejection: command rejected; no change confirmed.
- Missing/ambiguous device: say no match or ask which device/room, with zero actions.

No generative confirmation follows execution, and provider prose is not used as
proof of success. Existing Jeedom payload semantics are preserved (`{}` for binary
commands, `{"slider": value}` for supported percentage commands). Physical completion
tracking, model residency after idle, clock/capability answers, broader control grammar,
and general model factual accuracy remain separate work.

## Validation method

Tests use synthetic inventories and mocked providers; no broker connection or home
action. Coverage includes wrong/opposite IDs, missing/duplicate authorization, fresh
inventory changes, unknown rooms, accented names, disabled devices, finite/ranged
values, missing specs, compound/conditional requests, bounded references, false
provider success prose, and malicious model action calls on conversation inputs.
Voice and HTTP parity exercise the actual resolver and dispatcher, rather than
mocking away the newly added authority boundary.

GitNexus upstream impact was HIGH for tool dispatch and the new resolver, LOW for
the shared processor; inventory method callers were unresolved in the graph and
checked directly (`get_candidates` API and tool lookup). The change reaches both
shared request transports and diagnostics. Serena was unavailable without language
servers; targeted AST/source inspection was used. No files were staged or committed.

## Initial Apollo rerun

Initial run `20261017T230728Z-grounded-control-f18397` used a fresh isolated live server,
unchanged Qwen2.5 3B/80-token settings, the same cached inventory, and identical
baseline prompts and expected command IDs/payloads. Startup primed the tool prompt;
ASR, speaker profiles, and Piper were local. Mac microphone/playback stayed off.
The previous test instance was stopped; the deployed shared Ollama was untouched.

| Same baseline cases | Before | After |
| --- | ---: | ---: |
| Exact command traces, explicit controls | 1/8 | **8/8** |
| Correct command plus truthful control reply | 0/8 | **8/8** |
| Ambiguous/missing-room requests with appropriate abstention and reply | 1/4 | **4/4** |
| Median explicit-control start-to-audio-ready | 13.21 s | **0.71 s** |
| Explicit-control latency range | 8.26–30.87 s | **0.67–0.77 s** |
| Model calls across eight explicit controls | 19 | **0** |
| Mean generated control-reply duration | 12.68 s | **5.31 s** |
| Warm short-answer median, nine matched cases | 1.99 s | **1.99 s** |

These are processor-start-to-first-WAV-fetched timings, not speech-end-to-audible
playback. Control replies used fresh TTS cache entries in the new server. The
comparison is a small diagnostic run, not an SLO or proof of physical execution.
First short-answer coldness differed: the earlier model had idled/unloaded, while
the rerun followed warm-up immediately; that sample is excluded from the matched
warm short-answer comparison. No model-retention policy was changed.

All 28 baseline requests completed once with one audio response. Eight actions were
recorded exactly once; four ambiguous/missing-room requests produced zero actions.
Every recorded action had `dry_run=true`, `executed=false`. All control reply text
and TTS voices matched the request language. Automatic checks cleared 25/28;
manual review accepted 24/28. The remaining four failures are FR/EN clock and
capability questions: generated time, unhelpful answers, tool-shaped JSON, and an
irrelevant generated action attempt. That attempt was rejected **before the action
provider**, demonstrating the guard, not successful conversational intent detection.

Saved human EN/FR recordings still produced the confirmed words and same nearest
person, scores 0.528/0.651, `known=false/true`. With fresh TTS replies, explicit PCM
end-to-first-audio-ready measured **3.16 s EN / 3.42 s FR**. This bypasses microphone
acoustics and native endpoint detection; it does not test newly spoken control audio.
Actual voice/text control-handler parity is covered by the deterministic tests.

Private reports under the run ID retain `report.json`, `cases.json`, exact traces,
manual review, source hashes, benchmark output, and supplementary scripts/results.
Do not copy household names, command IDs, or speaker identities into shared context.


## Final-code validation

The initial eight challenge cases exposed poor negation replies despite zero actions.
Explicit EN/FR negation handling was added and 105 targeted tests then passed. Final
run `20261017T231411Z-control-final-8894d5` repeats the 28-case baseline, six supplementary
natural/voice turns, and eight challenge turns with fresh runtime source. The final runtime archive SHA-256 is
`b59560d71f7c80408a01068a17ff7ad609c0089c1670bf5a468b8c8e42ff062b`; hashes of all
five changed Python runtime files match local source. The final baseline retained
8/8 correct and truthful controls, median **0.70 s**, range **0.67–0.72 s**. Saved human
PCM with fresh TTS measured **3.07 s EN / 3.36 s FR** to audio-ready. The original
measurement boundaries and remaining utility/factual-quality failures still apply.

All **42 final turns** completed once with one audio response. The eight challenge
cases (polite requests, negation, compound requests, and references without prior
session context, in both languages) passed exact-command/abstention and manual reply
review. Two polite variants recorded commands; the other six produced none. Explicit
negation now acknowledges no command in the correct language, with zero model calls.
Cached replies in these repeat cases are excluded from the fresh-control latency table.
Across the complete final run, ten commands were recorded and none physically executed.

The supplemental natural conversation retained context and language but reproduced
two factual errors in French; its four audio-ready timings were 4.65–7.84 s. Cached
saved-voice repeats measured 2.55 s EN / 2.84 s FR and are distinct from fresh-TTS
measurements above. Manual review accepted 36/42 turns; the six remaining failures
are the four utility/capability cases and two French factual explanations. Do not
present successful execution-boundary checks as general conversation accuracy.

Final evidence: **105 targeted tests**, seven context-checker tests, context coverage
against HEAD, and diff whitespace checks passed. Runtime network audit observed 22
loopback destinations and zero external or DNS attempts under private-network isolation.
This is evidence for the isolated cached-inventory run, not production LAN/MQTT or
physical-device completion. Mac audio processes were absent. The final headless
log-only unit remains active with its existing three-hour lifetime; test clients exited.
