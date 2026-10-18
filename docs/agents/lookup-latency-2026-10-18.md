# Grounded lookup replies — 2026-10-18

Deployed and evaluated on Apollo. Tools remain log-only; only text and saved/synthetic
PCM were used, with physical microphone/speaker ingress off. This fixes lookup reply
latency and grounding; general science and some speech interpretation remain open.

## Implemented change

`app/llm/lookup_reply.py` renders a batch of `find_devices` results directly after
the model chooses its search arguments. This removes the second model round trip
for read-only lookup batches. User wording still goes through the existing model;
the renderer has no sentence matcher or benchmark-specific device names. First-call
prompt, offered tool schema, model weights, sampling and selection remain unchanged.

Replies name devices/rooms and translate provider capability identifiers into short
French/English descriptions. They do not speak internal command IDs, infer current
device state or claim an action. Same-named devices retain distinct inventory IDs.
Display limits are explicit; unavailable inventory, no matches and incomplete
identity are different outcomes. Unknown capabilities get a generic description,
not invented semantics. Multiple searches are deduplicated by device and capability;
partial failures/no-match searches remain explicit.

Mixed search/action batches retain the existing authorization guard. Generated
actions cannot authorize themselves. The shared request processor still handles
both voice and text, writes the exact final response into session history and
uses request-specific TTS language. No new background inference or runtime Internet
dependency is added. Existing general science defects are not fixed by this change.

## Evaluation

The fresh reference is `20261018T133802Z-mute-verified-bbb07c`. Its six lookup
audio-ready times are FR 15.08/13.57/14.95 s and EN 12.74/12.03/9.93 s, each with two
model calls. Several answers speak technical IDs/capability names. These measurements
exclude physical playback. The original [lookup investigation](freeform-lookup-2026-10-18.md)
retains prior template experiments and general-answer regressions.

The maintained `scripts/e2e/lookup_transfer.py` adds held-out paraphrases, shutters,
missing-device searches and current-state questions, optionally with saved speech.
It checks tool traces, grounded targets, lack of actions and spoken IDs, and explicit
state uncertainty; semantic correctness/language still require reviewing exact replies.
The candidate is `20261018T141513Z-lookup-fast-731426`. Its source archive SHA-256 is
`a221e11c3e59139bed533e4140387a5c273f6ad07686bfce0d7041e11d1cfa3f`.
Both runs use cached Qwen2.5 3B on Apollo CPU with the same first-call prompt/schema,
model settings, inventory and input fixtures. These are small paired samples, not
production percentiles or physical speech-to-speaker measurements.

| Lookup | Reference audio ready (s) | Candidate audio ready (s) |
| --- | ---: | ---: |
| FR 0 | 15.08 | 5.15 |
| FR 1 | 13.57 | 3.81 |
| FR 2 | 14.95 | 3.98 |
| EN 0 | 12.74 | 4.46 |
| EN 1 | 12.03 | 3.75 |
| EN 2 | 9.93 | 3.75 |

French median falls 14.949→3.981 s (73%); English 12.030→3.750 s (69%). All six
lookups use one model call instead of two. The removed second generation previously
took 5.61–10.03 s; the first call took 3.57–4.07 s. Ordinary model answers do not
enter the renderer.

- **32 functional cases:** no flags in either run; eight exact log-only controls
  remain correct. All non-clock, non-lookup replies are identical.
- **26 quality cases:** all replies identical, including 24 science/uncertainty
  questions and two saved arithmetic recordings. Science median 3.825→3.791 s;
  maximum 7.456→7.432 s. English saved voice 2.376→2.372 s; French 2.631→2.676 s.
  Raw wake-name transcription flags remain, but both arithmetic answers are 84.
- **Four queued trials:** median submitted-to-first-audio 4.014→4.031 s (+17 ms).
  All answers/status checks pass and overload returns 429. French individual times
  are 4.166/4.219→4.341/4.232 s; English 3.862/3.861→3.830/3.830 s. This small
  sample shows no broad latency regression, not a zero-variance guarantee.

The additional transfer suite has eight text cases plus two saved voice fixtures:

| Input | Reference → candidate audio ready (s) | Candidate outcome |
| --- | --- | --- |
| FR lamp paraphrase | 13.611→3.566 | Grounded capabilities |
| FR shutter paraphrase | 15.600→4.786 | Grounded capabilities |
| FR current state | 14.216→3.412 | Explicit state unknown |
| FR absent fixture | 7.684→4.929 | No match, asks name/room |
| EN lamp paraphrase | 12.911→3.248 | Grounded capabilities |
| EN shutter paraphrase | 14.778→4.262 | Grounded capabilities |
| EN current state | 12.345→3.191 | Explicit state unknown |
| EN absent fixture | 10.070→4.781 | No match, asks name/room |
| EN saved speech | 12.904→2.979 | Target found despite raw ASR mismatch |
| FR saved speech | 6.284→6.225 | Existing interpretation failure unchanged |

All nine successful lookups use one model call, avoid spoken internal IDs and take
no action. The reference English state case's `state_uncertainty_missing` flag is
a heuristic false positive: its exact reply already said no current state was
known. The reference French state case really omitted uncertainty and spoke IDs.

The saved French fixture asks what the inventory lamp permits. ASR splits the
device name into ordinary words; the unchanged first model call skips search and
responds in English with a clarification. Both versions have `transcript_mismatch`,
`missing_search` and `missing_target`. This is not fixed by formatting tool results.
The English fixture finds the device despite its raw transcription mismatch.

## Verification and remaining limits

Apollo: **304 unit/integration tests passed**, four existing warnings, 7.67 s.
Coverage includes provider-backed capability rendering, both request transports,
mixed search/action rejection, client cleanup/mute, monitor guards and shared queue.
JavaScript syntax, whitespace and maintained-context checks also pass.

Post-benchmark audits matched archive/runtime sources. Network traces saw only
loopback destinations and no attempted external/DNS destinations in either isolated
application run. Reference 112 accepted requests and candidate 84 each have exactly
one terminal outcome; overload rejection is explicit. This is scoped process/run
evidence, not a certification of every host service or normal production networking.

Private evidence remains under `/data/aria/e2e/reports/<run-id>` and
`/Users/kaneda/Library/Logs/Aria/e2e/<run-id>`: `speed-*`, `quality-*`, `queue-*`,
`transfer-*`, candidate `comparison.json`, and both `audit.json`/server evidence.
Fixtures use cached Apollo Piper and saved PCM; no sound was played or captured.

General science errors remain, including incorrect explanations of seasons, sky,
rain and magnetism. Exact equality preserves current quality; it does not make
those answers correct. First-search mistakes, arbitrary synonyms and state retrieval
remain open. No-match search now asks for clarification instead of spending another
model turn trying a revised query. Unknown capability vocabulary is described
generically; extend the provider mapping when new supported capability types appear.

The monitor deployed with this run adds offline-client clearing and a compact layout.
At 1280×720, the empty-list log viewport starts at 344 px rather than 740 px. Browser
checks covered desktop, 375 px mobile and short landscape with no horizontal overflow,
filter reset, reconnect visibility and preserved mute deadlines. See the
[monitor guide](../../monitor/README.md) and [client protocol](client-protocol.md).

## Follow-up: French transcription failure isolation, 14:29–14:31 UTC

Six text-only probes on the same deployed run varied the failed saved transcript.
Each used a fresh session, the normal shared request path and log-only tools; generated
WAVs were captured without playback. No application, prompt or model change was made.

| Input variation | Search/target | Audio ready (s) |
| --- | --- | ---: |
| Exact raw French transcript | Missing; English clarification | 6.716 |
| Intended French text, quoted names | Correct; French reply | 3.681 |
| Intended French text, unquoted names | Correct; French reply | 3.294 |
| Raw transcript with only device name corrected | Correct; French reply | 2.887 |
| Raw transcript with only room corrected | Missing; English clarification | 6.852 |
| English unquoted equivalent | Correct; English reply | 3.231 |

This isolates the split device-name spelling as the trigger for this fixture;
quotation marks and the malformed room are not required to explain it. Failure
happens before `find_devices`, so improving provider search alone cannot repair
this case. The same raw transcript also fails with the API's unknown-speaker tag,
corroborating the prior attributed-voice failure. These are text probes, not a new
ASR or acoustic evaluation, and they do not establish behavior for other names.

Next candidate: evaluate inventory-derived entity hints/recovery before model tool
selection, against held-out names, similar-name ambiguity, non-device conversation
and both languages. Preserve raw transcripts, require clarification for ambiguity,
keep lookup separate from action authorization, and measure prompt/queue cost before
deployment. No fixture-specific substitution belongs in production. The current
runtime is unchanged; private `lookup-diagnosis.json` contains exact inputs/replies.
