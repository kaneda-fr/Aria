# Inventory-grounded lighting questions

## Problem and implementation

The maintainer's live questions about household lights reached unrestricted model
lookup. A request for current state could become a name search, return no devices,
or mix real lights with similarly named network switches. A French spoken
`statut` was also transcribed as `statue`, losing the read intent. These are
server interpretation failures; the actual lighting states already exist in Jeedom.

The candidate adds [light_queries.py](../../app/plugins/inventory/light_queries.py)
to the shared voice/text worker, ahead of generic lookup. It recognizes reusable
French/English lighting questions: inventory, counts, state, on/off predicates,
capabilities, room subsets, explicit channels and adjacent follow-ups. Device and
command identity come from enabled, typed inventory entries. No model inference,
new model, external service or device action is used for these reads. Established
single-device reads retain their existing path when already resolvable.

Before classifying a read, the existing control grammar is consulted with **no
inventory**. Recognized imperatives, references and refusals return to the established
control path; this check cannot select a device or authorize execution. In particular,
“Now turn that same light off” must remain a command rather than an off-state query.

State requests select at most eight typed information commands before any I/O.
An oversized scope asks for a room instead of reading a truncated subset. Lists
have a separate 24-device display limit. Unknown names/rooms, ambiguous singular
references and unsupported historical/complex requests ask for clarification;
they cannot broaden into a household read. Explicit device/channel selection is
revalidated by [ReadOnlyInventory](../../app/plugins/inventory/state_read.py).
One bounded bridge read supplies group values. No write command is passed to it.

Responses count devices separately from channels and preserve partial, mixed and
unavailable states. A device with one on and one off channel is partly on; an off
channel remains visible in an off-state question. Missing values never become
off. Known positive evidence can answer an existential question despite other
unavailable values; universal claims require complete supporting evidence.
Retrieval does not establish a sensor's observation timestamp or physical completion.

Follow-ups reconstruct scope only from adjacent, marked lighting responses in
the worker's same-speaker/session history. Only short read/count ellipses, choices
and inventory-grounded locations inherit it. Unrelated arithmetic, factual or
language requests return to their normal paths. No global device context is added.
Household phrases are scope syntax rather than an equipment named “Home”. English
`garden` uses the existing shared French/English lexical alias mechanism. Spoken
`statue` is normalized only as a leading state label with a lighting noun and no
matching declared object name; raw transcripts/history remain unchanged. This
does not correct arbitrary ASR errors or text API typos.

## Validation and limits

Source: branch `codex/lights-questions`, based on merged main `7ba8250`.
The canonical development checkout contains unrelated inherited work and was
preserved. New tests use synthetic device identities and exact read/action oracles.

- Local safe suite: **753 passed**, including the new lighting and safety cases.
- Target server safe suite: **731 passed**; five Mac-only modules excluded.
- Broad runtime pass: **38/38 lighting cases**, eight exact simulated actions and two
  queues pass their functional oracles. Two literal spoken-name mismatches remain
  separately recorded; they are handled by the existing wake alias.

Tests cover both request transports, speaker/session isolation, ambiguity and
ordinal clarification, unknown rooms/channels, disabled and non-light devices,
budget refusal before I/O, malformed/missing states, mixed channels and language.
The complete existing suite guards action selection, one-use authorization,
read-bridge quarantine, history, queueing and speech behavior.

The server comparison uses the actual cached catalogue with deterministic synthetic
observations through a separate local read-only Unix socket. It checks exact info
IDs and expected values, not just successful HTTP responses. Saved offline TTS WAVs
exercise ASR/speaker/wake/request/TTS on the target CPU. Physical microphones and
players remain stopped during the comparison; tools remain log-only. These tests
cannot prove acoustic recognition, real sensor freshness or physical device effects.

No claim of unrestricted natural-language understanding is made. New lexical
constructions should be evaluated against held-out paraphrases and action/read
negative cases, rather than adding answers for individual benchmark questions.

## Target-server results

The broad paired runs used the same four-core Xeon, cached Qwen2.5 3B model/runtime, tool schema,
catalogue, synthetic values and six-message history. Baseline main had **9/32**
functional lighting passes. The final source passes **32/32 common cases plus six
held-out existential/collective variants**. The 60-case candidate also includes
22 non-lighting turns: their literal replies are unchanged from baseline. Four of
the lighting cases stream saved synthetic PCM through ASR, speaker filtering, wake
admission and TTS. English `Arya` versus reference `Aria` remains a raw transcript
mismatch in two cases; wake admission and state selection succeed in both versions.
All requested FR/EN TTS language checks pass.

| Measurement | Baseline | Final |
| --- | ---: | ---: |
| FR common text lighting median, accepted to first WAV (14 cases) | 1.668 s | 0.692 s |
| EN common text lighting median, accepted to first WAV (14 cases) | 0.793 s | 0.707 s |
| Exact simulated action median (8 cases) | 0.765 s | 0.748 s |
| FR queued reply, accepted to first WAV | 4.835 s | 4.897 s |
| EN queued reply, accepted to first WAV | 4.524 s | 4.423 s |

Baseline lighting timings include incorrect replies and are not a quality-equivalent
speed comparison. Final text-lighting maxima are 1.339 s FR and 3.016 s EN: the
first full English household answer synthesizes substantially more text than the
old failed lookup. Repeated answers benefit from the WAV cache. Saved-PCM-end to
first-WAV medians are 0.977 s FR and 0.867 s EN (only two cases per language).
These measurements exclude real microphone endpointing and physical playback;
this is one paired pass, not a production latency SLO or statistical guarantee.

The exact eight dry-run command IDs, payloads and execution counts pass unchanged.
Both queues return the correct answers, preserve accepted work and reject overload
with HTTP 429. Network traces show no non-loopback destinations or DNS attempts
for the isolated baseline/candidate runtime (42/55 observed loopback destinations).
This scoped cached-startup test does not certify every optional provider/client.

Two candidate defects were rejected before deployment: unrelated questions inherited
a lighting scope, and a prefixed reference command was taken for a state read. The
final guards have regression tests; no benchmark-specific answer was added.

**Remaining performance/quality limits:** the first fresh French factual request
after a short answer took 3.615 s versus 1.827 s in baseline. Prompt evaluation is
similar (0.923 versus 0.979 s); total model time contains about 1.7 s beyond
load/prompt/generation while earlier speculative prefill drains. The prior candidate
also showed this handoff cost. Exact history and prefill policy were preserved;
this deserves a separate cache-cancellation/scheduling fix. The two pre-existing
French science answers remain inaccurate/underspecified and unchanged; empty
automatic flags are not evidence of general factual correctness.

## Initial live deployment checks

The validated uncommitted branch source is deployed with a source-hash manifest,
log-only actions and the existing real read bridge. It is based on main, **not an
exact main checkout**. Five live lighting information channels return successfully;
one standalone bridge batch took 0.152 s. Four deployed FR/EN API checks take
0.124–0.309 s without speech, return exactly those five channels and correctly report
four devices separately from channel count. Two real queued lighting reads pass
with first captured WAV at 2.719/2.695 s, correct preceding arithmetic and overload
rejection. Physical audio was stopped throughout these tests. No device action is
executed. Sensor observation timestamps remain unknown.

Mac microphone/playback restoration is checked separately after the above tests.
The original live-session deadline is retained; no testing automation is extended.

## Live phrasing follow-up and final source

After the first Mac restoration, the real client successfully read a named light's
two channels (286 ms processing), but the raw request `donde moi le statut des
lumières` was rejected. This is evidence that the initial named cases did not cover
all conversational framing, not evidence that the Jeedom values were missing.

The final source recognizes reusable FR/EN request frames (`donne-moi`, `me donner`,
`m'indiquer`, `show me`, `give me`, and related forms). It separates “show the state”
from an equipment list. Voice-only repair accepts a single edit of the read verb
`donne`, followed by `moi`, with an explicit state label and a typed lighting noun.
Declared equipment names take priority. No fuzzy equipment/room selection or new
model was added; raw requests remain in history/logs. The bounded `statue` state-label
repair also handles its leading article. Negation, unknown scope and other explicit
device domains retain their guards/handlers.

Latest target transfer: **36/36 functional cases** plus **two queues** pass. This
includes 12 framed state requests, six refusal/unknown-scope cases, eight exact
dry-run commands, eight context/maths turns and two new saved-PCM requests. Thirty-two
API cases deliberately suppress speech; two text cases and two saved-audio cases
verify TTS delivery/language. The English saved request still transcribes `Arya`;
the literal mismatch is retained and wake/state processing succeeds. Queue first-WAV
times are 4.848 s FR / 4.519 s EN. Four additional single-device framing cases pass
in both the local and target-server suites.

The final deployment passes six real Jeedom reads (0.245–0.314 s without speech),
including French/English frames on a single-channel named light, with exact info IDs
and zero model calls. Two real queued lighting reads pass at 2.784/2.535 s to first
captured WAV; preceding arithmetic and overload rejection also pass. Mac native
capture and playback are restored only afterward, with the existing read-only HTTP
reconnect mitigation and original session deadline. ESP32 ingress/Sonos remain off.

Shutter status questions and merged ASR words observed in the live conversation
remain a separate issue; this lighting change does not claim to repair all device
classes. The speculative prefill handoff described above is also still open.

## Reproduction

Run `python -m pytest tests` locally with the documented client dependencies.
For a focused check, run `python -m pytest tests/test_light_questions.py
tests/test_light_question_safety.py tests/test_light_paraphrases.py tests/test_light_request_frames.py
tests/test_read_request_path.py
tests/test_grounded_control.py tests/test_lexical_selection.py`.
Use the [server E2E procedure](e2e-testing.md) for paired model/queue measurements,
six history messages, cached local models, log-only tools and physical audio paused.
Private run manifests retain source hashes, catalogue/value fixture, WAV hashes,
exact replies/events, network traces and per-turn queue-inclusive timings.
