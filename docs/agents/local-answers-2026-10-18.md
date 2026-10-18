# Clock, capability, and conversation regression checks — 2026-10-18

The maintainer requested continued fixes with quality and latency measured at each
step. All inference and speech synthesis run on Apollo's existing CPU-only Xeon.
Mac capture, playback, and live client stay off. Tools remain log-only inside the
isolated loopback network namespace; no physical control is claimed.

## Changes

`app/llm/local_answers.py` answers explicit French/English clock questions using
the server clock. `ARIA_TIMEZONE` selects an IANA zone; unset uses the host zone.
Explicit Zurich/UTC requests override it. Invalid configuration yields an unavailable
answer. Clock handling no longer depends on enabling the optional router.

Capability questions use the actual started inventory/action providers and tool
definitions, with no model or tool calls. Disabled/unavailable capabilities are not
advertised. Technical names are returned only when asked for tools; ordinary
capability questions receive spoken descriptions. The test provider declares its
log-only mode so descriptions do not imply real device changes.

Both paths retain the shared queue, session history, and current-request TTS language.
They are narrow grammars; unsupported wording still reaches the model. They do not
solve arbitrary current-world questions or factual hallucinations. Control selection,
single-use authorization, and truthful outcome rendering are unchanged.

## Method

Compare the same 42 turns with the [grounded-control baseline](grounded-control-2026-10-18.md):
28 primary cases, four natural explanatory turns, two repeated saved human PCM
turns, and eight control challenges. New instances use fresh TTS caches and the same
model, prompts, history limit, voices, recognition profiles, and inventory. Repeated
PCM/control replies within a run can hit TTS cache and are reported separately.
Timing is server request processing to the first WAV fetched by a silent capture
sink, not speech-end-to-audible playback. PCM timing starts after the last submitted
saved audio chunk; endpointing/acoustic capture is not exercised.

Manual semantic review remains necessary. The capability oracle now rejects spoken
tool JSON even if it contains both expected names. A separate model-only
`scripts/e2e/quality_probe.py` compares the same 16 French/English factual/uncertainty
prompts with and without tools. It requires the isolated log-only server, requests
no speech, and records exact answers and model timings. Alternating schemas changes
prefix-cache behavior, so this diagnostic is not a stable conversation latency test.

## Clock-only checkpoint

Private run: `20261017T232419Z-clock-404aac`. Manual acceptance improved 36/42 → 38/42;
both clock answers now pass. Controls remain 8/8 exact and truthful; all eight
challenge cases pass. The same capability and French science failures remain.

| Group | Before median | Clock-only median | Count |
| --- | ---: | ---: | ---: |
| Explicit controls | 0.702 s | 0.717 s | 8 |
| Clock | 3.013 s | 0.556 s | 2 |
| Warm short answers | 1.969 s | 1.966 s | 9 |
| Natural explanations | 5.804 s | 5.866 s | 4 |

The next topic-switch answers regress from about 2 s to 4.02/4.35 s: local answers
avoid model calls but leave more history to prefill at the next model turn. The warm
short-answer tail therefore regresses despite an unchanged median. Do not report
blanket latency non-regression. The primary 28-turn sum falls only 48.68 → 47.81 s.
Focused tests briefly overlapped the clock run, so tiny timing differences are noise;
the final run separates tests from latency measurements. The network audit observed
21 loopback destinations and no external/DNS attempts.

## Capability checkpoint

Private run: `20261017T232941Z-capabilities-21eaae`. All 28 primary cases and eight
challenges pass; 40/42 total are manually accepted. The two French science errors
are unchanged. Capability answers reach audio in 0.779/0.823 s with zero model/tool
calls, versus 6.214/4.753 s before. Clock median is 0.551 s; control median 0.737 s.
The 28-turn latency sum falls to 41.01 s, but the topic-switch turns regress further
to 5.79/4.96 s. Long capability replies add unnecessary speech and prefill work.
The next iteration returns only names when the user explicitly requests a tool list,
while general capability explanations retain the log-only qualification. Network
audit: 23 loopback destinations, no external/DNS attempts.

## Concise-answer checkpoint

Private run: `20261017T233500Z-concise-local-b33ab4`. The primary 28 cases pass,
including the stricter clock interval and spoken-JSON oracles. Clock replies take
0.541/0.539 s and tool-name replies 0.593/0.591 s (French/English). The next model
turn takes 5.01/4.44 s, down from 5.79/4.96 s with the verbose capability reply but
still slower than the original roughly 2 s. Its prompt processing alone costs
4.00/3.48 s. Shorter wording improves the regression without eliminating it.

The 28-case median is 0.752 s and sample p95 4.443 s, versus 1.784/4.753 s before.
Do not use the improved mixed median/p95 to hide the slower individual follow-ups.
All 123 targeted Apollo tests passed before this run; no unit tests ran alongside
its latency measurements. Seven local context tests also passed.

The complete run accepts 40/42 cases with no new semantic failures. Controls remain
8/8 exact/truthful (median 0.713 s, range 0.686–0.793), all eight challenges pass,
and every turn has one request and one audio response in the expected language.
All ten recorded commands are dry-run, executed=false. The two French science
explanations remain incorrect. Nine warm short answers retain a 1.975 s median,
but their sample p95 rises from 2.304 to 5.011 s. Natural explanation median is
5.850 s versus 5.804 s; these small samples do not establish statistical equivalence.
The primary 28-turn summed audio-ready delay drops from 48.68 to 39.30 s.

Saved human PCM has the same confirmed transcript and nearest-person attribution:
scores 0.527862 EN (below threshold) and 0.650695 FR (above). Fresh-response PCM-end
to first-audio time is 3.184/3.337 s EN/FR versus 3.070/3.358 s before. Repeated
cached-TTS responses are 2.488/2.831 s and are not mixed into the fresh comparison.
No microphone capture, acoustic echo handling, audible playback, or physical device
completion was measured in these headless runs.

**Acceptance:** observed quality improves; blanket latency non-regression is NOT
met because the next model turn after local replies is slower. Do not introduce
background prefill solely to hide this in request-start metrics: any later experiment
must include queue admission, first audio, CPU work, and idle/active memory or power.
Keep the truthful clock/capability fixes and prioritize bounded context/cache handling.

## Paired factual diagnostic

The same final server completed 32 additional **text-only, no-TTS** requests: 16
prompts with tools and the same 16 without tools, including independent sessions
and the sky→sunset follow-up. Every response used one model call and zero tool or
action calls. There was no token cutoff. These diagnostic requests are separate
from the 42-turn audio-ready benchmark; no routing change was adopted.

| French topic | Tools offered | No tools |
| --- | --- | --- |
| Blue sky | Incorrect absorption explanation | Still incorrect |
| Sunset | Reversed wavelength explanation | Incorrect absorption mechanism |
| Ice floats | Incorrect coefficient explanation | Correct lower-density explanation |
| Seasons | Incorrect orbit-shape explanation | Still incorrect |
| Moonlight | Correct reflected sunlight | Correct, but violates one-sentence instruction |
| Sound in vacuum | Incorrectly says yes | Correctly says no |
| Photosynthesis | Omits carbon dioxide | Says air; insufficiently specific |
| Unknown bicycle colour | Correct abstention, wrong language | Correct French abstention |

English scientific answers are correct in both modes. Without tools, the moon reply
also violates the requested sentence count, and the bicycle abstention invents a
previous interaction in a fresh session. Under the strict combined factual, language,
context-truthfulness, and format criteria, both modes have 9/16 passes, one partial,
and six failures, with different failure types. This is a small diagnostic, not a
claim about general accuracy. Exact replies and per-dimension notes are private in
`ablation.json` and `quality-review.json`.

Alternating schemas rebuilds the prefix cache: median request latency is 10.84 s
with tools and 3.08 s without, **not** a steady-profile speed comparison. For the
English ice case, tools cause 451 prompt tokens / 11.38 s prefill versus 141 / 0.58 s
without. The current model/prompt combination has wider French quality problems;
removing tools alone is insufficient. Evaluate routing/prompt candidates against
these held-out questions and control intent cases before changing the active profile.

Final network audit: 22 observed loopback destinations, no external/DNS attempts
through the benchmark and factual probe. All 51 runtime Python source hashes match
the local source/archive; extra archive entries are non-imported AppleDouble metadata.
Apollo's isolated unit is active; no Aria Mac client is running. Surviving service
process RSS totals about 5.10 GiB at the final sample (shared pages may be counted
twice). Python/ASR and model-runner RSS high-water marks are about 3.02/2.06 GiB;
the runner accumulated 988 CPU seconds over startup and all final-run work. These
are not whole-service peak RAM or power measurements; systemd accounting is unavailable.

## Reproduce

On Apollo, against an isolated instance started through the
[E2E workflow](e2e-testing.md), run `python -m scripts.e2e.conversation_bench
--server-run <run> --output <report> --captures <confirmed-capture-directory>`.
The factual diagnostic is `python -m scripts.e2e.quality_probe --server-run <run>
--output <report>`. Use the prepared Apollo environment and never run benchmarks
concurrently. The private run contains `supplement.py` and `challenges.py` for the
additional 14 cases, plus `comparison.json` and manual reviews. Preserve the same
session order, fresh-cache conditions, request language, and inventory when comparing.

## Evidence handling

Private reports, transcripts, inventory-derived cases, source archives/hashes, and
server/network logs live under `~/Library/Logs/Aria/e2e/<run-id>` and Apollo's
`/data/aria/e2e/{runs,reports}/<run-id>`. Keep household names, command IDs, and speaker
identity out of repository reports. No commits or production deployment are implied.
