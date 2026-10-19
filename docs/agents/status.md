# Current evidence, gaps, and history

**2026-10-19:** [Sentence voice routing](sentence-tts-2026-10-19.md) and acknowledged
Mac delivery are implemented. Review fixes release cancelled slots, invalidate
interrupted waiters and propagate muted scheduling. Apollo: 235 targeted tests pass.
Other UI/inventory/history/input-admission fixes remain separate. Audio is off;
tools are log-only. Live-state entries below are historical.

**Latest audio instruction — 2026-10-18:** the maintainer enabled Mac **playback only**
and ESP32 input. Mac capture remains OFF. The playback-only listener attached to the
current run and its readiness WAV completed successfully; LAN ingress was restored.
Do not run generated-speech benchmarks while live playback/ingress is active: pause
that path first or use non-audio analysis. This supersedes the prior all-audio-off
instruction; tools stay log-only. See [playback-only operation](../../client/macos/README.md).
**Latest [conversation/code review](conversation-regressions-2026-10-18.md):**
four English replies used French TTS; two feedback-like turns lost their openings
to ring overflow and entered history. Mac playback queued up to 9.7 seconds after
WAV readiness. Rolling 12-message history coincides with 13.6–15.9-second prompt
evaluation; weather was misrouted to device lookup. Recognition is active but
uncertain. Analysis only; three mocked reproductions, no restart or runtime fixes.
The earlier [prefix fix](feedback-prefix-2026-10-18.md) passed 324 Apollo tests but
did not establish acoustic feedback reliability. Priorities: playback/input turns,
language agreement, rolling history, tool purpose. Tools remain log-only.
[Name-matching evaluation](name-matching-options-2026-10-18.md) reviews Byname's
aliases, Damerau corrections and WordNet vocabulary. Recommend model-free layers
first; a read-only Apollo apostrophe-normalization probe recovered the failed name
in a 69-device scan at 0.015 ms median. This is not an integrated matcher or a
general accuracy result. ESP32 subsequently connected and produced the feedback evidence above.
The [paused-work snapshot](paused-work-2026-10-18.md) remains historical evidence.
Serena symbol lookup works; see [code intelligence](code-intelligence.md).

[Grounded lookup replies](lookup-latency-2026-10-18.md) remove the second model pass
after read-only search. Paired Apollo lookup medians fall FR 14.95→3.98 s and EN
12.03→3.75 s to generated audio. All 32 functional cases pass, eight exact log-only
controls remain correct, and all 26 general/saved-voice replies match the reference.
Queue median is 4.014→4.031 s. Existing science defects are unchanged, not fixed.
Eight additional text lookups and saved English speech work; saved French speech
still misrecognizes a device name, skips search and replies in English. The
six-probe text isolation in that report confirms that correcting only the device
name restores lookup; correcting only the room does not. No production name
correction has been added; inventory-derived recovery needs ambiguity evaluation.
The [earlier lookup investigation](freeform-lookup-2026-10-18.md) preserves template
changes and their general-answer regressions; this follow-up does not erase them.

Current isolated Apollo run is `20261018T145826Z-echo-prefix-01dafd`; the prior lookup
reference remains `20261018T141513Z-lookup-fast-731426`. The compact monitor adds
**Clear offline**, hiding disconnected clients until reconnect while preserving mute
deadlines, session keys and accepted requests. UI filter reset, reconnect, mute/unmute
and responsive layouts were verified with synthetic connections; no physical audio.
Confirmed mute's earlier saved-voice/expiry evidence remains in its report below.
The development service retains its three-hour lifetime (this start expires around
19:58 Europe/Zurich); monitor autostart remains disabled, but it is running.
A bounded thread heartbeat, `aria-apollo-validation`, schedules six half-hour follow-ups
for targeted tests/fixes. Any later stop instruction overrides restart permission.
See the [mute report](client-mute-2026-10-18.md) for precise evidence and limitations.

[Model residency fix](idle-residency-2026-10-18.md): Ollama requests now default to
`keep_alive=-1` (`ARIA_LLM_KEEP_ALIVE_S`), retaining the model/cache without periodic
inference. Two 310-second idle tests on Apollo: first French voice audio ready
19.82→3.16 s; English 19.09→2.85 s. About 2 GiB runner RSS, 0.02 CPU-seconds per idle
interval; whole-machine power unmeasured. 217 targeted tests, 28 functional cases
per policy, eight exact log-only controls per policy, and four queued trials per
policy pass. Non-clock replies match; queue median 3.984→3.978 s. Raw ASR name flags
remain. Added free-form lookups fail identically in both policies: French replies
in English/miscounts commands as devices; English returns empty. No claim that
these tool-quality defects are fixed. Network and request lifecycle audits pass.

[Live conversation review](live-wake-review-2026-10-18.md): wake/follow-ups and nearest
speaker attribution worked. Two-channel ambiguity, a wrong strict room filter,
and absent state querying broke device interaction. The idle reload is now fixed;
queueing, tool-response quality, and audible busy feedback remain work.

[Spoken wake and self-rejection fixes](spoken-wake-2026-10-18.md), D-011:
“Bonjour Ari” / “Hello area” address correction and confidence-based self rejection
remain; weak self candidates proceed as unknown. Nearest-human attribution, raw
scores, and Echo Guard are preserved. Bare-name and recorded farewell ASR failures
remain; see the report for original 202-test/32-wake-case evidence and limitations.
The previous live run `20261018T095025Z-live-449f47` was stopped for idle comparisons.
Mac mic/playback is ready on fixed Apollo run `20261018T105214Z-live-92c831`;
tools remain log-only, with the existing three-hour service TTL.

The [initial wake-flow report](wake-flow-2026-10-18.md), D-010, documents session
isolation, history/reference reset, the 30-second window measured after estimated
playback, and onset-based expiry. These behaviors remain; the later report above
supersedes its exact-name-only voice and unconditional self-suppression limitations.

[ESP32 microphone parity](../../client/esp32/README.md), D-012: local WebRTC VAD,
pre-roll, AGC/fixed gain, optional noise gates, and diagnostics use the existing
Apollo protocol; no server update is required. The rejected guard/cancel extension
and its server changes were removed. Host compatibility tests and cross-build pass.
ESP32 playback-aware muting is a gap; existing disconnect can finalize partial
client-VAD speech. After antenna installation, a 75-second stream had no drops and
Apollo accepted/completed requests. Speech quality, echo and latency remain unvalidated;
[board evidence](../../client/esp32/tests/hardware-2026-10-18.md).

[French science/model evaluation](science-2026-10-18.md): no tested candidate meets
the requested quality/latency no-regression condition, so the French science issue
is **not accepted as fixed** and no model default was changed. The final 4B/new-runtime
42-case comparison fixes sky/sunset but regresses two French voice replies to English
and slows English voice replies. Controls remain exact/log-only; queue median is
4.02 s versus 3.99 s. Rejected language/cache/profile changes were removed; the
model/prompt configuration remains at the previous baseline; the later wake change is above. Expanded reference science
scores 13/24; the 4B candidate reaches 20/24 but introduces an English science
regression and leaves three French failures. See the per-case review, not just totals.
Read-only model lookups also skip missing-device searches; known lookups take
17–20 s, unlike the fast deterministic controls. See the [interaction-surface audit](interaction-surface-2026-10-18.md) for synonym
restrictions; the formerly missing wake gate is now implemented above, with spoken-name limits. Tools stay log-only and
Mac audio off. The isolated test services are stopped; 132 targeted Apollo tests
and seven context tests pass. The [history-prefill baseline](history-prefill-2026-10-18.md) remains
the comparison point: 2.24/1.98 s FR/EN follow-ups and 0.71 s controls.

[Tool-capable warm-up comparison](warmup-2026-10-18.md): eight Apollo cases reduced
first-request audio-ready means from 17.13 to 6.39 s FR and 16.55 to 5.74 s EN with
identical replies. Startup costs about ten seconds more; ASR, endpointing, and physical
playback are excluded. Twenty-four targeted tests passed. Mac audio remains off.

The maintainer clarified the speech-only household interface and parked voice
interruption on October 18 ([D-008](decisions.md)). Mac remains development-only.
Keep microphone suppression during replies; earlier speech delivery can be evaluated
without barge-in. No keyboard/button workaround or interruption work is in scope.

On October 18 the maintainer authorized Apollo/Mac restart with nearest-person
attribution, confidence logs, and log-only tools, superseding the
[recognition review](speaker-review-2026-10-17.md)'s stop/rejection policy.
The live adapter has 12 history messages, exact private logs, cached inventory,
and actions with no MQTT client. Tool decisions/payloads are logged; physical
control is disabled; the narrow explicit-control fix is documented above.

[Conversation review](conversation-review-2026-10-18.md): attribution worked on three
French turns, but time/capability questions caused irrelevant searches; no voice
action call. Speech-to-player delay was 18.10, 11.22, and 12.10 s, dominated by LLM
work. Clock handling was router-gated at that review; the local-answer fix above
removes that dependency. General factual quality and concise replies remain open.

Conversational latency is a core requirement ([D-007](decisions.md)). Use the latest
live review above for measured speech-to-player and queue effects; historical
processor-only timings do not establish conversational responsiveness.

## Historical implementation evidence — 2026-10-17

The maintainer prioritized tool calling over capture redesign ([D-006](decisions.md)).
The [initial tool diagnosis](tool-calling-2026-10-17.md) reproduced wrong-device and
opposite-action selection (2/8 exact controls); later grounded-control work supersedes
that implementation status. [Controlled captures](controlled-capture.md) established
that the saved Paris answers were spoken by the maintainer, with no reproduced
server content insertion. Earlier unsaved microphone content remains unattributed.
[Turn/language work](turns-and-language.md) preserves request-specific TTS language
and bounded endpointing. [Shared admission](shared-request-path.md) unifies voice/text
through bounded FIFO processing. Their dated reports retain exact runs, quality flags,
network audits and test counts; those historical passes do not prove current quality
or physical device completion.

## Earlier baseline and setup

The [original bilingual baseline](baseline-2026-10-17.md) completed 18 pipelines
with nine quality flags and exposed the shared-path/queue defects fixed above.
The [E2E workflow](e2e-testing.md) stages an isolated Apollo instance, captures
WAV output and logs commands without real actions. Its early smoke runs established
infrastructure, not general conversation or control reliability. Later controlled
captures and dated comparisons above supersede those initial checks.

## Baseline and latest development

Remote heads were checked with `git ls-remote --heads origin`; they matched local refs:

| Branch | Head | Latest commit date | Meaning |
| --- | --- | --- | --- |
| `main` | `3ae33f9` | 2026-01-28 | Sonos selection and startup checks |
| `feature/control-bench` | `6efe669` | 2026-02-03 | Embedding classification and ONNX experiments |
| `feature/plugin-system` | `905fd38` | 2026-02-05 | Current checkout; benchmark tooling copied in at `0fbbb73` |

The current checkout also has pre-existing untracked router code, test reports,
operational scripts, and design docs. Those are working-tree evidence, not content
guaranteed to exist in a clean clone. Preserve them; do not stage everything blindly.
The GitHub all-state issue query returned no issues; the PR query returned only the
merged documentation PR #1. There is no issue backlog to substitute for this audit.

## Findings affecting the objective

| Finding | Evidence | Consequence / next verification |
| --- | --- | --- |
| Local-only operation is mandatory; compliance is unverified | Maintainer's instruction and startup-download clarification, [D-004](decisions.md) | Test startup downloads separately; after readiness, exercise required flows with Internet blocked and LAN available and inspect attempted external calls, including background activity |
| Apollo hardware and launch setup recovered | Read-only SSH verified Xeon E3-1225 v5, four cores/threads, nominal 64 GB RAM, AVX2; [deployment evidence](deployment.md) | Measure power/concurrent load and establish ESP32 variants/roles; deployed Ollama configuration differs from local llama.cpp configuration |
| Voice/text wiring divergence fixed in working tree | Both use `_process_request` through [shared admission](shared-request-path.md) | EN/FR fake-tool parity tested; real control interpretation/device completion still unproven |
| Pending requests preserved; overload explicit | [RequestQueue](../../app/llm/request_queue.py) rejects new work on full queue and expires stale work before execution | FIFO/cancellation/shutdown tested; per-session quotas and client busy-status UX remain |
| Inference stages can compete for CPU | Explicit ONNX backend in [ASR](../../app/asr/parakeet_onnx.py) defaults intra-op threads to CPU count; speaker model also uses Torch | Measure the complete pipeline and set shared resource budgets |
| Publish acceptance is called success | [actions provider](../../app/plugins/actions/provider.py), `execute_command` | Add explicit outcome semantics before claiming physical completion |
| Tool discipline is unreliable in historical runs | Local decision report; [analysis](../../TOOL_CALLING_DECISION_ANALYSIS.md) | Measure intended action and execution trace, not tool count alone |
| Historical pass criteria overstate completion | The report counts a one-tool, truncated clarification as a passing control case | Replace the acceptance oracle before selecting an architecture |
| Learning is not wired end to end | [router](../../app/routing/router.py), `update_centroids`; no caller in refreshed graph | Verify feedback collection, labels, persistence, and rollback before enabling adaptation |
| Packaging includes required modules | Wheel build/import outside checkout verified routing and speaker recognition | Keep package checks alongside checkout tests |
| Guidance contains conflicting status claims | [roadmap](../../IMPLEMENTATION_ROADMAP.md) versus [plugin documentation](../PLUGIN_SYSTEM.md) | Read these as historical plans/reports; verify current behavior in source |

The [architecture options review](architecture-options.md) maps these findings to a
proposed design and evaluation sequence. Its recommendation is not an accepted
decision or evidence of implemented fixes.

## Historical measurements

Historical results use incomplete action oracles and hardware provenance; they are
not current reliability scores. See [review history](reviews-history.md).

## Memory review and provenance

Serena returned no memories. Episodic/archive searches found no reliable newer Aria
architecture decision; unrelated-project entries and unavailable February sessions
are not authority. Missing history is a retrieval limit, not proof no discussion
occurred. Promote recovered decisions only with a user statement, date, project
match, and comparison against current source. See [decisions](decisions.md).
