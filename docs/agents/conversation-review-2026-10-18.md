# Live conversation review — 2026-10-18

Follow-up decision: [D-008](decisions.md) defers interruption and its echo-cancellation
work. The product remains speech-only; evaluate earlier sentence playback with
microphone suppression maintained across the entire reply. Barge-in discussion below
is a future consideration, not current scope.

Scope: `20261017T220249Z-live-b90112`, three transcribed French voice turns, excluding
the readiness API request and the developer's direct action-provider check. Logs
use UTC; this conversation occurred after midnight on October 18 in Europe/Zurich.
The service was left running; this review did not restart it or change runtime code.

Private evidence is in the run's `review/` directory under the existing Mac E2E
log root: copied Apollo conversation/server/model/network logs, `state.json` (97
events), and derived `latency-breakdown.json`. Raw conversation and identities stay
out of shared context. The Mac `timings.jsonl` supplies single-clock playback timing.

## Recognition and tool behavior

- Nearest-person attribution returned the maintainer's profile for all three
  transcribed turns. Scores were 0.483, 0.390, and 0.448; all `known=false` against
  the unchanged 0.65 threshold. The restored policy passed the nearest user to the
  LLM as requested. These are similarities, not calibrated probabilities. The
  runner-up was another language profile of the same person; do not interpret that
  small margin as separation from another person. Highest different-person profile
  scores were at most 0.078 in the logged candidates. No identity timeout was logged.
- Tool schemas were offered on every voice model call. Two `find_devices` calls
  completed, but both were inappropriate: a clock/time query and an empty query
  after a request to list capabilities. Both returned no candidates. The final
  response to the capability question continued the prior clock topic.
- No voice turn called `execute_command`. The sole `Live.Command` was the developer's
  earlier direct provider check (`request_id=null`), not the user's conversation.
  Do not count it as successful voice command selection. It was log-only and did
  not execute. The network snapshot found 16 loopback destinations and no external
  or DNS attempts. No physical device outcome is established.
- Three later finalized segments produced no text/request: one failed the ASR gate,
  two ran ASR but yielded no final text. Raw audio and raw-before-filter transcripts
  were not retained, so these cannot be labeled user speech, noise, or ASR failures.
  Add stage/reason diagnostics before drawing conclusions; no new capture was made.

## Where the delay went

All values are seconds. Speech end means the last positive client VAD frame;
playback start means launch of `afplay`, not measured hardware audible onset.

| Turn purpose | VAD speech end → player launch | Shared processing | Prompt evaluation | Token generation | Reply → server audio ready |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial time question | 18.10 | 15.62 | 11.33 | 4.00 | 0.84 |
| Time clarification | 11.22 | 8.87 | 1.97 | 6.30 | 0.80 |
| Capability/tool-list question | 12.10 | 9.47 | 3.44 | 5.40 | 0.83 |

The last two rows each include two model calls. The second calls alone took 5.57 s
and 4.77 s. The actual local tool handling was sub-millisecond between logged model
completion and tool-result events; the extra inference dominates the tool round trip.
Generation was about 11.5 tokens/s on the CPU Q4_K_M 3B model. Responses were not
truncated: all model calls ended with `stop`. Lowering the token cap would not repair
the wrong answers and could reintroduce truncation.

Admission wait was below 1 ms on all three turns. Observed endpoint contribution was
0.68–0.72 s. After subtracting shared processing and reply-to-audio readiness from
client-turn-end-to-player-launch, 0.87–1.09 s remains for ASR, speaker inference,
transport, polling, and playback launch preparation together. Existing logs cannot
split these components accurately; do not report a fabricated ASR/speaker breakdown.
The shared processing interval accounts for about 83–90% of post-turn-end delay.

Synthesized replies lasted 7.76–9.86 s. The Mac adapter mutes outgoing microphone
frames throughout playback plus 300 ms. Long replies therefore impose a long period
without voice interruption as well as delaying the first response.

Timing field interpretation follows [Ollama usage metrics](https://docs.ollama.com/api/usage).
The model was already loaded by readiness; voice-call `load_duration` was about
0.14 s per model call. A model reload does not explain the 11.33 s first-turn prompt
cost. Readiness deliberately used `use_tools=false`, whereas voice added tool schemas;
that changed prompt prefix is a plausible explanation for cold prompt evaluation,
not a measured cache-hit conclusion. Cache counters were not logged.

## Source causes and recommended fix order

1. **Restore cheap local capabilities independently of the optional router.**
   [`_chat_shortcut`](../../app/main.py) returns immediately when the router is absent,
   but the clock response is inside that function. Live mode disables the router.
   Its exact-phrase match would also miss greeting-prefixed questions and the
   clarification used here. Move narrow local time/date/capability handling before
   that gate, with explicit local timezone and FR/EN coverage. List capabilities
   from the registered tools rather than searching inventory. This is a targeted
   repair, not a decision to adopt the broader Option A architecture.
2. **Correct tool selection and reduce unnecessary model calls.** The advertised
   schemas contain only `find_devices` and `execute_command`; neither supplies the
   current time or tool introspection. Keep general conversation out of device
   search; validate blank searches and respond truthfully to unsupported requests.
   Avoid adding a second LLM solely to classify every turn. Preserve exact grounded
   command selection, dry-run semantics, and shared voice/text behavior. Test topic
   switches so a previous mistaken answer does not dominate the latest request.
3. **Warm and minimize the actual operational prompt.** Warm the tool-capable path
   without executing calls, keep stable prompt prefixes and bounded useful history,
   and compare first/warm timings on Apollo. Set an explicit bounded model residency
   policy if needed for pauses; residency is not the measured bottleneck in these
   adjacent turns. Keep the old Xeon resource envelope; no larger model is justified
   by this review. [Ollama chat options](https://docs.ollama.com/api/chat) document
   streaming and model residency controls.
4. **Make answers short and deliver complete sentences earlier.** Prefer concise
   local replies and truthful structured action confirmations. Implement actual
   sentence-to-TTS delivery for conversational generation before considering larger
   inference changes. Merely setting `ARIA_LLM_STREAM=1` will not suffice:
   [`generate_ollama_response`](../../app/llm/ollama_client.py) accumulates chunks and
   returns only after completion; `_process_request` schedules speech afterward.
   The live response observer also buffers the response and expects one JSON object,
   so it must be adapted for NDJSON streaming. Never stream an action-success claim
   before the action result. Preserve echo suppression while designing cancellation
   and barge-in; simply unmuting playback risks feedback.
5. **Instrument the remaining stages, then tune.** Add explicit final-audio-received,
   ASR start/end, speaker start/end, raw-vs-filtered outcome reason, model first token,
   sentence-ready, TTS, and player timings. Preserve confidence logging. Do not first
   shorten endpoint silence (previously fixed split turns), disable recognition, or
   rewrite the queue: these are not the dominant delays in this sample.

Validate these proposals with saved EN/FR inputs, then a user-controlled live session
on Apollo. Measure useful first audio, correct tool/no-tool selection, exact intended
command, and completion independently. Three French turns are not an English baseline
or a reliable p95. No performance improvement is claimed until the changes are measured.
