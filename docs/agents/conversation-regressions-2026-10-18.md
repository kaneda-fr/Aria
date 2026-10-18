# Live conversation and server code review — 2026-10-18

The latest live interaction exposes failures across language, capture, playback,
history and tool selection. These interact: an English reply spoken with a French
voice returns through the microphone, loses its opening, becomes a new request,
and contaminates history. Long replies accumulate at the Mac player while rolling
history later makes model prompt evaluation expensive. Passing the earlier
synthetic suites did not establish that this conversation works.

**Analysis only:** runtime unchanged, no generated audio or device actions. Tools
remain log-only; ESP32 input and Mac playback remain active, Mac capture off.

## Evidence and scope

- Active monitor pointer and runtime agree on run
  `20261018T145826Z-echo-prefix-01dafd`; archive SHA-256
  `d8e098ee889c78137691cc2fe8c3c4c9f4c29ba7f505ef5b33dd97bf9fb93611`.
- Reviewed the ten processed live turns beginning around **17:25:54 Europe/Zurich**
  through the last Mac player exit around **17:28:47**. Earlier unaddressed background
  transcripts were ignored by the wake gate. One activation and nine follow-ups
  appear in the reviewed conversation; there is no wake-state reset in that span.
- Copied exact request/reply/model/tool logs, server events and Ollama logs to the
  private run's `review-20261018-latest/` directory. `joined-turns.json` joins
  request status, speaker, model, TTS and Mac player timings. `review-provenance.json`
  records hashes; `offline-probes.py` retains reproductions. Private identities
  and raw conversation text remain there.
- Matched six central runtime/playback files against Apollo's immutable source.
  Analyzed startup, ingress, segmentation, ASR, recognition, echo, wake/session,
  shared admission/worker, history/prefill, local replies, control authorization,
  inventory selection, tool dispatch/rendering, Piper, speech scheduling, Mac
  playback and relevant evaluation code. This is a review of the active server
  conversation path, not an exhaustive firmware, provider or security audit.
- Three local **mocked** probes reproduce wrong-language speech scheduling,
  prefix changes at rolling-history capacity, and removal of an utterance opening
  by the ring buffer. All three reproduce the current behavior. Their passing
  assertions document defects; two existing dependency warnings remain.

## 1. Reply language and voice have separate, inconsistent contracts — confirmed

Four of the ten replies contain English prose but select the French Piper model.
The initial French request already exhibits this before history contamination.

[`_process_request`](../../app/main.py) detects the **input** language once, then
passes it unchanged through `_speak_reply` to
[`SpeechOutput.speak`](../../app/speech_output/speech_output.py). The generated
reply is not checked for language. The live prompt's general instruction to answer
in the user's language is insufficient; it is not an enforced per-turn output
contract. The detector selected French correctly on these turns. This is not a
cache collision: speech cache keys include the selected voice.

The existing `test_request_language_is_fixed_for_all_chunks` in
[`test_speech_language.py`](../../tests/test_speech_language.py) explicitly expects
an English chunk to use the French voice. It protects numeric continuity but also
accepts this failure. Conversation checks do not assert reply-language agreement.

**Recommended correction:** represent requested reply language, observed reply
language and synthesis language explicitly. Give generation a current-turn language
instruction; validate substantive prose before scheduling it. Select a voice that
can pronounce the actual text if generation still deviates, while keeping numbers
and language-neutral fragments in the turn's chosen language. Do not make a second
model call the default repair or return canned answers to individual questions.
Voice fallback alone does not satisfy French answer quality. Add paired FR/EN
tests for a deliberately wrong-language model response, alternating follow-ups,
numeric output, mixed content and explicit reply-language requests.

## 2. Feedback loses its opening before the echo matcher sees it — confirmed

Two apparent playback-feedback inputs were admitted as unknown speakers, answered,
and written into conversation history. Their text overlaps the preceding replies;
there is no retained acoustic recording to prove the exact mixture of human and
speaker audio. Both have **actual logged ring-buffer overflow**:

| Input | Oldest PCM discarded | Retained PCM |
| --- | ---: | ---: |
| First feedback-like turn | 39,040 bytes = 1.22 seconds | Last 10 seconds |
| Later mixed feedback-like turn | 589,440 bytes = 18.42 seconds | Last 10 seconds |

[`RingBuffer.append`](../../app/audio/ringbuffer.py) intentionally drops the oldest
bytes to stay bounded. [`ws_asr`](../../app/main.py) logs each drop but still
transcribes and admits the suffix as a complete request. The original onset timestamp
is retained even though it now describes audio that is no longer in the buffer.
Both feedback transcriptions have a different opening from the original reply;
the new three-word prefix rule cannot match missing words. Whole-text comparison
also misses: containment/fuzzy scores are approximately 0.56/0.60 and 0.67/0.15.

[`EchoGuardV2`](../../app/speech_output/echo_guard_v2.py) is therefore doing what
its implementation tests specify, but those tests did not cover the preceding
loss of the opening. Increasing the prefix tail or reducing thresholds does not
repair this data loss and risks suppressing humans.

**Recommended correction:** make utterance completeness an admission invariant.
Track overflow, retained onset and original duration; never silently treat an
arbitrary suffix as a complete control request. For this parked-interruption stage,
use actual playback state to exclude the relevant acoustic input during output,
then resume automatically. Bound long-turn handling explicitly with a tested
discard/clarification or segmentation policy. Do not grow buffers without limits.

## 3. Server playback estimates disagree with the Mac player — confirmed

[`SpeechOutput.speak`](../../app/speech_output/speech_output.py) schedules a task
and cancels its predecessor. The shared request worker completes without awaiting
physical playback. The live sink only fetches/records a WAV;
[`CaptureSink.stop_all`](../../scripts/e2e/server.py) only records a stop event.
[`mac_playback.listen`](../../scripts/e2e/mac_playback.py) instead waits for each
`afplay` process before polling/playing later WAV events. It does not consume
server stop events. Old replies can therefore remain queued on the Mac after the
server has replaced their speech task.

Observed additional WAV-ready-to-player-launch delays include **9.70, 7.54 and
6.46 seconds**. Long replies take roughly 12–18 seconds in the Mac player. The
server's TTS event interval starts when the WAV is prepared, not when the Mac
audibly starts it. `ConversationGate.finish` also uses the server speech task as
`playback_done`. Thus echo windows, conversation timeout and what a person hears
do not share one clocked lifecycle.

**Recommended correction:** define a server-owned output lifecycle with a playback
ID, destination and queued/started/finished/cancelled events. The Mac output adapter
must honor that policy and report progress; use a bounded fallback for sinks with
no acknowledgment. With interruption deferred, choose and test a consistent
non-overlapping playback policy. Keep microphone capture independent so transport
continues to drain while processing is suppressed. Do not call scheduled speech
"heard" or use synthesis completion as physical playback completion.

## 4. Rolling history reintroduces expensive prompt work

The last three model turns spend **15.56, 15.90 and 13.64 seconds** in prompt
evaluation; generation takes 5.22, 2.13 and 1.37 seconds. Model loading remains
approximately **0.14–0.17 seconds**. Ollama shows one runner startup and no reload
at this point. This is not a return of the idle-cold problem.

[`_process_request`](../../app/main.py) stores user/assistant pairs in a
`deque(maxlen=12)`. The first request using all six retained turns can still append
to the previous prefix. On the next request, the oldest pair disappears near the
front of the prompt, changing nearly the entire history prefix. The latency jump
aligns exactly with this first rollover. The offline probe demonstrates that input
change. Ollama's cached-token count is unavailable, so cache invalidation is a
strong source/timing explanation, not a directly measured cache-hit statistic.

[`schedule_prefill`](../../app/llm/history_prefill.py) is called only after a local
clock/capability answer with recent model use. It does not run on the general model
turns in this interaction. The earlier prefill work therefore does not cover this
rolling conversation. Adding speculative inference to every turn could increase
contention and must be measured before adoption.

| Late turn | Queue wait | Processing | Prompt evaluation | Admission → Mac launch |
| --- | ---: | ---: | ---: | ---: |
| Weather rephrasing | ~0 s | 21.24 s | 15.56 s | 23.74 s |
| Identity question | 3.41 s | 18.49 s | 15.90 s | 23.49 s |
| Name question | 15.39 s | 15.44 s | 13.64 s | 31.46 s |

These are descriptive live timings, not a paired performance experiment. The last
column measures process launch, not first audible audio; Mac/Apollo wall clocks
also have small skew. Endpoint receipt and ASR/speaker stage boundaries are not
logged precisely enough to reconstruct speech-end-to-audio latency.

**Recommended correction:** test history beyond capacity, including seven-plus
turns, rather than only short conversations. Evaluate bounded blocks of complete
turns with a token budget and infrequent, explicit rollover, retaining required
device references and relevant evidence. Avoid a second summarization model by
default. Compare identical histories and model settings on Apollo, including queues,
alternating clients, local answers and tool results. Do not drop useful context or
change factual replies merely to improve a median.

## 5. Tool selection confuses unavailable capabilities with device lookup — confirmed

One clear weather request produces `find_devices(query=weather)`, zero candidates,
then the new lookup renderer's request for an appliance name/room. The next request
gets the same device-oriented reply **without any tool call**. The model has copied
the prior assistant text. A synonym then gets another device-oriented answer.

[`render_lookup_results`](../../app/llm/lookup_reply.py) truthfully describes its
input, but `_process_request` short-circuits **every pure search batch** into that
renderer. A successful search is not evidence that search was appropriate to the
question. Persistent history keeps only user text and final prose; the search,
its purpose, scope and result provenance are not retained as structured evidence.

Only one lookup and **zero action calls** occur in this reviewed interaction.
There is no evidence here of a wrong physical action or of successful device
control. No local weather data provider is wired into this assistant tool path.
Local-only operation precludes silently adding an Internet weather service.

**Recommended correction:** provide an explicit available-capability contract and
validate tool purpose against it. Device lookup, command selection, current-state
queries and external-world facts are different operations. If local weather data
is absent, say so briefly in the requested language. Preserve bounded structured
tool evidence in history; do not treat yesterday's or a previous turn's prose as
a fresh observation. Keep the direct renderer for valid inventory questions and
test unrelated general questions, failed searches, rephrasing and topic changes.

## 6. Recognition is active; identity use and timeout visibility are weak

Runtime reports recognition enabled/ready with six profiles. Eight of the ten
requests are attributed to the same enrolled person, with similarity approximately
**0.37–0.57**, all `known=false`. This is the deliberately restored nearest-person
behavior, not a new disabling regression. Similarity is not a calibrated probability.
Two feedback-like turns have no timely speaker result. Later uncorrelated debug
events select the assistant profile; they are consistent with recognition finishing
after the one-second wait has expired.

[`_finalize_utterance`](../../app/main.py) silently converts timeout to no result.
Cancelling `asyncio.to_thread`'s waiter does not stop the underlying inference.
[`self_speech_decision`](../../app/speaker_recognition/recognizer.py) preserves
nearest-human attribution and rejects only confident self matches. Lowering that
threshold indiscriminately would reintroduce human rejection.

The model receives the speaker tag, yet one identity question describes Aria
instead of the person; the next name question uses the tag correctly. Recognition
and model use of identity are distinct checks. Confidence/status are not part of
the model message, so the reply can sound more certain than the evidence warrants.

**Recommended correction:** log timeout and stage duration with the utterance/request
key, bound concurrent recognition work, and carry attribution status/confidence
separately from the nearest name. Preserve the maintainer's nearest-person behavior;
do not turn it into verified identity. Evaluate identity questions and recognition
under overlapping ASR/LLM/TTS load, rather than interpreting a friendly name reply
as a biometric accuracy result.

## Wider code risks

The [code-risk appendix](conversation-code-risks-2026-10-18.md) covers the mismatch
between model tool instructions and server authorization, globally shared output,
unbounded pre-LLM inference admission, unchecked generation truncation, missing
stage correlation, test-profile drift and wheel packaging. These are source
findings; they are not all demonstrated regressions in this ten-turn interaction.

## Recommended implementation sequence

1. **Make input/output turns coherent:** agree on actual playback lifecycle;
   prevent self-output admission while interruption is parked; reject/handle
   truncated utterances explicitly. Keep client streaming operational. This stops
   feedback from poisoning subsequent quality and latency tests.
2. **Repair the language contract in the same priority group:** current-turn
   generation language, reply validation and matching speech voice, preserving
   numeric continuity. Demonstrate FR/EN turns with history and feedback present.
3. **Fix rolling-history latency:** paired seven-plus-turn Apollo sessions with
   exact reply/tool oracles and queue-inclusive timings. Recheck the idle residency
   result separately only if its code/configuration changes.
4. **Align tool purpose and supported capabilities:** correct unavailable-data
   replies, topic changes, structured search evidence and server authorization.
   Expand model-free inventory name recovery only after this baseline is stable.
5. **Close identity and capacity gaps:** confidence-aware context, correlated
   timeout metrics, independent output destinations and bounded ASR/speaker work.

Acceptance must include held-out FR/EN wording, silence/idle, long/overflow turns,
wrong-language model output, unknown and attributed speakers, topic changes, exact
log-only actions, more than six history turns, overlapping clients, queued requests
and delayed playback. Replay deterministic timing with fake players before a
controlled acoustic check; no individual benchmark sentence should become a canned
runtime answer. Use Apollo's existing Xeon, local assets and no extra inference
model as the starting point.
