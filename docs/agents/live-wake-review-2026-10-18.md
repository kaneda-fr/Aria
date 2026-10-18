# Live conversation review — 2026-10-18

Read-only review of Apollo run `20261018T095025Z-live-449f47`, with the Mac client.
Evidence was captured at 11:58:11 Europe/Zurich after the 11:57 conversation.
The server uses the [tested snapshot](spoken-wake-2026-10-18.md), not the later
concurrent ESP32 changes. Tools remain log-only. No inference probe, action,
restart, or runtime configuration change was made during this review.

## What worked

- The spoken name plus clock question activated the session without name repair.
  Three processed follow-ups without the name stayed active. An earlier unrelated
  transcript was ignored. Goodbye and expiry were not exercised in this conversation.
- Five consecutive French segments selected the maintainer's enrolled profile.
  Scores were 0.525, 0.628, 0.409, 0.561, and 0.660; only the last passed the 0.65
  recognition threshold. Nearest-person attribution worked, but high-confidence
  identification was inconsistent. Do not present these cosine scores as probabilities.
- A weak self candidate (0.129, known=false) proceeded as unknown rather than
  being rejected. Its unrelated transcript did not wake the assistant.
- All four spoken replies used French TTS. Five admitted requests completed once;
  one additional request was explicitly rejected busy. The network snapshot has
  17 loopback destinations and no observed external/DNS attempt.

## Control and conversation failures

The first off request resolved to one named light with **two independent off
channels**. The server correctly withheld execution, but its generic device/room
clarification failed to explain the actual channel ambiguity. Replaying selection
against the saved inventory reproduces both candidate commands without inference
or side effects. Do not select one arbitrarily or automatically execute both.

The short follow-up then made one `find_devices` call, using a conversational room
name as a strict room filter. The inventory files the named device under a floor
label instead. The lookup returned zero matches even though the device exists.
Its spoken absence claim overstated what that narrow search established. Pending
clarification is not represented as structured state; the generic model reconstructs
the context and can discard the already identified device/channel candidates.

A subsequent question asking which lights were on made no tool call and received
another control clarification. The current two tools expose command lookup and
execution, not a state-query operation. The live adapter's cached inventory states
map was empty. It cannot establish current on/off state from this evidence.

There were **zero `execute_command` calls and zero recorded actions**. This run
demonstrates wake/attribution and a lookup attempt, not successful home control.

## Latency breakdown

| Logged turn | Measured delay | Dominant work |
| --- | --- | --- |
| Wake + clock | 2.138 s last VAD speech → Mac player start | 0.641 s endpoint gap, 0.625 s reply-to-WAV; remaining capture/ASR/speaker/transport/player overhead not separately timed |
| Off request → clarification | 2.396 s last VAD speech → player | 0.714 s endpoint gap, 0.593 s reply-to-WAV; deterministic selection took milliseconds |
| Short clarification follow-up | 29.910 s server admission → first WAV | 28.581 s model/tool loop, then 1.328 s audio preparation |
| Which lights are on | 19.501 s admission → WAV | 15.048 s queue wait, 4.428 s processing, 0.024 s cached audio; Mac also waits for previous playback |

The slow loop contains **20.870 s prompt evaluation**, 5.329 s token generation,
and 2.040 s model loading across two model calls. Tool execution itself is under
1 ms. The first model call evaluates 516 prompt tokens in 19.542 s. This is not a
slow device/network round trip.

Ollama logs configure `OLLAMA_KEEP_ALIVE=5m`; startup warm-up completed at 11:50:53,
and a new runner/model load appears at 11:57:22. The warmed state did not survive
the idle gap. Earlier local clock/control replies made no model calls. History
prefill requires a recent model call in the same session, so it does not run after
that initial local clock. Deterministic control clarifications do not schedule it
either. Startup warm-up and the existing prefill optimization therefore do not
cover this real opening sequence.

The slow request blocks the single worker. One follow-up queues; another finalized
utterance is rejected busy while that pending slot is occupied. The queue remains
bounded, but the conversation experience fails. The client prints status in the
terminal; it provides no spoken busy/rejection feedback. Increasing queue capacity
would not remove the underlying delay.

## Evidence limits and next priorities

Eight utterances finalized: one filtered before ASR, one with empty ASR output,
five accepted requests (including the earlier ignored transcript), and one busy
rejection. Rejected raw text was not logged in `Live.Request`; only its length is
available. No captured PCM is retained, so the empty/filtered content cannot be
inferred. Late-turn client/request associations are explicitly ambiguous; do not
invent exact speech-to-player timings for them. ASR and speaker inference also
lack separate stage durations in this run.

1. Keep the small model and tool-prefix cache available across realistic idle gaps,
   within Apollo's RAM/power budget. Test an idle interval over five minutes followed
   by local clock/control turns and then a model/tool turn. Prefer model residency
   over periodic dummy inference; measure the effect rather than assuming warm-up
   at startup guarantees low first-use latency.
2. Maintain pending clarification with device/channel candidates and intended action.
   Ask about the unresolved channel; use inventory-backed room/device aliases.
   Keep explicit execution authorization and truthful outcomes.
3. Add read-only state retrieval with freshness and unavailable-state handling;
   do not answer state questions from cached command metadata or invented values.
4. Add appropriate speech-only busy feedback and reliable utterance/request/audio
   correlation, including empty/rejected inputs. Preserve bounded queues and guards.

Private exact transcripts, speaker names, command IDs, inventory, logs, analysis
script, and calculations are in the live run's `review-20261018T095811Z/` directory
under `~/Library/Logs/Aria/e2e/`. `review.md` contains the user-specific trace;
`analysis.json` contains per-request durations and the read-only selection replay.
Live `conversation.jsonl` and `timings.jsonl` remain in the parent run directory;
Apollo `server.log` contains wake/queue decisions absent from the compact Mac log.

Only context documents were changed for this review. Serena retrieval still fails
with no language servers; source findings were checked against the deployed
snapshot through targeted reads. No production code was edited.

Later observation at 12:00:37 Europe/Zurich: a non-name transcript was logged as
`ignored` with `expired=true`, confirming return to waiting after inactivity.
This is outside the eight-utterance conversation/audit window above; it was not a
new inference probe. Goodbye remains untested in this human conversation.
