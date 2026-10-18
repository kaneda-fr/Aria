# Unknown-speaker feedback rejection — 2026-10-18

The maintainer reported feedback during ESP32 input/Mac playback and requested
rejection when the person is unknown and the utterance starts like Aria's reply.
Tools remain log-only; Mac capture is off. This adds a bounded text heuristic,
not acoustic echo cancellation or voice interruption.

## Observed failure, 16:47–16:49 Europe/Zurich

The ESP32 connected under its stable client name. Two user time questions retained
nearest-human attribution even below the recognition threshold. Aria's played clock
reply then returned as an unknown-speaker transcript. A repeated no-device reply
returned twice, once almost verbatim. Both reached the model, causing unnecessary
searches/replies and adding feedback to session history. A later polluted-history
turn took 22.47 s; 16.73 s of that was prompt evaluation.

Raw self-profile scores around 0.24–0.41 were below the 0.75 self-rejection threshold.
They correctly became effective `unknown` under the existing policy. Lowering that
threshold would reintroduce the earlier problem of rejecting human speech merely
because its nearest profile was Aria. The new rule combines identity with text/time.

Echo Guard v2 was enabled but checked the ASR-completion time against a two-second
tail after a word-count duration estimate. Short clock replies also failed its
five-content-word/20-character minimum. The clock WAV was 1.88 s; Mac player launch
to exit was 6.06 s in the first observed turn. Player launch is not audible onset.
Even the longer exact repeat could finish ASR after the estimated matching window.

## Change

- Voice finalization passes effective speaker identity and server-captured utterance
  onset to Echo Guard. Monotonic onset is converted to wall time at evaluation;
  at most 60 seconds of processing delay is accepted. API text is unaffected.
- For `unknown` only, match the first three normalized words against either of the
  two retained TTS events. Case, accents and punctuation are normalized. For a short
  numeric reply, two opening words suffice only if the incoming continuation starts
  with digits or a supported FR/EN numeral word. This keeps "It is possible..."
  from matching the two-word opening of a numeric clock reply.
- Each reference must cover the speech onset or its own bounded extra tail:
  `ARIA_ECHO_V2_UNKNOWN_PREFIX_TAIL_SEC`, default eight seconds, clamped 0–30.
  The prefix branch runs before whole-text minimum lengths. It logs
  `reason=unknown_speaker_prefix`, prefix count, reference TTS ID, capture age,
  effective speaker and raw similarity score. No new model is loaded or called.
- Existing whole-text containment/fuzzy checks remain. They use capture time and
  each reference's own normal tail; an old reply cannot borrow a new reply's window.
  Existing disabled/bypass configuration remains effective.
- TTS timing now uses generated WAV duration instead of word count. Word count is
  only a fallback when duration cannot be read. Output transport/player delay is
  still estimated, which is why the bounded prefix tail is separate.

Suppression occurs before request admission, conversation history and tool/model
processing. Nearest-human attribution, raw confidence, confident-self rejection,
voice/text processing and action authorization remain unchanged.

## Validation

81 focused local tests and **324 Apollo tests** passed. New coverage includes FR/EN
opening matches, delayed ASR, numeric replies, stale/future references, attributed
human preservation, unrelated openings, disable/bypass behavior, voice rejection
before the queue and WAV-based TTS timing. Existing wake, identity, mute, lookup,
control, history, queue, monitor and speech-language suites pass.

Offline chronological replay of the captured conversation's text, timestamps and
WAV durations would suppress four feedback-like turns, including the first clock
echo and both no-device repetitions. It preserves both attributed user questions,
the unknown-speaker wake/question turns, and leaves three badly garbled or mid-reply
feedback-like transcripts unmatched. Utterance numbers reset on reconnect, so onset
correlation uses the latest matching start before that request's speaker event.
This is a retrospective guard decision, not a new acoustic end-to-end pass.

Private evidence resides in the previous run
`20261018T141513Z-lookup-fast-731426/feedback-review` on the Mac, including the log
snapshot and `candidate-log-replay.json`. New source is staged in Apollo run
`20261018T145826Z-echo-prefix-01dafd`, archive SHA-256
`d8e098ee889c78137691cc2fe8c3c4c9f4c29ba7f505ef5b33dd97bf9fb93611`.
Live playback/ingress were paused during tests; no automated audio was played.

The deployed smoke check returned **10/10 identical replies** to the previous run:
FR/EN facts, on/off controls, lookup and saved arithmetic speech. The two saved voice
inputs retain their raw ASR mismatch flags. Four queue trials pass; median submitted
to first generated audio is 4.241 s (previous suite 4.031 s). Test ordering/cache
state differs, and saved voice request-to-audio increased by about 0.4–0.5 s; this is
not a controlled no-latency-regression claim. The guard adds no model/tool round trip.
Network audit saw only loopback destinations; all 26 accepted requests completed,
with four explicit overload rejections. Archive/runtime source hashes match.
Mac playback-only and LAN ingress were restored after this evaluation; Mac capture
stays off. Physical echo behavior awaits the next live conversation.

GitNexus was refreshed before edits: voice finalization LOW (caller `ws_asr`), TTS
worker LOW (`speak`, five speech flows), config LOW. The dynamic guard call was
unresolved in the graph and verified directly in voice finalization. Serena edits
preserved inherited changes; no commit or bulk staging.

## Limits

An unknown human repeating the same opening near playback can still be suppressed.
Three common opening words are evidence, not identity proof. The numeric word list
supports common FR/EN forms, not every number/language. Prefixes that ASR changes
into another language, completely garbles, or misses at the start remain unmatched.
There is still no physical playback acknowledgment or ESP32 playback-aware muting.
Do not report the whole acoustic feedback problem solved from these tests.
