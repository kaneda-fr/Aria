# Speaker recognition regression review — 2026-10-17

**Superseded on 2026-10-18:** the maintainer requested restoring nearest-person
attribution while keeping confidence logs, and restarting with Mac playback and
log-only tools. The historical stop and policy recommendation below are not current
instructions. See [current status](status.md).

The maintainer requested stopping the live test and investigating why previously
working person detection disappeared. The Mac microphone/playback process, SSH
tunnel, and owned Apollo live service are stopped. Do not restart without a new
instruction. The pending control test remains **log commands only**, with no real
device actuation.

## Established cause of the visible identity change

Two development changes affected the experience:

1. The first live adapter reused the minimal E2E profile, which disabled speaker
   recognition. That was a test-profile omission, not a hardware limitation.
2. After enabling recognition, this task changed `SpeakerRecognizer.identify` from
   always returning the best candidate's base user to returning `unknown` when
   `known=False`. This is the **only source difference** between the deployed
   `/data/aria/dev` recognizer and the current recognizer. The existing threshold
   remains 0.65. This changed identity passed to the LLM, not just console formatting.

The latest stopped run loaded the local encoder and all six existing profiles.
All five logged identification results ranked the maintainer's profiles first,
with scores ranging from 0.248 to 0.537, below threshold. No identification timeout
or model error was observed for those results. Other finalized segments had empty
transcripts; the existing voice path only runs identification for nonempty filtered
text. Enabled recognition therefore does not imply a result for every segment.

## Same-input comparison on Apollo

Offline replay used the two previously confirmed saved recordings, unchanged
private profiles, the same locally cached ECAPA model, and networking disabled.
It did not reopen the microphone or run the conversation server.

| Recording | Deployed recognizer, four threads | Current recognizer, two threads |
| --- | --- | --- |
| English | Same correct nearest profile; 0.527862; known=false; candidate user returned | Same nearest profile; 0.527862; known=false; unknown returned |
| French | Same correct nearest profile; 0.650695; known=true; candidate user returned | Same nearest profile; 0.650695; known=true; candidate user returned |

Scores agree within floating-point rounding. On these inputs, embedding computation
has not regressed; the rejection policy explains the missing English identity.
Four-thread identification took 0.32–0.34 s, versus 0.56–0.62 s with two threads in
this small sequential probe. This is not an end-to-end latency comparison or proof
that four threads are optimal under concurrent ASR/LLM load.

Enrollment source is byte-identical to Apollo's deployed copy and uses the same
model family, 16 kHz mono audio, normalized per-recording embeddings, and a normalized
mean profile. Existing profiles contain only name and embedding; they do not record
model revision or enrollment audio provenance. Historical weights and original human
enrollment WAVs were not recovered in the inspected locations. The deployed event
log contains no speaker events. This review cannot establish historical score quality
or explain every low score from the unsaved latest live audio.

Private evidence: run `20261017T214203Z-live-f569ac`, `speaker-comparison.jsonl`,
`conversation-apollo.jsonl`, and `server.log` under the existing Mac E2E log directory.
The comparison includes actual private candidate names; keep those files out of Git.
Final network audit reports 19 loopback destinations and no external/DNS attempts.

## Recommended next change

Separate **nearest candidate** from **accepted identity** explicitly in the live UI,
logs, and conversation context. Show why a match was rejected instead of implying
the recognizer is absent. Decide the conversational identity policy deliberately;
do not silently treat nearest-match attribution as confirmed identity or lower the
threshold simply to force a name. This distinction also matters to self-voice filtering.

Calibrate acceptance with known-speaker and other-speaker recordings before changing
the threshold or replacing enrollments. Reuse saved audio first; a new capture is not
needed to demonstrate the regression above. Preserve recognition in the live profile
and compare four versus two threads within the full Apollo latency budget. Device
tools remain a separate pending dry-run task; they were not enabled by this review.
