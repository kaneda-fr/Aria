# Short-interaction language detection on Apollo

2026-10-19. Evaluation only; no serving detector, prompt, voice or history change.
This extends the [initial screening](language-detector-evaluation-2026-10-18.md).

## Finding

The best tested resource/accuracy candidate is the existing French/English voice
evidence combined with Lingua low-accuracy mode, neutral-text handling and bounded
sentence/window checks. It costs about **0.25 ms p95 and 20 MiB added resident RAM**
on Apollo for short replies. It is **not accepted for production**: a fresh borrowed-word
example and longer saved replies produce false fallbacks. Latency is affordable;
false rejection of good speech is the remaining decision blocker.

The October 18 recommendation to evaluate Lingua high alone is superseded by this
comparison of complete decision policies. This does not establish that low mode
is intrinsically more accurate than high mode: vocabulary, thresholds, segmentation
and neutral handling differ in their effect on short conversations.

## What was compared

All candidates run locally on the existing Xeon E3-1225 v5, Python 3.11.13, with
pycld2 0.42, langid 1.1.6 and lingua-language-detector 2.1.1 from the previously
prepared private dependency directory. No downloads during measurement.

1. **Current:** unchanged `speech_language(text, requested)` extracted from source.
2. **Rules:** broader function-word evidence, French clitic handling, shared neutral
   handling and bounded checks for both French and English within the reply.
3. **Raw libraries:** full-reply classification without neutral/context protection.
4. **Library guards:** common neutral handling, confidence plus runner-up margin,
   sentence checks and optional overlapping six-word windows, at most 12 segments.
   Low/high Lingua initially use FR/EN/PT/ES/DE/IT; an additional low-mode profile
   includes all 75 supported languages. langid uses its full inventory.
5. **Conservative combination:** the six-language low guard, retaining the current
   detector's language when both request hints give the same answer; additional
   localized French/English evidence; short unknown replies keep the current voice
   decision. Mixed/unsupported results still request a fallback.

Neutral protection retains the current request's language for numbers, identifiers,
isolated unrecognized Latin words and exact inventory-name matches. Common short
confirmations have lexical language evidence. It does not use another request's
language. Five synthetic inventory names are supplied to every improved policy;
the current API does not receive an inventory-name set. Non-Latin prose is screened
before treating an isolated token as a name; known inventory names take precedence.
Unrecognized single words are inherently ambiguous under this policy.

These are response-language/voice decisions, not input-language or speech-recognition
accuracy. A pure English reply to a French request is scored for the English voice,
matching current server policy. Enforcing French generation is a separate problem.
Mixed or unsupported prose should trigger the maintainer's short honest fallback;
neutral content should preserve the request voice. Fallback text is not generated
or played by this experiment. No extra LLM call is used.

## Corpus and calibration

The committed [corpus](../../scripts/e2e/fixtures/language_cases.json) has 480 cases,
240 unique synthetic texts, each with both French and English request hints. These
are paired conditions, not 480 independent utterances. Texts are at most 68 characters.
Categories include confirmations, device status, names/numbers, borrowed words,
unsupported languages, mixtures across sentences and mixtures within a sentence.

- 96 development cases choose thresholds. Loss is twice false fallbacks plus wrong
  voice choices plus missed mixed/unsupported text. Search: confidence .5/.7/.85/.95,
  margin .1/.25/.4, sentence-only versus sentence plus windows. These library scores
  are not calibrated probabilities and cannot be compared directly across libraries.
- 192 test cases initially compare candidates. An experimental neutral helper was
  then fixed for non-Latin strings/ASCII identifiers; these 192 cases subsequently
  became regression evidence, not untouched validation for that fix.
- 112 new confirmation cases were frozen before the corrected comparison. Thresholds
  still used only the original 96 development cases.
- After reviewing those results, the conservative combination was added. Its 80 fresh
  cases were frozen before its run, with no text overlap with earlier partitions.
  Its results on the earlier 304 cases are development/regression evidence; the
  separate 80-case result is the fresh check. No further tuning followed that check.

This is an authored challenge corpus, with deliberately frequent mixtures and
opposite-language hints. Scores are not estimates of household failure rates.
No benchmark question is answered by a canned application response.

## Paired comparison on the earlier 304 cases

216 valid/neutral cases and 88 mixed/unsupported cases. A correct decision means
the proper voice or an appropriate fallback, not factual answer correctness.
Guard rows include their protection logic, not just the raw library call.

| Policy | Correct / 304 | False fallback / 216 | Wrong voice / 216 | Bad text missed / 88 | Warm p95 ms | Added MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 180 | 0 | 36 | 88 | 0.0055 | ~2 |
| Improved rules | 202 | 0 | 30 | 72 | 0.064 | ~2 |
| CLD2 guard | 265 | 20 | 7 | 12 | 0.058 | ~5 |
| langid guard | 255 | 26 | 5 | 18 | 2.295 | ~45 |
| Lingua low, six languages | 285 | 4 | 11 | 4 | 0.242 | ~17 |
| Lingua high, six languages | 251 | 30 | 1 | 22 | 0.301 | ~194 |
| Lingua low, 75 languages | 261 | 26 | 5 | 12 | 0.845 | ~92 |
| Conservative current + Lingua low | 297 | 0 | 3 | 4 | 0.253* | ~20 |

*Conservative timing includes the additional 80 fresh cases. Its p50 is 0.068 ms,
p99 0.335 ms, observed maximum 1.716 ms. The earlier rows share the 304-case timing
distribution. These are process-level wall times, not end-to-end queue/TTS latency.

The conservative combination introduces no errors on the earlier cases that the
current detector got right. This alone is insufficient: those cases informed its
design. Remaining errors include an isolated English confirmation with a French
hint, ambiguous short follow-ups, a Dutch sentence classified as English, and a
French/English mixture whose English clause has weak lexical evidence.

Raw libraries perform worse as direct replacements because names, numbers and
short confirmations get arbitrary languages or unknown results. For example,
raw CLD2 rejects 48 of the 136 valid cases in the initial 192-case partition;
its guarded version rejects four. Library-only accuracy cannot decide integration.

## Fresh 80-case check

56 valid/neutral and 24 mixed/unsupported cases; 40 unique texts, both hints.

| Policy | Correct / 80 | False fallback | Wrong voice | Bad text missed |
| --- | ---: | ---: | ---: | ---: |
| Current | 45 | 0 | 11 | 24 |
| Improved rules | 50 | 0 | 10 | 20 |
| Lingua low guard | 66 | 10 | 4 | 0 |
| Conservative combination | 74 | 2 | 4 | 0 |

The two new regressions are the two hints for one French phrase: “Le streaming
audio est suspendu pour cinq minutes.” It is falsely flagged mixed. The four
wrong voices are short monolingual replies with the opposite hint, already wrong
in the current detector. All eight neutral cases preserve the request voice.
Do not claim no quality regression or general 92.5% production accuracy.

## Replay of actual saved benchmark outputs

48 saved outputs from the previous six/eight-message history experiments, 1–282
characters, replayed without generation or audio. Contains six clear mixtures.
Manual labels assess language only: factual errors and truncation remain defects.
One dominant-French answer containing isolated “ao” is treated as French with a
typo/foreign intrusion; this ambiguous case is not proof of pure-language safety.

| Policy | Correct / 48 | False fallback / 42 | Wrong voice | Mixed missed / 6 |
| --- | ---: | ---: | ---: | ---: |
| Current | 41 | 0 | 1 | 6 |
| Improved rules | 43 | 0 | 0 | 5 |
| Lingua low guard | 45 | 3 | 0 | 0 |
| Conservative combination | 45 | 3 | 0 | 0 |

The combination catches all six old mixtures but rejects three otherwise
dominantly French/English replies (including the ambiguous “ao” case). At least
two false fallbacks are unambiguous regressions. Longer-reply p95 rises to 1.082 ms,
still small; localized false language detections are the problem.

## Operational evidence and limits

- Final comparison: `20261019T054744Z-language-context-0e1221`; conservative/fresh
  run and saved replay: `20261019T055128Z-language-context-3d1f94`.
  Initial screening with the neutral/RSS defects is retained separately as
  `20261019T054419Z-language-context-d660a8`; do not use its RSS figures.
- Fresh process per backend, two BLAS/OpenMP/Rayon threads, all texts warmed before
  timing, seeded shuffled rounds for at least two seconds/profile. Added RSS is
  measured before accumulating timing samples. It includes small harness/corpus
  overhead; peak RSS after timing is not detector memory. Six-language low startup
  was ~37 ms plus ~7 ms first call; initialize and warm before readiness.
- Seven comparison network traces plus three final-profile traces and a saved-replay
  trace show no IP socket or DNS attempts. Namespace isolation blocks external
  networking. Existing Apollo live services were not restarted/stopped or messaged;
  their background CPU load was not controlled. Load averages are saved per profile.
- Apollo's pointer was a newer log-only live run than the local status snapshot.
  No test opened a microphone, played audio, executed a device action or sent a
  request to that live instance. No Mac model inference. Private results contain
  corpus/source hashes, selected policies, errors, predictions, memory and timings.
- Bounded segments can miss late or small mixtures. A six-language inventory cannot
  reliably reject all other Latin-script languages. Neutral/known-name protection
  depends on trustworthy metadata and is not a general language recognizer.

## Next choice

Keep the current production guard while evaluating the conservative combination
in shadow mode or an isolated integration. It is the best measured short-reply
trade-off here, but false fallbacks must be resolved before activating rejection.
Prioritize corroborating a suspicious segment in its sentence context and separating
borrowed terms from language switching. Avoid adding phrase-specific exceptions.
Use another untouched corpus for that change; the cases above are now regression
fixtures. A second LLM call is neither needed nor justified by these timings.

## Reproduction and verification

The [runner](../../scripts/e2e/language_eval.py) is evaluation-only; application code
does not import it. With the pinned dependencies prepared, stage the runner, corpus
and the [frozen baseline](../../scripts/e2e/fixtures/language_baseline.py) on Apollo, then run each backend sequentially in
an empty network namespace, e.g.:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
PYTHONPATH=/path/to/prepared/deps unshare --user --map-root-user --net \
python language_eval.py --corpus language_cases.json --source language_baseline.py \
  --detector lingua_low --output result.json --seconds 2
```

The baseline contains the two unchanged functions extracted from the measured
working-tree source; that source was ahead of `main`. Its original full-file hash is
recorded in the fixture. Extraction changes the file hash, so compare function ASTs
and predictions when checking provenance. The fixture lets this evaluation run
without merging unrelated production work.

Capture network traces when verifying offline operation. The runner reports raw,
guarded and conservative profiles where applicable. No serving imports or TTS.

Test strategy: unit accounting/neutral boundaries and integration with real local
detector libraries are selected. UI/e2e, public contracts and deploy smoke are
inapplicable to this non-deployed evaluation tool; property-based testing is deferred
for this low-impact tool. Six oracle/boundary tests pass both locally and on Apollo,
covering independent error categories, identifiers, inventory names, short replies,
non-Latin text and the segment work bound. See [tests](../../tests/test_language_eval.py).
GitNexus marked neutral-helper impact HIGH but confined it to this runner's
classification/timing flows; factory and main edits were LOW. No app symbol edited.

Library references: [Lingua](https://github.com/pemistahl/lingua-py),
[langid](https://github.com/saffsd/langid.py), [pycld2](https://pypi.org/project/pycld2/).
The results above are our Apollo measurements, not upstream performance claims.
