# Apollo response-language detector screening

2026-10-18. Maintainer requested latency evidence before selecting a pre-TTS
detector. No production detector, prompt, history or voice configuration changed.

## Method and evidence

Private run `20261018T202742Z-language-eval-c9c288` on the existing Apollo Xeon
E3-1225 v5, Python 3.11.13. Dependencies prepared outside the serving venv:
pycld2 0.42, langid 1.1.6, lingua-language-detector 2.1.1. Official PyPI asset
hashes checked; downloads completed before offline execution. Lingua used six
languages: French, English, Portuguese, Spanish, German and Italian. This does
not establish rejection of every unsupported language.

72 texts, 1–282 characters: 48 saved benchmark replies and 24 synthetic cases,
including short/neutral replies, mismatched request/reply language and mixtures.
18 synthetic monolingual labels, three mixed labels, three neutral labels. Saved
replies are available for inspection, not all scored against language labels.
The current `speech_language` function was extracted unchanged using AST to avoid
application startup/audio side effects. It receives the request hint; other
detectors receive no hint. This compares detector calls, not identical fallback
policies or a new TTS integration.

Fresh process per profile; NumPy loaded before baseline RSS; BLAS/OpenMP limited
to two threads. Initialization and first call measured separately. All texts
exercised before seeded shuffled timing: 1,152–2,880 calls/profile, 23,112 total
across nine profiles. Fast profiles have only 21–42 ms timed loops: quick screening,
not sustained-load tail guarantees. Filesystem caches were not flushed. No parallel
inference load or physical audio. Network namespaces had no external route; four
traces show no IPv4/IPv6 socket operations or DNS attempts. A local netlink route
query is not external traffic. The initial empty-route-table probe failed before
measurement and was corrected; no failed run contributed latency samples.

Private artifacts under the run ID include `cases.json`, `bench.py`, `summary.json`,
per-profile JSON with raw classifications, host/package metadata and network traces.
Serving Aria stayed active, tools log-only, monitor pointer unchanged; physical
input/playback remained paused.

## Measured latency and resources

Milliseconds per reply. Added RSS is warm process RSS minus the NumPy-loaded
baseline, not package size or exclusive physical memory.

| Detector | Warm p50 | Warm p95 | Full reply + sentences p95 | Added resident MiB |
| --- | ---: | ---: | ---: | ---: |
| Current heuristic | 0.0054 | 0.0162 | not measured | 1.2 |
| CLD2 | 0.0062 | 0.0140 | 0.0353 | 5.6 |
| CLD2 with byte-range vectors | 0.0068 | 0.0162 | not measured | 7.3 |
| langid, full inventory | 0.7507 | 0.9673 | 2.7176 | 45.1 |
| Lingua low accuracy, six languages | 0.0623 | 0.2126 | not measured | 19.7 |
| Lingua high accuracy, six languages | 0.1326 | 0.2408 | 0.6509 | 208.3 |

Sentence profiles also classify the whole reply. Split at sentence-ending
punctuation followed by whitespace; inspect up to four sentences; avoid an extra
call for one sentence. Their separate-process RSS deltas: CLD2 7.0 MiB, langid
45.0 MiB, Lingua high 202.6 MiB.

Import/construction: CLD2 1.5 ms, langid 1,318 ms, Lingua low 33 ms, high 364 ms.
First calls: CLD2 0.07 ms, langid 86.5 ms, Lingua low/high about 6.3 ms. Warm-up
must finish before readiness. langid initialization peaked at 164.4 MiB total
versus 73.2 MiB warm; Lingua high peaked at 256.2 MiB versus 236.3 MiB warm.
Full-reply timed-loop CPU/wall seconds: heuristic 0.026/0.021, CLD2 0.022/0.022,
vectors 0.029/0.025, langid 2.735/1.396, Lingua low 0.230/0.230, high 0.323/0.323.
Counts differ; totals are resource evidence, not a direct speed ranking.

## Quality cautions and recommendation

Exact labels on the 18 synthetic monolingual cases: heuristic 15/18 (cannot express
unsupported languages), CLD2 14/18, langid 17/18, Lingua low 13/18, high 17/18.
Unknown counts as a non-match, not necessarily an unsafe result. Neutral cases
excluded. No confidence threshold or neutral-text policy tuned.

- CLD2 confidently labels synthetic French “La capitale de la France est Paris.”
  as English; “Oui.” and “Yes.” return unknown. Default reliability is insufficient
  for a sole short-reply guard. Vectors leave each synthetic short mixture in one
  detected range, without exposing both languages.
- Whole-reply detection hides mixtures. Sentence checks expose differing labels
  in five of six saved mixed replies for CLD2, all six for langid and Lingua high.
  CLD2 leaves one second sentence unknown. Lingua calls the first fragment of one
  mixture Spanish: this is mixture evidence, not perfect component identification.
- All three sentence profiles expose both synthetic mixtures across sentences;
  all miss the French/Portuguese mixture inside a single sentence.
- Short/neutral handling remains necessary. Lingua high calls the synthetic French
  secret-word sentence Portuguese. Low accuracy mislabels several ordinary short
  English/French sentences. A strict mismatch rule would cause false fallbacks.

Latency is affordable compared with multi-second generation. CLD2 is fastest and
smallest in measured added RSS, but is not ready as a sole guard. Lingua high plus
sentence inspection is a useful next quality candidate at about 0.65 ms p95 and
200 MiB added RSS; langid offers broader language coverage and lower steady RSS
at about 2.72 ms p95. Evaluate confidence/unknown, neutral handling and mixtures
within sentences before choosing. Do not restrict classes to French/English only.
Keep the maintainer's short honest fallback preference rather than generation retry
if a guard is accepted. This screen does not deploy one or measure end-to-end TTS
latency/quality regression.

API references: [pycld2](https://pypi.org/project/pycld2/),
[CLD2](https://github.com/CLD2Owners/cld2),
[langid](https://github.com/saffsd/langid.py),
[Lingua 2.1.1](https://pypi.org/project/lingua-language-detector/2.1.1/).
API descriptions come from these sources; measurements above come from Apollo.
