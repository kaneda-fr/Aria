# Apollo conversation and control baseline — 2026-10-18

## Scope and reproduction

Maintainer requested an autonomous conversation/latency benchmark with **Mac capture
and playback off**. All inference, saved-audio ingestion, and WAV generation ran on
Apollo. Tools remained log-only; no MQTT client or physical action. No production
runtime fix is claimed by this task.

Runner: [conversation_bench.py](../../scripts/e2e/conversation_bench.py).
It generates inventory-grounded expected commands independently of model replies,
uses unique conversation sessions, and tests the shared text processor with tools
and speech enabled. Optional saved human PCM traverses `/ws/asr` using explicit
client utterance boundaries. This excludes microphone acoustics and native VAD.

```sh
cd <staged-source>
<apollo-python> -m scripts.e2e.conversation_bench \
  --server-run <isolated-live-run> --output <private-report-directory> \
  --captures <confirmed-capture-directory>
```

`--captures` is optional. The live adapter must report log-only actions, ready speaker
recognition, and completed warm-up. Never point this runner at a physical-action
server. Artifacts include exact prompts/replies, command arguments/results, request
IDs, model timings, audio metadata, inventory hash, runner hash, and capture hashes.
Keep them private: they contain household inventory and identity data.

Baseline ID: `20261017T224427Z-conversation-bench`; server ID:
`20261017T223829Z-headless-live-788648`. Private artifacts live under the maintainer's
Aria log directory on the Mac and `/data/aria/e2e/reports/<baseline-id>` on Apollo.
`report.json` is the original 28-case run; `manual-review.json` records semantic
review without overwriting automatic results. `supplement.py`/`supplement.json`
add four natural questions/follow-ups and two voice timing repeats. Archived runner
versions preserve the hashes before and after first-audio instrumentation.

Hardware/configuration: Xeon E3-1225 v5, four CPU threads, nominal 64 GB RAM;
Qwen2.5 3B Instruct Q4_K_M, model digest prefix `357c53fb659c`, temperature 0,
80 generated tokens per model call, 2048 context; 12 history messages; actual
cached inventory, six speaker profiles, Parakeet CPU ASR, local Piper FR/EN.
The isolated server source archive SHA-256 is
`1157cc626eb0c8532abf542f03757d5cf407e3cc08ea935290312fe8f1fa6dd8`.
The working tree is dirty; HEAD alone does not identify the tested implementation.

## Results: transport completion is not conversation success

All **28/28** baseline requests generated and served audio. Automatic guards cleared
17; manual review accepted only **13/28** for their particular fixture. This is a
small diagnostic suite, not a production accuracy estimate. No repeated statistical
comparison or held-out model selection experiment was performed.

| Category | Observed result |
| --- | --- |
| Short facts, arithmetic, secret-word recall, topic change | 10/10 expected answers; no tools |
| Tool-capability questions | FR attempts an invented action; EN speaks tool-call-shaped JSON instead of a natural explanation |
| Local clock | 0/2 correct; FR fabricated log-only JSON, EN “Not applicable” |
| Explicit on/off/dim/close controls | **1/8 exact command traces**, 0/8 complete correct and truthful interactions |
| Nonexistent room | Neither executes; EN returns unrelated candidates and invents a device-state claim |
| Ambiguous light request | FR falsely says no light found; EN selects an arbitrary light and claims success |
| Confirmed saved human recordings | 2/2 exact ASR transcripts, one request each, attributed to the enrolled person, appropriate reply and TTS voice |

Four French control replies were in English although the request-specific TTS voice
remained French. This is model response-language failure, not evidence that the TTS
language fix regressed. No wrong TTS voice occurred in the baseline.

Eight action attempts: four invented IDs rejected, one slider without a required
value rejected, and three valid IDs recorded. Those three were: a wrong **open**
action for requested **close**, the correct off follow-up, and an ungrounded light
choice for an ambiguous request. All had `executed=false` and `dry_run=true`.
The correct off call was followed by a false claim that the light had changed.
The French shutter reply described the wrong-direction action as closing.

The EN on request retrieved the correct on/off candidates, described them as separate
devices, did not execute, and hit the 80-token output limit. The EN dim request first
invented an ID, then obtained the correct candidate but asked the user for a command
ID. IDs are implementation details, not a viable speech-only clarification.

The four supplementary natural turns ask why the sky is blue and follow up about
sunset, with one-sentence answers. All kept context, stayed in the requested language,
and avoided tools. The two French explanations were factually wrong: the first
attributed blue sky to absorption by the Earth; the second reversed blue/red
wavelength ordering. English gave adequate simplified scattering explanations.
Fluency and successful transport are insufficient answer-quality checks.

Both saved recordings, repeated twice, had exact ASR and the same nearest enrolled
person. Scores were 0.528 EN (`known=false`) and 0.651 FR (`known=true`), with the
configured 0.65 threshold. Attribution below threshold follows the restored legacy
policy; this is not proof of high-confidence verification or unknown-speaker rejection.

## Latency on Apollo

Times below are **processor start to first generated WAV fetched by the capture
sink**, excluding ASR, endpointing, network playback transport, and physical sound.
Every baseline reply used one audio chunk, so first and last audio-ready coincide.
The runner's `observed_total_s` and original `saved_pcm_end_to_audio_observed_s`
include waiting for `speech_busy` to clear, including simulated speech duration;
**do not interpret those fields as first-response latency**. The supplementary runner
adds explicit first-audio timestamps using the shared Apollo host clock.

| Baseline subset | n | Median | Observed range |
| --- | ---: | ---: | ---: |
| Short answers with warm model | 9 | 1.99 s | 1.51–2.33 s |
| Explicit controls | 8 | 13.21 s | 8.26–30.87 s |
| All device requests, including ambiguity/missing room | 12 | 11.46 s | 8.26–30.87 s |
| Complete baseline, mixed cases | 28 | 6.51 s | 1.51–30.87 s |
| Natural questions/follow-ups, supplemental | 4 | 5.84 s | 4.64–7.96 s |

Mixed-suite nearest-rank sample p95 is 26.15 s. This small, heterogeneous sample is
not a service p95 or a comparison of French versus English quality. FR ran first,
and all requests were sequential. No contention/overload latency was measured.

- Explicit controls spent **93.8%** of measured pre-audio time in the model: mean
  15.13 s model work, including 5.52 s prompt processing and 8.88 s generation.
  They required 2–4 model calls. The wrong-direction shutter case took 30.87 s.
- Reply-to-audio averaged 0.99 s for explicit controls. Queue admission-to-running
  was below 1.1 ms across the sequential baseline. Neither dominates this run.
- Control replies themselves averaged **12.68 s of generated speech**. With
  interruption parked, verbose failure explanations also postpone listening again.
- The first short answer took **17.91 s** after the model had idled: 15.34 s prompt
  processing and 1.87 s load. Ollama logs show a new runner starting. Startup warm-up
  is effective while retained but does not keep the model resident indefinitely;
  a true-after-startup `warmup_ready` flag is not a current residency check.

Instrumented saved-voice repeats measured **2.54 s EN / 2.74 s FR** from explicit
end of submitted PCM to first audio-ready. About 1.15 / 1.05 s preceded the shared
request processor (ASR/recognition/handling combined, not individually profiled).
These repeated replies reused cached TTS WAVs; they are a warm lower-scope check,
not representative fresh synthesis or live speech-end-to-speaker latency. No new
microphone recording, acoustic endpointing, or audible playback was involved.

## Why these failures occur

Source and traces agree on distinct defects:

1. [InventoryProvider.get_candidates](../../app/plugins/inventory/provider.py)
   only visits commands after detecting capability keywords. An exact device-name
   query can therefore return nothing. Accent handling misses `lumière`.
2. `_score_candidate` treats room as a bonus, gives unrelated devices a positive
   fallback score, and does not enforce requested action. The tool's top-five result
   can mix rooms, directions, sliders, and commands belonging to one device.
3. [Tool execution](../../app/llm/tools.py) does not require an inventory match before
   delegating to actions. The live log-only adapter rejected invented IDs; that is
   **not** evidence that the production fallback is safe. A known ID alone also
   cannot prove requested device, room, direction, or authorization by this utterance.
4. The model sometimes skips lookup, ignores correct results, or performs actions
   on a non-action question. Prompt instructions alone did not enforce the contract.
5. A generative final reply can contradict a structured result, select the wrong
   language, or spend tokens explaining internals. Clock handling remains gated by
   the disabled router, as documented in the prior conversation review.

## Recommended fix order

1. **Ground and validate execution in server code.** Fail closed on missing IDs or
   specs; require device, room, action, value/range, and ambiguity checks before any
   action. Make exact-name and accent-normalized retrieval work, apply explicit room
   and action constraints, deduplicate by command, and return compact relevant
   candidates. A known command from an unrelated request is insufficient. Cover the
   reproduced wrong-direction, arbitrary-device, invented-ID, and missing-room cases.
2. **Make outcomes truthful and brief.** Render concise FR/EN feedback from structured
   results for recorded/rejected/accepted/observed actions. Do not let the model
   turn dry-run or publish acceptance into physical completion. Provide local time
   and capability answers through reliable local handling. These changes can remove
   unnecessary final model turns while improving correctness.
3. **Retain the warm model within a measured idle resource budget**, then compare
   long-idle first turns and ongoing turns. Keep prompt/tools stable and compact;
   measure reduced model round trips before considering a larger model. Do not lower
   token limits blindly: this run already contains a truncated response.
4. Rerun this suite plus paraphrases and fresh sessions, then resume a user-controlled
   acoustic test when requested. Require both correct traces and truthful speech.
   Maintain log-only tools until reliability is demonstrated. No ESP32, interruption,
   or hardware-upgrade work is required to address these immediate server defects.

This order hardens the current tool loop; it does not select the open future control
architecture. CPU/power and whole-system RAM budgets remain unmeasured by this suite.

## Verification

Three benchmark-oracle tests passed on Apollo, covering wrong/opposite/duplicate or
missing commands, extraneous tools, missing audio, wrong answers, and token cutoffs.
Python compilation passed. No production implementation changed in this task.
The final runtime network trace observed 38 loopback destinations and zero external
or DNS attempts, under private-network isolation. This demonstrates this isolated
cached-inventory run, not production LAN/MQTT operation or startup model downloads.
Mac capture/playback processes were absent. Apollo's headless log-only service was
left active with its existing three-hour maximum runtime; the benchmark client exited.
