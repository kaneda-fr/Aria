# Architecture review: existing Xeon and ESP32, local operation

Date: 2026-10-17. Status: **RECOMMENDATION, not an accepted architecture decision**.
Decision owner: project maintainer. [D-001 remains open](decisions.md).
Current delivery scope is accepted separately in D-005: Apollo server development
with the Mac microphone client; ESP32 integration is deferred.
Baseline: `feature/plugin-system` at `905fd38` plus the existing untracked router.
This is a source/history review; no new hardware benchmark or live-device test.

Follow-up: the [2026-10-17 bilingual baseline](baseline-2026-10-17.md) now supplies
measured conversation evidence and a prioritized fix list. It confirms shared
request handling and admission as the first work; it does not compare the control
architecture candidates or accept D-001.

Subsequent implementation: [shared request handling](shared-request-path.md) now
resolves the voice/text wiring divergence, silent pending-request displacement,
and generic CHAT response branch. Findings below describe the pre-refactor source
unless noted; the architecture comparison and D-001 remain open.

The next [turn/language iteration](turns-and-language.md) keeps all eight WAV cases
to one request and removes wrong-voice selections. Physical capture still includes
unexplained extra content; four stories still truncate. These are conversation
pipeline findings, not evidence selecting either control architecture.

## Recommendation

Use one server-owned control service for voice and text. Let a compact local model
interpret language; let ordinary code resolve devices, validate requests, execute
permitted actions, and track outcomes. Keep the existing audio pipeline and providers.
Migrate incrementally from the current tool loop rather than rewriting the system.

The preferred candidate is a revised Option A: one constrained interpretation for
ordinary controls, deterministic execution, and a short factual response. A bounded
tool proposal can express the same interpretation; JSON versus tool syntax is a
secondary choice. The important boundary is that the model cannot choose its own
execution loop or turn a fabricated result into a success claim.

This is the best-supported direction from the current failure modes, not proof that
a particular model will meet accuracy or latency requirements. Compare it with a
hardened bounded tool candidate using the same model and held-out requests before
accepting D-001. Keep unrestricted multi-step tool use out of the initial control path.

## Options against the constraints

All candidates must operate without external calls after readiness. A local LLM
endpoint alone does not establish that its loaders, dependencies, or providers comply.

| Option | Fit for constrained Xeon | Reliability and product tradeoff | Recommendation |
| --- | --- | --- | --- |
| Current model-controlled tool loop | Repeated inference and growing tool/history context; API allows five iterations | Flexible, but historical runs skip actions or call tools during conversation; voice does not share this path | Preserve as a comparison baseline, not the preferred target |
| Compact model interpretation + deterministic control | Usually one interpretation; optional bounded repair before any side effect; no model call needed for routine confirmation | Explicit validation and execution trace; ambiguity and follow-ups must be designed carefully | Preferred candidate |
| Bounded tool proposal + deterministic control | Can have the same one-call budget if inventory lookup/execution stay in code | Reuses schemas; only useful if it performs as well without restoring model-controlled loops | Compare as an alternative interface to the same control service |
| Rules or fixed commands only | Lowest inference cost for supported requests | Narrow command grammar limits natural French/English conversation | Use for a small explicit fast path, not the whole assistant |

A larger model may improve language behavior, but the old report's suggested 7B
upgrade is not a justified default. First reduce unnecessary model work. Evaluate
the existing small-model baseline and compact quantized candidates on the actual CPU;
do not assume AVX512, VNNI, a GPU, a RAM capacity, or a specific tokens/second rate.

Subsequent read-only [Apollo inspection](deployment.md) establishes four CPU
cores/threads (Xeon E3-1225 v5), nominal 64 GB RAM, and AVX2 without AVX512. Use this
baseline for the comparison. The deployed script uses Ollama and differs from the
checkout's llama.cpp configuration; preserve both as distinct evidence. RAM capacity
does not remove the need to limit competing inference work on four cores.

## Current issues that drive this recommendation

| Evidence | Architectural response |
| --- | --- |
| `process_text` and `_llm_worker` in [main.py](../../app/main.py) differ: only the text API supplies the tool loop | Both adapters call one interpretation/control service; preserve session identity, speech options, and error semantics |
| The local historical decision report records 14/20 passes and about 7.10 s mean control latency; some supposed passes are only clarification/tool lookup | Replace the test oracle with expected intent, target, payload, execution count, and outcome; measure current latency afresh |
| [_enqueue_final_llm](../../app/main.py) replaces the previous waiting item when the shared queue is full | Bounded admission with explicit busy/cancel/expired outcomes and session fairness; never silently lose an accepted command |
| [ActionsProvider.execute_command](../../app/plugins/actions/provider.py) reports MQTT publish acceptance as success | Separate submitted, observed completion, failed, and unknown/timeout; confirmations use these facts |
| [ASR](../../app/asr/parakeet_onnx.py) can default to all CPU cores; [speaker recognition](../../app/speaker_recognition/recognizer.py) also uses Torch threading | Budget total inference concurrency and threads across stages; reserve capacity for audio and other server work |
| [Router](../../app/routing/router.py) defaults to separate English/French embedding models; [loaders](../../app/routing/embeddings.py) use model identifiers | Measure whether routing saves more work than it adds; preload selected assets and enforce offline operation |
| [llama.cpp adapter](../../app/llm/ollama_client.py) currently has no JSON-schema request field and streaming tool deltas are overwritten rather than accumulated by index | Verify the chosen backend/version contract; implement and test constrained output or complete tool reconstruction before relying on it |

The [evidence log](status.md) distinguishes historical benchmark claims from current
runtime behavior. GitHub supplied no issue backlog; the issues above come from source
and local reports. Recovered memories did not establish a newer architecture decision.

## Proposed processing and authority boundary

```text
Mac microphone client -> bounded LAN audio -> VAD / ASR / echo guards
text API ---------------------------------------------------------> request service
request service -> explicit fast path OR compact local interpretation
                -> conversation: local response, no actuation capability
                -> state query: inventory/provider read, factual response
                -> control: resolve -> validate -> execute -> observe -> response
                -> ambiguity/unsupported: clarify or explain, no execution
response -> existing local Piper / Sonos path where speech is requested
```

Use a small interpretation schema distinguishing chat, read-only query, control,
clarification, and unsupported requests. Include action, target hints, and values;
validate them against the actual provider capabilities. Do not force every request
into the historical Option A's five-action vocabulary or accept model-invented IDs.
For a large inventory, retrieve a bounded candidate set locally instead of placing
all devices and accumulated history in every prompt.

Store short, session-scoped pending clarification state tied to actual candidates.
Resolve follow-ups such as “the kitchen one” against that state; expire it and
revalidate current inventory before execution. A model's self-reported confidence
must not authorize actuation. Ambiguous targets and invalid payloads cause no action.

For the first milestone, support one explicit action per accepted request. Execute
it at most once at the application boundary; clarify or reject unsupported compound
requests. A syntax repair may run once before execution. Never retry a side effect
because a later model response failed. Request IDs and deduplication help, but MQTT
delivery and reconnect behavior still require integration-specific tests: an
application guard is not an exactly-once physical-device guarantee.

Use templates for routine outcomes. “Command sent” and “light is on” mean different
things. Confirm completion only with suitable state/acknowledgement evidence from
the integration; otherwise report the uncertainty. Do not require a second LLM call
to invent a friendly confirmation. Preserve natural conversation in a separate local
response path that has no actuator access.

Schema-constrained decoding is a candidate implementation aid, not a semantic safety
guarantee. llama.cpp documents JSON-schema-to-grammar support, with limitations; the
actual installed server and this repository's adapter need a compatibility test.
See the [official grammar documentation](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).

## Work placement and resource budget

Keep broader ASR, bilingual interpretation, inventory, execution, outcome tracking,
and speech synthesis on the Xeon initially. Reuse the existing provider indexes and
bounded in-memory state; this milestone does not need another database, vector
service, or distributed agent framework.

The maintainer has identified a Seeed XIAO ESP32-S3 camera/Sense listing; the combined
Sense/Plus seller title leaves the exact SKU uncertain. See [hardware evidence](deployment.md).
Firmware and roles remain unverified. If they are audio
satellites, assign capture, bounded buffering/pre-roll, transport, and only supported
local wake/VAD work. If they are actuator controllers, retain their local command/state
role. Use the documented Sense capabilities as a candidate baseline, verifying actual
PSRAM/flash and microphone capture before sizing firmware. The inspected repository does not yet
provide a verified ESP32 implementation to assess.

Espressif's documented MultiNet is limited command recognition in English/Chinese;
it does not meet Aria's French/English conversational objective by itself. Its
resource figures are board/model-specific. See the official
[MultiNet documentation](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/speech_command_recognition/README.html)
and [benchmarks](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/benchmark/README.html).
Retain the current echo guard; do not assume an ESP32 has a synchronized playback
reference for audio coming from a separate Sonos speaker.

Start with one bounded LLM inference slot, but measure the entire pipeline under
simultaneous audio capture, ASR, speaker recognition, and playback. Set explicit
thread limits for the actual ASR backend, Torch, and LLM runtime rather than allowing
each to claim the whole machine. Keep blocking inference off audio/network handling.
Use queue age limits and explicit overload responses; stale device commands must
not run much later simply because capacity became available.

Run an ablation with and without the embedding router. Keep a router only if it
improves end-to-end correctness or total cost under realistic traffic. A compact
multilingual model is a candidate for reducing duplicate resident models, not an
assumed improvement. Defer online learning until labels, persistence, evaluation,
and rollback exist; silence is not verified success.

Measure idle/active power as well as latency and memory. An always-resident model
can avoid cold starts but consumes RAM; unloading can introduce latency. Choose from
measurements on the existing server rather than automatically enabling every model.

## Startup may download; ready operation cannot

Proposed lifecycle: `PREPARING -> WARMING -> READY`, with local failure/degraded
states. A preparation process may download missing model assets, pinned to recorded
revisions with integrity checks. Resolve all selected language/model/tokenizer/voice
assets before handing them to operational workers. No background downloader may
continue after Aria becomes operational; adding models requires an explicit return
to a non-operational maintenance/preparation phase.

Launch runtime workers with offline settings already present, use local asset paths,
warm required models, then expose readiness. Hugging Face documents that environment
variables are read at import time; setting `HF_HUB_OFFLINE` after imports is too late.
Disable its telemetry too, but do not mistake library flags for coverage of every
dependency. See [official environment-variable documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

Enforce a LAN-only egress policy for the runtime services, including model servers
and integrations, scoped so other host workloads are unaffected. Audit attempted
external connections: a blocked update check is still a defect under the requirement
of no external calls. A missing model during operation yields a local error or an
unavailable optional feature, never an online fallback. Readiness must distinguish
which features are available. Test both startup downloads and cached offline startup.

## Staged delivery and decision gate

1. **Establish the baseline.** Use Apollo's verified CPU/RAM/instruction support;
   record other server workloads, models and revisions with the Mac client. Correct the test
   oracle; record stage timings, peak RAM, concurrency, and idle/active power.
2. **Fix architecture-independent gaps.** Share voice/text orchestration, make queue
   admission explicit, introduce truthful outcome states, and implement the
   preparation/readiness boundary. Verify clean packaging includes required modules.
3. **Compare interpretation candidates.** Run revised Option A and bounded tool
   proposals against the current loop on the same held-out bilingual cases and small
   local model. Hold hardware, prompts/data versions, and provider fixtures constant.
   Compare routing on/off and quantization separately so causes remain identifiable.
4. **Validate the current deployment.** Use fake executors for broad regression,
   then an explicitly scoped real-device trial on Apollo with the Mac client. Include
   overlapping sessions, sustained audio, echo, ambiguous follow-ups, device failure,
   timeouts, reconnects, duplicate requests, noisy ASR, and Internet disconnection.
   ESP32 firmware and integration validation are deferred to a later milestone.
5. **Accept or revise D-001 from results.** Record the chosen design, measurements,
   limitations, and rejected alternatives. Expand integrations and learning later.

Acceptance requires zero unintended/duplicate actuation and zero false completion
claims in the evaluation suite, correct supported actions or justified clarification,
no silent accepted-request loss, and no external calls after readiness. These are
proposed test gates, not claims of proven universal reliability. Record latency
p50/p95 from speech end to acknowledgement and observed outcome, plus stage timings.
Agree numerical latency, capacity, memory, and power budgets after the baseline;
historical results are insufficient to promise a performance target now.
