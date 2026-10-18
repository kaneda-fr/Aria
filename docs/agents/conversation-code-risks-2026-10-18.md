# Server code-risk appendix — 2026-10-18

Companion to the [live conversation review](conversation-regressions-2026-10-18.md).
These are source findings and follow-up risks, not all newly proven live failures.
The main review records the evidence, unchanged runtime and recommended priorities.

## Control contract mismatch

The live/default model prompts teach search then execution, while generated
`execute_command` calls receive no authorization and are rejected. Only the explicit
parser in [`control.py`](../../app/llm/control.py) can create authorization;
[`tools.py`](../../app/llm/tools.py) validates/consumes it before dispatch.
This protects exact device/action selection but limits free-form controls even
when a model proposes the right action. Align the model/tool contract with
server-owned authorization; preserve D-001 as open. Do not remove authorization
merely to make a benchmark pass. The reviewed conversation has no action calls.

## Multiple clients share output

There is one [`speech_output`](../../app/speech_output/speech_output.py) and one
[`echo guard`](../../app/speech_output/echo_guard_v2.py) with only two recent events,
no client/destination association. Serialized inference does not serialize playback.
A second client's response can cancel another's server speech while the Mac queues
both. Named input sessions alone do not establish independent output sessions.
Define output destination, ownership, queue policy and cancellation consistently.

## CPU admission ends too late

The LLM FIFO is bounded, but ASR and speaker inference happen before it, on shared
engines from per-connection tasks in [`ws_asr`](../../app/main.py). Socket limits
do not constitute a measured four-core inference budget. A slow ASR final also
pauses reads on its own socket. Define bounded stage admission and preserve
drain/backpressure; verify concurrency before claiming multi-client capacity.
Speaker timeout cancels its waiter, not the underlying inference thread.

## Generation completion is unchecked

The [`LLM client`](../../app/llm/ollama_client.py) discards `done_reason` and returns
text even at a token limit. The live observer logs this, but the shared processor
can persist/schedule incomplete output as successful. This conversation stops
normally; historical story truncation remains a separate known defect. Carry
completion metadata to the response policy and test incomplete output explicitly.

## Success and observability are misleading

All ten reviewed requests finish `success=true` with no warning/error despite
semantic failures. TTS language events lack explicit request/playback IDs, speaker
timeout is silent, overflow is not an admission outcome, and playback completion
is absent. Add correlated stage events and end-to-end semantic checks. Queue
completion, scheduled speech, player start and first audible audio are different
observations; existing logs do not measure all of them.

## Profile drift

Bilingual prompting, log-only providers, warmup, history and Mac delivery live
partly in [`scripts/e2e`](../../scripts/e2e/live_server.py), while the default prompt
and providers differ. Promote tested runtime policies into normal modules and
configuration rather than accumulating indispensable monkey patches in a test
adapter. Keep log-only enforcement independently verifiable. The active runtime's
six central source files matched the working tree in this review.

## Packaging gap

[`pyproject.toml`](../../pyproject.toml) explicitly omits `app.speaker_recognition`
and `app.routing` from the wheel package list. Checkout-based tests cannot establish
that an installed wheel imports or behaves like Apollo's source run. This is
already a documented gap, not the cause of the present running instance's errors.
