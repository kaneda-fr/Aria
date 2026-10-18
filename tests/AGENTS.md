# Test and evaluation context

Read [workflow](../docs/agents/workflow.md), [status](../docs/agents/status.md), and
[decision criteria](../docs/agents/decisions.md).

- For wake-enabled runs, activate each session and assert ignored requests have zero
  model/tool/audio effects. Keep short-name ASR failures separate from state tests;
  see [wake benchmark](../docs/agents/wake-flow-2026-10-18.md).
- Separate offline unit tests, model benchmarks, API integration, and live-device tests.
- Use the [Mac + Apollo runner](../docs/agents/e2e-testing.md) for isolated conversation
  pipeline checks; its capture sink does not prove physical Sonos or home-device control.
- Test startup model downloads separately, then block Internet egress with LAN
  available after readiness. Inspect attempted calls during requests and background
  work; no downloads, update checks, telemetry, or remote fallback are allowed then.
  Also test cached-model startup offline. Mocks alone do not establish compliance.
- Initial offline selection: `python -m pytest tests/plugins/test_plugin_system.py tests/test_sonos_http_sink.py`.
- Context checker only (standard library): `python3 -m unittest discover -s tests -p test_context.py -v`.
- Root-level test scripts can actuate real devices; inspect before running them.
- A tool count or HTTP success is not a correct-action oracle. Assert tool names,
  grounded IDs, validated payload, execution count, and truthful outcome separately.
- Compare latency by turn as well as median: a local answer can defer history prefill
  to the next model turn. Spoken tool JSON is not a valid capability answer.
- Compare prefill enabled/disabled with the same source, prompt/history, and tools.
  Include real overlap/queue wait, overload rejection, and prefill cancellation;
  count discarded prefill separately from generated conversation replies.
- Evaluate text and voice entry points; keep data splits held out from seed/training examples.
- Record model/version, hardware, configuration, dataset, sample size, and p50/p95 timings.
- Acceptance uses Apollo/Mac plus the new ESP32 client (D-012); board validation is pending.
  Eventual hardware acceptance includes the existing Xeon and actual ESP32 variants: peak RAM,
  CPU/concurrency, buffering, reconnect behavior, and idle/active power where measurable.
  Record unmeasured budgets explicitly; a faster development machine is not acceptance evidence.
- For microphone baselines, review extra/missing input content as well as answer,
  voice, and request count; a merged contaminated transcript can pass those checks.
- Saved manual captures need a user-confirmed spoken reference bound to the original
  audio hash; never assume the displayed prompt was spoken from a “clean” note alone.
- Synthetic benchmark accuracy is not production control reliability or online-learning evidence.
- Update status when evidence changes; preserve historical reports as dated evidence.
