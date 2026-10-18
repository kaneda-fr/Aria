# Routing context

Read [status](../../docs/agents/status.md) and [open strategy](../../docs/agents/decisions.md).

- Keep CHAT, CONTROL, and UNKNOWN distinct; measure false actuation separately from accuracy.
- Startup may download missing models before readiness. Once operational, use local
  models only: no lazy/request-time downloads, update checks, remote classification,
  external telemetry, or export of utterances/learning data.
- Verify runtime language/model selection; benchmark classifiers are separate implementations.
- Compare thresholds only on the same model, normalization, language split, and dataset.
- An `update_centroids` method is not a verified-outcome learning system.
- Do not learn from the router's own prediction. Resolve label quality, persistence,
  rollback, and consent/data handling before implementing online adaptation.
- Historical [learning rules](<../../SELF-LEARNING ROUTER RULES.md>) are a proposal;
  silence-as-confirmation needs review, not automatic acceptance.
- Report warm/cold timing and hardware with benchmark results. Do not infer end-to-end latency.
- Evaluate model memory and CPU cost on the existing Xeon alongside the other speech
  stages; extra embedding models or learning work must fit the shared resource budget.
- Update status/architecture for routing behavior and tests/AGENTS.md for evaluation changes.
