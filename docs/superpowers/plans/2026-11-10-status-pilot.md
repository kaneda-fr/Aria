# Lights and covers status pilot

> **For agentic workers:** Use superpowers:executing-plans, sequentially with test-first verification.

**Goal:** truthful everyday FR/EN status replies and scoped, traceable facts on the existing server.
**Architecture:** pure shared projection after existing bounded reads; direct language renderers; read-only cover resolver. Keep existing action authorization/history/queues.
**Tech stack:** Python 3.11+, existing dataclasses/httpx/pytest; no dependency or model added.
**Spec:** ../../agents/jeedom-status-contract-2026-10-21.md and ../../agents/jeedom-status-semantics-2026-10-21.md.

## Global constraints

Local-only; tools log-only; no hardware audio. Maximum eight info reads; preserve all source observations and exact history. Cover polarity requires an identity-bound explicit mapping; unknown is not closed/off. Never call action providers for a status. Everyday vocabulary in both languages, technical details in structured facts/logs. Respect catalogue incompleteness.

## Review focus

Missing/foreign/duplicate readings; invalid numbers and false polarities; source identity change; ambiguous and cross-domain followups; existential versus universal claims with incomplete coverage.

## Task 1: projection and shared rendering

Files: new app/plugins/inventory/status_facts.py; update state_read.py/light_queries.py/tools.py; tests/test_status_facts.py.
Interface: project_status(snapshot, identities, result, domain, mappings=None) -> report; render_status(report, language, wanted=None, count=False, quantifier=None) -> str. Local mappings are bound to an exact command/device fingerprint, loaded from ARIA_STATUS_MAP_FILE only.
- [x] Write and observe failing tests for mixed/missing states, fresh/unknown timestamps, coverage, both cover orientations and invalid mappings/numbers.
- [x] Implement pure projection and natural-language rendering. Attach reports to reads and tool output; reuse for light groups and individual reads.
- [x] Run targeted tests; update prior wording assertions only where the new user-approved vocabulary changes the contract.

## Task 2: cover request path

Files: new app/plugins/inventory/cover_queries.py; update main.py/state_read.py; tests/test_cover_status.py.
Interface: resolve_cover_query(text,snapshot,history=(),language='fr') -> plan|None; answer_cover_query(plan,provider) -> (text,result). Use bounded typed info selection and existing device forms, strict qualifiers and adjacent trusted history. No state/action ambiguity or generated authority.
- [x] Write failing FR/EN API/read/clarification tests; assert exact info IDs and zero model/action calls.
- [x] Implement group and singular cover status routing, ordinal/location followups, honest unknown mapping.
- [x] Run full safe tests directory; root integration scripts are excluded because some actuate devices.

## Task 3: target hardware comparison and evidence

Files: synthetic evaluation helpers/evidence in ignored .local/status-pilot; public report in docs/agents; environment/architecture/status/workflow updates.
- [x] Stage archived pre-change source and candidate with identical cached models/catalogue and synthetic read values. No actual audio clients or device action providers.
- [x] Compare held-out FR/EN texts, saved PCM, exact log-only controls, longer history and queue-inclusive first-WAV latency; report failures separately from latency gains.
- [x] Profile projection p50/p95 and CPU/RSS; audit offline network and stop isolated services afterward.
- [x] Run context/link/whitespace checks and review the final code and evidence.

Existing lighting edits are inherited and uncommitted: preserve them and record the pilot separately; do not commit combined inherited work as a task bookkeeping step. User has authorized implementation and target tests; proceed without a separate plan approval pause.
