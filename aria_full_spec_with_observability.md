# Aria – Plugin System, Inventory & Action Observability (Copilot Instructions)

> **Authoritative Copilot specification**  
> This document is the single source of truth for Copilot when implementing Aria plugins, providers, action execution, and observability.
>
> It defines:
> - a **generic plugin system**
> - a **Tooling Provider model** (inventory, actions, future providers)
> - an **Inventory provider** (Jeedom MQTT first source)
> - an **Actions / Executor provider**
> - an **Action Observability & async completion model**
>
> Tone, structure, and rigor are aligned with the existing Aria Copilot instructions.

---

## 0. Scope & intent

This specification MUST be followed by Copilot when implementing:

1. A **pluggable subsystem** inside Aria
2. A **provider model** enabling future providers (weather, news, calendar, etc.)
3. An **Inventory provider** implemented on top of that subsystem
4. An **Actions / Executor provider** for LLM-driven execution
5. **Action observability** with async feedback and TTL-based resolution

The design MUST support adding **other providers later** without changing the LLM integration layer.

---

## 1. Core design principles (non-negotiable)

1. **Local-first & deterministic routing**  
   All hot-path routing and down-selection must be local, synchronous, and single-digit ms.  
   No network calls, no LLM calls on the hot path.

2. **Stable LLM contract**  
   The LLM must see a **stable tool interface** regardless of enabled providers.

3. **Providers produce data, not tool schemas**  
   Providers never define LLM tool schemas directly.  
   They produce **structured data** consumed by a central tooling layer.

4. **Enable / disable on demand**  
   Providers are enabled via config and can be disabled without code changes.

5. **Crash-safe, cache-safe & replayable**  
   Provider state, caches, and expectations must be atomic and recoverable.

6. **LLM never decides truth**  
   LLMs may phrase messages, but **never decide execution success/failure**.

---

## 2. Plugin system (generic)

### 2.1 Plugin lifecycle

All plugins/providers MUST implement:

```python
class AriaPlugin:
    def start(self) -> None
    def stop(self) -> None
    def health(self) -> dict
```

Optional:

```python
    def on_config_reload(self, new_config: dict) -> None
```

---

## 3. Tooling Provider model (extensible)

A **Tooling Provider** is a plugin exposing structured data to the tooling layer.

```python
class ToolingProvider(AriaPlugin):
    name: str

    def get_snapshot(self) -> dict
    def get_candidates(self, text: str, context: dict | None = None) -> dict
```

Rules:
- Providers MUST NOT define LLM tool schemas
- Providers MUST NOT call the LLM
- `get_candidates()` MUST be deterministic and fast
- Network providers MUST use caching

---

## 4. Provider registration & configuration

### 4.1 Registration

```python
PROVIDER_REGISTRY = {
  "inventory": InventoryProvider,
  "actions": ActionsProvider,
  # future: "weather", "news", "calendar", ...
}
```

### 4.2 Configuration

```yaml
providers:
  inventory:
    enabled: true

  actions:
    enabled: true
```

---

## 5. Inventory Provider

### 5.1 Responsibilities

- ingest discovery data from one or more **sources**
- normalize into a common model
- maintain `inventory.json`
- build in-memory indexes
- ingest **state / event updates**
- expose candidates and state snapshots

Non-responsibilities:
- executing actions
- defining LLM tools

---

### 5.2 Inventory sources

```python
class InventorySource:
    def start(self) -> None
    def stop(self) -> None
    def normalize(self, raw_event) -> list[DeviceRecord]
```

---

### 5.3 Normalized data model

#### DeviceRecord

```python
DeviceRecord = {
  "source": str,
  "eq_id": str,
  "name": str,
  "room": str,
  "domain": str,
  "enabled": bool,
  "visible": bool,
  "commands": list[CommandRecord],
  "tags": set[str],
  "seen_at": int,
}
```

#### CommandRecord

```python
CommandRecord = {
  "cmd_id": str,
  "device_id": str,
  "type": "info" | "action",
  "subType": str,
  "capability": str,
  "unit": str | None,
  "range": tuple | None,
  "tags": set[str],
  "execution": dict | None,
}
```

---

## 6. Command tagging (MANDATORY)

Commands are the **unit of intent**.

Each command MUST have:
- exactly one `cap:<CAPABILITY>` tag
- one `type:<info|action>` tag
- one `subtype:<slider|binary|numeric|other>` tag

Capabilities are canonical and language-agnostic.

---

## 7. Text → tags → candidates (hot path)

- deterministic text normalization
- multilingual packs loaded simultaneously
- no LLM usage
- O(1) / O(log n) index intersections

---

## 8. Actions / Executor Provider

### 8.1 Responsibilities

- execute commands by `cmd_id`
- validate values
- enforce policies
- create **execution expectations**
- emit execution events

---

### 8.2 ExecutionSpec

```python
ExecutionSpec = {
  "backend": "jeedom_http" | "jeedom_mqtt" | "sonos_http" | "custom",
  "args": {
    "value_required": bool,
    "value_type": "int" | "float" | "bool" | "string" | None,
    "range": tuple | None,
    "template": str | None
  }
}
```

---

### 8.3 Stable LLM tools

- `actions.execute(cmd_id, value?)`
- `actions.dry_run(cmd_id, value?)`

---

## 9. Action Observability & Async Completion (authoritative)

### 9.1 Execution context

When an action is executed, the Actions provider MUST create an **Execution Context**:

```python
ExecutionContext = {
  "execution_id": str,
  "requested_by": str,
  "utterance": str,
  "cmd_id": str,
  "value": any | None,
  "created_at": int,
  "expires_at": int,
  "status": "pending" | "success" | "failed" | "timeout"
}
```

---

### 9.2 Expectation

An **Expectation** defines what observation resolves an execution.

```python
Expectation = {
  "execution_id": str,
  "observer_cmd_id": str | None,
  "predicate": "equals" | "within_tolerance" | "changed",
  "target": any | None,
  "tolerance": float | None,
  "after_ts": int,
  "timeout_ms": int
}
```

Rules:
- Expectations are optional
- If no expectation exists → optimistic execution
- Timeout ≠ failure

---

### 9.3 Event ingestion & matching

- Inventory ingests state/events and normalizes them
- Events are matched against expectations by `observer_cmd_id`
- Matching is deterministic
- No LLM involvement

---

### 9.4 Resolution outcomes

An execution resolves to one of:

- `success`
- `timeout`
- `failed`

Resolution emits:

```json
{
  "type": "actions.execution_completed",
  "execution_id": "...",
  "status": "success|timeout|failed",
  "observed": {
    "cmd_id": "...",
    "value": 42,
    "ts": 123456
  }
}
```

---

### 9.5 Notification & LLM usage

- Orchestrator decides whether to notify user
- LLM may be used **only to phrase** the message
- LLM input = execution context + deterministic result
- LLM MUST NOT infer status

---

## 10. Concurrency & safety

- Single writer per provider
- Immutable snapshots for readers
- Expectation tracker indexed by observer cmd_id
- Timeouts enforced centrally

---

## 11. Observability & metrics

Providers MUST expose counters:

- executions_total
- executions_success
- executions_timeout
- executions_failed
- expectations_active
- events_received

---

## 12. Non-goals

- No dynamic tool schemas
- No per-device tools
- No LLM decision-making
- No blocking on long-running actions

---

## 13. Summary

This design turns Aria into a **deterministic, provider-driven assistant**.

- Inventory discovers and tracks capabilities
- Actions execute side effects safely
- Observability confirms outcomes asynchronously
- LLMs are used for language, not logic

---

**This document is authoritative for Copilot.**
