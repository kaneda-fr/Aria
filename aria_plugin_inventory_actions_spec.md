# Aria – Plugin System & Inventory Tooling (Copilot Instructions)

> **Authoritative Copilot specification**\
> This document extends Aria’s existing Copilot instructions. It defines:
>
> - a **generic plugin system** for Aria (extensible, enable-on-demand)
> - a concrete **Inventory Tool Plugin** (Jeedom MQTT first source)
>
> Tone, structure, and rigor are aligned with the existing Aria Copilot instructions.

---

## 0. Scope & intent

This specification MUST be followed by Copilot when implementing:

1. A **pluggable subsystem** inside Aria
2. An **Inventory Tool Plugin** implemented on top of that subsystem

The design MUST support adding **other discovery sources later** (e.g. Home Assistant, HTTP APIs, file-based inventories, cloud providers) **without changing the LLM integration layer**.

---

## 1. Core design principles (non-negotiable)

1. **Local-first & deterministic**\
   All hot-path logic must be local, synchronous, and single-digit ms. No network calls, no LLM calls.

2. **Stable LLM contract**\
   The LLM must see a **stable tool interface** regardless of how many plugins are enabled.

3. **Plugins produce data, not tools**\
   Plugins never define LLM tools directly. They produce **capability data** consumed by a central tooling layer.

4. **Enable / disable on demand**\
   Plugins are loaded from config and can be disabled without code changes.

5. **Crash-safe & replayable**\
   Plugin state persistence must be atomic and recoverable.

---

## 2. Aria plugin system (generic)

### 2.1 Plugin lifecycle

All plugins MUST implement the same lifecycle:

```python
class AriaPlugin:
    def start(self) -> None
    def stop(self) -> None
    def health(self) -> dict
```

Optional hooks:

```python
    def on_config_reload(self, new_config: dict) -> None
```

---

### 2.2 Plugin registration

Plugins are registered centrally:

```python
aria/plugins/
  registry.py
```

```python
PLUGIN_REGISTRY = {
  "inventory": InventoryPlugin,
  # future: "calendar", "weather", "media", ...
}
```

---

### 2.3 Plugin configuration

Plugins are enabled via Aria config:

```yaml
plugins:
  inventory:
    enabled: true
    sources:
      - type: jeedom_mqtt
        enabled: true
        broker: 192.168.100.60
        topic: jeedom/discovery/#
```

Aria MUST:

- instantiate only enabled plugins
- pass each plugin its config subtree

---

### 2.4 Plugin boundaries

Plugins:

- MAY maintain internal state
- MAY subscribe to external systems
- MUST expose **read-only snapshots** to the rest of Aria

Plugins MUST NOT:

- call the LLM
- block the main event loop
- modify global state outside their namespace

---

## 3. Inventory Tool Plugin (overview)

The Inventory plugin is a **core plugin** providing a unified inventory view to Aria.

Responsibilities:

- ingest discovery data from one or more **sources**
- normalize into a common device/command model
- maintain an incremental `inventory.json`
- build fast in-memory indexes
- expose **candidate selections** for LLM use

Non-responsibilities:

- executing actions
- triggering discovery
- defining LLM tools

---

## 4. Inventory plugin – multi-source architecture

### 4.1 Sources

A source is a sub-plugin inside inventory:

```python
class InventorySource:
    def start(self) -> None
    def stop(self) -> None
    def normalize(self, raw_event) -> list[DeviceRecord]
```

Examples:

- `JeedomMqttSource`
- `HomeAssistantMqttSource`
- `HttpInventorySource`

Each source:

- produces **normalized DeviceRecords**
- is fully isolated

---

### 4.2 Inventory core

The Inventory plugin core:

- merges records from all sources
- deduplicates by `(source_id, eq_id)`
- maintains `seen_at` per source

---

## 5. Normalized data model (authoritative)

### 5.1 Device record

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

### 5.2 Command record

```python
CommandRecord = {
  "cmd_id": str,
  "device_id": str,
  "type": "info" | "action",
  "subType": str,
  "capability": str,          # canonical, language-agnostic
  "unit": str | None,
  "range": tuple | None,
  "tags": set[str],           # mandatory command-level tags
  "execution": dict | None,   # ExecutionSpec (required for action execution)
}
```

---

## 6. Inventory persistence

- File: `data/inventory.json`
- Atomic writes (`.tmp` + rename)
- Keep last known good backup
- Never auto-delete devices
- Mark stale when `now - seen_at > STALE_TTL`

---

## 7. Tags & capabilities

### 7.1 Device-level tags (mandatory)

Derived automatically from inventory:

- `room:<room>`
- `domain:<domain>`
- device name tokens

Device tags are used **only for scoping and grouping**.

---

### 7.2 Command-level tags (MANDATORY)

All commands MUST be tagged. Command tags are the **primary mechanism** for intent matching and down-selection.

Each command MUST have:

- exactly **one canonical capability tag**: `cap:<CAPABILITY>`
- one `type:<info|action>` tag
- one `subtype:<slider|binary|numeric|other>` tag

Canonical capabilities are **language-agnostic** and stable, for example:

```text
ON, OFF
OPEN, CLOSE, UP, DOWN, STOP
SET_LEVEL, SET_VALUE
READ_VALUE, READ_TEMP, READ_POWER, READ_CONSUMPTION
PLAY, PAUSE
VOLUME_UP, VOLUME_DOWN, SET_VOLUME
```

Capabilities MUST be derived deterministically, in this priority order:

1. `generic_type` (highest confidence)
2. `type + subType`
3. command name heuristics

Commands without a valid capability MUST be excluded from the LLM tooling view.

---

### 7.3 Optional LLM enrichment (offline only)

LLM-based tagging is optional and MUST:

- run only on inventory update (never hot path)
- only add **alias / synonym tags**
- never override canonical capability tags
- be cached and versioned

---

## 8. Text → tags (hot path, no LLM)

### 8.1 Language handling

- Load FR and EN packs simultaneously
- No language detection
- Score both

### 8.2 Automatic dictionaries

Built from inventory:

- room names
- device names
- command names

Expanded with small curated synonym packs.

---

## 9. Candidate selection

Steps:

1. extract room
2. extract domain
3. extract capability
4. intersect indexes
5. rank
6. top-K

All operations must be O(1) or O(log n).

---

## 10. LLM integration contract

### 10.1 Stable tools (defined centrally)

Plugins NEVER define tool schemas. Tool schemas are defined in a single central module (tooling layer) and remain stable.

### 10.2 Data exposed per request

Inventory plugin exposes candidate data only:

```json
{
  "scope": {"room": "Salon"},
  "devices": [...],
  "commands": [...]
}
```

---

## 11. Action execution (LLM-driven)

Aria MUST implement a separate **Actions / Executor plugin** to execute commands selected by the LLM.

### 11.1 Responsibilities

- Validate and execute action commands by `cmd_id`
- Route execution to the correct backend (Jeedom HTTP first)
- Enforce safety policies (type checks, ranges, allow/deny)
- Return structured execution results

### 11.2 Non-responsibilities

- Discovery ingestion
- Inventory normalization/tagging
- Dynamic tool schema creation

### 11.3 ExecutionSpec (authoritative)

Inventory MUST attach an `execution` spec for action commands. This allows the executor to be deterministic.

```python
ExecutionSpec = {
  "backend": "jeedom_http" | "jeedom_mqtt" | "sonos_http" | "custom",
  "target": {
    # backend-specific routing info
  },
  "args": {
    "value_required": bool,
    "value_type": "int" | "float" | "bool" | "string" | None,
    "range": tuple | None,        # (min,max) for sliders/setpoints
    "enum": list[str] | None,
    "template": str | None        # e.g. "type=setvalue&value=#slider#"
  }
}
```

Rules:
- `execution` MUST be present for commands that can be executed.
- Commands without `execution` MUST NOT be executable.
- Executor MUST NOT accept arbitrary payloads; it only accepts `cmd_id` + validated `value`.

### 11.4 Actions plugin configuration

```yaml
plugins:
  actions:
    enabled: true
    backends:
      jeedom_http:
        enabled: true
        base_url: http://192.168.100.60
        api_key_env: JEEDOM_API_KEY
        timeout_ms: 1500
```

### 11.5 Stable LLM tool contract (authoritative)

Tool schemas MUST be stable and minimal:

- `actions.execute(cmd_id: str, value?: any)`
- `actions.dry_run(cmd_id: str, value?: any)` (recommended)

Executor behavior:
- Validate against `ExecutionSpec.args`
- Reject missing/invalid values
- Enforce allow/deny + enabled/stale checks
- Return structured result:

```json
{
  "ok": true,
  "cmd_id": "2",
  "executed": true,
  "backend": "jeedom_http",
  "message": "Set Volets Salon to 30%",
  "observed": {"optional": "state"}
}
```

---

## 12. Concurrency & safety

- Single writer per inventory plugin
- Immutable snapshots for readers
- Source callbacks push into a queue
- Actions plugin MUST be thread-safe and MUST apply timeouts

---

## 13. Observability

Maintain counters:

- events_received
- devices_added
- devices_updated
- devices_unchanged
- parse_errors
- executions_total
- executions_failed

Expose via plugin health().

---

## 14. Non-goals

- No per-device tools
- No dynamic tool schemas
- No hot-path LLM calls
- Inventory plugin never executes actions directly (execution is delegated to the Actions plugin)

---

## 15. Summary

This design turns Aria into a **plugin-driven capability platform**.

Inventory becomes one producer of structured capabilities, enabling:

- aggressive pre-LLM down-selection
- stable LLM behavior
- future discovery sources without refactoring

Action execution is implemented via a separate executor plugin with a stable tool contract.

---

**This document is authoritative for Copilot.**
