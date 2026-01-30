# Aria – Plugin System, Providers & Tooling (Copilot Instructions)

> **Authoritative Copilot specification**  
> This document extends Aria’s existing Copilot instructions. It defines:
>
> - a **generic plugin system** for Aria (extensible, enable-on-demand)
> - a **provider model** to add future capabilities (inventory, actions, weather, news, etc.)
> - a concrete **Inventory provider** (Jeedom MQTT first source)
> - a concrete **Actions / Executor provider** for LLM-driven execution
>
> Tone, structure, and rigor are aligned with the existing Aria Copilot instructions.

---

## 0. Scope & intent

This specification MUST be followed by Copilot when implementing:

1. A **pluggable subsystem** inside Aria
2. A **provider model** enabling future tooling providers (weather/news/etc.)
3. An **Inventory provider** implemented on top of that subsystem
4. An **Actions / Executor provider** for executing commands selected by the LLM

The design MUST support adding **other providers later** (e.g. weather, external news feeds, calendar, HTTP APIs, file-based inventories, cloud providers) **without changing the LLM integration layer**.

---

## 1. Core design principles (non-negotiable)

1. **Local-first & deterministic routing**  
   All hot-path routing and down-selection must be local, synchronous, and single-digit ms. No network calls, no LLM calls.

2. **Stable LLM contract**  
   The LLM must see a **stable tool interface** regardless of how many providers are enabled.

3. **Providers produce data, not tool schemas**  
   Providers never define LLM tool schemas directly. They produce **structured data** (snapshots, candidates, execution specs) consumed by a central tooling layer.

4. **Enable / disable on demand**  
   Providers are enabled by config and can be disabled without code changes.

5. **Crash-safe, cache-safe & replayable**  
   Provider state persistence and caches must be atomic and recoverable.

---

## 2. Plugin system (generic)

### 2.1 Plugin lifecycle

All plugins/providers MUST implement the same lifecycle:

```python
class AriaPlugin:
    def start(self) -> None
    def stop(self) -> None
    def health(self) -> dict
```

Optional hook:

```python
    def on_config_reload(self, new_config: dict) -> None
```

---

## 3. Provider model (extensible)

A **Provider** is a plugin exposing structured data to the tooling layer. This enables adding capabilities like weather/news later without refactoring the LLM integration.

```python
class ToolingProvider(AriaPlugin):
    name: str

    # Stable read-only interface for tooling integration
    def get_snapshot(self) -> dict

    # Optional: deterministic candidate selection for context down-select
    def get_candidates(self, text: str, context: dict | None = None) -> dict
```

Notes:
- `get_candidates()` MUST be deterministic and low-latency.
- Providers that require network I/O (weather/news) MUST implement caching so repeated calls are fast.
- Providers MUST NOT define tool schemas.

---

## 4. Provider registration

Providers are registered centrally:

```python
aria/providers/registry.py
```

```python
PROVIDER_REGISTRY = {
  "inventory": InventoryProvider,
  "actions": ActionsProvider,
  # future: "weather", "news", "calendar", "media", ...
}
```

---

## 5. Provider configuration

Providers are enabled via Aria config:

```yaml
providers:
  inventory:
    enabled: true
    sources:
      - type: jeedom_mqtt
        enabled: true
        broker: 192.168.100.60
        topic: jeedom/discovery/#

  actions:
    enabled: true
    backends:
      jeedom_http:
        enabled: true
        base_url: http://192.168.100.60
        api_key_env: JEEDOM_API_KEY
        timeout_ms: 1500

  # future examples (placeholders, not implemented in this spec)
  weather:
    enabled: false
    cache_ttl_sec: 900

  news:
    enabled: false
    cache_ttl_sec: 300
```

Aria MUST:
- instantiate only enabled providers
- pass each provider its config subtree

---

## 6. Provider boundaries

Providers:
- MAY maintain internal state
- MAY subscribe to external systems
- MUST expose **read-only snapshots** to the rest of Aria

Providers MUST NOT:
- call the LLM
- block the main event loop
- modify global state outside their namespace

---

## 7. Inventory Provider (overview)

The Inventory provider is a **core provider** that maintains a unified inventory view.

Responsibilities:
- ingest discovery data from one or more **sources**
- normalize into a common device/command model
- maintain an incremental `inventory.json`
- build fast in-memory indexes
- expose **candidate selections** for tooling/LLM use

Non-responsibilities:
- executing actions
- triggering discovery
- defining LLM tool schemas

---

## 8. Inventory – multi-source architecture

### 8.1 Sources

A source is a sub-component inside inventory:

```python
class InventorySource:
    def start(self) -> None
    def stop(self) -> None
    def normalize(self, raw_event) -> list[DeviceRecord]
```

Examples:
- `JeedomMqttSource`
- `HomeAssistantMqttSource` (future)
- `HttpInventorySource` (future)

Each source:
- produces **normalized DeviceRecords**
- is fully isolated

---

### 8.2 Inventory core

The Inventory provider core:
- merges records from all sources
- deduplicates by `(source_id, eq_id)`
- maintains `seen_at` per source

---

## 9. Normalized data model (authoritative)

### 9.1 Device record

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

### 9.2 Command record

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

## 10. Inventory persistence

- File: `data/inventory.json`
- Atomic writes (`.tmp` + rename)
- Keep last known good backup
- Never auto-delete devices
- Mark stale when `now - seen_at > STALE_TTL`

---

## 11. Tags & capabilities

### 11.1 Device-level tags (mandatory)

Derived automatically from inventory:
- `room:<room>`
- `domain:<domain>`
- device name tokens

Device tags are used **only for scoping and grouping**.

---

### 11.2 Command-level tags (MANDATORY)

All commands MUST be tagged. Command tags are the **primary mechanism** for intent matching and down-selection.

Each command MUST have:
- exactly **one canonical capability tag**: `cap:<CAPABILITY>`
- one `type:<info|action>` tag
- one `subtype:<slider|binary|numeric|other>` tag

Canonical capabilities are **language-agnostic** and stable, e.g.:

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

Commands without a valid capability MUST be excluded from candidate outputs.

---

### 11.3 Optional LLM enrichment (offline only)

LLM-based tagging is optional and MUST:
- run only on inventory update (never hot path)
- only add **alias / synonym tags**
- never override canonical capability tags
- be cached and versioned

---

## 12. Text → tags (hot path, no LLM)

- Load FR and EN packs simultaneously
- No language detection
- Score both interpretations

Automatic dictionaries built from inventory:
- room names
- device names
- command names

Expanded with small curated synonym packs.

---

## 13. Candidate selection

Steps:
1. extract room
2. extract domain
3. extract capability
4. intersect indexes
5. rank
6. top-K

All operations must be O(1) or O(log n).

---

## 14. Tooling layer (stable LLM contract)

### 14.1 Tool schemas are centralized and stable

Providers NEVER define tool schemas. Tool schemas are defined in a single central module and remain stable.

### 14.2 Provider data exposed per request

Inventory exposes candidate data only:

```json
{
  "scope": {"room": "Salon"},
  "devices": [...],
  "commands": [...]
}
```

Other providers (weather/news) expose snapshots/candidates in their own schema, but the tooling layer decides what to include.

---

## 15. Actions / Executor Provider (LLM-driven)

Aria MUST implement a separate **Actions / Executor provider** to execute commands selected by the LLM.

### 15.1 Responsibilities
- Validate and execute action commands by `cmd_id`
- Route execution to the correct backend (Jeedom HTTP first)
- Enforce safety policies (type checks, ranges, allow/deny)
- Return structured execution results

### 15.2 Non-responsibilities
- Discovery ingestion
- Inventory normalization/tagging
- Dynamic tool schema creation

### 15.3 ExecutionSpec (authoritative)

Inventory MUST attach an `execution` spec for action commands.

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

### 15.4 Stable LLM tool contract (authoritative)

Tool schemas MUST be stable and minimal:
- `actions.execute(cmd_id: str, value?: any)`
- `actions.dry_run(cmd_id: str, value?: any)` (recommended)

---

## 16. Future providers (weather/news/etc.) – requirements

When implementing future providers:
- implement `ToolingProvider`
- provide caching with TTL
- keep network I/O out of hot path
- expose `get_snapshot()` (always)
- optionally expose `get_candidates()` for deterministic scoping (e.g. location inference)

Tooling layer will expose stable tools like:
- `weather.get(location?, when?)`
- `news.search(query, recency_days?)`

Tool schemas remain stable even if provider implementations change.

---

## 17. Concurrency & safety

- Single writer per provider for mutable state
- Immutable snapshots for readers
- Source callbacks push into a queue
- Actions provider MUST be thread-safe and apply timeouts

---

## 18. Observability

Maintain counters:
- events_received
- devices_added
- devices_updated
- devices_unchanged
- parse_errors
- executions_total
- executions_failed
- cache_hits / cache_misses (for network providers)

Expose via provider `health()`.

---

## 19. Non-goals

- No per-device tools
- No dynamic tool schemas
- No hot-path LLM calls
- Inventory provider never executes actions directly (execution is delegated)

---

## 20. Summary

This design turns Aria into a **provider-driven capability platform**.

- Inventory produces structured capabilities for home control.
- Actions executes commands selected by the LLM via stable tools.
- Future providers (weather/news/etc.) plug in without refactoring the LLM integration.

---

**This document is authoritative for Copilot.**
