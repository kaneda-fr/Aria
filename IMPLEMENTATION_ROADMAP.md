# Aria Plugin System Implementation Roadmap

> **Status**: Ready to implement
> **Based on**: Jeedom MQTT analysis + Plugin system spec
> **Date**: 2026-01-30

---

## Context

We have analyzed your Jeedom MQTT discovery and event logs. The data structure is **well-suited** for implementing the Aria plugin system as specified in `aria_full_spec_with_observability.md`.

**Key Documents Created:**
1. [`docs/jeedom_mqtt_analysis.md`](docs/jeedom_mqtt_analysis.md) - Complete protocol analysis
2. [`docs/jeedom_normalized_examples.py`](docs/jeedom_normalized_examples.py) - Python data structures
3. [`docs/jeedom_capability_quick_ref.md`](docs/jeedom_capability_quick_ref.md) - Quick reference guide

---

## Current State

### ✅ Fully Implemented (Production)
- macOS client audio streaming
- Server ASR + LLM pipeline
- TTS to Sonos
- Speaker recognition
- Echo suppression v2
- Multi-language support

### ❌ Not Yet Implemented (Plugin System)
- Generic plugin lifecycle
- Tooling provider model
- Inventory provider (Jeedom MQTT source)
- Actions provider (command execution)
- Action observability (async completion tracking)

---

## Implementation Phases

### Phase 1: Plugin System Foundation (1-2 days)

**Goal**: Generic plugin infrastructure

**Tasks:**
1. Create `app/plugins/` directory structure
2. Implement `AriaPlugin` base class
   ```python
   class AriaPlugin:
       def start(self) -> None
       def stop(self) -> None
       def health(self) -> dict
       def on_config_reload(self, new_config: dict) -> None
   ```
3. Implement `ToolingProvider` interface
   ```python
   class ToolingProvider(AriaPlugin):
       name: str
       def get_snapshot(self) -> dict
       def get_candidates(self, text: str, context: dict | None) -> dict
   ```
4. Create plugin registry and loader
5. Add plugin lifecycle management to `app/main.py`

**Files to Create:**
- `app/plugins/__init__.py`
- `app/plugins/base.py`
- `app/plugins/registry.py`
- `app/plugins/tooling_provider.py`

**Acceptance Criteria:**
- [ ] Plugins can be registered and loaded
- [ ] `start()` and `stop()` lifecycle methods work
- [ ] `health()` endpoint returns plugin status
- [ ] Config hot-reload triggers `on_config_reload()`

---

### Phase 2: Inventory Provider (3-4 days)

**Goal**: Jeedom device discovery and state tracking

**Tasks:**
1. Implement `InventorySource` interface
2. Create `JeedomMQTTSource`
   - Subscribe to `jeedom/discovery/eqLogic/#`
   - Subscribe to `jeedom/cmd/event/#`
   - Parse and normalize discovery messages
   - Track state updates
3. Implement normalized data models
   - `DeviceRecord`
   - `CommandRecord`
4. Build in-memory indexes
   - By `eq_id`
   - By `cmd_id`
   - By capability tags
   - By room/location
5. Implement tag derivation logic
6. Create `inventory.json` persistence

**Files to Create:**
- `app/plugins/inventory/__init__.py`
- `app/plugins/inventory/provider.py`
- `app/plugins/inventory/sources/base.py`
- `app/plugins/inventory/sources/jeedom_mqtt.py`
- `app/plugins/inventory/models.py`
- `app/plugins/inventory/indexer.py`
- `app/plugins/inventory/normalizer.py`

**Configuration (ENV):**
```bash
ARIA_INVENTORY_ENABLED=1
ARIA_INVENTORY_SOURCES=jeedom_mqtt
ARIA_JEEDOM_MQTT_HOST=192.168.100.60
ARIA_JEEDOM_MQTT_PORT=1883
ARIA_JEEDOM_MQTT_USERNAME=
ARIA_JEEDOM_MQTT_PASSWORD=
ARIA_INVENTORY_CACHE_PATH=/var/aria/inventory.json
```

**Acceptance Criteria:**
- [ ] Discovery messages parsed and normalized
- [ ] Devices and commands indexed by capability tags
- [ ] State events update in-memory inventory
- [ ] `get_candidates(text)` returns relevant commands
- [ ] Inventory persists to disk and reloads on startup
- [ ] Health check shows device count and last update

---

### Phase 3: Actions Provider (1-2 days)

**Goal**: Command execution via Jeedom MQTT

**Tasks:**
1. Implement `ExecutionBackend` interface
2. Create `JeedomMQTTBackend`
   - Reuse existing MQTT client from inventory provider
   - Publish to `jeedom/cmd/set/{cmd_id}` with proper payload format
   - Handle payload templates (slider, color, message, select, other)
3. Implement `ActionsProvider`
   - `execute(cmd_id, value)` → ExecutionContext
   - `dry_run(cmd_id, value)` → validation
   - Create expectations for async actions
4. Expose stable LLM tools
   - `actions.execute`
   - `actions.dry_run`

**Files to Create:**
- `app/plugins/actions/__init__.py`
- `app/plugins/actions/provider.py`
- `app/plugins/actions/backends/base.py`
- `app/plugins/actions/backends/jeedom_mqtt.py`
- `app/plugins/actions/execution_context.py`
- `app/plugins/actions/llm_tools.py`

**Configuration (ENV):**
```bash
ARIA_ACTIONS_ENABLED=1
ARIA_ACTIONS_BACKEND=jeedom_mqtt
ARIA_MQTT_QOS_PUBLISH=1  # At least once delivery
ARIA_ACTIONS_DRY_RUN_MODE=0
```

**Acceptance Criteria:**
- [ ] LLM can call `actions.execute(cmd_id, value)`
- [ ] Commands execute via MQTT publish to `jeedom/cmd/set/{cmd_id}`
- [ ] Payload formatted correctly per subtype (slider, other, etc.)
- [ ] ExecutionContext created with unique ID
- [ ] `dry_run` validates without executing
- [ ] Reuses existing MQTT connection (no separate client)
- [ ] Errors handled gracefully (MQTT disconnect, invalid cmd_id)

---

### Phase 4: Action Observability (3-4 days)

**Goal**: Async completion tracking with expectations

**Tasks:**
1. Implement `Expectation` model
   - Predicates: `equals`, `within_tolerance`, `changed`
   - Observer cmd_id linkage
   - Timeout tracking
2. Create `ExpectationTracker`
   - Index expectations by observer cmd_id
   - Match incoming events against predicates
   - Resolve to success/timeout/failed
   - TTL-based cleanup
3. Integrate with `InventoryProvider`
   - Route state events to expectation tracker
4. Emit resolution events
   - `actions.execution_completed`
5. (Optional) LLM notification phrasing
   - Use LLM to phrase success/timeout messages
   - LLM input = deterministic result, not decision

**Files to Create:**
- `app/plugins/actions/expectation.py`
- `app/plugins/actions/tracker.py`
- `app/plugins/actions/resolution.py`

**Configuration (ENV):**
```bash
ARIA_OBSERVABILITY_ENABLED=1
ARIA_OBSERVABILITY_DEFAULT_TIMEOUT_MS=60000
ARIA_OBSERVABILITY_TOLERANCE_LIGHT=2.0
ARIA_OBSERVABILITY_TOLERANCE_SHUTTER=5.0
```

**Acceptance Criteria:**
- [ ] Expectations created for actions with observer cmd_id
- [ ] State events matched against active expectations
- [ ] Success resolution when predicate matches
- [ ] Timeout resolution after TTL expires
- [ ] Resolution events emitted with deterministic status
- [ ] LLM phrases user notifications (optional)

---

### Phase 5: Integration & Testing (2-3 days)

**Goal**: End-to-end plugin system validation

**Tasks:**
1. Create integration tests
   - Mock MQTT broker for discovery/events
   - Mock Jeedom HTTP API
   - Test full flow: discovery → selection → execution → observation
2. Create real-device tests
   - Test with your actual shutters (Volets Salle a Manger)
   - Test with your actual lights (Lumiere Salle a Manger New)
   - Validate timeouts and tolerances
3. Performance testing
   - Candidate selection latency (target: <10ms)
   - Discovery processing (135 devices)
   - Event ingestion throughput
4. Error resilience
   - MQTT broker disconnect/reconnect
   - Jeedom API failures
   - Malformed discovery messages
5. Documentation
   - Update README with plugin configuration
   - Add plugin system architecture diagram
   - Document capability tags and voice patterns

**Files to Create:**
- `tests/plugins/test_inventory_provider.py`
- `tests/plugins/test_actions_provider.py`
- `tests/plugins/test_jeedom_mqtt_source.py`
- `tests/plugins/test_expectation_tracker.py`
- `tests/integration/test_full_flow.py`
- `docs/plugin_architecture.md`

**Acceptance Criteria:**
- [ ] All unit tests passing
- [ ] Integration tests covering happy path + error cases
- [ ] Real-device tests successful (shutters, lights)
- [ ] Latency targets met (<10ms hot path)
- [ ] Error recovery validated
- [ ] Documentation complete

---

### Phase 6: LLM Integration (1-2 days)

**Goal**: Expose inventory and actions to LLM

**Tasks:**
1. Update LLM system prompt
   - Add plugin-awareness
   - Explain available capabilities
2. Create LLM tool definitions
   - `inventory.search(query)` → candidates
   - `actions.execute(cmd_id, value?)` → execution
   - `actions.dry_run(cmd_id, value?)` → preview
3. Implement tool result formatting
   - Human-readable messages
   - Error messages with suggestions
4. (Optional) Multi-step confirmations
   - "I found 3 lights in the living room. Turn off all?"
   - Disambiguation prompts

**Files to Update:**
- `app/llm/ollama_client.py`
- `app/main.py` (tool registration)

**Configuration (ENV):**
```bash
ARIA_LLM_TOOLS_ENABLED=1
ARIA_LLM_REQUIRE_CONFIRMATION=0
```

**Acceptance Criteria:**
- [ ] LLM can discover available devices
- [ ] LLM can execute actions via tools
- [ ] LLM receives execution results
- [ ] Errors are user-friendly
- [ ] Confirmations work (if enabled)

---

## Development Guidelines

### Code Style
- Follow existing Aria patterns
- Type hints everywhere
- Async/await for I/O operations
- Structured logging (JSON format)

### Testing
- Unit tests for pure logic
- Integration tests with mocks
- Real-device tests (marked as `@pytest.mark.integration`)
- No tests should require cloud services

### Performance Targets
- Hot path (text → candidates): <10ms
- Discovery processing (135 devices): <500ms
- Event ingestion: <5ms per event
- Execution latency: <100ms (HTTP request)

### Error Handling
- Never crash the main server
- Log errors with context
- Graceful degradation (plugin disabled if source fails)
- User-facing errors in LLM responses

---

## Environment Variables Summary

```bash
# Plugin System
ARIA_PLUGINS_ENABLED=1
ARIA_PLUGINS_CONFIG_PATH=/etc/aria/plugins.yaml

# Inventory Provider
ARIA_INVENTORY_ENABLED=1
ARIA_INVENTORY_SOURCES=jeedom_mqtt
ARIA_JEEDOM_MQTT_HOST=192.168.100.60
ARIA_JEEDOM_MQTT_PORT=1883
ARIA_JEEDOM_MQTT_USERNAME=
ARIA_JEEDOM_MQTT_PASSWORD=
ARIA_INVENTORY_CACHE_PATH=/var/aria/inventory.json

# Actions Provider
ARIA_ACTIONS_ENABLED=1
ARIA_ACTIONS_BACKEND=jeedom_mqtt
ARIA_MQTT_QOS_PUBLISH=1
ARIA_ACTIONS_DRY_RUN_MODE=0

# Observability
ARIA_OBSERVABILITY_ENABLED=1
ARIA_OBSERVABILITY_DEFAULT_TIMEOUT_MS=60000
ARIA_OBSERVABILITY_TOLERANCE_LIGHT=2.0
ARIA_OBSERVABILITY_TOLERANCE_SHUTTER=5.0

# LLM Tools
ARIA_LLM_TOOLS_ENABLED=1
ARIA_LLM_REQUIRE_CONFIRMATION=0
```

---

## Dependencies to Add

```toml
# pyproject.toml additions

[project]
dependencies = [
  # ... existing dependencies ...

  # MQTT client for Jeedom
  "paho-mqtt>=2.0.0",

  # (Optional) For advanced text normalization
  "unidecode>=1.3.0",

  # (Optional) For fuzzy matching in candidate selection
  "rapidfuzz>=3.0.0"
]
```

---

## Risk Mitigation

### Risk 1: MQTT Broker Instability
**Mitigation:**
- Auto-reconnect with exponential backoff
- Persistent inventory cache (survive restarts)
- Health check shows connection status

### Risk 2: MQTT Publish Failures
**Mitigation:**
- QoS 1 (at least once delivery)
- Connection status monitoring
- Retry logic with exponential backoff

### Risk 3: Expectation Matching False Positives
**Mitigation:**
- Require `after_ts` (ignore events before execution)
- Use strict predicates for critical actions
- Log all matches for debugging

### Risk 4: Memory Growth (Unbounded Events)
**Mitigation:**
- TTL-based cleanup (remove old expectations)
- Bounded caches (LRU eviction)
- Periodic inventory compaction

---

## Success Metrics

**Phase 1-3 (Foundation):**
- [ ] Plugin system boots without errors
- [ ] Jeedom discovery loads 135 devices
- [ ] Actions execute via HTTP API
- [ ] Inventory indexes built in <500ms

**Phase 4-5 (Observability):**
- [ ] Shutter actions resolve correctly (success/timeout)
- [ ] Light actions resolve in <5s
- [ ] Zero false positives in expectation matching
- [ ] Error recovery (MQTT disconnect) works

**Phase 6 (LLM):**
- [ ] User can control devices via voice in EN/FR
- [ ] LLM selects correct candidates (>90% accuracy)
- [ ] Execution confirmations work
- [ ] Error messages are actionable

---

## Timeline Estimate

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Foundation | 1-2 days | Day 1 | Day 2 |
| Phase 2: Inventory | 3-4 days | Day 3 | Day 6 |
| Phase 3: Actions | 1-2 days | Day 7 | Day 8 |
| Phase 4: Observability | 3-4 days | Day 9 | Day 12 |
| Phase 5: Integration | 2-3 days | Day 13 | Day 15 |
| Phase 6: LLM | 1-2 days | Day 16 | Day 17 |

**Total: 11-17 days** (2-3.5 weeks)

**Note:** Phase 3 is faster due to MQTT reuse (no separate HTTP client needed).

---

## Next Steps (Immediate)

1. **Review** this roadmap and the analysis documents
2. **Decide** on implementation approach:
   - Option A: I implement it for you (step-by-step with your approval)
   - Option B: You implement it (I provide guidance and reviews)
   - Option C: Hybrid (you start, I help when stuck)
3. **Clarify** any open questions
4. **Begin Phase 1** (Plugin System Foundation)

---

## Questions for You

1. Do you have a Jeedom API key ready for testing?
2. What's your preferred development workflow? (TDD, implement-then-test, etc.)
3. Should we implement all phases or start with MVP (Phase 1-3 only)?
4. Any specific devices you want to prioritize beyond shutters/lights?
5. Do you want multilingual support in Phase 6 or defer it?

---

**Ready to start implementation. What's your decision?**
