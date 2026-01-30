# Jeedom MQTT Discovery & Events Analysis

> **Purpose**: Technical analysis of Jeedom MQTT protocol for implementing Aria Inventory Provider
> **Data Source**: Live MQTT dumps from `jeedom/discovery/#` and `jeedom/cmd/event/#`
> **Date**: 2026-01-30

---

## Executive Summary

Jeedom publishes device discovery and state events over MQTT with a well-structured, rich schema:

- **Discovery**: Complete device + command catalog published at `jeedom/discovery/eqLogic/{eq_id}`
- **Events**: Real-time state updates published at `jeedom/cmd/event/{cmd_id}`
- **Capabilities**: Standardized `generic_type` tags (e.g., `LIGHT_SLIDER`, `FLAP_STATE`) enabling deterministic routing
- **Action/Info Separation**: Commands are either `type: "action"` (executable) or `type: "info"` (state)

This protocol is **ideal** for Aria's Inventory Provider model:
- Single-source discovery (no polling required after initial load)
- Deterministic capability mapping via `generic_type`
- Native action execution metadata (`logicalId`, `value` templates)
- Real-time state updates for observability

---

## 1. MQTT Topic Structure

### 1.1 Discovery Topics

```
jeedom/discovery/eqLogic/{eq_id}
```

**Characteristics:**
- Published once per device on Jeedom startup or device configuration change
- Contains complete device metadata + all commands
- Single JSON payload per device (can be large: 5-100KB)
- Retained messages (new subscribers get full catalog immediately)

**Example:**
```
jeedom/discovery/eqLogic/255  →  {...device + commands...}
jeedom/discovery/eqLogic/273  →  {...device + commands...}
```

### 1.2 Event Topics

```
jeedom/cmd/event/{cmd_id}
```

**Characteristics:**
- Published on every state change
- Single command per topic
- Small payloads (100-500 bytes)
- NOT retained (only current state, no history)

**Example:**
```
jeedom/cmd/event/5454  →  {"value":64,"humanName":"[RDC][Volets Salle a Manger][Etat]",...}
jeedom/cmd/event/5633  →  {"value":99,"humanName":"[RDC][Lumiere Salle a Manger New][Etat]",...}
```

---

## 2. Discovery Message Structure

### 2.1 Device (eqLogic) Schema

```json
{
  "id": "255",                          // Device ID (unique)
  "name": "Volets Salle a Manger",      // Display name
  "logicalId": "11",                    // Backend-specific ID (e.g., Z-Wave node)
  "generic_type": null,                 // Usually null at device level
  "object_id": "2",                     // Room/location ID
  "object_name": "RDC",                 // Room/location name
  "eqType_name": "zwavejs",             // Plugin/integration type
  "isVisible": "1",                     // UI visibility flag
  "isEnable": "1",                      // Enabled/disabled
  "configuration": {
    "product_name": "QNSH-001P10 - Wave Shutter",
    "manufacturer_id": 1120,
    "product_type": 3,
    "product_id": 130,
    "firmwareVersion": "12.17.0",
    "createtime": "2026-01-24 23:30:38",
    "updatetime": "2026-01-25 20:36:29",
    "real_eqType": "zwavejs"
  },
  "category": {                         // Device categories (bitmask-like)
    "opening": "1",                     // Shutters/blinds
    "light": "0",
    "heating": "0",
    "security": "0",
    "energy": "0",
    "automatism": "0"
  },
  "status": {
    "lastCommunication": "2026-01-30 10:15:20",
    "timeout": 0,
    "warning": 0,
    "danger": 0
  },
  "cmds": {                             // All commands (see section 2.2)
    "5450": {...},
    "5451": {...}
  }
}
```

### 2.2 Command (cmd) Schema

Commands are nested under `cmds` keyed by `cmd_id`.

#### Action Command Example (Shutter Positioning)

```json
{
  "id": "5450",
  "logicalId": "38-1-targetValue-#slider#",
  "generic_type": "FLAP_SLIDER",                    // ← CAPABILITY TAG
  "eqType": "zwavejs",
  "name": "Positionnement",
  "order": "1",
  "type": "action",                                 // ← ACTION (executable)
  "subType": "slider",                              // ← VALUE TYPE
  "eqLogic_id": "255",
  "isHistorized": "0",
  "unite": "",
  "configuration": {
    "minValue": "0",
    "maxValue": "99",
    "class": "38",
    "endpoint": "1",
    "property": "targetValue",
    "value": "#slider#",                            // ← EXECUTION TEMPLATE
    "logicalId": "38-1-targetValue-#slider#"
  },
  "value": "5454",                                  // ← LINKED STATE cmd_id
  "isVisible": "1"
}
```

#### Info Command Example (Shutter State)

```json
{
  "id": "5454",
  "logicalId": "38-1-currentValue",
  "generic_type": "FLAP_STATE",                     // ← CAPABILITY TAG
  "eqType": "zwavejs",
  "name": "Etat",
  "order": "8",
  "type": "info",                                   // ← INFO (read-only state)
  "subType": "numeric",                             // ← VALUE TYPE
  "eqLogic_id": "255",
  "isHistorized": "0",
  "unite": "%",
  "configuration": {
    "class": "38",
    "endpoint": "1",
    "property": "currentValue",
    "logicalId": "38-1-currentValue"
  },
  "value": "",                                      // No linked state (is the state)
  "isVisible": "0"
}
```

---

## 3. Event Message Structure

Events are published to `jeedom/cmd/event/{cmd_id}` with minimal payloads:

```json
{
  "value": 64,                                      // Current state value
  "humanName": "[RDC][Volets Salle a Manger][Etat]",
  "unite": "%",                                     // Unit (optional)
  "name": "Etat",                                   // Command name
  "type": "info",
  "subtype": "numeric"
}
```

**Key Observations:**
- `cmd_id` is in the topic, not the payload
- No `generic_type` in events (use discovery mapping)
- `humanName` format: `[Room][Device][Command]`
- Events only for `type: "info"` commands (no events for actions)

---

## 4. Capability Mapping (generic_type)

Jeedom uses standardized `generic_type` tags for capabilities. These are **language-agnostic** and enable deterministic routing.

### 4.1 Observed Capabilities

#### Lighting
| generic_type | Description | Type | SubType |
|--------------|-------------|------|---------|
| `LIGHT_SLIDER` | Dimmer/intensity control | action | slider |
| `LIGHT_STATE` | Current brightness | info | numeric |
| `LIGHT_ON` | Turn on (full brightness) | action | other |
| `LIGHT_OFF` | Turn off | action | other |
| `POWER` | Power consumption | info | numeric |
| `CONSUMPTION` | Energy consumption | info | numeric |

#### Shutters/Blinds (Flaps)
| generic_type | Description | Type | SubType |
|--------------|-------------|------|---------|
| `FLAP_SLIDER` | Position control (0-99%) | action | slider |
| `FLAP_STATE` | Current position | info | numeric |
| `FLAP_UP` | Open/raise | action | other |
| `FLAP_DOWN` | Close/lower | action | other |
| `FLAP_STOP` | Stop movement | action | other |
| `POWER` | Power consumption | info | numeric |
| `CONSUMPTION` | Energy consumption | info | numeric |

#### Generic
| generic_type | Description | Type | SubType |
|--------------|-------------|------|---------|
| `GENERIC_INFO` | Generic sensor | info | numeric/string/binary |
| `DONT` | Explicitly ignored | action/info | * |

### 4.2 Unmapped Capabilities

Some commands have `generic_type: null` or technical types:
- `0-0-pingNode`: Network diagnostics
- `0-0-healNode`: Network repair
- `0-0-nodeStatus`: Node status string
- `0-0-isFailedNode`: Node health check
- Z-Wave class-specific metadata (113-*, 50-*, etc.)

**Recommendation**: Filter out `null` and unmapped types during normalization unless explicitly required.

---

## 5. Device Categories

Devices declare categories via the `category` object:

```json
{
  "heating": "0",
  "security": "0",
  "energy": "0",
  "light": "1",          // ← This is a light
  "opening": "0",
  "automatism": "0",
  "multimedia": "0",
  "default": "0"
}
```

**Usage:**
- **Primary domain hint**: Use the `"1"` category as the device domain
- Combine with `generic_type` for precise capability tagging

**Observed Domains:**
- `light` → lighting control
- `opening` → shutters, blinds, doors, gates
- `energy` → meters, monitoring
- `heating` → thermostats, HVAC

---

## 6. Action Execution Metadata

Action commands include execution details:

### 6.1 Value Templates

Commands use template placeholders for dynamic values:

| Template | Meaning | Example |
|----------|---------|---------|
| `#slider#` | Slider value (0-99) | `"value": "#slider#"` |
| `true` | Boolean true | `"value": "true"` |
| `false` | Boolean false | `"value": "false"` |
| `99` | Fixed value (full on) | `"value": "99"` |
| `0` | Fixed value (off) | `"value": "0"` |
| `255` | "Last value" sentinel | `"value": "255"` |

### 6.2 Execution Backend Mapping

From `logicalId` patterns, we can infer backend routing:

| Pattern | Backend | Example |
|---------|---------|---------|
| `{class}-{endpoint}-{property}-{value}` | Z-Wave JS | `38-1-targetValue-#slider#` |
| `{class}.{index}.{instance}` | OpenZWave (legacy) | `1.38.0` |
| Custom | Jeedom HTTP | Direct API call with `cmd_id` |

**Recommendation**: Use `cmd_id` + Jeedom HTTP API as the primary execution backend for simplicity. MQTT command publishing is possible but requires deeper integration.

---

## 7. State-Action Linkage

Action commands often reference their corresponding state command:

```json
{
  "id": "5450",
  "name": "Positionnement",
  "type": "action",
  "subType": "slider",
  "value": "5454"        // ← Points to state cmd_id
}
```

```json
{
  "id": "5454",
  "name": "Etat",
  "type": "info",
  "subType": "numeric",
  "value": ""            // ← Is the state itself
}
```

**Usage for Observability:**
- When executing action `5450`, create expectation on observer `5454`
- Match incoming events on `jeedom/cmd/event/5454` against expectation predicate

---

## 8. Observed Devices in Sample Data

### Shutters (Volets)
- **Device**: "Volets Salle a Manger" (eq_id: 255)
- **Room**: RDC
- **Type**: Z-Wave JS shutter controller
- **Capabilities**:
  - `FLAP_SLIDER` (action/slider): Position 0-99%
  - `FLAP_STATE` (info/numeric): Current position
  - `FLAP_UP`, `FLAP_DOWN`, `FLAP_STOP` (action/other)
  - `POWER` (info/numeric): Instantaneous power (W)
  - `CONSUMPTION` (info/numeric): Cumulative energy (kWh)

### Lights (Lumiere)
- **Device**: "Lumiere Salle a Manger New" (eq_id: 273)
- **Room**: RDC
- **Type**: Z-Wave JS dimmer (Mini Dimmer)
- **Capabilities**:
  - `LIGHT_SLIDER` (action/slider): Intensity 0-99%
  - `LIGHT_STATE` (info/numeric): Current brightness
  - `LIGHT_ON`, `LIGHT_OFF` (action/other)
  - `POWER` (info/numeric): Instantaneous power (W)
  - `CONSUMPTION` (info/numeric): Cumulative energy (kWh)

- **Device**: "Lumiere Sofa" (visible in events)
- **Capabilities**:
  - Binary state (`Etat-1`, info/binary)

---

## 9. Event Patterns from Sample Data

### Shutter Movement Sequence

```
1. Power spike (movement starts)
   jeedom/cmd/event/5455 {"value":142.9, "name":"Puissance"}

2. State updates during movement
   jeedom/cmd/event/5454 {"value":64, "name":"Etat"}

3. Target reached
   jeedom/cmd/event/5454 {"value":99, "name":"Etat"}

4. Power drops to zero (motor stops)
   jeedom/cmd/event/5455 {"value":0, "name":"Puissance"}
```

**Observability Insight:**
- Shutter actions are **asynchronous** (motor takes time to move)
- Expectation should use `predicate: "equals"` matching target value
- Optional: Use power drop as secondary confirmation
- Timeout: ~30-60s for full open/close

### Light Dimming Sequence

```
1. Power change (load adjusts)
   jeedom/cmd/event/5635 {"value":11.2, "name":"Puissance"}

2. State updates
   jeedom/cmd/event/5633 {"value":99, "name":"Etat"}
   jeedom/cmd/event/5633 {"value":38, "name":"Etat"}
   jeedom/cmd/event/5633 {"value":0, "name":"Etat"}

3. Power stabilizes
   jeedom/cmd/event/5635 {"value":24.8, "name":"Puissance"}
   jeedom/cmd/event/5635 {"value":17.8, "name":"Puissance"}
```

**Observability Insight:**
- Light actions are **fast** (sub-second response typical)
- Expectation: `predicate: "equals"` for on/off, `predicate: "within_tolerance"` for dimming
- Timeout: ~3-5s

---

## 10. Normalization Requirements

### 10.1 Device Record Mapping

| Aria Field | Jeedom Source | Transformation |
|------------|---------------|----------------|
| `source` | Fixed | `"jeedom_mqtt"` |
| `eq_id` | `id` | Direct mapping |
| `name` | `name` | Direct mapping |
| `room` | `object_name` | Direct mapping |
| `domain` | `category` | Map first `"1"` → `"light"`, `"opening"`, etc. |
| `enabled` | `isEnable` | Parse `"1"` → `true` |
| `visible` | `isVisible` | Parse `"1"` → `true` |
| `commands` | `cmds` | Iterate and normalize (see 10.2) |
| `tags` | Derived | Extract from category, generic_type, eqType |
| `seen_at` | `status.lastCommunication` | Parse ISO → UNIX timestamp |

### 10.2 Command Record Mapping

| Aria Field | Jeedom Source | Transformation |
|------------|---------------|----------------|
| `cmd_id` | `id` | Direct mapping |
| `device_id` | `eqLogic_id` | Direct mapping |
| `type` | `type` | Direct mapping (`"action"` or `"info"`) |
| `subType` | `subType` | Direct mapping (`"slider"`, `"numeric"`, `"binary"`, `"other"`) |
| `capability` | `generic_type` | Direct mapping (skip if `null`) |
| `unit` | `unite` | Direct mapping |
| `range` | `configuration.minValue`, `maxValue` | Parse to tuple `(min, max)` |
| `tags` | Derived | `["cap:{generic_type}", "type:{type}", "subtype:{subType}"]` |
| `execution` | `configuration`, `logicalId`, `value` | Build ExecutionSpec (see 10.3) |

### 10.3 ExecutionSpec Construction

```python
ExecutionSpec = {
  "backend": "jeedom_mqtt",   # MQTT execution (not HTTP)
  "args": {
    "cmd_id": cmd["id"],
    "value_required": cmd["subType"] == "slider",
    "value_type": {
      "slider": "int",
      "numeric": "float",
      "binary": "bool",
      "other": None
    }[cmd["subType"]],
    "range": (
      int(cmd["configuration"].get("minValue", 0)),
      int(cmd["configuration"].get("maxValue", 99))
    ) if cmd["subType"] == "slider" else None,
    "payload_template": cmd["subType"],  # slider, color, message, select, other
    "observer_cmd_id": cmd.get("value") if cmd["type"] == "action" else None
  }
}
```

---

## 11. Implementation Recommendations

### 11.1 Inventory Source: Jeedom MQTT

```python
class JeedomMQTTSource(InventorySource):
    def start(self):
        # Subscribe to discovery and events
        self.mqtt.subscribe("jeedom/discovery/eqLogic/#")
        self.mqtt.subscribe("jeedom/cmd/event/#")

    def normalize(self, raw_event) -> list[DeviceRecord]:
        # Parse discovery payload
        # Build DeviceRecord + CommandRecords
        # Apply tagging rules
        # Return normalized list
```

### 11.2 Tag Derivation

```python
def derive_tags(device, command):
    tags = set()

    # Capability tag (mandatory for commands with generic_type)
    if command.get("generic_type"):
        tags.add(f"cap:{command['generic_type'].lower()}")

    # Type tags
    tags.add(f"type:{command['type']}")
    tags.add(f"subtype:{command['subType']}")

    # Domain tags from device category
    for cat, enabled in device["category"].items():
        if enabled == "1":
            tags.add(f"domain:{cat}")

    # Backend tag
    tags.add(f"backend:{device['eqType_name']}")

    return tags
```

### 11.3 Event Ingestion

```python
def on_mqtt_event(self, topic, payload):
    # Extract cmd_id from topic
    cmd_id = topic.split('/')[-1]

    # Parse payload
    event = json.loads(payload)

    # Lookup command in inventory by cmd_id
    command = self.inventory.get_command(cmd_id)
    if not command:
        return  # Unknown command

    # Normalize event
    normalized_event = {
        "cmd_id": cmd_id,
        "value": event["value"],
        "unit": event.get("unite"),
        "ts": int(time.time())
    }

    # Update inventory state
    self.inventory.update_command_state(cmd_id, normalized_event)

    # Match against expectations
    self.expectation_tracker.match(normalized_event)
```

### 11.4 Execution Backend: Jeedom MQTT

```python
def execute_command(self, cmd_id, value=None):
    # Get execution spec for payload formatting
    execution_spec = self.get_execution_spec(cmd_id)
    subtype = execution_spec["args"]["payload_template"]

    # Build MQTT topic
    topic = f"{self.root_topic}/cmd/set/{cmd_id}"

    # Build payload based on subtype
    if subtype == "slider":
        payload = json.dumps({"slider": int(value)})
    elif subtype == "color":
        payload = json.dumps({"color": str(value)})
    elif subtype == "message":
        payload = json.dumps(value)  # {title, message}
    elif subtype == "select":
        payload = json.dumps({"select": int(value)})
    else:  # "other" or default
        payload = ""

    # Publish to MQTT (async-friendly)
    self.mqtt_client.publish(topic, payload, qos=1)

    # Create execution context + expectation
    # ...
```

---

## 12. Open Questions & Future Work

### 12.1 Discovery Refresh
- **Q**: How often does Jeedom republish discovery?
- **A**: On device add/remove/config change. Not periodic.
- **Impl**: Store discovery in persistent cache, reload on startup.

### 12.2 MQTT Command Execution
- **Q**: Can we execute actions via MQTT publish (vs HTTP)?
- **A**: YES - via `jeedom/cmd/set/{cmd_id}` with payload format per subtype.
- **Impl**: MQTT-only execution (see `docs/jeedom_mqtt_command_execution.md`).

### 12.3 Multi-Endpoint Devices
- **Q**: How to handle devices with multiple endpoints (e.g., multi-relay)?
- **A**: Each endpoint becomes a separate command. Group by `eqLogic_id`.
- **Impl**: Inventory should support multiple commands per device naturally.

### 12.4 Scenes/Scenarios
- **Q**: Does Jeedom expose scenes via discovery?
- **A**: Not observed in sample data. Scenes may be separate plugin.
- **Impl**: Defer to future provider or manual configuration.

---

## 13. Summary & Next Steps

### Key Findings
✅ Jeedom MQTT protocol is **well-suited** for Aria Inventory Provider
✅ Discovery provides **complete device catalog** in single messages
✅ `generic_type` enables **deterministic capability routing**
✅ State-action linkage supports **robust observability**
✅ Event stream provides **real-time state updates** for expectations

### Implementation Path
1. **Implement JeedomMQTTSource**
   - Subscribe to discovery + events
   - Normalize to DeviceRecord/CommandRecord
   - Build in-memory indexes

2. **Implement Jeedom HTTP Executor**
   - Execute actions via Jeedom API
   - Create ExecutionContext + Expectations
   - Link observer cmd_id for state tracking

3. **Integrate with Actions Provider**
   - Expose stable LLM tools (`actions.execute`, `actions.dry_run`)
   - Route to Jeedom backend via ExecutionSpec

4. **Build Expectation Tracker**
   - Match incoming MQTT events against active expectations
   - Resolve success/timeout/failure deterministically
   - Emit resolution events for LLM notification

5. **Test with Real Devices**
   - Volets (shutters): Async movement, power monitoring
   - Lumiere (lights): Fast dimming, binary on/off
   - Validate timeout values, tolerance thresholds

---

**This analysis is ready for implementation.**
