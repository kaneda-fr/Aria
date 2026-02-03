# Jeedom MQTT Command Execution

> **Authoritative specification for MQTT-based command execution**
> Based on: https://doc.jeedom.com/en_US/plugins/programming/mqtt2/
> Replaces HTTP execution backend

---

## Overview

All command execution with Jeedom will use **MQTT publish/subscribe**, not HTTP API calls.

**Benefits:**
- Single MQTT connection for discovery, events, and execution
- Lower latency (no HTTP overhead)
- Native pub/sub for async observability
- Consistent with existing discovery/event pattern

---

## Command Execution Topic Structure

### Root Topic
Default: `jeedom` (configurable in Jeedom MQTT2 plugin)

### Execution Topics

| Purpose | Topic Pattern | Direction |
|---------|---------------|-----------|
| Execute command | `jeedom/cmd/set/{cmd_id}` | Aria → Jeedom |
| Request value | `jeedom/cmd/get/{cmd_id}` | Aria → Jeedom |
| Value response | `jeedom/cmd/value/{cmd_id}` | Jeedom → Aria |
| Event broadcast | `jeedom/cmd/event/{cmd_id}` | Jeedom → Aria |
| Discovery | `jeedom/discovery/eqLogic/{eq_id}` | Jeedom → Aria |

---

## Execution Payload Formats

### 1. Slider/Cursor Commands (Dimmers, Shutters)

**Command Type:** `subType: "slider"`
**Generic Types:** `LIGHT_SLIDER`, `FLAP_SLIDER`

**Topic:**
```
jeedom/cmd/set/5450
```

**Payload:**
```json
{
  "slider": 50
}
```

**Examples:**
```bash
# Set light to 75%
mosquitto_pub -t "jeedom/cmd/set/5630" -m '{"slider":75}'

# Set shutter to 0% (closed)
mosquitto_pub -t "jeedom/cmd/set/5450" -m '{"slider":0}'

# Set shutter to 99% (open)
mosquitto_pub -t "jeedom/cmd/set/5450" -m '{"slider":99}'
```

---

### 2. Binary/Default Commands (On/Off, Up/Down)

**Command Type:** `subType: "other"`
**Generic Types:** `LIGHT_ON`, `LIGHT_OFF`, `FLAP_UP`, `FLAP_DOWN`, `FLAP_STOP`

**Topic:**
```
jeedom/cmd/set/{cmd_id}
```

**Payload:**
```
(empty message)
```
or
```json
{}
```

**Examples:**
```bash
# Turn light on
mosquitto_pub -t "jeedom/cmd/set/5631" -m ''

# Open shutter
mosquitto_pub -t "jeedom/cmd/set/5451" -m ''

# Close shutter
mosquitto_pub -t "jeedom/cmd/set/5453" -m ''

# Stop shutter
mosquitto_pub -t "jeedom/cmd/set/5452" -m ''
```

---

### 3. Color Commands

**Command Type:** `subType: "color"`
**Generic Types:** `LIGHT_SET_COLOR`

**Payload:**
```json
{
  "color": "#96c927"
}
```

**Example:**
```bash
# Set RGB light color
mosquitto_pub -t "jeedom/cmd/set/1234" -m '{"color":"#ff0000"}'
```

---

### 4. Message Commands

**Command Type:** `subType: "message"`

**Payload:**
```json
{
  "title": "Alert",
  "message": "Motion detected"
}
```

---

### 5. Select/List Commands

**Command Type:** `subType: "select"`

**Payload:**
```json
{
  "select": 1
}
```

---

### 6. Info Commands (State Updates)

**Command Type:** `type: "info"`

**Note:** Info commands are typically **read-only** via events. However, you can manually update their state via MQTT:

**Payload:**
```json
{
  "value": "new_value",
  "datetime": "2026-01-30 10:15:00"
}
```

`datetime` is optional (defaults to now).

---

## Value Request Pattern (Polling)

If you need to explicitly request the current value of a command (vs. waiting for an event):

**Request Topic:**
```
jeedom/cmd/get/{cmd_id}
```

**Request Payload:**
```
(empty)
```

**Response Topic:**
```
jeedom/cmd/value/{cmd_id}
```

**Response Payload:**
```
{current_value}
```

**Example:**
```bash
# Request shutter position
mosquitto_pub -t "jeedom/cmd/get/5454" -m ''

# Listen for response
mosquitto_sub -t "jeedom/cmd/value/5454"
# → 64  (current position)
```

**Usage in Aria:**
- Subscribe to `jeedom/cmd/value/{cmd_id}` before publishing to `jeedom/cmd/get/{cmd_id}`
- Use for initial state queries (if needed)
- **Not required** if using event subscriptions (`jeedom/cmd/event/#`)

---

## Event Subscription (Recommended for Observability)

**Already implemented in your logs!**

**Topic:**
```
jeedom/cmd/event/{cmd_id}
```

**Payload:**
```json
{
  "value": 64,
  "humanName": "[RDC][Volets Salle a Manger][Etat]",
  "unite": "%",
  "name": "Etat",
  "type": "info",
  "subtype": "numeric"
}
```

**Subscription Pattern:**
```bash
# Subscribe to all events
mosquitto_sub -t "jeedom/cmd/event/#"

# Subscribe to specific command
mosquitto_sub -t "jeedom/cmd/event/5454"
```

**Aria Usage:**
- Already subscribed via `JeedomMQTTSource`
- Events drive observability (expectation matching)

---

## Authentication

**Required for local mode** (recommended for security).

**MQTT Broker Configuration:**
```bash
ARIA_JEEDOM_MQTT_HOST=192.168.100.60
ARIA_JEEDOM_MQTT_PORT=1883
ARIA_JEEDOM_MQTT_USERNAME=aria_client
ARIA_JEEDOM_MQTT_PASSWORD=<secure_password>
```

**Paho MQTT Client:**
```python
client = mqtt.Client()
client.username_pw_set(username, password)
client.connect(host, port)
```

---

## Execution Backend Implementation

### Revised ExecutionSpec

```python
ExecutionSpec = {
  "backend": "jeedom_mqtt",  # Changed from jeedom_http
  "args": {
    "cmd_id": "5450",
    "value_required": True,
    "value_type": "int",
    "range": (0, 99),
    "payload_template": "slider",  # New field
    "observer_cmd_id": "5454"
  }
}
```

### Payload Template Mapping

| Command SubType | Payload Template | Payload Format |
|-----------------|------------------|----------------|
| `slider` | `slider` | `{"slider": value}` |
| `color` | `color` | `{"color": value}` |
| `message` | `message` | `{"title": t, "message": m}` |
| `select` | `select` | `{"select": value}` |
| `other` | `default` | `{}` or empty |

---

## Python Implementation

### JeedomMQTTBackend

```python
import paho.mqtt.client as mqtt
import json
from typing import Any, Optional

class JeedomMQTTBackend:
    def __init__(self, host: str, port: int, username: str, password: str, root_topic: str = "jeedom"):
        self.host = host
        self.port = port
        self.root_topic = root_topic

        self.client = mqtt.Client()
        self.client.username_pw_set(username, password)
        self.client.connect(host, port)
        self.client.loop_start()

    def execute(self, cmd_id: str, value: Optional[Any] = None, subtype: str = "other") -> None:
        """Execute a command via MQTT."""
        topic = f"{self.root_topic}/cmd/set/{cmd_id}"

        # Build payload based on subtype
        if subtype == "slider":
            if value is None:
                raise ValueError("Slider commands require a value")
            payload = json.dumps({"slider": int(value)})

        elif subtype == "color":
            if value is None:
                raise ValueError("Color commands require a value")
            payload = json.dumps({"color": str(value)})

        elif subtype == "message":
            if not isinstance(value, dict) or "message" not in value:
                raise ValueError("Message commands require {title, message} dict")
            payload = json.dumps(value)

        elif subtype == "select":
            if value is None:
                raise ValueError("Select commands require a value")
            payload = json.dumps({"select": int(value)})

        else:  # "other" or default
            payload = ""

        # Publish
        self.client.publish(topic, payload, qos=1)

    def request_value(self, cmd_id: str) -> None:
        """Request current value (response via jeedom/cmd/value/{cmd_id})."""
        topic = f"{self.root_topic}/cmd/get/{cmd_id}"
        self.client.publish(topic, "", qos=1)

    def stop(self):
        """Clean shutdown."""
        self.client.loop_stop()
        self.client.disconnect()
```

### Example Usage

```python
# Initialize backend
backend = JeedomMQTTBackend(
    host="192.168.100.60",
    port=1883,
    username="aria_client",
    password="secure_password"
)

# Execute slider command (set light to 50%)
backend.execute(cmd_id="5630", value=50, subtype="slider")

# Execute binary command (turn light on)
backend.execute(cmd_id="5631", subtype="other")

# Execute shutter down
backend.execute(cmd_id="5453", subtype="other")

# Request current value
backend.request_value(cmd_id="5454")
```

---

## Observability Flow (MQTT-Based)

### 1. Execute Action
```python
# Aria publishes action
topic: jeedom/cmd/set/5450
payload: {"slider": 0}
```

### 2. Create Expectation
```python
expectation = {
    "execution_id": "exec-123",
    "observer_cmd_id": "5454",
    "predicate": "equals",
    "target": 0,
    "timeout_ms": 60000
}
```

### 3. Wait for Event
```python
# Jeedom publishes state change
topic: jeedom/cmd/event/5454
payload: {"value": 0, "name": "Etat", ...}
```

### 4. Match Expectation
```python
# ExpectationTracker matches event to expectation
if event.cmd_id == "5454" and event.value == 0:
    resolve_execution(exec_id="exec-123", status="success")
```

---

## Updated Configuration

```bash
# MQTT Connection (single client for all operations)
ARIA_JEEDOM_MQTT_HOST=192.168.100.60
ARIA_JEEDOM_MQTT_PORT=1883
ARIA_JEEDOM_MQTT_USERNAME=aria_client
ARIA_JEEDOM_MQTT_PASSWORD=<password>
ARIA_JEEDOM_MQTT_ROOT_TOPIC=jeedom

# Execution Backend
ARIA_ACTIONS_BACKEND=jeedom_mqtt  # Changed from jeedom_http

# QoS Settings
ARIA_MQTT_QOS_PUBLISH=1  # At least once delivery for commands
ARIA_MQTT_QOS_SUBSCRIBE=1  # At least once delivery for events
```

---

## Comparison: MQTT vs HTTP

| Aspect | MQTT | HTTP (Previous) |
|--------|------|-----------------|
| Latency | ~5-10ms | ~20-50ms |
| Connection | Persistent | Per-request |
| Observability | Native (events) | Polling required |
| Authentication | Username/password | API key |
| Complexity | Single client | Separate HTTP client |
| Real-time | Yes | No |
| Bandwidth | Low | Higher |

**Verdict:** MQTT is superior for Aria's real-time use case.

---

## Testing Commands

### Test Light Dimming
```bash
# Subscribe to events
mosquitto_sub -h 192.168.100.60 -p 1883 -u aria_client -P password -t "jeedom/cmd/event/5633"

# Execute dimming
mosquitto_pub -h 192.168.100.60 -p 1883 -u aria_client -P password -t "jeedom/cmd/set/5630" -m '{"slider":75}'

# Observe event
# → {"value":75,"humanName":"[RDC][Lumiere Salle a Manger New][Etat]",...}
```

### Test Shutter Control
```bash
# Subscribe to state
mosquitto_sub -h 192.168.100.60 -p 1883 -u aria_client -P password -t "jeedom/cmd/event/5454"

# Close shutter
mosquitto_pub -h 192.168.100.60 -p 1883 -u aria_client -P password -t "jeedom/cmd/set/5453" -m ''

# Observe events during movement
# → {"value":64,...}
# → {"value":32,...}
# → {"value":0,...}
```

---

## Implementation Updates Required

### Files to Update

1. **`docs/jeedom_mqtt_analysis.md`**
   - Section 8.4: Change execution backend from HTTP to MQTT
   - Update payload examples

2. **`docs/jeedom_normalized_examples.py`**
   - Update `ExecutionSpec` to use `jeedom_mqtt` backend
   - Add `payload_template` field

3. **`IMPLEMENTATION_ROADMAP.md`**
   - Phase 3: Change "JeedomHTTPBackend" to "JeedomMQTTBackend"
   - Remove HTTP dependencies
   - Update env vars

### New Implementation

**Phase 3 (Actions Provider) becomes simpler:**
- No separate HTTP client needed
- Reuse existing MQTT connection from inventory provider
- Single connection handles discovery + events + execution

---

## Summary

✅ **All Jeedom operations via MQTT:**
- Discovery: `jeedom/discovery/eqLogic/#`
- Events: `jeedom/cmd/event/#`
- Execution: `jeedom/cmd/set/{cmd_id}`
- Value requests: `jeedom/cmd/get/{cmd_id}` (optional)

✅ **Simpler architecture:**
- One MQTT client (vs. MQTT + HTTP)
- Consistent pub/sub pattern
- Lower latency

✅ **Better observability:**
- Native event-driven confirmation
- No polling needed

---

**Ready to update implementation plan with MQTT-only execution.**
