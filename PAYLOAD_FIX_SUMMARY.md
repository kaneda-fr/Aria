# Jeedom MQTT2 Payload Format Fix

**Date:** 2026-01-31
**Issue:** Incorrect payload format for slider commands
**Status:** ✅ FIXED

---

## Problem

The initial implementation used `{"value": X}` for all command types, but the Jeedom MQTT2 protocol requires different payload formats based on command subtype.

---

## Correct Payload Formats (Jeedom MQTT2 Protocol)

Based on: [docs/jeedom_mqtt_command_execution.md](docs/jeedom_mqtt_command_execution.md)

### 1. Slider Commands (Dimmers, Shutters)
**SubType:** `slider`
**Payload Template:** `slider`
**Format:** `{"slider": <value>}`

**Examples:**
```bash
# Set dimmer to 50%
Topic: jeedom/cmd/set/5630
Payload: {"slider": 50}

# Set shutter position to 75%
Topic: jeedom/cmd/set/5450
Payload: {"slider": 75}
```

### 2. Binary/Default Commands (On/Off, Up/Down, Stop)
**SubType:** `other`
**Payload Template:** `other`
**Format:** `{}` (empty JSON) or `""` (empty string)

**Examples:**
```bash
# Turn light ON
Topic: jeedom/cmd/set/5631
Payload: {}

# Turn light OFF
Topic: jeedom/cmd/set/5632
Payload: {}

# Open shutter
Topic: jeedom/cmd/set/5451
Payload: {}
```

### 3. Select Commands
**SubType:** `select`
**Payload Template:** `select`
**Format:** `{"select": <value>}`

**Example:**
```bash
# Select option 2
Topic: jeedom/cmd/set/1234
Payload: {"select": 2}
```

### 4. Message Commands
**SubType:** `message`
**Payload Template:** `message`
**Format:** `{"title": "...", "message": "..."}`

**Example:**
```bash
# Send notification
Topic: jeedom/cmd/set/5678
Payload: {"title": "Alert", "message": "Motion detected"}
```

### 5. Color Commands
**SubType:** `color`
**Payload Template:** `color`
**Format:** `{"color": "#rrggbb"}`

**Example:**
```bash
# Set RGB color to red
Topic: jeedom/cmd/set/9012
Payload: {"color": "#ff0000"}
```

---

## Files Modified

### 1. `/app/plugins/actions/provider.py`

#### Changed: `_build_payload()` method (lines 252-270)

**Before:**
```python
def _build_payload(self, cmd_id, value, execution_spec) -> str:
    if value is not None:
        return json.dumps({"value": value})
    else:
        return json.dumps({})
```

**After:**
```python
def _build_payload(self, cmd_id, value, execution_spec) -> str:
    # Get payload template from execution spec
    payload_template = None
    if execution_spec:
        args = execution_spec.get("args", {})
        payload_template = args.get("payload_template")

    # Build payload based on template
    if payload_template == "slider":
        if value is not None:
            return json.dumps({"slider": int(value)})
        else:
            return json.dumps({})

    elif payload_template == "select":
        if value is not None:
            return json.dumps({"select": value})
        else:
            return json.dumps({})

    # ... (handles message, color, other, etc.)
```

#### Changed: MQTT topic pattern (line 152)

**Before:**
```python
topic = f"{self.root_topic}/cmd/action/{cmd_id}"
```

**After:**
```python
topic = f"{self.root_topic}/cmd/set/{cmd_id}"
```

### 2. `test_mqtt_command.py`

Updated test payloads to use correct formats:
- ON/OFF commands: `{}` (empty)
- Dimmer command: `{"slider": 50}`

---

## Updated Command Reference

### Lumiere Salle a Manger New (Dining Room Light)

| Action | cmd_id | Capability | Topic | Payload | Example |
|--------|--------|------------|-------|---------|---------|
| **Turn ON** | 5631 | light_on | `jeedom/cmd/set/5631` | `{}` | `mosquitto_pub -t "jeedom/cmd/set/5631" -m '{}'` |
| **Turn OFF** | 5632 | light_off | `jeedom/cmd/set/5632` | `{}` | `mosquitto_pub -t "jeedom/cmd/set/5632" -m '{}'` |
| **Set to 50%** | 5630 | light_slider | `jeedom/cmd/set/5630` | `{"slider": 50}` | `mosquitto_pub -t "jeedom/cmd/set/5630" -m '{"slider":50}'` |
| **Set to 100%** | 5630 | light_slider | `jeedom/cmd/set/5630` | `{"slider": 99}` | `mosquitto_pub -t "jeedom/cmd/set/5630" -m '{"slider":99}'` |
| **Set to 0%** | 5630 | light_slider | `jeedom/cmd/set/5630` | `{"slider": 0}` | `mosquitto_pub -t "jeedom/cmd/set/5630" -m '{"slider":0}'` |

---

## Testing

### Test with mosquitto_pub (from server)

```bash
# Connect to your server
ssh user@192.168.100.100

# Turn ON dining room light
mosquitto_pub -h 192.168.100.60 -t "jeedom/cmd/set/5631" -m '{}'

# Set brightness to 50%
mosquitto_pub -h 192.168.100.60 -t "jeedom/cmd/set/5630" -m '{"slider":50}'

# Turn OFF
mosquitto_pub -h 192.168.100.60 -t "jeedom/cmd/set/5632" -m '{}'
```

### Test with Python script

```bash
# On your server (192.168.100.100)
cd /path/to/aria

# Edit test_mqtt_command.py and uncomment the test line
# Then run:
python3 test_mqtt_command.py
```

### Test with ARIA API

```bash
# Start ARIA with plugins enabled
bash start.sh

# Test LLM tool calling
curl -X POST http://192.168.100.100:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "mets la lumière de la salle à manger à 75%",
    "use_tools": true
  }'
```

Expected LLM behavior:
1. Calls `find_devices(query="salle a manger light")`
2. Gets `cmd_id: 5630` (light_slider capability)
3. Calls `execute_command(cmd_id="5630", value=75)`
4. Actions provider builds payload: `{"slider": 75}`
5. Publishes to: `jeedom/cmd/set/5630`

---

## Verification Checklist

- [x] `_build_payload()` method updated to handle all payload templates
- [x] MQTT topic changed from `cmd/action/` to `cmd/set/`
- [x] Test script updated with correct payloads
- [x] Documentation updated
- [ ] Tested on server with real Jeedom MQTT broker
- [ ] Verified slider commands work (0-99%)
- [ ] Verified binary commands work (on/off)

---

## Impact

This fix ensures that:
✅ Dimmer/brightness commands use `{"slider": X}` format
✅ Binary on/off commands use empty payload `{}`
✅ Other command types (select, message, color) use correct formats
✅ Full compatibility with Jeedom MQTT2 protocol

---

## References

- **Jeedom MQTT2 Docs**: https://doc.jeedom.com/en_US/plugins/programming/mqtt2/
- **Local Spec**: [docs/jeedom_mqtt_command_execution.md](docs/jeedom_mqtt_command_execution.md)
- **Execution Spec**: See `inventory.json` → devices → commands → execution.args.payload_template
