# ARIA Plugin System — Tool Calling Architecture

## Overview

ARIA implements a **plugin-based tool calling system** that enables the LLM to interact with external systems (e.g., Jeedom home automation) via function calling. The plugin system consists of:

1. **Plugin Registry** — Manages plugin lifecycle and discovery
2. **Inventory Provider** — Discovers and caches available devices/commands from external systems
3. **Actions Provider** — Executes commands on devices via MQTT
4. **LLM Tools** — Exposes inventory and actions as callable functions for the LLM

This architecture decouples ARIA's core ASR/LLM/TTS pipeline from external integrations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  ARIA Server (FastAPI)                                      │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  /api/llm/process                                     │  │
│  │                                                         │  │
│  │  1. User text → LLM (Ollama)                          │  │
│  │  2. LLM decides: call tools or respond directly       │  │
│  │  3. Tool execution loop (max 5 iterations)            │  │
│  │  4. LLM generates final response                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                              ↓                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  LLM Tools (app/llm/tools.py)                         │  │
│  │                                                         │  │
│  │  • find_devices(query, room?) → candidates            │  │
│  │  • execute_command(cmd_id, value?) → result           │  │
│  └───────────────────────────────────────────────────────┘  │
│                              ↓                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Plugin Registry                                       │  │
│  │                                                         │  │
│  │  inventory_provider ← Inventory Plugin                │  │
│  │  actions_provider   ← Actions Plugin                  │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                             ↓
┌────────────────────┐                    ┌────────────────────┐
│ Inventory Provider │                    │ Actions Provider   │
│ (app/plugins/      │                    │ (app/plugins/      │
│  inventory/)       │                    │  actions/)         │
│                    │                    │                    │
│ • Jeedom MQTT      │                    │ • MQTT Publisher   │
│   source           │                    │ • Payload builder  │
│ • Device discovery │                    │ • Command executor │
│ • Command indexing │                    │                    │
│ • Caching          │                    │                    │
└────────────────────┘                    └────────────────────┘
        ↓                                             ↓
┌────────────────────────────────────────────────────────────┐
│ Jeedom MQTT Broker (192.168.100.60:1883)                  │
│                                                              │
│ • jeedom/inventory → Device inventory                      │
│ • jeedom/cmd/set/{cmd_id} → Command execution             │
└────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Plugin Registry (`app/plugins/registry.py`)

Central manager for all plugins. Provides plugin lifecycle management and dependency injection.

**Responsibilities:**
- Register plugin classes
- Initialize plugins with configuration
- Provide plugin instances to API endpoints
- Handle plugin health checks

**Example:**
```python
from app.plugins.registry import PluginRegistry
from app.plugins.inventory.provider import InventoryProvider
from app.plugins.actions.provider import ActionsProvider

registry = PluginRegistry()
registry.register_plugin_class("inventory", InventoryProvider)
registry.register_plugin_class("actions", ActionsProvider)

# Access plugins
inventory = registry.get_plugin("inventory")
actions = registry.get_plugin("actions")
```

---

### 2. Inventory Provider (`app/plugins/inventory/`)

Discovers and catalogs available devices and commands from Jeedom.

**Key Files:**
- `provider.py` — Main provider class
- `sources/jeedom.py` — Jeedom MQTT source implementation

**Data Flow:**
1. Subscribe to `jeedom/inventory` MQTT topic
2. Parse inventory messages (69 devices, 2,562 commands)
3. Build searchable index by device name, room, type
4. Cache to `ARIA_INVENTORY_CACHE_PATH` for fast startup
5. Provide search API to LLM tools

**Inventory Schema:**
```json
{
  "devices": [
    {
      "id": "123",
      "name": "Lumiere Salle a Manger New",
      "room": "RDC",
      "type": "light",
      "commands": [
        {
          "cmd_id": "5631",
          "name": "ON",
          "type": "action",
          "subtype": "other",
          "execution": {
            "protocol": "mqtt",
            "topic": "jeedom/cmd/set/5631",
            "args": {
              "payload_template": "binary"
            }
          }
        },
        {
          "cmd_id": "5630",
          "name": "Slider",
          "type": "action",
          "subtype": "slider",
          "execution": {
            "protocol": "mqtt",
            "topic": "jeedom/cmd/set/5630",
            "args": {
              "payload_template": "slider",
              "min": 0,
              "max": 99
            }
          }
        }
      ]
    }
  ]
}
```

**Search Algorithm:**
- Tokenize query (lowercase, remove accents)
- Match against device names, room names, command names
- Score by:
  - Exact match > partial match > fuzzy match
  - Device name > room > command name
- Return top N candidates (default: 5)

---

### 3. Actions Provider (`app/plugins/actions/`)

Executes commands on devices via MQTT.

**Key Files:**
- `provider.py` — Main provider class with execution logic

**Jeedom MQTT2 Protocol:**

Commands are executed by publishing to `jeedom/cmd/set/{cmd_id}` with payloads based on command type:

| Command Type | Payload Template | Example |
|--------------|------------------|---------|
| Binary (ON/OFF) | `{}` | `{}` (empty object) |
| Slider (0-99) | `{"slider": X}` | `{"slider": 50}` |
| Select | `{"select": "value"}` | `{"select": "auto"}` |
| Message | `{"title": "...", "message": "..."}` | `{"title": "Alert", "message": "Door open"}` |
| Color | `{"color": "#rrggbb"}` | `{"color": "#FF5733"}` |

**Execution Flow:**
1. Receive `execute_command(cmd_id, value?)` call
2. Lookup command in inventory
3. Build MQTT payload using `_build_payload()` method
4. Publish to `jeedom/cmd/set/{cmd_id}` with QoS 1
5. Log structured execution message
6. Return success/failure

**Error Handling:**
- Unknown cmd_id → Return error to LLM
- MQTT publish failure → Logged, return error
- Invalid value for slider → Clamp to min/max range

---

### 4. LLM Tools (`app/llm/tools.py`)

Exposes plugin functionality as OpenAI-compatible function schemas.

**Tool Definitions:**

#### `find_devices`
```python
{
  "name": "find_devices",
  "description": "Search for devices and commands that match a user request. Returns candidates with cmd_id for execution.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The user's request (e.g., 'living room light', 'lumière salon')"
      },
      "room": {
        "type": "string",
        "description": "Optional room name to filter results"
      }
    },
    "required": ["query"]
  }
}
```

**Example call:**
```json
{
  "name": "find_devices",
  "arguments": {
    "query": "lumière salle à manger"
  }
}
```

**Example response:**
```json
{
  "candidates": [
    {
      "cmd_id": "5631",
      "device_name": "Lumiere Salle a Manger New",
      "command_name": "ON",
      "room": "RDC",
      "description": "Turn on Lumiere Salle a Manger New in RDC"
    },
    {
      "cmd_id": "5632",
      "device_name": "Lumiere Salle a Manger New",
      "command_name": "OFF",
      "room": "RDC",
      "description": "Turn off Lumiere Salle a Manger New in RDC"
    },
    {
      "cmd_id": "5630",
      "device_name": "Lumiere Salle a Manger New",
      "command_name": "Slider",
      "room": "RDC",
      "description": "Set brightness for Lumiere Salle a Manger New in RDC (0-99)"
    }
  ]
}
```

#### `execute_command`
```python
{
  "name": "execute_command",
  "description": "Execute a command on a device using its cmd_id. Call find_devices first to get the cmd_id.",
  "parameters": {
    "type": "object",
    "properties": {
      "cmd_id": {
        "type": "string",
        "description": "The command ID from find_devices"
      },
      "value": {
        "type": "number",
        "description": "Optional value for slider commands (0-99)"
      }
    },
    "required": ["cmd_id"]
  }
}
```

**Example call:**
```json
{
  "name": "execute_command",
  "arguments": {
    "cmd_id": "5630",
    "value": 50
  }
}
```

**Example response:**
```json
{
  "success": true,
  "message": "Command executed successfully",
  "cmd_id": "5630",
  "value": 50
}
```

---

## LLM Integration

### Tool Calling Flow

1. **User Request:**
   ```
   POST /api/llm/process
   {
     "text": "allume la lumière du salon",
     "use_tools": true
   }
   ```

2. **LLM Response (Tool Call):**
   ```json
   {
     "tool_calls": [
       {
         "id": "call_1",
         "name": "find_devices",
         "arguments": "{\"query\": \"lumière salon\"}"
       }
     ]
   }
   ```

3. **Tool Execution:**
   Server executes `find_devices("lumière salon")` → Returns candidates

4. **LLM Response (Second Tool Call):**
   ```json
   {
     "tool_calls": [
       {
         "id": "call_2",
         "name": "execute_command",
         "arguments": "{\"cmd_id\": \"5587\"}"
       }
     ]
   }
   ```

5. **Tool Execution:**
   Server executes `execute_command("5587")` → MQTT publish

6. **LLM Final Response:**
   ```json
   {
     "content": "J'ai allumé la lumière du salon."
   }
   ```

### System Prompt

The LLM is instructed via system prompt to:
- **ALWAYS** use tools for device control
- **NEVER** claim to control devices without calling tools
- Follow the workflow: `find_devices()` → `execute_command()` → confirm
- Not rely on session history for device IDs

Example system prompt (from `start.sh`):
```
You are ARIA, a home automation assistant.

CRITICAL RULES:
1. EVERY device control request MUST use find_devices + execute_command tools
2. NEVER claim to have controlled a device without calling execute_command
3. Even if you remember device IDs from earlier in the conversation, you MUST still call find_devices and execute_command for EVERY new request
4. Session history is for context only - you MUST use tools for every action
5. If you answer without calling tools, you are lying to the user

WORKFLOW FOR EVERY REQUEST:
User: 'turn on the light' → find_devices() → execute_command() → confirm
User: 'turn it off' → find_devices() → execute_command() → confirm
User: 'set to 50%' → find_devices() → execute_command() → confirm

No shortcuts. No assumptions. Always call the tools.
```

---

## Configuration

### Environment Variables

All plugin configuration is via environment variables.

#### Plugin System
```bash
# Enable/disable plugins
export ARIA_PLUGINS_ENABLED=1
export ARIA_INVENTORY_ENABLED=1
export ARIA_ACTIONS_ENABLED=1
```

#### Inventory
```bash
# Cache path for device inventory
export ARIA_INVENTORY_CACHE_PATH=/data/aria/inventory.json
```

#### Jeedom MQTT
```bash
# MQTT broker connection
export ARIA_JEEDOM_MQTT_HOST=192.168.100.60
export ARIA_JEEDOM_MQTT_PORT=1883
export ARIA_JEEDOM_MQTT_ROOT_TOPIC=jeedom

# MQTT Quality of Service
export ARIA_MQTT_QOS_SUBSCRIBE=1  # Inventory subscription
export ARIA_MQTT_QOS_PUBLISH=1    # Command execution
```

#### LLM (for tool calling)
```bash
export ARIA_LLM_PROVIDER=ollama
export ARIA_LLM_URL=http://localhost:11434
export ARIA_LLM_MODEL=qwen2.5:3b-instruct  # Minimum for basic tool calling
# export ARIA_LLM_MODEL=qwen2.5:7b-instruct  # Recommended for reliable tool calling

export ARIA_LLM_NUM_PREDICT=1024  # Allow enough tokens for tool calls
export ARIA_LLM_TEMPERATURE=0.1   # Low temperature for deterministic tool calling
export ARIA_LLM_TOP_P=0.85

export ARIA_LLM_SYSTEM_PROMPT="<your system prompt with tool calling rules>"
```

---

## Testing

### Health Check
```bash
curl http://localhost:8000/healthz | jq
```

Expected response:
```json
{
  "status": "ok",
  "plugins": {
    "status": "healthy",
    "plugins": {
      "inventory": {
        "status": "healthy",
        "message": "69 devices, 2562 commands",
        "details": {
          "device_count": 69,
          "command_count": 2562,
          "sources": {
            "jeedom": {
              "connected": true
            }
          }
        }
      },
      "actions": {
        "status": "healthy",
        "message": "Connected to MQTT broker",
        "details": {
          "host": "192.168.100.60",
          "port": 1883,
          "connected": true
        }
      }
    }
  }
}
```

### Inventory API
```bash
# Get full snapshot
curl http://localhost:8000/api/inventory/snapshot | jq '.devices | length'
# Expected: 69

# Search for devices
curl -X POST http://localhost:8000/api/inventory/candidates \
  -H "Content-Type: application/json" \
  -d '{"query": "lumière salon"}' | jq
```

### Actions API
```bash
# Execute a command (turn on dining room light)
curl -X POST http://localhost:8000/api/actions/execute \
  -H "Content-Type: application/json" \
  -d '{"cmd_id": "5631"}' | jq
```

### LLM Tool Calling
```bash
# Test full LLM flow with tools
curl -X POST http://localhost:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "allume la lumière du salon",
    "use_tools": true
  }' | jq
```

Expected flow (check logs):
1. LLM calls `find_devices(query="lumière salon")`
2. LLM calls `execute_command(cmd_id="5587")`
3. MQTT publish to `jeedom/cmd/set/5587`
4. LLM returns "J'ai allumé la lumière du salon."

### Decision-Making Test Suite
```bash
# Run comprehensive test suite (20 test cases)
python test_decision_making.py --yes
```

Tests validate:
- Tool calling for device control (expected: 2 tools per request)
- No tools for conversation (expected: 0 tools)
- Proper intent classification

**Current baseline (3b model):** ~30% pass rate
**Target (7b model + pre-filter):** 85%+ pass rate

---

## Known Issues & Improvements

### Current Issues (as of 2026-02-01)

1. **3b Model Unreliable (20-30% success rate)**
   - Calls tools for greetings/questions
   - Incomplete tool sequences (1 instead of 2 calls)
   - **Solution:** Upgrade to qwen2.5:7b-instruct

2. **No Pre-Filter for Conversation**
   - All text goes to LLM, even "bonjour"
   - Wastes latency and causes false tool calls
   - **Solution:** Add intent classifier (see QUICK_START_CHECKLIST.md)

3. **Session History Optimization**
   - LLM sometimes skips find_devices if it "remembers" from history
   - **Solution:** Stronger system prompt (already implemented)

### Planned Improvements

1. **Phase 1: Pre-Filter Layer** (30 min)
   - Fast intent classification before LLM
   - Return instant responses for greetings/questions
   - Only send device control to LLM
   - Expected: Conversation 23% → 95%

2. **Phase 2: Upgrade to 7b Model** (1 hour)
   - Better instruction following
   - More reliable tool calling
   - Expected: Control 43% → 85%+

3. **Phase 3: Response Validation** (optional)
   - Verify tool calls match intent
   - Retry on incomplete sequences
   - Expected: Control 85% → 95%+

See [QUICK_START_CHECKLIST.md](../QUICK_START_CHECKLIST.md) for step-by-step implementation guide.

---

## Extending the Plugin System

### Adding a New Plugin

1. Create plugin directory:
   ```
   app/plugins/my_plugin/
   ├── __init__.py
   ├── provider.py
   └── config.py
   ```

2. Implement provider class:
   ```python
   # app/plugins/my_plugin/provider.py
   from app.plugins.base import BasePlugin

   class MyPlugin(BasePlugin):
       def __init__(self, config: dict):
           self.config = config

       async def initialize(self):
           # Setup connections, load data
           pass

       async def health_check(self) -> dict:
           return {
               "status": "healthy",
               "message": "Plugin is operational"
           }

       async def my_method(self, arg: str) -> dict:
           # Your plugin logic
           return {"result": arg}
   ```

3. Register in `app/main.py`:
   ```python
   from app.plugins.my_plugin.provider import MyPlugin

   app.state.plugin_registry.register_plugin_class("my_plugin", MyPlugin)
   ```

4. Add LLM tool in `app/llm/tools.py`:
   ```python
   TOOLS.append({
       "type": "function",
       "function": {
           "name": "my_function",
           "description": "...",
           "parameters": {...}
       }
   })

   async def execute_my_function(args: dict, ...):
       plugin = registry.get_plugin("my_plugin")
       return await plugin.my_method(args["arg"])
   ```

5. Update tool executor in `app/llm/tools.py`:
   ```python
   if tool_name == "my_function":
       result = await execute_my_function(tool_args, ...)
   ```

---

## Architecture Decisions

### Why Plugins?
- **Modularity:** Core ARIA (ASR/TTS/LLM) independent of integrations
- **Testability:** Each plugin can be tested in isolation
- **Extensibility:** Easy to add new sources (Zigbee, Z-Wave, HomeAssistant)
- **Configuration:** Plugin system can be fully disabled

### Why MQTT?
- **Asynchronous:** Non-blocking command execution
- **Reliable:** QoS 1 ensures delivery confirmation
- **Standard:** Jeedom MQTT2 protocol is well-documented
- **Decoupled:** ARIA doesn't need direct Jeedom API access

### Why Two-Step Tool Calling?
`find_devices()` → `execute_command()` instead of single call

**Pros:**
- Disambiguation: User says "light", system shows options
- Verification: LLM can confirm which device before executing
- Flexibility: LLM can handle partial matches

**Cons:**
- Higher latency (2 LLM calls instead of 1)
- More complex prompt engineering

**Decision:** Two-step is more robust for natural language ambiguity.

### Why Inventory Caching?
- 2,562 commands = 1.1MB JSON
- MQTT inventory can be slow to receive on startup
- Cache enables instant queries during development
- Cache invalidated on MQTT inventory update

---

## Troubleshooting

### Plugins Not Healthy

**Check logs:**
```bash
tail -f aria.log | grep "ARIA.Plugins"
```

**Common issues:**
- MQTT broker unreachable → Check `ARIA_JEEDOM_MQTT_HOST`
- Inventory not received → Check Jeedom MQTT2 plugin
- Cache file permissions → Check `ARIA_INVENTORY_CACHE_PATH`

### LLM Not Calling Tools

**Symptoms:**
- "I can't control devices" response
- No tool calls in logs

**Fixes:**
1. Verify `use_tools=true` in API request
2. Check LLM system prompt includes tool calling rules
3. Increase `ARIA_LLM_NUM_PREDICT` (min 1024)
4. Upgrade to 7b model

### MQTT Commands Not Executing

**Check:**
1. Actions provider healthy: `curl localhost:8000/healthz | jq '.plugins.plugins.actions'`
2. MQTT broker accessible: `mosquitto_pub -h 192.168.100.60 -t test -m "hello"`
3. Payload format correct: Check logs for `ARIA.Actions.Publishing`
4. Jeedom receiving: Check Jeedom MQTT plugin logs

---

## References

- [Jeedom MQTT Analysis](jeedom_mqtt_analysis.md)
- [Jeedom Command Execution](jeedom_mqtt_command_execution.md)
- [Quick Start Checklist](../QUICK_START_CHECKLIST.md)
- [Next Steps Action Plan](../NEXT_STEPS_ACTION_PLAN.md)
- [Tool Calling Decision Analysis](../TOOL_CALLING_DECISION_ANALYSIS.md)
