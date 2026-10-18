# ARIA Plugin System — Documentation Summary

**Date:** 2026-02-01
**Branch:** `feature/plugin-system`
**Status:** Documentation Complete

---

## Overview

This document summarizes the comprehensive documentation created for ARIA's plugin-based tool calling system.

The plugin system enables ARIA's LLM to interact with external home automation systems (currently Jeedom via MQTT) through OpenAI-compatible function calling.

---

## Documentation Files Created/Updated

### 1. **`docs/PLUGIN_SYSTEM.md`** (NEW)
Comprehensive architecture documentation covering:
- System architecture and component overview
- Plugin Registry, Inventory Provider, Actions Provider
- LLM tool definitions (`find_devices`, `execute_command`)
- Jeedom MQTT2 protocol payload formats
- Tool calling flow and integration
- Configuration reference
- Testing procedures
- Known issues and improvements
- Extension guide for new plugins
- Architecture decisions and rationale
- Troubleshooting guide

**Size:** ~1,000 lines
**Audience:** Developers implementing or extending the plugin system

---

### 2. **`docs/ENVIRONMENT.md`** (UPDATED)
Added new sections:
- **Plugin System** — Enable/disable flags
- **Inventory Provider** — Cache configuration
- **Jeedom MQTT Integration** — Broker connection, QoS settings
- **LLM Integration** — Updated with tool calling recommendations

**New variables documented:**
```bash
ARIA_PLUGINS_ENABLED
ARIA_INVENTORY_ENABLED
ARIA_ACTIONS_ENABLED
ARIA_INVENTORY_CACHE_PATH
ARIA_INVENTORY_REFRESH_INTERVAL_SEC
ARIA_JEEDOM_MQTT_HOST
ARIA_JEEDOM_MQTT_PORT
ARIA_JEEDOM_MQTT_ROOT_TOPIC
ARIA_MQTT_QOS_SUBSCRIBE
ARIA_MQTT_QOS_PUBLISH
```

**Audience:** Users configuring ARIA

---

### 3. **`.copilot-instructions.md`** (UPDATED)
Added **PART 4 — Plugin System** quick reference:
- Architecture diagram
- Plugin component overview
- Tool calling flow example
- Required environment variables
- Jeedom MQTT2 payload formats
- System prompt requirements
- Testing commands
- Link to full documentation

**Audience:** AI assistants (Copilot/Claude) building or modifying ARIA

---

### 4. **`README.md`** (UPDATED)
Added/Updated:
- **High-level architecture** diagram — Added plugin system components
- **Configuration** section — Added link to PLUGIN_SYSTEM.md
- **Plugin System (Tool Calling)** section (NEW) — Overview, example, quickstart
- **Roadmap** — Updated with plugin improvements

**Audience:** General users and contributors

---

## Architecture Summary

### Components

```
ARIA Server (FastAPI)
    └─ /api/llm/process
        └─ LLM Tools (app/llm/tools.py)
            ├─ find_devices(query, room?)
            └─ execute_command(cmd_id, value?)
                └─ Plugin Registry
                    ├─ Inventory Provider
                    │   └─ Jeedom MQTT Source
                    │       └─ jeedom/inventory (subscribe)
                    └─ Actions Provider
                        └─ MQTT Publisher
                            └─ jeedom/cmd/set/{cmd_id} (publish)
```

### Data Flow

1. **User Request:** "allume la lumière du salon"
2. **LLM Call 1:** `find_devices(query="lumière salon")`
3. **Inventory Search:** Returns candidates with `cmd_id`
4. **LLM Call 2:** `execute_command(cmd_id="5587")`
5. **MQTT Publish:** `jeedom/cmd/set/5587` ← `{}`
6. **LLM Response:** "J'ai allumé la lumière du salon."

### Key Technologies

- **Protocol:** Jeedom MQTT2 (QoS 1)
- **LLM:** Ollama (qwen2.5:7b-instruct recommended)
- **Tool Format:** OpenAI function calling
- **Inventory:** Cached JSON (69 devices, 2,562 commands)

---

## Configuration Quick Reference

### Minimal Setup

```bash
# Enable plugins
export ARIA_PLUGINS_ENABLED=1
export ARIA_INVENTORY_ENABLED=1
export ARIA_ACTIONS_ENABLED=1

# MQTT connection
export ARIA_JEEDOM_MQTT_HOST=192.168.100.60
export ARIA_JEEDOM_MQTT_PORT=1883
export ARIA_JEEDOM_MQTT_ROOT_TOPIC=jeedom

# LLM for tool calling
export ARIA_LLM_PROVIDER=ollama
export ARIA_LLM_URL=http://localhost:11434
export ARIA_LLM_MODEL=qwen2.5:7b-instruct
export ARIA_LLM_NUM_PREDICT=1024
export ARIA_LLM_TEMPERATURE=0.1
```

### Mac Development

See [`start-osx.sh`](../start-osx.sh) for complete Mac development configuration.

---

## Testing

### Health Check
```bash
curl http://localhost:8000/healthz | jq '.plugins'
```

### Inventory Query
```bash
curl -X POST http://localhost:8000/api/inventory/candidates \
  -H "Content-Type: application/json" \
  -d '{"query": "lumière salon"}' | jq
```

### Tool Calling
```bash
curl -X POST http://localhost:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "allume la lumière", "use_tools": true}' | jq
```

### Decision-Making Test Suite
```bash
python test_decision_making.py --yes
```

**Current baseline (qwen2.5:3b-instruct):** 30% pass rate
**Target (qwen2.5:7b-instruct):** 85%+ pass rate

---

## Known Issues & Roadmap

### Current Issues (as of 2026-02-01)

1. **3b Model Unreliable**
   - Calls tools for greetings/questions (should be 0 tools)
   - Incomplete tool sequences (1 instead of 2 calls)
   - **Fix:** Upgrade to qwen2.5:7b-instruct

2. **No Pre-Filter**
   - All text sent to LLM (wastes latency)
   - **Fix:** Add intent classifier (Phase 1 in QUICK_START_CHECKLIST.md)

### Improvement Plan

See comprehensive guides:
- **[QUICK_START_CHECKLIST.md](../QUICK_START_CHECKLIST.md)** — 2-hour implementation guide (20% → 85%+ success)
- **[NEXT_STEPS_ACTION_PLAN.md](../NEXT_STEPS_ACTION_PLAN.md)** — Detailed analysis and 4-phase plan
- **[TOOL_CALLING_DECISION_ANALYSIS.md](../TOOL_CALLING_DECISION_ANALYSIS.md)** — LLM decision-making analysis

---

## Files Structure

```
aria/
├── README.md                              # Updated: Added plugin system section
├── .copilot-instructions.md              # Updated: Added Part 4 plugin reference
├── start-osx.sh                           # NEW: Mac development configuration
├── docs/
│   ├── PLUGIN_SYSTEM.md                  # NEW: Complete architecture documentation
│   ├── ENVIRONMENT.md                    # Updated: Added plugin variables
│   ├── jeedom_mqtt_analysis.md          # Existing: Jeedom protocol analysis
│   ├── jeedom_mqtt_command_execution.md # Existing: Command execution guide
│   └── ...
├── app/
│   ├── plugins/
│   │   ├── registry.py                   # Plugin lifecycle management
│   │   ├── inventory/
│   │   │   ├── provider.py              # Device discovery
│   │   │   └── sources/jeedom.py        # MQTT source
│   │   └── actions/
│   │       └── provider.py              # Command execution
│   ├── llm/
│   │   ├── tools.py                     # LLM function schemas
│   │   └── ollama_client.py             # Tool calling support
│   └── main.py                          # FastAPI app with plugin integration
├── test_decision_making.py              # Tool calling test suite
├── QUICK_START_CHECKLIST.md             # 2-hour fix guide
├── NEXT_STEPS_ACTION_PLAN.md            # Comprehensive improvement plan
└── TOOL_CALLING_DECISION_ANALYSIS.md    # Decision-making analysis
```

---

## For Future Contributors

### Building As-Is

To recreate this exact system:

1. **Read documentation:**
   - [README.md](../README.md) — Project overview
   - [docs/PLUGIN_SYSTEM.md](PLUGIN_SYSTEM.md) — Plugin architecture
   - [docs/ENVIRONMENT.md](ENVIRONMENT.md) — Configuration reference
   - [.copilot-instructions.md](../.copilot-instructions.md) — Build specifications

2. **Setup environment:**
   ```bash
   # Python 3.11-3.12 (NOT 3.13 due to audioop)
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Configure:**
   ```bash
   cp start.sh.example start.sh
   # Edit start.sh with your configuration
   # See docs/ENVIRONMENT.md for all variables
   ```

4. **Install LLM:**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull recommended model
   ollama pull qwen2.5:7b-instruct
   ```

5. **Run:**
   ```bash
   bash start.sh
   # Or on Mac: bash start-osx.sh
   ```

### Extending Plugins

See [docs/PLUGIN_SYSTEM.md](PLUGIN_SYSTEM.md) section "Extending the Plugin System" for:
- Creating new plugins
- Adding new LLM tools
- Integration patterns

---

## Jeedom MQTT2 Protocol Reference

### Payload Formats

| Command Type | Payload | Example |
|--------------|---------|---------|
| Binary (ON/OFF) | `{}` | `{}` |
| Slider (0-99) | `{"slider": X}` | `{"slider": 50}` |
| Select | `{"select": "value"}` | `{"select": "auto"}` |
| Message | `{"title": "...", "message": "..."}` | `{"title": "Alert", "message": "Door open"}` |
| Color | `{"color": "#rrggbb"}` | `{"color": "#FF5733"}` |

### Topics

- **Inventory (subscribe):** `jeedom/inventory`
- **Command execution (publish):** `jeedom/cmd/set/{cmd_id}`

---

## Documentation Standards

All documentation follows these principles:

1. **Completeness:** Every component has implementation details
2. **Examples:** Real-world examples with expected outputs
3. **Configuration:** All environment variables documented
4. **Testing:** Verification procedures included
5. **Troubleshooting:** Common issues and solutions
6. **Privacy:** No real IPs, names, or PII (use placeholders)

---

## Next Steps

### Immediate (This Week)
- [ ] Implement Phase 1: Pre-filter layer (30 min)
- [ ] Upgrade to qwen2.5:7b-instruct (1 hour)
- [ ] Re-run test suite (target: 85%+)

### Short-term (This Month)
- [ ] Optimize system prompt with examples
- [ ] Add response validation
- [ ] Monitor real usage patterns

### Long-term
- [ ] Add support for additional protocols (Zigbee, Z-Wave)
- [ ] Implement pre-filter layer
- [ ] Add metrics dashboard
- [ ] Fine-tune model on ARIA-specific data

---

## References

- **Architecture:** [docs/PLUGIN_SYSTEM.md](PLUGIN_SYSTEM.md)
- **Configuration:** [docs/ENVIRONMENT.md](ENVIRONMENT.md)
- **Build Specs:** [.copilot-instructions.md](../.copilot-instructions.md)
- **Improvements:** [QUICK_START_CHECKLIST.md](../QUICK_START_CHECKLIST.md)
- **Analysis:** [TOOL_CALLING_DECISION_ANALYSIS.md](../TOOL_CALLING_DECISION_ANALYSIS.md)

---

**Documentation Version:** 1.0
**Last Updated:** 2026-02-01
**Maintainer:** See git history
