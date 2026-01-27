> **ARCHIVED / OBSOLETE**
>
> This file describes an experimental tools/Jeedom design that is not the current repo baseline.
> Use **[.copilot-instructions.md](../../.copilot-instructions.md)** as the single authoritative specification.

# ARIA — Conversational AI + Tool Execution (Local, Xeon)  
Copilot Implementation Instructions

## Objective
Implement a **low-latency conversational AI** for ARIA that:
- Maintains conversation per session
- Executes commands (Jeedom / APIs) when required
- Runs fully local on Xeon
- Preserves existing audio pipeline (Parakeet STT + current TTS)
- Keeps LLM + tools off the WebSocket audio thread

---

## Target Architecture

```
Parakeet STT → Conversation LLM → Tool Router → Jeedom / APIs → LLM → TTS
```

Two execution paths:

1) Fast conversational path (always-on)  
2) Tool execution path (only when needed)

---

## Output Contract (STRICT)

The LLM must output **exactly one** of the following:

### A) Assistant message (spoken)
Plain text only.

### B) Tool call (pure JSON)
```json
{
  "type": "tool_call",
  "name": "jeedom.execute",
  "arguments": {
    "cmd_id": 1234,
    "value": null
  }
}
```

Rules:
- JSON only, no markdown, no extra text.
- If parameters are missing → ask a question instead of calling tools.

---

## Conversation State

### Files to create

```
app/conversation/state.py
app/conversation/manager.py
```

### Responsibilities

**ConversationState**
- messages: list[dict]
- summary: str (optional)
- add_user(text)
- add_assistant(text)
- add_tool(name, result)
- get_context_messages()

**ConversationManager**
- session_id → ConversationState
- get(session_id)
- reset(session_id) (optional)

### Memory limits

Env vars:
```
ARIA_CONVO_MAX_TURNS=20
ARIA_CONVO_MAX_CHARS=12000
ARIA_CONVO_SUMMARY_ENABLED=0
```

---

## Tool Framework

### Files to create

```
app/tools/registry.py
app/tools/types.py
app/tools/jeedom.py
app/tools/http_tool.py
```

### Tool Registry

- allowlisted tools only
- async execution
- safe failure

### Jeedom Tool

Env vars:
```
ARIA_TOOLS_ENABLED=1
ARIA_JEEDOM_URL=http://jeedom.local
ARIA_JEEDOM_API_KEY=xxxx
ARIA_JEEDOM_TIMEOUT_S=8
```

Function:
```
async def jeedom_execute(cmd_id: int, value=None) -> ToolResult
```

---

## Device Mapping (optional but recommended)

```
config/devices.json
```

Example:
```json
{
  "living_room_light": {
    "jeedom_cmd_id_on": 1234,
    "jeedom_cmd_id_off": 1235
  }
}
```

Load into:
```
app.state.device_map
```

---

## LLM Client Upgrade

Modify:

```
app/llm/ollama_client.py
```

Add:

```
async def generate_chat(messages: list[dict], config, client=None) -> str
```

Use Ollama `/api/chat`.

### System Prompt Requirements

- ARIA identity
- strict tool-call rules
- concise conversational style
- list of available tools + device aliases

---

## Queue Payload Change

Current:
```
_enqueue_final_llm(app, transcript)
```

Change to:
```
_enqueue_final_llm(app, session_id, transcript)
```

Queue item:
```python
{
  "session_id": session_id,
  "transcript": text
}
```

Update WebSocket handler accordingly.

---

## LLM Worker Pipeline

Modify `_llm_worker(app)`

### Steps

1) Get session state
2) Add user transcript
3) Build chat context
4) Call LLM
5) Parse response

### If normal message:
- add assistant message
- speak via existing TTS

### If tool_call:
- validate tool name
- execute tool
- add tool result to conversation
- call LLM again for final spoken confirmation

### Logging format

```
ARIA.TRANSCRIPT: ...
ARIA.LLM: ...
ARIA.TOOL.CALL: ...
ARIA.TOOL.RESULT: ...
ARIA.LLM: ...
```

---

## Non-Blocking Rule

- LLM + tools run only in `_llm_worker`
- WebSocket audio loop must never block
- Queue remains bounded (latest transcript wins)

---

## Recommended Models (CPU / Xeon)

Primary chat model:
- Llama 3.1 8B Instruct (GGUF Q4_K_M)
or
- Qwen2.5 7B Instruct (GGUF Q4_K_M)

Optional planner model:
- Qwen2.5 14B (used only for tool reasoning)

---

## Acceptance Tests

1) Conversation memory
- “My name is Sebastien” → “What’s my name?”

2) Tool execution
- “Turn on living room light” → Jeedom call + spoken confirmation

3) Clarification
- “Set thermostat” → ask temperature

4) Queue stability
- rapid speech does not crash

5) Barge-in preserved
- interrupt TTS with speech

---

## Implementation Order

1) Conversation state + manager
2) Ollama chat API support
3) Tool registry + Jeedom tool
4) Queue payload update
5) Tool-call parsing + execution
6) Device mapping loader
7) Tests and logging

---

## Design Principle

Conversation must feel instant.  
Commands must be correct.  
Audio must never wait for tools.
