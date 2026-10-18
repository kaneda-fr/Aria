> **Current status (2026-10-17): proposal, not the selected architecture.** The
> maintainer has reopened this choice. Read [decision D-001](docs/agents/decisions.md)
> and [current evidence](docs/agents/status.md). The original specification below
> is preserved as historical design material; its authority claims are not current.

# ARIA Option A — Deterministic Orchestrator + LLM-as-Parser (No LLM Tool Calling)

**Status:** Implementation spec (single source of truth)\
**Date:** 2026-02-01\
**Target:** ARIA on llama.cpp (CPU), Qwen2.5-3B-Instruct Q4_K_M\
**Goal:** Reliable device control with minimal latency; eliminate “tools for conversation” and “incomplete tool sequences” without brittle keyword lists.

---

## 1) Why Option A

Observed failures:
- **Tools for conversation**: greetings/questions trigger `find_devices()`
- **Incomplete workflows**: control requests run `find_devices()` but skip `execute_command()`

Root cause:
- The LLM is acting as **router + orchestrator + conversationalist**.
- Small models are not reliable at multi-step tool discipline.

**Option A fix:** LLM does **NLU only** (classification + extraction). Server executes tools deterministically.

---

## 2) Non-Negotiable Principle

> **The LLM must never call tools.**

Tools are executed by server code only.

Implications:
- No tool schemas in the LLM prompt.
- LLM output is strictly JSON matching a schema.
- Orchestrator enforces “find then execute” invariants.

---

## 3) System Overview

### 3.1 Pipeline

User text → Router →\
- **CHAT** → respond (no tools)\
- **CONTROL** → LLM parse (JSON) → `find_devices()` → select command → `execute_command()` → confirm/clarify

### 3.2 Key Modules
- `router`: decides CHAT vs CONTROL (fast)
- `parser`: calls LLM for JSON parse only (tools disabled)
- `orchestrator`: deterministic FSM; executes tools; manages disambiguation state
- `providers`: inventory + execution providers (existing)

---

## 4) Router Spec (Fast Intent Routing)

### 4.1 Interface
```ts
type Mode = "chat" | "control" | "unknown";

route_intent(text: string): {
  mode: Mode;
  confidence: number; // 0..1
  reason?: string;    // for logs
}
```

### 4.2 Routing Policy (must implement exactly)
1. **Bias toward CHAT** unless clearly imperative device control.
2. Return **CONTROL** only if confidence ≥ `CONTROL_THRESHOLD`.
3. Return **CHAT** only if confidence ≥ `CHAT_THRESHOLD`.
4. Otherwise return **UNKNOWN**.

Recommended thresholds (initial):
- `CONTROL_THRESHOLD = 0.70`
- `CHAT_THRESHOLD = 0.70`

### 4.3 UNKNOWN Handling (must implement exactly)
If `mode == "unknown"`:
- Run the **LLM parser** (tools still disabled).
- If parser returns `mode="control"`, proceed through CONTROL orchestrator.
- Else respond as CHAT.

This avoids keyword lists while keeping safety.

### 4.4 Implementation Options (choose one; keep swappable)
- **Embedding router** (preferred): cosine similarity against intent centroids
- **Tiny classifier** (best long-term)
- **Minimal heuristics** (temporary fallback only; isolated module)

---

## 5) LLM Parser Spec (JSON Only, No Tools)

### 5.1 Output Schema (ParseResult)
```json
{
  "mode": "chat" | "control",
  "chat": { "response": "string" } | null,
  "control": {
    "query": "string",
    "action": "turn_on" | "turn_off" | "set_level" | "open" | "close",
    "value": number | null
  } | null
}
```

### 5.2 Parser Rules (strict)
- Tools disabled.
- Output JSON only: **no prose, no markdown, no code fences**.
- If not clearly device control, return `mode="chat"`.
- `value` only allowed when `action="set_level"`; else must be `null`.
- Clamp `value` to `[0, 100]`.
- Never invent device IDs or command IDs.

### 5.3 Parser Prompt (must use verbatim)

**SYSTEM (verbatim):**
You are an NLU parser for a home automation assistant. You must output ONLY a valid JSON object, and nothing else. No markdown. No prose. No explanations.

Return a JSON object matching this schema exactly:
{
  "mode": "chat" | "control",
  "chat": { "response": string } | null,
  "control": { "query": string, "action": "turn_on"|"turn_off"|"set_level"|"open"|"close", "value": number|null } | null
}

Rules:
- If the user is not clearly requesting a device action, mode must be "chat".
- For "set_level", value must be 0..100. Otherwise value must be null.
- Never invent devices or ids. query is only a short search phrase (room + device type).
- Output JSON only.

Examples:
User: bonjour
{"mode":"chat","chat":{"response":"Bonjour !"},"control":null}

User: merci
{"mode":"chat","chat":{"response":"De rien !"},"control":null}

User: allume la lumière du salon
{"mode":"control","chat":null,"control":{"query":"lumière salon","action":"turn_on","value":null}}

User: mets la lumière du salon à 50%
{"mode":"control","chat":null,"control":{"query":"lumière salon","action":"set_level","value":50}}

User: ouvre les volets
{"mode":"control","chat":null,"control":{"query":"volets","action":"open","value":null}}

**USER:** `<user_text_here>`

### 5.4 Validation + Repair (must implement exactly)
- Attempt to parse JSON.
- If invalid JSON or schema violation:
  - Retry **once** with a repair prompt:
    - “Your previous output was invalid. Output ONLY a valid JSON object matching the schema. No other text.”
- If still invalid: return an error response (CHAT mode) like:
  - “Désolé, je n’ai pas compris.”

---

## 6) Deterministic Orchestrator (FSM)

### 6.1 States
- ROUTE
- PARSE
- FIND_DEVICES
- DISAMBIGUATE
- SELECT_COMMAND
- EXECUTE
- CONFIRM
- ERROR

### 6.2 Core Invariants (hard requirements)
If CONTROL:
- Must call `find_devices()` at least once.
- Must call `execute_command()` exactly once for a single action.
- Must not confirm success without `execute_command().ok == true`.

### 6.3 Session State (disambiguation support)
Store per-session (e.g., in memory keyed by `session_id`):
```ts
type PendingDisambiguation = {
  kind: "device_choice";
  original_control: { query: string; action: string; value: number|null };
  options: Array<{ device_id: string; label: string }>; // top 3
  created_at_ms: number;
}
```

If a session has pending disambiguation:
- The next user message is interpreted as:
  - selection (1/2/3, or option label match), OR
  - a new command (route again)

Timeout pending disambiguation after e.g. 2 minutes.

---

## 7) Tool Interfaces (Existing Providers)

### 7.1 Inventory
```ts
find_devices(query: string): {
  devices: Array<{
    device_id: string;
    name: string;
    room?: string;
    commands: Array<{
      cmd_id: string;
      name: string;
      // optional metadata if available:
      kind?: "toggle" | "slider" | "cover";
    }>;
  }>;
}
```

### 7.2 Execution
```ts
execute_command(cmd_id: string, payload: object | null): {
  ok: boolean;
  message?: string;
}
```

---

## 8) Deterministic Command Selection (No LLM)

### 8.1 Normalize Action → Capability
- `turn_on` / `turn_off` → **toggle**
- `set_level` → **slider**
- `open` / `close` → **cover**

### 8.2 Capability Inference (when `kind` missing)
Infer `kind` from command `name` (lowercased), using simple maps:

- toggle keywords: `on`, `off`, `toggle`, `power`, `allume`, `éteins`, `marche`, `arrêt`
- slider keywords: `brightness`, `dimmer`, `level`, `%`, `intensité`, `luminosité`, `slider`
- cover keywords: `open`, `close`, `up`, `down`, `ouvre`, `ferme`, `monte`, `descend`, `volet`, `store`

This is **not routing**; it’s internal command mapping.

### 8.3 Scoring (deterministic)
For each command candidate:
- +100 if inferred/explicit kind matches required capability
- +20 token overlap between `(device.name + device.room + command.name)` and `control.query`
- +5 action synonym hit (same keyword maps as above)
Tie-breakers:
1. higher score
2. if tie: stable order by `cmd_id` ascending

### 8.4 Auto-select vs Disambiguate
If top device/command score is not clearly better:
- If best score - second best score < `MARGIN_THRESHOLD` (start with 20),
  ask user to disambiguate (top 3 options).
- Otherwise auto-select.

---

## 9) Disambiguation UX (must be short)
Example:
> “Je trouve plusieurs lumières : 1) Salon 2) Cuisine 3) Chambre. Laquelle ?”

If user answers `2` → select option 2 and continue to EXECUTE.

---

## 10) Payload Normalization

### 10.1 set_level
- payload must be `{"slider": X}` (X in 0..100)

### 10.2 turn_on / turn_off
Use whatever your provider expects, but it must be deterministic.
Examples (choose one and standardize):
- `{"toggle": 1}` / `{"toggle": 0}`
- or `{"value": true}` / `{"value": false}`

### 10.3 open / close
Likewise deterministic, e.g.:
- `{"value": "open"}` / `{"value": "close"}`
- or provider-specific fields

---

## 11) Observability (required)

Emit structured logs:

- `ARIA.Route.Decision` { text, mode, confidence, reason }
- `ARIA.Parse.Result` { text, ok, mode, action, value, query }
- `ARIA.Tool.FindDevices` { query, device_count, top_labels }
- `ARIA.Select.Command` { action, chosen_device_id, chosen_cmd_id, score, margin }
- `ARIA.Tool.Execute` { cmd_id, payload, ok, duration_ms }
- `ARIA.Control.Outcome` { ok, reason, response }

Return a dev-only `tool_trace` in API responses (optional).

---

## 12) Latency Targets

CONTROL path targets:
- Router: < 20 ms
- LLM parse: 0.5–1.5 s (CPU)
- Inventory lookup: < 20 ms
- Execute: < 200 ms
- Total target: < 2 s (best effort on CPU)

Primary levers:
- Keep parser prompt short and constant.
- Cap parser output (`max_tokens` ~ 120).
- Avoid sending history/tool schemas to parser.

---

## 13) Tests (must exist)

### 13.1 Invariant Tests
- CHAT requests must produce **0 tool calls**
- CONTROL requests must produce **exactly 1 find + 1 execute**
- Never confirm success unless execute ok

### 13.2 Acceptance Set (minimum)
- “bonjour” → chat, no tools
- “merci” → chat, no tools
- “what time is it?” → chat, no tools (or explicit “I only do home control”)
- “allume la lumière du salon” → find + execute
- “éteins la salle à manger” → find + execute
- “mets la lumière du salon à 50%” → find + execute with slider=50
- “ouvre les volets” → find + execute

### 13.3 Disambiguation Test
If `find_devices("lumière")` returns multiple rooms:
- system asks choice (top 3)
- reply “2” executes correct device

---

## 14) Implementation Checklist (do in order)

1. Implement `router` with thresholds + UNKNOWN handling.
2. Implement `parser` with the verbatim prompt + validation + single repair retry.
3. Implement `orchestrator` FSM with invariants + deterministic selection + disambiguation session state.
4. Wire into `/api/llm/process`:
   - If pending disambiguation: handle selection
   - Else: route → (chat or control)
5. Add logging events + (optional) tool_trace.
6. Add tests for invariants + acceptance set + disambiguation.

---

## 15) Definition of Done

- Conversation: **0** tool calls in all acceptance tests.
- Control: **100%** of acceptance control tests call `find_devices()` then `execute_command()`.
- No hallucinated confirmations.
- Measured pass rate ≥ 85% on your existing test suite.
- Latency improved by removing tool schemas/history from the parse model call.

---
