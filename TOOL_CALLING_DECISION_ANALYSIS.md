# ARIA LLM Tool Calling Decision Analysis

**How does ARIA decide when to use tools vs when to respond directly?**

---

## Decision Factors

### 1. **System Prompt Rules**
From [start.sh](start.sh:22-36):
```
CRITICAL RULES:
1. EVERY device control request MUST use find_devices + execute_command tools
2. NEVER claim to have controlled a device without calling execute_command
3. Even if you remember device IDs from earlier in the conversation, you MUST still call find_devices and execute_command for EVERY new request
```

### 2. **Available Tools**
From [app/llm/tools.py](app/llm/tools.py:15-58):
- `find_devices(query, room?)` - Search for devices
- `execute_command(cmd_id, value?)` - Execute device commands

### 3. **LLM Decision Process**

The LLM (qwen2.5:3b-instruct) analyzes the user input and decides:

```
Input → Intent Classification → Action
  ↓           ↓                    ↓
"allume"   CONTROL              find_devices() + execute_command()
"quelle"   QUESTION             Direct response
"merci"    CONVERSATION         Direct response
```

---

## Expected Behavior by Sentence Type

### ✅ **Should ALWAYS Use Tools** (Device Control)

| Category | French Examples | English Examples | Expected Tools |
|----------|----------------|------------------|----------------|
| **Turn ON** | "allume la lumière", "active la lampe" | "turn on the light", "switch on lamp" | 2 calls |
| **Turn OFF** | "éteins la lumière", "coupe la lampe" | "turn off the light", "switch off lamp" | 2 calls |
| **Dimming** | "mets à 50%", "baisse à 30%" | "set to 50%", "dim to 30%" | 2 calls |
| **Open/Close** | "ouvre les volets", "ferme les stores" | "open the shutters", "close the blinds" | 2 calls |
| **Brightness** | "augmente la lumière", "réduis l'intensité" | "increase brightness", "reduce intensity" | 2 calls |

### ❌ **Should NEVER Use Tools** (Information/Conversation)

| Category | French Examples | English Examples | Expected Behavior |
|----------|----------------|------------------|-------------------|
| **Questions** | "quelle heure est-il?", "où est la cuisine?" | "what time is it?", "where is the kitchen?" | Direct response |
| **Status** | "est-ce que la lumière est allumée?", "quel est l'état?" | "is the light on?", "what's the status?" | Direct response (no state tracking) |
| **Greetings** | "bonjour", "salut", "merci" | "hello", "hi", "thank you" | Direct response |
| **Help** | "aide-moi", "que peux-tu faire?" | "help me", "what can you do?" | Direct response |
| **Conversation** | "comment ça va?", "tu es sympa" | "how are you?", "you're nice" | Direct response |

### ⚠️ **Ambiguous Cases** (Context-Dependent)

| Input | Interpretation | Should Use Tools? | Notes |
|-------|----------------|-------------------|-------|
| "la lumière" | Incomplete | No | Too vague, LLM should ask for clarification |
| "salon" | Just a room name | No | No action verb |
| "plus fort" | Louder? Brighter? | Depends | Needs context from history |
| "encore" | Repeat last action | Maybe | Requires session history |
| "c'est bon" | Confirmation | No | Acknowledgment, not action |

---

## Test Cases

### Test Suite 1: Clear Device Control (Should Use Tools)

```python
test_cases_control = [
    # French
    ("allume la lumière du salon", 2),           # Turn on
    ("éteins la lumière de la salle à manger", 2), # Turn off
    ("mets la lumière à 50%", 2),                # Dimming
    ("ouvre les volets", 2),                     # Open shutters
    ("ferme les stores de la chambre", 2),       # Close blinds

    # English
    ("turn on the kitchen light", 2),            # Turn on
    ("turn off the bedroom lamp", 2),            # Turn off
    ("set dining room to 75%", 2),               # Dimming
    ("open the living room shutters", 2),        # Open
    ("close all blinds", 2),                     # Close (if multi-device supported)
]
```

**Expected Flow for Each:**
1. LLM calls `find_devices(query="...")`
2. Receives candidate list
3. LLM calls `execute_command(cmd_id="...", value=...)`
4. Confirms action in response

### Test Suite 2: Conversational (Should NOT Use Tools)

```python
test_cases_conversation = [
    # Questions
    ("quelle heure est-il?", 0),
    ("what time is it?", 0),

    # Greetings
    ("bonjour", 0),
    ("hello", 0),
    ("merci", 0),
    ("thank you", 0),

    # Status Queries (ARIA has no state tracking)
    ("est-ce que la lumière du salon est allumée?", 0),
    ("is the kitchen light on?", 0),

    # Help
    ("aide-moi", 0),
    ("what can you do?", 0),

    # Conversation
    ("comment ça va?", 0),
    ("how are you?", 0),
]
```

**Expected Behavior:**
- 0 tool calls
- Direct text response
- No MQTT messages sent

### Test Suite 3: Edge Cases

```python
test_cases_edge = [
    # Incomplete commands
    ("la lumière", "should ask for clarification"),
    ("salon", "should ask what to do"),

    # Implicit references (requires history)
    ("éteins-la", "needs context from previous turn"),
    ("augmente", "needs to know what to increase"),

    # Multi-device
    ("allume toutes les lumières", "should handle multiple devices"),
    ("éteins tout", "should handle 'all'"),

    # Invalid devices
    ("allume le four", "no oven in inventory - should say not found"),
    ("active la machine à café", "no coffee machine - should say not found"),

    # Complex commands
    ("allume la lumière et mets-la à 50%", "2 actions? or 1 with value?"),
]
```

---

## Current Issues (From Test Results)

### Issue 1: Session History Optimization ❌

**Problem:** LLM skips tool calls after first successful execution

**Evidence:**
```
Test 1 (Salon ON): 2 tool calls ✅
Test 2 (Salon OFF): 0 tool calls ❌  (should be 2)
Test 3 (Salle a Manger ON): 1 tool call ⚠️  (should be 2)
Test 4-6: 0 tool calls ❌
```

**Root Cause:**
- LLM sees device info in session history
- Decides "I already know the cmd_id"
- Skips find_devices call
- Claims to execute but doesn't call execute_command

**Solution:**
- Stronger system prompt (already added) ✅
- Increase token limit (changed from 48 → 1024) ✅
- Consider using larger model (7b instead of 3b)

### Issue 2: Token Limit Too Low ⚠️

**Problem:** Response truncated before completing tool calls

**Evidence:**
```
Response: "Je trouve plusieurs lampes... (cmd_id: 5"
[TRUNCATED - hit 48 token limit]
```

**Solution:**
- Changed `ARIA_LLM_NUM_PREDICT` from 48 → 1024 ✅

### Issue 3: Model Size ⚠️

**Problem:** 3b model struggles with complex tool calling logic

**Current:** `qwen2.5:3b-instruct`
**Recommended:** `qwen2.5:7b-instruct`

**Trade-offs:**
- 3b: Faster (lower latency), but less reliable tool calling
- 7b: Slower (~2x latency), but much better at following instructions

---

## Decision Tree

```
User Input
    ↓
Contains action verb? (allume, éteins, mets, ouvre, ferme, etc.)
    ↓ YES
Contains device reference? (lumière, volets, lampe, etc.)
    ↓ YES
    → USE TOOLS (find_devices + execute_command)

    ↓ NO (just action, no device)
    → ASK FOR CLARIFICATION

    ↓ NO (no action verb)
Is it a question? (quelle, où, est-ce que, etc.)
    ↓ YES
    → DIRECT RESPONSE (no tools)

    ↓ NO
Is it a greeting/thanks? (bonjour, merci, etc.)
    ↓ YES
    → DIRECT RESPONSE

    ↓ NO
Is it ambiguous? (just device name, incomplete)
    → ASK FOR CLARIFICATION
```

---

## Verification Script

Run this to test decision-making:

```bash
# Test clear commands (should use tools)
python3 test_single_command.py

# Test questions (should NOT use tools)
curl -X POST http://192.168.100.100:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "quelle heure est-il?", "use_tools": true, "session_id": "test-question"}'

# Should return 0 tool calls

curl -X POST http://192.168.100.100:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "bonjour", "use_tools": true, "session_id": "test-greeting"}'

# Should return 0 tool calls
```

---

## Recommendations

### 1. **Improve System Prompt** ✅ Done
Added explicit rules about ALWAYS using tools for device control.

### 2. **Increase Token Limit** ✅ Done
Changed from 48 → 1024 tokens.

### 3. **Upgrade Model** ⚠️ Recommended
Switch to `qwen2.5:7b-instruct` for better reliability.

### 4. **Disable Session History for Tool Calls** 🤔 Consider
Option: Don't include history when use_tools=true, forcing fresh tool calls every time.

### 5. **Add Validation Layer** 💡 Future
Before returning response, check:
- If input contains action verb + device → verify tool calls were made
- If tool_calls_made < 2 for control request → reject response and retry

### 6. **Add Intent Classification** 💡 Future
Pre-process input with lightweight classifier:
- `control` → MUST use tools
- `question` → MUST NOT use tools
- `conversation` → MUST NOT use tools
- Force LLM to comply based on classification

---

## Key Takeaway

**The LLM should distinguish between:**

1. **Action Requests** (allume, éteins, mets) → TOOLS REQUIRED
2. **Information Requests** (quelle, où, est-ce que) → NO TOOLS
3. **Social Interaction** (bonjour, merci) → NO TOOLS

The stronger system prompt + higher token limit should improve this, but the 3b model may still struggle. Consider upgrading to 7b for production use.
