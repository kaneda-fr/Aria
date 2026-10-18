# ARIA Router Testing Guide

## Quick Test: Router Performance

### Option 1: Test with Full Server

**Terminal 1: Start ARIA Server with Router**
```bash
cd /Users/kaneda/dev/Aria
bash start-test-router.sh
```

**Terminal 2: Run Test Suite**
```bash
cd /Users/kaneda/dev/Aria
source .venv/bin/activate
python test_decision_making.py --yes
```

### Option 2: Quick API Tests (Manual)

**Start server first (Terminal 1):**
```bash
bash start-test-router.sh
```

**Test CHAT routing (should be instant, 0 tools):**
```bash
# Test 1: Greeting (French) - should route to CHAT
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "bonjour", "use_tools": true}' | jq '.tool_calls_made'

# Expected output: 0 (router bypasses LLM)

# Test 2: Thanks (English) - should route to CHAT
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "thank you", "use_tools": true}' | jq '.tool_calls_made'

# Expected output: 0 (router bypasses LLM)
```

**Test CONTROL routing (should use LLM + tools):**
```bash
# Test 3: Device control (French) - should route to CONTROL
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "allume la lumière", "use_tools": true}' | jq

# Expected: tool_calls_made > 0, routes to LLM with tools

# Test 4: Device control (English) - should route to CONTROL
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "turn on the light", "use_tools": true}' | jq

# Expected: tool_calls_made > 0, routes to LLM with tools
```

### Option 3: Check Router Logs

**Start server and watch logs:**
```bash
bash start-test-router.sh 2>&1 | grep "ARIA.Router"
```

**Send test requests and watch routing decisions:**
```bash
# In another terminal
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "bonjour", "use_tools": true}'
```

**Look for log entries like:**
```json
{
  "event": "ARIA.Router.Decision",
  "mode": "chat",
  "confidence": 0.84,
  "distance_to_control": 0.478,
  "distance_to_chat": 0.078,
  "latency_ms": 15.2
}
```

---

## Expected Results (with Router Enabled)

### Before Router (Baseline - 30% success)
- **Conversation tests**: 23.1% (3/13) - greetings call tools incorrectly
- **Control tests**: 42.9% (3/7) - LLM tool calling unreliable
- **Overall**: 30% (6/20)

### After Router (Expected - 60-70% success)
- **Conversation tests**: **95%+ (12/13)** - router filters CHAT, no tools
- **Control tests**: **~45% (3-4/7)** - still using 3b LLM (unchanged)
- **Overall**: **60-70% (13-14/20)**

### Key Improvements to Look For
1. ✅ **CHAT requests**: 0 tool calls, instant response (<1s)
2. ✅ **Greetings bypassed**: "bonjour", "hello", "merci" → direct response
3. ✅ **Routing latency**: <50ms (vs 10-15s LLM call)
4. ⚠️ **CONTROL unchanged**: Still ~45% (3b model + LLM tool calling)

---

## Interpreting Test Results

### Success Metrics

**Conversation Success** (Target: 95%+)
- Test: `"bonjour"`, `"hello"`, `"thank you"`, `"merci"`
- Expected: `tool_calls_made: 0`
- Response: Instant (<1s)

**Control Pass-through** (Target: Keep working)
- Test: `"allume la lumière"`, `"turn on the light"`
- Expected: `tool_calls_made: 1-2` (router → LLM → tools)
- Response: 10-15s (unchanged from before)

### Router Decision Logs

Look for these patterns in logs:

**Good CHAT classification:**
```
ARIA.Router.Decision | mode=CHAT, confidence=0.84, dist_control=0.478, dist_chat=0.078
```

**Good CONTROL classification:**
```
ARIA.Router.Decision | mode=CONTROL, confidence=0.77, dist_control=0.105, dist_chat=0.468
```

**UNKNOWN (needs tuning):**
```
ARIA.Router.Decision | mode=UNKNOWN, confidence=0.35, dist_control=0.726, dist_chat=0.469
```

---

## Troubleshooting

### Problem: Server won't start

**Check embedding model download:**
```bash
ls -lh ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
```

If missing, model downloads on first start (~118MB, takes 1-2 minutes).

### Problem: Router not enabled

**Check logs for:**
```
ARIA.Router.Enabled | model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

If you see `ARIA.Router.Disabled`, check environment variables.

### Problem: All requests route to UNKNOWN

**Possible causes:**
1. Thresholds too high (0.70 may be strict)
2. Seed data not loading
3. Centroid separation too small (<0.3)

**Fix: Lower thresholds**
```bash
export ARIA_ROUTER_CONTROL_THRESHOLD=0.60
export ARIA_ROUTER_CHAT_THRESHOLD=0.60
```

### Problem: CHAT requests still call tools

**Check router decision in logs:**
- If `mode=UNKNOWN`: Thresholds too strict
- If `mode=CHAT` but tools called: Router not integrated (bug)

---

## Comparison Commands

### Test WITHOUT Router
```bash
# Disable router
export ARIA_ROUTER_ENABLED=0

# Start server
bash start-test-router.sh

# Run test
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "bonjour", "use_tools": true}' | jq '.tool_calls_made'

# Expected: 1 or more (LLM incorrectly calls tools)
```

### Test WITH Router
```bash
# Enable router
export ARIA_ROUTER_ENABLED=1

# Start server
bash start-test-router.sh

# Run test
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "bonjour", "use_tools": true}' | jq '.tool_calls_made'

# Expected: 0 (router bypasses LLM)
```

---

## Full Test Suite Commands

### Run Complete Test Suite

**With router enabled:**
```bash
cd /Users/kaneda/dev/Aria
source .venv/bin/activate

# Make sure server is running in another terminal
# (bash start-test-router.sh)

# Run test suite
python test_decision_making.py --yes

# Or with verbose output
python test_decision_making.py --yes -v
```

**Expected output:**
```json
{
  "control_tests": {
    "passed": 3,
    "total": 7,
    "pass_rate": 0.429
  },
  "conversation_tests": {
    "passed": 12,
    "total": 13,
    "pass_rate": 0.923
  },
  "overall": {
    "passed": 15,
    "total": 20,
    "pass_rate": 0.75
  }
}
```

### Save Results for Comparison
```bash
# Run test and save results
python test_decision_making.py --yes > router_test_results_$(date +%Y%m%d_%H%M%S).json

# Compare with baseline
diff decision_test_results.json router_test_results_*.json
```

---

## Performance Benchmarks

### Latency Targets

| Request Type | Before Router | After Router | Target |
|--------------|---------------|--------------|--------|
| CHAT (greeting) | 10-15s | **<1s** | <1s ✅ |
| CONTROL (device) | 10-15s | 10-15s | <20s ✅ |
| Router overhead | N/A | <50ms | <50ms ✅ |

### Success Rate Targets

| Test Category | Before | After | Target |
|---------------|--------|-------|--------|
| Conversation | 23.1% | **95%+** | >95% ✅ |
| Control | 42.9% | ~45% | >90% (Phase 2) |
| Overall | 30% | **65-70%** | >85% (Phase 2) |

---

## Next Steps Based on Results

### If Conversation Success ≥ 95%
✅ **Router works!** Phase 1 complete.
- Document results
- Consider Phase 2 (Parser + FSM) if control success <80%

### If Conversation Success 70-94%
⚠️ **Router helps but needs tuning**
- Lower thresholds (0.70 → 0.60)
- Add more seed examples
- Check centroid separation

### If Conversation Success <70%
❌ **Router not helping**
- Check logs for UNKNOWN classifications
- Validate seed data loaded
- Check embedding model loaded correctly

---

## Quick Reference: Test Commands

```bash
# 1. Start server with router
bash start-test-router.sh

# 2. Quick test (another terminal)
curl -X POST http://127.0.0.1:8000/api/llm/process \
  -H "Content-Type: application/json" \
  -d '{"text": "bonjour", "use_tools": true}' | jq '.tool_calls_made'

# 3. Full test suite
python test_decision_making.py --yes

# 4. Check logs
grep "ARIA.Router" aria.log | tail -20
```
