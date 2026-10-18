#!/usr/bin/env python3
"""Test single command without session history."""

import requests
import json

ARIA_API_URL = "http://192.168.100.100:8000"

def test_command(text):
    """Test command with unique session ID (no history)."""

    import time
    session_id = f"test-{int(time.time())}"  # Unique session every time

    payload = {
        "text": text,
        "session_id": session_id,
        "use_tools": True,
        "speak_response": False
    }

    print(f"\n{'='*80}")
    print(f"Command: {text}")
    print(f"Session: {session_id} (fresh, no history)")
    print(f"{'='*80}")

    response = requests.post(f"{ARIA_API_URL}/api/llm/process", json=payload, timeout=30)
    result = response.json()

    print(f"\n✅ Response: {result.get('response')}")
    print(f"🔧 Tool calls: {result.get('tool_calls_made', 0)}")

    if result.get('tool_calls_made', 0) == 0:
        print("⚠️  WARNING: No tools were called! LLM just pretended to do it.")

    return result

# Test each command independently (fresh session each time)
print("\n" + "="*80)
print("Testing commands WITHOUT session history")
print("="*80)

test_command("allume la lumière du salon")
test_command("éteins la lumière du salon")
test_command("mets la lumière de la salle à manger à 50%")
