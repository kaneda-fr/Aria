#!/usr/bin/env python3
"""
Test ARIA API with LLM tool calling for real device control.

This script sends requests to ARIA server running on 192.168.100.100
to control lights via LLM tool calling.
"""

import requests
import json
import time

# ARIA API configuration
ARIA_API_URL = "http://192.168.100.100:8000"

def test_llm_command(text: str, session_id: str = "test-session"):
    """
    Send text to ARIA LLM processing endpoint with tool calling.

    Args:
        text: User command in natural language
        session_id: Optional session ID for conversation history

    Returns:
        API response dict
    """

    url = f"{ARIA_API_URL}/api/llm/process"

    payload = {
        "text": text,
        "session_id": session_id,
        "use_tools": True,  # Enable tool calling
        "speak_response": False  # Don't speak response for testing
    }

    print(f"\n{'=' * 80}")
    print(f"Request: {text}")
    print(f"{'=' * 80}")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()

        print(f"\n✅ Success!")
        print(f"Response: {result.get('response')}")
        print(f"Tool calls made: {result.get('tool_calls_made', 0)}")
        if result.get('error'):
            print(f"Error: {result.get('error')}")

        print(f"\nFull response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return result

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error: Cannot reach {ARIA_API_URL}")
        print("Make sure:")
        print("  1. ARIA server is running on 192.168.100.100:8000")
        print("  2. You can reach the server from this machine")
        print("  3. Plugins are enabled (ARIA_PLUGINS_ENABLED=1)")
        return None

    except requests.exceptions.Timeout:
        print(f"\n❌ Timeout: Request took longer than 30 seconds")
        return None

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"Response: {e.response.text}")
        return None

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def test_health_check():
    """Check if ARIA server is running and plugins are loaded."""

    print(f"\n{'=' * 80}")
    print("Health Check")
    print(f"{'=' * 80}")

    url = f"{ARIA_API_URL}/healthz"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        health = response.json()

        print(f"✅ ARIA server is running")
        print(f"Status: {health.get('status')}")
        print(f"ASR loaded: {health.get('asr_loaded')}")

        plugins = health.get('plugins', {})
        if plugins:
            print(f"\nPlugin Status:")
            print(f"  Overall: {plugins.get('status')}")

            for plugin_name, plugin_status in plugins.get('plugins', {}).items():
                print(f"\n  {plugin_name}:")
                print(f"    Status: {plugin_status.get('status')}")
                print(f"    Message: {plugin_status.get('message')}")

                details = plugin_status.get('details', {})
                if 'device_count' in details:
                    print(f"    Devices: {details['device_count']}")
                if 'command_count' in details:
                    print(f"    Commands: {details['command_count']}")
                if 'connected' in details:
                    print(f"    MQTT Connected: {details['connected']}")
        else:
            print("\n⚠️  No plugins loaded - make sure ARIA_PLUGINS_ENABLED=1")

        return health

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return None


def main():
    """Run API tests."""

    print("=" * 80)
    print("ARIA API Command Test")
    print("Testing LLM tool calling with real devices")
    print("=" * 80)

    # Step 1: Health check
    health = test_health_check()
    if not health:
        print("\n❌ Cannot proceed - server not reachable")
        return

    # Check if plugins are enabled
    if not health.get('plugins'):
        print("\n❌ Plugins not enabled - cannot test device control")
        print("Start ARIA with: bash start.sh")
        return

    time.sleep(1)

    # Test 1: Turn on Lumiere Salon (binary on/off, no dimmer)
    print("\n\n" + "=" * 80)
    print("TEST 1: Turn ON Lumiere Salon (binary on/off)")
    print("=" * 80)

    result1 = test_llm_command("allume la lumière du salon")
    time.sleep(3)

    # Test 2: Turn off Lumiere Salon
    print("\n\n" + "=" * 80)
    print("TEST 2: Turn OFF Lumiere Salon")
    print("=" * 80)

    result2 = test_llm_command("éteins la lumière du salon")
    time.sleep(3)

    # Test 3: Turn on Lumiere Salle a Manger (dimmable)
    print("\n\n" + "=" * 80)
    print("TEST 3: Turn ON Lumiere Salle a Manger (dimmable)")
    print("=" * 80)

    result3 = test_llm_command("allume la lumière de la salle à manger")
    time.sleep(3)

    # Test 4: Set Salle a Manger to 50%
    print("\n\n" + "=" * 80)
    print("TEST 4: Set Lumiere Salle a Manger to 50%")
    print("=" * 80)

    result4 = test_llm_command("mets la lumière de la salle à manger à 50%")
    time.sleep(3)

    # Test 5: Set Salle a Manger to 75%
    print("\n\n" + "=" * 80)
    print("TEST 5: Set Lumiere Salle a Manger to 75%")
    print("=" * 80)

    result5 = test_llm_command("mets la lumière de la salle à manger à 75 pourcent")
    time.sleep(3)

    # Test 6: Turn off Salle a Manger
    print("\n\n" + "=" * 80)
    print("TEST 6: Turn OFF Lumiere Salle a Manger")
    print("=" * 80)

    result6 = test_llm_command("éteins la lumière de la salle à manger")

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    tests = [
        ("Salon ON", result1),
        ("Salon OFF", result2),
        ("Salle a Manger ON", result3),
        ("Salle a Manger 50%", result4),
        ("Salle a Manger 75%", result5),
        ("Salle a Manger OFF", result6),
    ]

    for name, result in tests:
        if result:
            status = "✅" if result.get('success') else "❌"
            tools = result.get('tool_calls_made', 0)
            response = result.get('response', '')[:50]
            print(f"{status} {name}: {tools} tool calls - {response}...")
        else:
            print(f"❌ {name}: Failed to connect")


if __name__ == "__main__":
    print("\n⚠️  WARNING: This will control your real lights!")
    print("Make sure ARIA server is running on 192.168.100.100:8000")
    print("Press Ctrl+C to cancel, or Enter to continue...\n")

    try:
        input()
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
