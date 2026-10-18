#!/usr/bin/env python3
"""
Test ARIA's routing: when to use tools vs when to respond directly.

This script validates the router's classification:
- Conversation (CHAT): Should use 0 tools (direct response)
- Device Control (CONTROL): Should use >0 tools (1, 2, or 3 all acceptable)

Note: This testw routing behavior only. The exact number of tool calls
(1 vs 2 vs 3) is out of scope - that's a separate tool discipline issue.
"""

import requests
import json
import time
from typing import List, Tuple

ARIA_API_URL = "http://127.0.0.1:8000"


def test_sentence(text: str, expected_tools: int, category: str, session_id: str = None) -> dict:
    """
    Test a single sentence and verify routing behavior (tool vs no-tool).

    Args:
        text: Input sentence
        expected_tools: Expected tool behavior (0 = no tools, >0 = any tools)
        category: Test category for reporting
        session_id: Optional session ID (use unique ID to avoid history)

    Returns:
        Test result dict
    """
    if session_id is None:
        # Use timestamp for unique session
        session_id = f"test-{int(time.time() * 1000)}"

    payload = {
        "text": text,
        "session_id": session_id,
        "use_tools": True,
        "speak_response": False
    }

    try:
        # Measure latency
        start_time = time.perf_counter()
        response = requests.post(
            f"{ARIA_API_URL}/api/llm/process",
            json=payload,
            timeout=30
        )
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        response.raise_for_status()
        result = response.json()

        actual_tools = result.get("tool_calls_made", 0)
        success = result.get("success", False)
        response_text = result.get("response", "")

        # Determine if test passed (routing-focused criteria)
        # For conversation (expected_tools=0): Pass if no tools were called
        # For control (expected_tools>0): Pass if any tools were called (1, 2, or 3 all OK)
        if expected_tools == 0:
            passed = (actual_tools == 0)  # CHAT must have exactly 0 tools
        else:
            passed = (actual_tools > 0)   # CONTROL must have >0 tools (count doesn't matter)

        return {
            "text": text,
            "category": category,
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "passed": passed,
            "success": success,
            "response": response_text[:100],  # Truncate for display
            "session_id": session_id,
            "latency_ms": round(latency_ms, 2)
        }

    except Exception as e:
        return {
            "text": text,
            "category": category,
            "expected_tools": expected_tools,
            "actual_tools": -1,
            "passed": False,
            "success": False,
            "response": f"ERROR: {str(e)}",
            "session_id": session_id,
            "latency_ms": 0.0
        }


def run_test_suite():
    """Run comprehensive test suite."""

    print("=" * 80)
    print("ARIA Routing Test Suite")
    print("Testing router classification: CHAT (0 tools) vs CONTROL (>0 tools)")
    print("Includes latency benchmarks for performance analysis")
    print("=" * 80)

    # Test Suite 1: Device Control (SHOULD use tools)
    print("\n" + "=" * 80)
    print("TEST SUITE 1: Device Control Commands (Should use >0 tools)")
    print("=" * 80)

    control_tests = [
        # French - Pass if >0 tools (1, 2, or 3 all acceptable)
        ("allume la lumière du salon", 1, "Turn ON (French)"),
        ("éteins la lumière de la salle à manger", 1, "Turn OFF (French)"),
        ("mets la lumière à 50%", 1, "Dimming (French)"),
        ("règle la lumière du salon à 75%", 1, "Set Level (French)"),

        # English - Pass if >0 tools (1, 2, or 3 all acceptable)
        ("turn on the kitchen light", 1, "Turn ON (English)"),
        ("turn off the bedroom light", 1, "Turn OFF (English)"),
        ("set the dining room light to 50%", 1, "Dimming (English)"),
    ]

    control_results = []
    for text, expected, category in control_tests:
        time.sleep(0.5)  # Rate limiting
        result = test_sentence(text, expected, category)
        control_results.append(result)

        status = "✅" if result["passed"] else "❌"
        tools_ok = "✅" if result['actual_tools'] > 0 else "❌"
        print(f"{status} {category}: {text[:40]:<40} "
              f"→ {result['actual_tools']} tools {tools_ok} | {result['latency_ms']:.0f}ms")

    # Test Suite 2: Conversation (Should NOT use tools)
    print("\n" + "=" * 80)
    print("TEST SUITE 2: Conversation (Should use 0 tools)")
    print("=" * 80)

    conversation_tests = [
        # Questions
        ("quelle heure est-il?", 0, "Time Question (French)"),
        ("what time is it?", 0, "Time Question (English)"),
        ("où est la cuisine?", 0, "Location Question (French)"),

        # Greetings
        ("bonjour", 0, "Greeting (French)"),
        ("hello", 0, "Greeting (English)"),
        ("merci", 0, "Thanks (French)"),
        ("thank you", 0, "Thanks (English)"),

        # Status queries (ARIA has no state)
        ("est-ce que la lumière du salon est allumée?", 0, "Status Query (French)"),
        ("is the light on?", 0, "Status Query (English)"),

        # Help
        ("aide-moi", 0, "Help Request (French)"),
        ("what can you do?", 0, "Capabilities (English)"),

        # Conversation
        ("comment ça va?", 0, "Small Talk (French)"),
        ("how are you?", 0, "Small Talk (English)"),
    ]

    conversation_results = []
    for text, expected, category in conversation_tests:
        time.sleep(0.5)  # Rate limiting
        result = test_sentence(text, expected, category)
        conversation_results.append(result)

        status = "✅" if result["passed"] else "❌"
        tools_status = "✅" if result['actual_tools'] == 0 else "❌"
        print(f"{status} {category}: {text[:40]:<40} "
              f"→ {result['actual_tools']} tools {tools_status} | {result['latency_ms']:.0f}ms")

    # Test Suite 3: Edge Cases
    print("\n" + "=" * 80)
    print("TEST SUITE 3: Edge Cases (Variable expectations)")
    print("=" * 80)

    edge_tests = [
        ("la lumière", 0, "Incomplete Command"),  # Too vague, should ask
        ("salon", 0, "Just Room Name"),           # No action
        ("plus fort", 0, "Ambiguous"),            # Needs context
    ]

    edge_results = []
    for text, expected, category in edge_tests:
        time.sleep(0.5)
        result = test_sentence(text, expected, category)
        edge_results.append(result)

        status = "⚠️"  # Edge cases are informational
        print(f"{status} {category}: {text[:40]:<40} "
              f"→ {result['actual_tools']} tools | {result['latency_ms']:.0f}ms | {result['response'][:40]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_results = control_results + conversation_results

    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    failed = total - passed

    control_passed = sum(1 for r in control_results if r["passed"])
    control_total = len(control_results)

    conversation_passed = sum(1 for r in conversation_results if r["passed"])
    conversation_total = len(conversation_results)

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"  Control Commands: {control_passed}/{control_total} "
          f"({control_passed/control_total*100:.1f}%)")
    print(f"  Conversation: {conversation_passed}/{conversation_total} "
          f"({conversation_passed/conversation_total*100:.1f}%)")

    # Latency statistics
    control_latencies = [r["latency_ms"] for r in control_results if r["latency_ms"] > 0]
    conversation_latencies = [r["latency_ms"] for r in conversation_results if r["latency_ms"] > 0]
    all_latencies = control_latencies + conversation_latencies

    if all_latencies:
        print(f"\n📊 Latency Benchmarks:")
        print(f"  Overall: {sum(all_latencies)/len(all_latencies):.1f}ms avg, "
              f"{min(all_latencies):.1f}ms min, {max(all_latencies):.1f}ms max")
        if control_latencies:
            print(f"  Control: {sum(control_latencies)/len(control_latencies):.1f}ms avg, "
                  f"{min(control_latencies):.1f}ms min, {max(control_latencies):.1f}ms max")
        if conversation_latencies:
            print(f"  Conversation: {sum(conversation_latencies)/len(conversation_latencies):.1f}ms avg, "
                  f"{min(conversation_latencies):.1f}ms min, {max(conversation_latencies):.1f}ms max")

    if failed > 0:
        print(f"\n❌ {failed} tests failed:")
        for result in all_results:
            if not result["passed"]:
                print(f"  - {result['category']}: '{result['text'][:50]}' "
                      f"(got {result['actual_tools']}, expected {result['expected_tools']})")

    # Detailed failures
    if failed > 0:
        print("\n" + "=" * 80)
        print("FAILURE ANALYSIS")
        print("=" * 80)

        control_failures = [r for r in control_results if not r["passed"]]
        if control_failures:
            print("\n⚠️  Control commands that didn't use any tools (expected >0):")
            for r in control_failures:
                print(f"\nInput: {r['text']}")
                print(f"Expected: >0 tools")
                print(f"Actual: {r['actual_tools']} tools")
                print(f"Response: {r['response']}")

        conversation_failures = [r for r in conversation_results if not r["passed"]]
        if conversation_failures:
            print("\n⚠️  Conversation that incorrectly used tools:")
            for r in conversation_failures:
                print(f"\nInput: {r['text']}")
                print(f"Expected: {r['expected_tools']} tools")
                print(f"Actual: {r['actual_tools']} tools")
                print(f"Response: {r['response']}")

    # Calculate latency statistics for JSON output
    latency_stats = {}
    if all_latencies:
        latency_stats = {
            "overall": {
                "avg_ms": round(sum(all_latencies)/len(all_latencies), 2),
                "min_ms": round(min(all_latencies), 2),
                "max_ms": round(max(all_latencies), 2),
                "median_ms": round(sorted(all_latencies)[len(all_latencies)//2], 2)
            }
        }
        if control_latencies:
            latency_stats["control"] = {
                "avg_ms": round(sum(control_latencies)/len(control_latencies), 2),
                "min_ms": round(min(control_latencies), 2),
                "max_ms": round(max(control_latencies), 2),
                "median_ms": round(sorted(control_latencies)[len(control_latencies)//2], 2)
            }
        if conversation_latencies:
            latency_stats["conversation"] = {
                "avg_ms": round(sum(conversation_latencies)/len(conversation_latencies), 2),
                "min_ms": round(min(conversation_latencies), 2),
                "max_ms": round(max(conversation_latencies), 2),
                "median_ms": round(sorted(conversation_latencies)[len(conversation_latencies)//2], 2)
            }

    # Save detailed results
    with open("decision_test_results.json", "w") as f:
        json.dump({
            "control_tests": control_results,
            "conversation_tests": conversation_results,
            "edge_tests": edge_results,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed/total if total > 0 else 0
            },
            "latency_benchmarks": latency_stats
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed results saved to: decision_test_results.json")


if __name__ == "__main__":
    import sys

    # Check if running in non-interactive mode
    non_interactive = "--yes" in sys.argv or "-y" in sys.argv

    if not non_interactive:
        print("⚠️  This will send ~20 requests to ARIA server")
        print(f"Make sure ARIA is running on {ARIA_API_URL}")
        print("Press Enter to continue, Ctrl+C to cancel...\n")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled by user")
            sys.exit(0)

    try:
        run_test_suite()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
