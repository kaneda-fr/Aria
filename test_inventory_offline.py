#!/usr/bin/env python3
"""
Test ARIA Inventory Provider offline (no MQTT required).

Uses cached inventory.json to test device search functionality.
"""

import asyncio
import json
from app.plugins.inventory.provider import InventoryProvider
from app.llm.tools import execute_tool, format_tool_result_for_llm


async def test_inventory_offline():
    """Test inventory search with local cache only."""

    print("=" * 80)
    print("ARIA Inventory Provider - Offline Test")
    print("=" * 80)

    # Configure inventory provider (using local cache only)
    inventory_config = {
        "cache_path": "inventory.json",
        "sources": {}  # No MQTT sources - offline mode
    }

    print("\n1. Creating inventory provider...")
    inventory = InventoryProvider(name="inventory", config=inventory_config)

    print("\n2. Starting inventory provider (loading from cache)...")
    inventory.start()

    # Check health
    print("\n3. Checking health...")
    health = inventory.health()
    print(f"Status: {health['status']}")
    print(f"Message: {health.get('message', 'OK')}")
    if 'details' in health:
        details = health['details']
        print(f"Devices: {details.get('device_count', 0)}")
        print(f"Commands: {details.get('command_count', 0)}")

    print("\n" + "=" * 80)
    print("TEST 1: Search 'salle a manger lumiere' (dining room light)")
    print("=" * 80)

    result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "salle a manger lumiere"},
        inventory_provider=inventory,
        actions_provider=None
    )

    print("\nRaw result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nFormatted for LLM:")
    print(format_tool_result_for_llm(result))

    # Show the top candidate details
    if result.get("success") and result.get("candidates"):
        print("\n📋 Top candidate details:")
        top = result["candidates"][0]
        print(f"  Device: {top.get('device_name')}")
        print(f"  Room: {top.get('room')}")
        print(f"  Capability: {top.get('capability')}")
        print(f"  Command ID: {top.get('cmd_id')}")
        print(f"  Match Score: {top.get('score'):.2f}")

    print("\n" + "=" * 80)
    print("TEST 2: Search 'dining room light' (English)")
    print("=" * 80)

    result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "dining room light"},
        inventory_provider=inventory,
        actions_provider=None
    )

    print("\nFormatted for LLM:")
    print(format_tool_result_for_llm(result))

    print("\n" + "=" * 80)
    print("TEST 3: Search for ON command")
    print("=" * 80)

    result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "salle a manger light on"},
        inventory_provider=inventory,
        actions_provider=None
    )

    print("\nFormatted for LLM:")
    print(format_tool_result_for_llm(result))

    if result.get("success") and result.get("candidates"):
        top = result["candidates"][0]
        print(f"\n💡 To turn ON: use cmd_id={top.get('cmd_id')}")

    print("\n" + "=" * 80)
    print("TEST 4: Search for OFF command")
    print("=" * 80)

    result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "salle a manger light off"},
        inventory_provider=inventory,
        actions_provider=None
    )

    print("\nFormatted for LLM:")
    print(format_tool_result_for_llm(result))

    if result.get("success") and result.get("candidates"):
        top = result["candidates"][0]
        print(f"\n💡 To turn OFF: use cmd_id={top.get('cmd_id')}")

    print("\n" + "=" * 80)
    print("TEST 5: Search for dimmer/intensity")
    print("=" * 80)

    result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "salle a manger dimmer intensity"},
        inventory_provider=inventory,
        actions_provider=None
    )

    print("\nFormatted for LLM:")
    print(format_tool_result_for_llm(result))

    if result.get("success") and result.get("candidates"):
        top = result["candidates"][0]
        print(f"\n💡 To set brightness: use cmd_id={top.get('cmd_id')} with value 0-99")

    print("\n" + "=" * 80)
    print("TEST 6: Search shutters")
    print("=" * 80)

    result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "volets salle a manger"},
        inventory_provider=inventory,
        actions_provider=None
    )

    print("\nFormatted for LLM:")
    print(format_tool_result_for_llm(result))

    # Cleanup
    print("\n" + "=" * 80)
    print("Stopping inventory provider...")
    inventory.stop()
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_inventory_offline())
