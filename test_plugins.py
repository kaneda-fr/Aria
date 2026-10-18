#!/usr/bin/env python3
"""
Test ARIA Plugin System with real Jeedom server.

Tests inventory and actions providers without starting full server.
"""

import asyncio
import json
from app.plugins.registry import PluginRegistry
from app.plugins.inventory import InventoryProvider
from app.plugins.actions import ActionsProvider
from app.llm.tools import execute_tool, format_tool_result_for_llm


async def test_plugin_system():
    """Test plugin system with local inventory cache."""

    print("=" * 80)
    print("ARIA Plugin System Test")
    print("=" * 80)

    # Create plugin registry
    registry = PluginRegistry()

    # Register plugin classes
    registry.register_plugin_class("inventory", InventoryProvider)
    registry.register_plugin_class("actions", ActionsProvider)

    # Configure inventory provider (using local cache)
    inventory_config = {
        "cache_path": "inventory.json",
        "sources": {}  # No MQTT sources for now, just use cache
    }

    # Configure actions provider
    actions_config = {
        "host": "192.168.100.60",  # Jeedom MQTT server
        "port": 1883,
        "username": None,  # Set if needed
        "password": None,  # Set if needed
        "root_topic": "jeedom",
        "qos": 1,
    }

    # Create plugins
    print("\n1. Creating plugins...")
    inventory = registry.create_plugin("inventory", inventory_config)
    actions = registry.create_plugin("actions", actions_config)

    # Start plugins
    print("\n2. Starting plugins...")
    registry.start_all()

    # Check health
    print("\n3. Checking plugin health...")
    health = registry.health_all()
    print(f"Overall status: {health['status']}")
    for plugin_name, plugin_health in health['plugins'].items():
        print(f"  {plugin_name}: {plugin_health['status']} - {plugin_health.get('message', 'OK')}")
        if 'details' in plugin_health:
            details = plugin_health['details']
            if 'device_count' in details:
                print(f"    Devices: {details['device_count']}")
            if 'command_count' in details:
                print(f"    Commands: {details['command_count']}")

    # Test 1: Search for dining room light
    print("\n" + "=" * 80)
    print("TEST 1: Search for dining room light")
    print("=" * 80)

    tool_args = {
        "query": "lumiere salle a manger"
    }

    result = await execute_tool(
        tool_name="find_devices",
        tool_args=tool_args,
        inventory_provider=inventory,
        actions_provider=actions
    )

    print("\nRaw result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nFormatted for LLM:")
    formatted = format_tool_result_for_llm(result)
    print(formatted)

    # Test 2: Search with English query
    print("\n" + "=" * 80)
    print("TEST 2: Search with English query")
    print("=" * 80)

    tool_args = {
        "query": "dining room light"
    }

    result = await execute_tool(
        tool_name="find_devices",
        tool_args=tool_args,
        inventory_provider=inventory,
        actions_provider=actions
    )

    print("\nFormatted for LLM:")
    formatted = format_tool_result_for_llm(result)
    print(formatted)

    # Test 3: Execute command (turn on light)
    print("\n" + "=" * 80)
    print("TEST 3: Execute command (TURN ON dining room light)")
    print("=" * 80)
    print("⚠️  This will send MQTT command to your Jeedom server!")

    # Get the ON command ID for dining room light
    search_result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "salle a manger light on"},
        inventory_provider=inventory,
        actions_provider=actions
    )

    if search_result.get("success") and search_result.get("candidates"):
        cmd = search_result["candidates"][0]
        cmd_id = cmd.get("cmd_id")

        print(f"\nFound command: {cmd.get('device_name')} - {cmd.get('capability')}")
        print(f"Command ID: {cmd_id}")

        # Execute the command
        print("\nExecuting command...")
        exec_result = await execute_tool(
            tool_name="execute_command",
            tool_args={"cmd_id": cmd_id},
            inventory_provider=inventory,
            actions_provider=actions
        )

        print("\nResult:")
        print(json.dumps(exec_result, indent=2, ensure_ascii=False))
        print("\nFormatted for LLM:")
        print(format_tool_result_for_llm(exec_result))
    else:
        print("❌ Could not find 'on' command for dining room light")

    # Test 4: Set dimmer to 50%
    print("\n" + "=" * 80)
    print("TEST 4: Set dimmer to 50%")
    print("=" * 80)

    search_result = await execute_tool(
        tool_name="find_devices",
        tool_args={"query": "salle a manger dimmer"},
        inventory_provider=inventory,
        actions_provider=actions
    )

    if search_result.get("success") and search_result.get("candidates"):
        cmd = search_result["candidates"][0]
        cmd_id = cmd.get("cmd_id")

        print(f"\nFound command: {cmd.get('device_name')} - {cmd.get('capability')}")
        print(f"Command ID: {cmd_id}")

        # Execute with value
        print("\nSetting intensity to 50%...")
        exec_result = await execute_tool(
            tool_name="execute_command",
            tool_args={"cmd_id": cmd_id, "value": 50},
            inventory_provider=inventory,
            actions_provider=actions
        )

        print("\nResult:")
        print(json.dumps(exec_result, indent=2, ensure_ascii=False))

    # Cleanup
    print("\n" + "=" * 80)
    print("Stopping plugins...")
    registry.stop_all()
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_plugin_system())
