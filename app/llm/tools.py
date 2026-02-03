"""
LLM Tool Definitions and Execution

Provides function calling tools for inventory and actions.
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# Tool definitions in OpenAI format (compatible with Ollama, llama.cpp, etc.)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_devices",
            "description": "Search for devices and commands that match a user request. Use this to find lights, shutters, or other devices to control. Returns a list of matching commands with their IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's request (e.g., 'lights in kitchen', 'bedroom shutter')"
                    },
                    "room": {
                        "type": "string",
                        "description": "Optional room name to filter results (e.g., 'kitchen', 'bedroom')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a command on a device (turn on/off lights, open/close shutters, set dimmer value, etc.). Use the cmd_id from find_devices results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd_id": {
                        "type": "string",
                        "description": "The command ID from find_devices results"
                    },
                    "value": {
                        "type": "number",
                        "description": "Optional value for the command (e.g., brightness 0-100 for dimmers, position 0-100 for shutters)"
                    }
                },
                "required": ["cmd_id"]
            }
        }
    }
]


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    inventory_provider: Optional[Any] = None,
    actions_provider: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Execute a tool call.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        inventory_provider: InventoryProvider instance
        actions_provider: ActionsProvider instance

    Returns:
        Dict with tool execution result
    """
    from app.aria_logging import get_logger
    log = get_logger("ARIA.LLM.Tools")

    try:
        if tool_name == "find_devices":
            return await _find_devices(tool_args, inventory_provider, log)
        elif tool_name == "execute_command":
            return await _execute_command(tool_args, inventory_provider, actions_provider, log)
        else:
            log.error("ARIA.LLM.Tools.UnknownTool", extra={"fields": {"tool_name": tool_name}})
            return {
                "error": f"Unknown tool: {tool_name}",
                "success": False
            }

    except Exception as e:
        log.error("ARIA.LLM.Tools.ExecutionError", extra={"fields": {
            "tool_name": tool_name,
            "error": str(e)
        }})
        return {
            "error": f"Tool execution failed: {str(e)}",
            "success": False
        }


async def _find_devices(
    args: Dict[str, Any],
    inventory_provider: Optional[Any],
    log: Any
) -> Dict[str, Any]:
    """Execute find_devices tool."""
    if not inventory_provider or not inventory_provider.is_started:
        return {
            "error": "Inventory not available",
            "success": False,
            "candidates": []
        }

    query = args.get("query", "")
    room = args.get("room")

    context = {}
    if room:
        context["room"] = room

    # Get candidates from inventory
    result = inventory_provider.get_candidates(text=query, context=context if context else None)

    candidates = result.get("candidates", [])
    log.info("ARIA.LLM.Tools.FindDevices", extra={"fields": {
        "query": query,
        "room": room,
        "candidate_count": len(candidates)
    }})

    # Simplify candidate data for LLM
    simplified = []
    for candidate in candidates[:5]:  # Limit to top 5
        data = candidate.get("data", {})
        simplified.append({
            "cmd_id": data.get("cmd_id"),
            "device_name": data.get("device_name"),
            "room": data.get("room"),
            "capability": data.get("capability"),
            "score": candidate.get("score")
        })

    # Debug logging (if enabled)
    import os
    debug_enabled = os.environ.get("ARIA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    if debug_enabled and simplified:
        log.info("ARIA.LLM.Tools.Debug.CandidatesFound", extra={"fields": {
            "query": query,
            "room": room,
            "candidates": simplified
        }})

    return {
        "success": True,
        "candidates": simplified,
        "total_found": len(candidates)
    }


async def _execute_command(
    args: Dict[str, Any],
    inventory_provider: Optional[Any],
    actions_provider: Optional[Any],
    log: Any
) -> Dict[str, Any]:
    """Execute execute_command tool."""
    if not actions_provider or not actions_provider.is_started:
        return {
            "error": "Actions not available",
            "success": False
        }

    cmd_id = args.get("cmd_id")
    value = args.get("value")

    if not cmd_id:
        return {
            "error": "Missing cmd_id parameter",
            "success": False
        }

    # Get execution spec from inventory (optional)
    execution_spec = None
    device_info = None
    if inventory_provider and inventory_provider.is_started:
        cmd = inventory_provider.indexer.get_command(cmd_id)
        if cmd:
            if cmd.execution:
                execution_spec = cmd.execution

            # Get device info for logging
            device = inventory_provider.indexer.get_device(cmd.device_id)
            if device:
                device_info = {
                    "device_name": device.name,
                    "room": device.room,
                    "capability": cmd.capability
                }

    # Debug logging (if enabled)
    import os
    debug_enabled = os.environ.get("ARIA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    if debug_enabled:
        log.info("ARIA.LLM.Tools.Debug.ExecutingCommand", extra={"fields": {
            "cmd_id": cmd_id,
            "value": value,
            "device_info": device_info,
            "has_execution_spec": execution_spec is not None
        }})

    # Execute command
    result = actions_provider.execute_command(
        cmd_id=cmd_id,
        value=value,
        execution_spec=execution_spec
    )

    log.info("ARIA.LLM.Tools.ExecuteCommand", extra={"fields": {
        "cmd_id": cmd_id,
        "value": value,
        "success": result.get("success", False)
    }})

    if debug_enabled:
        log.info("ARIA.LLM.Tools.Debug.ExecutionResult", extra={"fields": {
            "cmd_id": cmd_id,
            "result": result
        }})

    return result


def format_tool_result_for_llm(tool_result: Dict[str, Any]) -> str:
    """
    Format tool execution result for LLM consumption.

    Converts JSON result to a concise text description.
    """
    if not tool_result.get("success", False):
        error = tool_result.get("error", "Unknown error")
        return f"Error: {error}"

    # find_devices result
    if "candidates" in tool_result:
        candidates = tool_result.get("candidates", [])
        if not candidates:
            return "No matching devices found."

        lines = [f"Found {len(candidates)} device(s):"]
        for i, candidate in enumerate(candidates, 1):
            device_name = candidate.get("device_name", "Unknown")
            room = candidate.get("room", "")
            cmd_id = candidate.get("cmd_id", "")
            capability = candidate.get("capability", "")

            location = f" in {room}" if room else ""
            lines.append(f"{i}. {device_name}{location} (cmd_id: {cmd_id}, capability: {capability})")

        return "\n".join(lines)

    # execute_command result
    if "cmd_id" in tool_result:
        cmd_id = tool_result.get("cmd_id")
        message = tool_result.get("message", "Command executed")
        return f"Success: {message} (cmd_id: {cmd_id})"

    # Fallback
    return json.dumps(tool_result)
