"""
Jeedom MQTT Normalized Data Examples

These examples show the expected output after normalizing Jeedom discovery
messages into Aria's DeviceRecord and CommandRecord formats.

Based on real MQTT dumps from 2026-01-30.
"""

from typing import TypedDict, Literal

# Example 1: Shutter Device (Volets Salle a Manger)
SHUTTER_DEVICE = {
    "source": "jeedom_mqtt",
    "eq_id": "255",
    "name": "Volets Salle a Manger",
    "room": "RDC",
    "domain": "opening",
    "enabled": True,
    "visible": True,
    "commands": [
        # Action: Position slider
        {
            "cmd_id": "5450",
            "device_id": "255",
            "type": "action",
            "subType": "slider",
            "capability": "flap_slider",
            "unit": None,
            "range": (0, 99),
            "tags": {
                "cap:flap_slider",
                "type:action",
                "subtype:slider",
                "domain:opening",
                "backend:zwavejs"
            },
            "execution": {
                "backend": "jeedom_mqtt",
                "args": {
                    "cmd_id": "5450",
                    "value_required": True,
                    "value_type": "int",
                    "range": (0, 99),
                    "payload_template": "slider",
                    "observer_cmd_id": "5454"
                }
            }
        },
        # Action: Up
        {
            "cmd_id": "5451",
            "device_id": "255",
            "type": "action",
            "subType": "other",
            "capability": "flap_up",
            "unit": None,
            "range": None,
            "tags": {
                "cap:flap_up",
                "type:action",
                "subtype:other",
                "domain:opening",
                "backend:zwavejs"
            },
            "execution": {
                "backend": "jeedom_mqtt",
                "args": {
                    "cmd_id": "5451",
                    "value_required": False,
                    "value_type": None,
                    "range": None,
                    "payload_template": "other",
                    "observer_cmd_id": "5454"
                }
            }
        },
        # Action: Down
        {
            "cmd_id": "5453",
            "device_id": "255",
            "type": "action",
            "subType": "other",
            "capability": "flap_down",
            "unit": None,
            "range": None,
            "tags": {
                "cap:flap_down",
                "type:action",
                "subtype:other",
                "domain:opening",
                "backend:zwavejs"
            },
            "execution": {
                "backend": "jeedom_mqtt",
                "args": {
                    "cmd_id": "5453",
                    "value_required": False,
                    "value_type": None,
                    "range": None,
                    "payload_template": "other",
                    "observer_cmd_id": "5454"
                }
            }
        },
        # Action: Stop
        {
            "cmd_id": "5452",
            "device_id": "255",
            "type": "action",
            "subType": "other",
            "capability": "flap_stop",
            "unit": None,
            "range": None,
            "tags": {
                "cap:flap_stop",
                "type:action",
                "subtype:other",
                "domain:opening",
                "backend:zwavejs"
            },
            "execution": {
                "backend": "jeedom_mqtt",
                "args": {
                    "cmd_id": "5452",
                    "value_required": False,
                    "value_type": None,
                    "range": None,
                    "payload_template": "other",
                    "observer_cmd_id": "5454"
                }
            }
        },
        # Info: Position state
        {
            "cmd_id": "5454",
            "device_id": "255",
            "type": "info",
            "subType": "numeric",
            "capability": "flap_state",
            "unit": "%",
            "range": None,
            "tags": {
                "cap:flap_state",
                "type:info",
                "subtype:numeric",
                "domain:opening",
                "backend:zwavejs"
            },
            "execution": None  # Info commands are not executable
        },
        # Info: Power consumption
        {
            "cmd_id": "5455",
            "device_id": "255",
            "type": "info",
            "subType": "numeric",
            "capability": "power",
            "unit": "W",
            "range": (0, 2500),
            "tags": {
                "cap:power",
                "type:info",
                "subtype:numeric",
                "domain:opening",
                "domain:energy",
                "backend:zwavejs"
            },
            "execution": None
        }
    ],
    "tags": {
        "domain:opening",
        "backend:zwavejs",
        "manufacturer:1120",
        "product:QNSH-001P10"
    },
    "seen_at": 1738232120  # Unix timestamp from lastCommunication
}


# Example 2: Dimmable Light (Lumiere Salle a Manger New)
LIGHT_DEVICE = {
    "source": "jeedom_mqtt",
    "eq_id": "273",
    "name": "Lumiere Salle a Manger New",
    "room": "RDC",
    "domain": "light",
    "enabled": True,
    "visible": True,
    "commands": [
        # Action: Intensity slider
        {
            "cmd_id": "5630",
            "device_id": "273",
            "type": "action",
            "subType": "slider",
            "capability": "light_slider",
            "unit": None,
            "range": (0, 99),
            "tags": {
                "cap:light_slider",
                "type:action",
                "subtype:slider",
                "domain:light",
                "backend:zwavejs"
            },
            "execution": {
                "backend": "jeedom_mqtt",
                "args": {
                    "cmd_id": "5630",
                    "value_required": True,
                    "value_type": "int",
                    "range": (0, 99),
                    "payload_template": "slider",
                    "observer_cmd_id": "5633"
                }
            }
        },
        # Action: On (full brightness)
        {
            "cmd_id": "5631",
            "device_id": "273",
            "type": "action",
            "subType": "other",
            "capability": "light_on",
            "unit": None,
            "range": None,
            "tags": {
                "cap:light_on",
                "type:action",
                "subtype:other",
                "domain:light",
                "backend:zwavejs"
            },
            "execution": {
                "backend": "jeedom_mqtt",
                "args": {
                    "cmd_id": "5631",
                    "value_required": False,
                    "value_type": None,
                    "range": None,
                    "payload_template": "other",
                    "observer_cmd_id": "5633"
                }
            }
        },
        # Action: Off
        {
            "cmd_id": "5632",
            "device_id": "273",
            "type": "action",
            "subType": "other",
            "capability": "light_off",
            "unit": None,
            "range": None,
            "tags": {
                "cap:light_off",
                "type:action",
                "subtype:other",
                "domain:light",
                "backend:zwavejs"
            },
            "execution": {
                "backend": "jeedom_mqtt",
                "args": {
                    "cmd_id": "5632",
                    "value_required": False,
                    "value_type": None,
                    "range": None,
                    "payload_template": "other",
                    "observer_cmd_id": "5633"
                }
            }
        },
        # Info: Current brightness
        {
            "cmd_id": "5633",
            "device_id": "273",
            "type": "info",
            "subType": "numeric",
            "capability": "light_state",
            "unit": "%",
            "range": (0, 100),
            "tags": {
                "cap:light_state",
                "type:info",
                "subtype:numeric",
                "domain:light",
                "backend:zwavejs"
            },
            "execution": None
        },
        # Info: Power consumption
        {
            "cmd_id": "5635",
            "device_id": "273",
            "type": "info",
            "subType": "numeric",
            "capability": "power",
            "unit": "W",
            "range": (0, 1840),
            "tags": {
                "cap:power",
                "type:info",
                "subtype:numeric",
                "domain:light",
                "domain:energy",
                "backend:zwavejs"
            },
            "execution": None
        }
    ],
    "tags": {
        "domain:light",
        "backend:zwavejs",
        "manufacturer:345",
        "product:ZMNHHD"
    },
    "seen_at": 1738232111
}


# Example 3: Event Normalized Structure
EVENT_SHUTTER_STATE = {
    "cmd_id": "5454",
    "value": 64,
    "unit": "%",
    "ts": 1738232145,
    "source": "jeedom_mqtt"
}

EVENT_LIGHT_STATE = {
    "cmd_id": "5633",
    "value": 99,
    "unit": "%",
    "ts": 1738232150,
    "source": "jeedom_mqtt"
}

EVENT_POWER = {
    "cmd_id": "5455",
    "value": 142.9,
    "unit": "W",
    "ts": 1738232145,
    "source": "jeedom_mqtt"
}


# Example 4: Execution Context (created when action is executed)
EXECUTION_CONTEXT_EXAMPLE = {
    "execution_id": "exec-1738232200-5450-abc123",
    "requested_by": "llm",
    "utterance": "ferme les volets de la salle à manger",
    "cmd_id": "5450",
    "value": 0,  # 0 = fully closed
    "created_at": 1738232200,
    "expires_at": 1738232260,  # 60s timeout
    "status": "pending"
}


# Example 5: Expectation (created alongside execution context)
EXPECTATION_EXAMPLE = {
    "execution_id": "exec-1738232200-5450-abc123",
    "observer_cmd_id": "5454",
    "predicate": "equals",
    "target": 0,
    "tolerance": None,
    "after_ts": 1738232200,
    "timeout_ms": 60000
}


# Example 6: Resolution Event (emitted when expectation matches)
RESOLUTION_EVENT_SUCCESS = {
    "type": "actions.execution_completed",
    "execution_id": "exec-1738232200-5450-abc123",
    "status": "success",
    "observed": {
        "cmd_id": "5454",
        "value": 0,
        "ts": 1738232215  # 15 seconds after execution
    }
}

RESOLUTION_EVENT_TIMEOUT = {
    "type": "actions.execution_completed",
    "execution_id": "exec-1738232200-5450-abc123",
    "status": "timeout",
    "observed": None  # No matching observation within timeout
}


# Example 7: LLM Tool Call (what the LLM sees)
LLM_TOOL_CALL_EXAMPLE = {
    "tool": "actions.execute",
    "arguments": {
        "cmd_id": "5450",
        "value": 0
    }
}

LLM_TOOL_RESPONSE_IMMEDIATE = {
    "status": "pending",
    "execution_id": "exec-1738232200-5450-abc123",
    "message": "Execution initiated. Awaiting confirmation."
}

LLM_TOOL_RESPONSE_RESOLVED = {
    "status": "success",
    "execution_id": "exec-1738232200-5450-abc123",
    "observed_value": 0,
    "message": "Action completed successfully."
}


# Example 8: Candidate Selection (text → tags → candidates)
CANDIDATE_SELECTION_EXAMPLE = {
    "input_text": "ferme les volets de la salle à manger",
    "normalized_text": "ferme volets salle manger",
    "intent_tags": {
        "cap:flap_slider",
        "cap:flap_down",
        "domain:opening"
    },
    "location_tags": {
        "room:rdc"
    },
    "candidates": [
        {
            "cmd_id": "5453",  # FLAP_DOWN action
            "device_name": "Volets Salle a Manger",
            "room": "RDC",
            "capability": "flap_down",
            "score": 0.95
        },
        {
            "cmd_id": "5450",  # FLAP_SLIDER action (can set to 0)
            "device_name": "Volets Salle a Manger",
            "room": "RDC",
            "capability": "flap_slider",
            "score": 0.85
        }
    ],
    "selected": "5453"  # Highest score with exact capability match
}


if __name__ == "__main__":
    import json

    print("=== SHUTTER DEVICE ===")
    print(json.dumps(SHUTTER_DEVICE, indent=2, ensure_ascii=False))

    print("\n=== LIGHT DEVICE ===")
    print(json.dumps(LIGHT_DEVICE, indent=2, ensure_ascii=False))

    print("\n=== EVENT EXAMPLES ===")
    print(json.dumps(EVENT_SHUTTER_STATE, indent=2))

    print("\n=== EXECUTION FLOW ===")
    print(json.dumps(EXECUTION_CONTEXT_EXAMPLE, indent=2))
    print(json.dumps(EXPECTATION_EXAMPLE, indent=2))
    print(json.dumps(RESOLUTION_EVENT_SUCCESS, indent=2))
