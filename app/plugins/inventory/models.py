"""
Data models for inventory provider.

These are the normalized, canonical representations of devices and commands.
"""

from typing import Optional, Set, Tuple, Any, Dict
from dataclasses import dataclass, field


@dataclass
class CommandRecord:
    """
    Normalized command representation.

    Commands are the unit of intent - each command represents a single
    capability that can be queried (info) or executed (action).
    """

    cmd_id: str  # Unique command identifier
    device_id: str  # Parent device ID
    type: str  # "info" (read-only state) or "action" (executable)
    subType: str  # "slider", "numeric", "binary", "string", "other"
    capability: Optional[str]  # Generic capability (e.g., "light_slider", "flap_state")
    unit: Optional[str]  # Unit of measurement (e.g., "%", "W", "°C")
    range: Optional[Tuple[float, float]]  # Valid value range (min, max)
    tags: Set[str]  # Capability tags for matching
    execution: Optional[Dict[str, Any]]  # ExecutionSpec (for action commands)

    # Metadata
    name: str = ""  # Human-readable name
    visible: bool = True  # UI visibility flag


@dataclass
class DeviceRecord:
    """
    Normalized device representation.

    Devices group related commands and provide context (room, domain).
    """

    source: str  # Source identifier (e.g., "jeedom_mqtt")
    eq_id: str  # Equipment ID (unique within source)
    name: str  # Device name
    room: str  # Room/location name
    domain: str  # Primary domain (light, opening, heating, etc.)
    enabled: bool  # Device enabled/disabled
    visible: bool  # UI visibility flag
    commands: list[CommandRecord]  # All commands for this device
    tags: Set[str]  # Device-level tags
    seen_at: int  # Last communication timestamp (UNIX)

    # Optional metadata
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    firmware: Optional[str] = None


@dataclass
class InventorySnapshot:
    """
    Complete inventory state at a point in time.

    Immutable snapshot for concurrent reads.
    """

    devices: Dict[str, DeviceRecord]  # eq_id → DeviceRecord
    commands: Dict[str, CommandRecord]  # cmd_id → CommandRecord
    last_update: int  # Timestamp of last update
    device_count: int = 0
    command_count: int = 0

    def __post_init__(self):
        self.device_count = len(self.devices)
        self.command_count = len(self.commands)


@dataclass
class CommandState:
    """
    Current state of a command (from events).
    """

    cmd_id: str
    value: Any  # Current value
    unit: Optional[str]
    timestamp: int  # UNIX timestamp


# Type aliases for clarity
DeviceID = str
CommandID = str
TagSet = Set[str]
