"""
In-memory indexer for fast candidate selection.

Builds indexes by capability, room, device, tags for O(1)/O(log n) lookups.
"""

import logging
from typing import Dict, List, Set
from collections import defaultdict

from app.plugins.inventory.models import (
    DeviceRecord,
    CommandRecord,
    InventorySnapshot,
    CommandID,
    DeviceID,
)

logger = logging.getLogger(__name__)


class InventoryIndexer:
    """
    Builds and maintains in-memory indexes for fast candidate selection.

    Indexes:
    - By cmd_id (O(1) lookup)
    - By device_id (O(1) → O(n commands))
    - By capability tag (O(1) → O(m commands))
    - By room (O(1) → O(k commands))
    - By tag (general) (O(1) → O(j commands))
    """

    def __init__(self):
        self._devices: Dict[DeviceID, DeviceRecord] = {}
        self._commands: Dict[CommandID, CommandRecord] = {}

        # Indexes
        self._by_capability: Dict[str, Set[CommandID]] = defaultdict(set)
        self._by_room: Dict[str, Set[CommandID]] = defaultdict(set)
        self._by_device: Dict[DeviceID, Set[CommandID]] = defaultdict(set)
        self._by_tag: Dict[str, Set[CommandID]] = defaultdict(set)

    def rebuild(self, devices: List[DeviceRecord]) -> None:
        """
        Rebuild all indexes from device list.

        Args:
            devices: Complete list of devices
        """
        logger.info(f"Rebuilding indexes for {len(devices)} devices")

        # Clear existing
        self._devices.clear()
        self._commands.clear()
        self._by_capability.clear()
        self._by_room.clear()
        self._by_device.clear()
        self._by_tag.clear()

        # Process each device
        for device in devices:
            self._index_device(device)

        logger.info(
            f"Indexes rebuilt: {len(self._commands)} commands, "
            f"{len(self._by_capability)} capabilities, "
            f"{len(self._by_room)} rooms"
        )

    def add_device(self, device: DeviceRecord) -> None:
        """
        Add or update a single device.

        Args:
            device: Device to index
        """
        # Remove existing commands for this device (if any)
        if device.eq_id in self._devices:
            self.remove_device(device.eq_id)

        # Index new device
        self._index_device(device)

    def remove_device(self, device_id: DeviceID) -> None:
        """
        Remove a device and its commands from indexes.

        Args:
            device_id: Device to remove
        """
        device = self._devices.get(device_id)
        if not device:
            return

        # Remove commands
        for cmd in device.commands:
            self._remove_command_from_indexes(cmd.cmd_id)

        # Remove device
        del self._devices[device_id]

    def _index_device(self, device: DeviceRecord) -> None:
        """Internal: Index a device and all its commands."""
        # Store device
        self._devices[device.eq_id] = device

        # Index each command
        for cmd in device.commands:
            self._index_command(cmd, device)

    def _index_command(self, cmd: CommandRecord, device: DeviceRecord) -> None:
        """Internal: Index a single command."""
        # Store command
        self._commands[cmd.cmd_id] = cmd

        # Index by device
        self._by_device[device.eq_id].add(cmd.cmd_id)

        # Index by room
        if device.room:
            normalized_room = device.room.lower().strip()
            self._by_room[normalized_room].add(cmd.cmd_id)

        # Index by capability
        if cmd.capability:
            self._by_capability[cmd.capability].add(cmd.cmd_id)

        # Index by all tags
        for tag in cmd.tags:
            self._by_tag[tag].add(cmd.cmd_id)

    def _remove_command_from_indexes(self, cmd_id: CommandID) -> None:
        """Internal: Remove command from all indexes."""
        cmd = self._commands.get(cmd_id)
        if not cmd:
            return

        # Remove from device index
        self._by_device[cmd.device_id].discard(cmd_id)

        # Remove from capability index
        if cmd.capability:
            self._by_capability[cmd.capability].discard(cmd_id)

        # Remove from tag indexes
        for tag in cmd.tags:
            self._by_tag[tag].discard(cmd_id)

        # Remove from room index (requires device lookup)
        device = self._devices.get(cmd.device_id)
        if device and device.room:
            normalized_room = device.room.lower().strip()
            self._by_room[normalized_room].discard(cmd_id)

        # Remove command itself
        del self._commands[cmd_id]

    def get_snapshot(self) -> InventorySnapshot:
        """
        Get immutable snapshot of current inventory.

        Returns:
            InventorySnapshot with copies of device/command dicts
        """
        import time

        return InventorySnapshot(
            devices=dict(self._devices),
            commands=dict(self._commands),
            last_update=int(time.time()),
        )

    def find_by_capability(self, capability: str) -> List[CommandRecord]:
        """Find all commands with a specific capability."""
        cmd_ids = self._by_capability.get(capability, set())
        return [self._commands[cid] for cid in cmd_ids if cid in self._commands]

    def find_by_room(self, room: str) -> List[CommandRecord]:
        """Find all commands in a specific room."""
        normalized_room = room.lower().strip()
        cmd_ids = self._by_room.get(normalized_room, set())
        return [self._commands[cid] for cid in cmd_ids if cid in self._commands]

    def find_by_tags(self, tags: Set[str]) -> List[CommandRecord]:
        """
        Find commands matching ANY of the given tags.

        Args:
            tags: Set of tags to match

        Returns:
            List of matching commands (union of tag matches)
        """
        matching_ids: Set[CommandID] = set()

        for tag in tags:
            matching_ids.update(self._by_tag.get(tag, set()))

        return [self._commands[cid] for cid in matching_ids if cid in self._commands]

    def find_by_device(self, device_id: DeviceID) -> List[CommandRecord]:
        """Find all commands for a specific device."""
        cmd_ids = self._by_device.get(device_id, set())
        return [self._commands[cid] for cid in cmd_ids if cid in self._commands]

    def get_command(self, cmd_id: CommandID) -> CommandRecord | None:
        """Get a specific command by ID."""
        return self._commands.get(cmd_id)

    def get_device(self, device_id: DeviceID) -> DeviceRecord | None:
        """Get a specific device by ID."""
        return self._devices.get(device_id)

    @property
    def device_count(self) -> int:
        """Get total device count."""
        return len(self._devices)

    @property
    def command_count(self) -> int:
        """Get total command count."""
        return len(self._commands)
