"""
Base interface for inventory sources.

Sources ingest discovery data and normalize it into DeviceRecords.
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Any
from app.plugins.inventory.models import DeviceRecord, CommandState


class InventorySource(ABC):
    """
    Base class for inventory sources.

    Sources are responsible for:
    - Connecting to external systems (MQTT, HTTP, etc.)
    - Ingesting discovery data
    - Normalizing into DeviceRecord format
    - Providing state updates via callbacks
    """

    def __init__(self, name: str, config: dict):
        """
        Initialize source.

        Args:
            name: Source identifier
            config: Source-specific configuration
        """
        self.name = name
        self.config = config
        self._device_callback: Callable[[List[DeviceRecord]], None] = lambda devices: None
        self._state_callback: Callable[[CommandState], None] = lambda state: None

    @abstractmethod
    def start(self) -> None:
        """
        Start the source.

        Should:
        - Establish connections (MQTT, HTTP, etc.)
        - Subscribe to relevant topics/endpoints
        - Trigger initial discovery

        Raises:
            Exception: If source fails to start
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the source.

        Should:
        - Close connections gracefully
        - Unsubscribe from topics
        - Release resources
        """
        pass

    def set_device_callback(self, callback: Callable[[List[DeviceRecord]], None]) -> None:
        """
        Set callback for device discovery/updates.

        Args:
            callback: Function called with list of DeviceRecords
        """
        self._device_callback = callback

    def set_state_callback(self, callback: Callable[[CommandState], None]) -> None:
        """
        Set callback for state updates.

        Args:
            callback: Function called with CommandState
        """
        self._state_callback = callback

    def _on_devices_discovered(self, devices: List[DeviceRecord]) -> None:
        """Internal: Notify about discovered devices."""
        self._device_callback(devices)

    def _on_state_update(self, state: CommandState) -> None:
        """Internal: Notify about state update."""
        self._state_callback(state)
