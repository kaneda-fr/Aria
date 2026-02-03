"""
Base plugin interface for Aria.

All plugins must implement the AriaPlugin lifecycle.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class AriaPlugin(ABC):
    """
    Base class for all Aria plugins.

    Plugins provide a consistent lifecycle:
    - start(): Initialize resources, connect to services
    - stop(): Clean shutdown, release resources
    - health(): Report plugin health status
    - on_config_reload(): React to configuration changes (optional)
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize plugin with name and configuration.

        Args:
            name: Unique plugin identifier
            config: Plugin-specific configuration dict
        """
        self.name = name
        self.config = config
        self._started = False
        self._logger = logging.getLogger(f"aria.plugin.{name}")

    @abstractmethod
    def start(self) -> None:
        """
        Start the plugin.

        Called during server startup. Must be idempotent.
        Should perform:
        - Resource initialization
        - Service connections
        - Cache loading

        Raises:
            Exception: If startup fails (will prevent server boot)
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the plugin.

        Called during server shutdown. Must be idempotent.
        Should perform:
        - Clean disconnection
        - Resource release
        - Cache persistence
        """
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """
        Report plugin health status.

        Returns:
            Dict with health information:
            {
                "status": "healthy" | "degraded" | "unhealthy",
                "message": str,
                "details": dict  # plugin-specific metrics
            }
        """
        pass

    def on_config_reload(self, new_config: Dict[str, Any]) -> None:
        """
        React to configuration changes (optional).

        Args:
            new_config: Updated configuration dict

        Default implementation: Update config, no action.
        Override to implement hot-reload behavior.
        """
        self._logger.info(f"Config reload triggered for {self.name}")
        self.config = new_config

    def _mark_started(self) -> None:
        """Internal: Mark plugin as started."""
        self._started = True
        self._logger.info(f"Plugin {self.name} started")

    def _mark_stopped(self) -> None:
        """Internal: Mark plugin as stopped."""
        self._started = False
        self._logger.info(f"Plugin {self.name} stopped")

    @property
    def is_started(self) -> bool:
        """Check if plugin is currently started."""
        return self._started
