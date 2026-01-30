"""
Plugin registry and loader.

Manages plugin lifecycle and provides centralized access.
"""

from typing import Dict, List, Type, Optional, Any
import logging
from app.plugins.base import AriaPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for all Aria plugins.

    Responsibilities:
    - Plugin registration and discovery
    - Lifecycle management (start/stop all)
    - Health aggregation
    - Plugin lookup by name or type
    """

    def __init__(self):
        self._plugins: Dict[str, AriaPlugin] = {}
        self._plugin_classes: Dict[str, Type[AriaPlugin]] = {}

    def register_plugin_class(
        self,
        name: str,
        plugin_class: Type[AriaPlugin]
    ) -> None:
        """
        Register a plugin class for later instantiation.

        Args:
            name: Unique plugin identifier
            plugin_class: Plugin class (not instance)

        Raises:
            ValueError: If plugin name already registered
        """
        if name in self._plugin_classes:
            raise ValueError(f"Plugin class '{name}' already registered")

        self._plugin_classes[name] = plugin_class
        logger.info(f"Registered plugin class: {name}")

    def create_plugin(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> AriaPlugin:
        """
        Create and register a plugin instance.

        Args:
            name: Plugin name (must be registered)
            config: Plugin configuration

        Returns:
            Created plugin instance

        Raises:
            ValueError: If plugin class not registered
            Exception: If plugin instantiation fails
        """
        if name not in self._plugin_classes:
            raise ValueError(
                f"Plugin class '{name}' not registered. "
                f"Available: {list(self._plugin_classes.keys())}"
            )

        plugin_class = self._plugin_classes[name]

        try:
            plugin = plugin_class(name=name, config=config)
            self._plugins[name] = plugin
            logger.info(f"Created plugin instance: {name}")
            return plugin

        except Exception as e:
            logger.error(f"Failed to create plugin '{name}': {e}")
            raise

    def get_plugin(self, name: str) -> Optional[AriaPlugin]:
        """
        Get plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)

    def get_plugins_by_type(
        self,
        plugin_type: Type[AriaPlugin]
    ) -> List[AriaPlugin]:
        """
        Get all plugins of a specific type.

        Args:
            plugin_type: Plugin class/interface

        Returns:
            List of matching plugins
        """
        return [
            plugin for plugin in self._plugins.values()
            if isinstance(plugin, plugin_type)
        ]

    def start_all(self) -> None:
        """
        Start all registered plugins.

        Plugins are started in registration order.
        If any plugin fails to start, raises exception and stops.

        Raises:
            Exception: If any plugin fails to start
        """
        logger.info(f"Starting {len(self._plugins)} plugins...")

        for name, plugin in self._plugins.items():
            try:
                logger.info(f"Starting plugin: {name}")
                plugin.start()
                plugin._mark_started()

            except Exception as e:
                logger.error(f"Failed to start plugin '{name}': {e}")
                # Stop already-started plugins
                self._stop_started_plugins()
                raise

        logger.info("All plugins started successfully")

    def stop_all(self) -> None:
        """
        Stop all registered plugins.

        Plugins are stopped in reverse registration order.
        Continues stopping even if individual plugins fail.
        """
        logger.info(f"Stopping {len(self._plugins)} plugins...")

        # Stop in reverse order
        for name, plugin in reversed(list(self._plugins.items())):
            try:
                if plugin.is_started:
                    logger.info(f"Stopping plugin: {name}")
                    plugin.stop()
                    plugin._mark_stopped()

            except Exception as e:
                logger.error(f"Error stopping plugin '{name}': {e}")
                # Continue stopping other plugins

        logger.info("All plugins stopped")

    def _stop_started_plugins(self) -> None:
        """Internal: Stop all currently started plugins (cleanup on error)."""
        for name, plugin in self._plugins.items():
            if plugin.is_started:
                try:
                    logger.warning(f"Cleanup: stopping plugin {name}")
                    plugin.stop()
                    plugin._mark_stopped()
                except Exception as e:
                    logger.error(f"Error during cleanup of '{name}': {e}")

    def health_all(self) -> Dict[str, Any]:
        """
        Aggregate health status from all plugins.

        Returns:
            Dict with overall health and per-plugin status:
            {
                "status": "healthy" | "degraded" | "unhealthy",
                "plugins": {
                    "plugin_name": {
                        "status": str,
                        "message": str,
                        "details": dict
                    }
                }
            }
        """
        plugin_health = {}
        overall_status = "healthy"

        for name, plugin in self._plugins.items():
            try:
                health = plugin.health()
                plugin_health[name] = health

                # Aggregate status (worst wins)
                if health["status"] == "unhealthy":
                    overall_status = "unhealthy"
                elif health["status"] == "degraded" and overall_status != "unhealthy":
                    overall_status = "degraded"

            except Exception as e:
                logger.error(f"Health check failed for '{name}': {e}")
                plugin_health[name] = {
                    "status": "unhealthy",
                    "message": f"Health check error: {e}",
                    "details": {}
                }
                overall_status = "unhealthy"

        return {
            "status": overall_status,
            "plugins": plugin_health
        }

    def reload_config(self, plugin_name: str, new_config: Dict[str, Any]) -> None:
        """
        Reload configuration for a specific plugin.

        Args:
            plugin_name: Plugin to reload
            new_config: New configuration

        Raises:
            ValueError: If plugin not found
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        logger.info(f"Reloading config for plugin: {plugin_name}")
        plugin.on_config_reload(new_config)

    @property
    def plugin_count(self) -> int:
        """Get number of registered plugins."""
        return len(self._plugins)

    @property
    def plugin_names(self) -> List[str]:
        """Get list of registered plugin names."""
        return list(self._plugins.keys())
