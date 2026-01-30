"""
Tests for plugin system foundation.

Tests plugin lifecycle, registry, and health checks.
"""

import pytest
from typing import Dict, Any

from app.plugins.base import AriaPlugin
from app.plugins.tooling_provider import ToolingProvider
from app.plugins.registry import PluginRegistry


class DummyPlugin(AriaPlugin):
    """Simple test plugin for lifecycle testing."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.start_called = False
        self.stop_called = False

    def start(self) -> None:
        self.start_called = True
        self._mark_started()

    def stop(self) -> None:
        self.stop_called = True
        self._mark_stopped()

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "message": "Dummy plugin is fine",
            "details": {"start_called": self.start_called}
        }


class DummyToolingProvider(ToolingProvider):
    """Simple test tooling provider."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.data = config.get("data", {})

    def start(self) -> None:
        self._mark_started()

    def stop(self) -> None:
        self._mark_stopped()

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "message": "Tooling provider OK",
            "details": {"data_keys": list(self.data.keys())}
        }

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "timestamp": 123456
        }

    def get_candidates(self, text: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # Simple keyword matching
        matches = [k for k in self.data.keys() if text.lower() in k.lower()]
        return {
            "candidates": [
                {"id": k, "score": 0.9, "data": self.data[k]}
                for k in matches
            ]
        }


class TestPluginBase:
    """Test AriaPlugin base class."""

    def test_plugin_creation(self):
        plugin = DummyPlugin("test", {"foo": "bar"})
        assert plugin.name == "test"
        assert plugin.config == {"foo": "bar"}
        assert not plugin.is_started

    def test_plugin_lifecycle(self):
        plugin = DummyPlugin("test", {})

        # Start
        plugin.start()
        assert plugin.start_called
        assert plugin.is_started

        # Stop
        plugin.stop()
        assert plugin.stop_called
        assert not plugin.is_started

    def test_plugin_health(self):
        plugin = DummyPlugin("test", {})
        health = plugin.health()
        assert health["status"] == "healthy"
        assert "message" in health
        assert "details" in health

    def test_config_reload(self):
        plugin = DummyPlugin("test", {"a": 1})
        assert plugin.config == {"a": 1}

        plugin.on_config_reload({"a": 2, "b": 3})
        assert plugin.config == {"a": 2, "b": 3}


class TestToolingProvider:
    """Test ToolingProvider interface."""

    def test_tooling_provider_creation(self):
        provider = DummyToolingProvider("inventory", {"data": {"light": "on"}})
        assert provider.name == "inventory"
        assert provider.data == {"light": "on"}

    def test_get_snapshot(self):
        provider = DummyToolingProvider("inventory", {"data": {"a": 1, "b": 2}})
        provider.start()

        snapshot = provider.get_snapshot()
        assert "data" in snapshot
        assert snapshot["data"] == {"a": 1, "b": 2}

    def test_get_candidates(self):
        provider = DummyToolingProvider(
            "inventory",
            {
                "data": {
                    "light_kitchen": {"state": "on"},
                    "light_bedroom": {"state": "off"},
                    "shutter_living": {"position": 50}
                }
            }
        )
        provider.start()

        # Search for "light"
        result = provider.get_candidates("light")
        assert len(result["candidates"]) == 2
        assert all("light" in c["id"] for c in result["candidates"])

        # Search for "shutter"
        result = provider.get_candidates("shutter")
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["id"] == "shutter_living"

        # Search for non-existent
        result = provider.get_candidates("foobar")
        assert len(result["candidates"]) == 0


class TestPluginRegistry:
    """Test PluginRegistry."""

    def test_registry_creation(self):
        registry = PluginRegistry()
        assert registry.plugin_count == 0
        assert registry.plugin_names == []

    def test_register_plugin_class(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy", DummyPlugin)

        # Duplicate registration should fail
        with pytest.raises(ValueError, match="already registered"):
            registry.register_plugin_class("dummy", DummyPlugin)

    def test_create_plugin(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy", DummyPlugin)

        plugin = registry.create_plugin("dummy", {"test": "config"})
        assert isinstance(plugin, DummyPlugin)
        assert plugin.name == "dummy"
        assert plugin.config == {"test": "config"}
        assert registry.plugin_count == 1

    def test_create_unregistered_plugin(self):
        registry = PluginRegistry()

        with pytest.raises(ValueError, match="not registered"):
            registry.create_plugin("nonexistent", {})

    def test_get_plugin(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy", DummyPlugin)
        plugin = registry.create_plugin("dummy", {})

        retrieved = registry.get_plugin("dummy")
        assert retrieved is plugin

        assert registry.get_plugin("nonexistent") is None

    def test_get_plugins_by_type(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy", DummyPlugin)
        registry.register_plugin_class("tooling", DummyToolingProvider)

        registry.create_plugin("dummy", {})
        registry.create_plugin("tooling", {"data": {}})

        # Get all ToolingProviders
        providers = registry.get_plugins_by_type(ToolingProvider)
        assert len(providers) == 1
        assert isinstance(providers[0], DummyToolingProvider)

        # Get all AriaPlugins (should include both)
        all_plugins = registry.get_plugins_by_type(AriaPlugin)
        assert len(all_plugins) == 2

    def test_start_all(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy1", DummyPlugin)
        registry.register_plugin_class("dummy2", DummyPlugin)

        plugin1 = registry.create_plugin("dummy1", {})
        plugin2 = registry.create_plugin("dummy2", {})

        assert not plugin1.is_started
        assert not plugin2.is_started

        registry.start_all()

        assert plugin1.is_started
        assert plugin2.is_started
        assert plugin1.start_called
        assert plugin2.start_called

    def test_stop_all(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy1", DummyPlugin)
        registry.register_plugin_class("dummy2", DummyPlugin)

        plugin1 = registry.create_plugin("dummy1", {})
        plugin2 = registry.create_plugin("dummy2", {})

        registry.start_all()
        registry.stop_all()

        assert not plugin1.is_started
        assert not plugin2.is_started
        assert plugin1.stop_called
        assert plugin2.stop_called

    def test_health_all(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy", DummyPlugin)
        registry.register_plugin_class("tooling", DummyToolingProvider)

        registry.create_plugin("dummy", {})
        registry.create_plugin("tooling", {"data": {}})

        health = registry.health_all()

        assert health["status"] == "healthy"
        assert "plugins" in health
        assert "dummy" in health["plugins"]
        assert "tooling" in health["plugins"]
        assert health["plugins"]["dummy"]["status"] == "healthy"
        assert health["plugins"]["tooling"]["status"] == "healthy"

    def test_reload_config(self):
        registry = PluginRegistry()
        registry.register_plugin_class("dummy", DummyPlugin)

        plugin = registry.create_plugin("dummy", {"a": 1})
        assert plugin.config == {"a": 1}

        registry.reload_config("dummy", {"a": 2, "b": 3})
        assert plugin.config == {"a": 2, "b": 3}

        # Non-existent plugin
        with pytest.raises(ValueError, match="not found"):
            registry.reload_config("nonexistent", {})
