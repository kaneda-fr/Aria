"""
Aria Plugin System

Provides a generic plugin infrastructure for extending Aria's capabilities.
"""

from app.plugins.base import AriaPlugin
from app.plugins.tooling_provider import ToolingProvider
from app.plugins.registry import PluginRegistry

__all__ = [
    "AriaPlugin",
    "ToolingProvider",
    "PluginRegistry",
]
