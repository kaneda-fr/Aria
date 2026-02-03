"""
Tooling Provider interface.

Tooling Providers expose structured data to the LLM tooling layer
without defining LLM tool schemas directly.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
from app.plugins.base import AriaPlugin


class ToolingProvider(AriaPlugin):
    """
    Base class for tooling providers.

    Tooling Providers:
    - Produce structured data (not tool schemas)
    - Enable deterministic, local routing
    - Support future providers without LLM contract changes

    Design Principles:
    - Providers MUST NOT define LLM tool schemas
    - Providers MUST NOT call the LLM
    - get_candidates() MUST be deterministic and fast (<10ms)
    - Network providers MUST use caching
    """

    @abstractmethod
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of provider state.

        Returns:
            Dict with provider-specific data structure.
            Format depends on provider type.

        Example (Inventory):
            {
                "devices": [...],
                "commands": [...],
                "last_update": timestamp
            }

        Note:
            Should return immutable snapshot (safe for concurrent reads).
        """
        pass

    @abstractmethod
    def get_candidates(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get candidates matching user input.

        This is the HOT PATH for LLM tool execution.
        Must be:
        - Deterministic (same input → same output)
        - Fast (<10ms target)
        - Local (no network calls, no LLM calls)

        Args:
            text: User input text (normalized)
            context: Optional context (speaker, location, history)

        Returns:
            Dict with candidates:
            {
                "candidates": [
                    {
                        "id": str,
                        "score": float,
                        "data": dict  # provider-specific
                    }
                ],
                "debug": dict  # optional debug info
            }

        Example (Inventory):
            {
                "candidates": [
                    {
                        "id": "5450",
                        "score": 0.95,
                        "data": {
                            "cmd_id": "5450",
                            "device_name": "Volets Salle a Manger",
                            "capability": "flap_slider"
                        }
                    }
                ]
            }
        """
        pass
