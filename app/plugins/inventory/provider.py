"""
Inventory Provider - main plugin implementation.

Coordinates sources, indexer, and provides ToolingProvider interface.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.plugins.tooling_provider import ToolingProvider
from app.plugins.inventory.indexer import InventoryIndexer
from app.plugins.inventory.sources.base import InventorySource
from app.plugins.inventory.sources.jeedom_mqtt import JeedomMQTTSource
from app.plugins.inventory.models import DeviceRecord, CommandState

logger = logging.getLogger(__name__)


class InventoryProvider(ToolingProvider):
    """
    Inventory Provider plugin.

    Manages device discovery, state tracking, and candidate selection.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        self.indexer = InventoryIndexer()
        self.sources: Dict[str, InventorySource] = {}
        self.cache_path: Optional[Path] = None

        # Command state tracking (cmd_id → latest state)
        self._command_states: Dict[str, CommandState] = {}

    def start(self) -> None:
        """Start inventory provider."""
        logger.info("Starting InventoryProvider")

        # Configure cache path
        cache_path_str = self.config.get("cache_path", "/tmp/aria_inventory.json")
        self.cache_path = Path(cache_path_str)

        # Load from cache if exists
        self._load_from_cache()

        # Initialize sources
        source_configs = self.config.get("sources", {})
        for source_name, source_config in source_configs.items():
            self._start_source(source_name, source_config)

        self._mark_started()
        logger.info(f"InventoryProvider started with {len(self.sources)} sources")

    def stop(self) -> None:
        """Stop inventory provider."""
        logger.info("Stopping InventoryProvider")

        # Stop all sources
        for source_name, source in list(self.sources.items()):
            try:
                logger.info(f"Stopping source: {source_name}")
                source.stop()
            except Exception as e:
                from app.aria_logging import get_logger
                log = get_logger("ARIA.Inventory")
                log.error("ARIA.Inventory.SourceStopError", extra={"fields": {
                    "source": source_name,
                    "error": str(e)
                }})

        # Save to cache
        self._save_to_cache()

        self._mark_stopped()
        logger.info("InventoryProvider stopped")

    def health(self) -> Dict[str, Any]:
        """Report health status."""
        status = "healthy"
        details = {
            "device_count": self.indexer.device_count,
            "command_count": self.indexer.command_count,
            "sources": {},
        }

        # Check source health
        for source_name, source in self.sources.items():
            if isinstance(source, JeedomMQTTSource):
                source_healthy = source.is_connected
                details["sources"][source_name] = {
                    "connected": source_healthy,
                }
                if not source_healthy:
                    status = "degraded"
            else:
                details["sources"][source_name] = {"status": "unknown"}

        return {
            "status": status,
            "message": f"{self.indexer.device_count} devices, {self.indexer.command_count} commands",
            "details": details,
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete inventory snapshot."""
        snapshot = self.indexer.get_snapshot()

        return {
            "devices": {eq_id: self._serialize_device(device) for eq_id, device in snapshot.devices.items()},
            "commands": {cmd_id: self._serialize_command(cmd) for cmd_id, cmd in snapshot.commands.items()},
            "states": {cmd_id: self._serialize_state(state) for cmd_id, state in self._command_states.items()},
            "last_update": snapshot.last_update,
            "device_count": snapshot.device_count,
            "command_count": snapshot.command_count,
        }

    def get_candidates(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get command candidates matching user input.

        This is the HOT PATH for LLM tool execution (<10ms target).

        Args:
            text: User input (e.g., "turn on light in kitchen")
            context: Optional context (room, speaker, etc.)

        Returns:
            Dict with candidates and scoring info
        """
        # For MVP, implement simple keyword matching
        # Future: Use semantic embeddings, ML models, etc.

        normalized_text = text.lower().strip()
        words = set(normalized_text.split())

        # Extract potential capability keywords
        # Map keywords to domain prefixes (broader matching)
        # Action keywords (on/off) map to domain "light_" to include all light capabilities
        capability_keywords = {
            # Domain keywords
            "light": "light_",
            "lumiere": "light_",
            "shutter": "flap_",
            "volet": "flap_",
            "blind": "flap_",

            # Action keywords - map to domain to find all device types
            "on": "light_",
            "allume": "light_",
            "off": "light_",
            "eteins": "light_",

            # Specific capability keywords (for more precise matching)
            "dim": "light_slider",
            "brighten": "light_slider",
            "dimmer": "light_slider",
            "open": "flap_up",
            "ouvre": "flap_up",
            "close": "flap_down",
            "ferme": "flap_down",
        }

        # Match capabilities
        matched_caps = set()
        for keyword, cap_prefix in capability_keywords.items():
            if keyword in words or keyword in normalized_text:
                matched_caps.add(cap_prefix)

        # Find commands by capability
        candidates = []
        for cap_prefix in matched_caps:
            for capability in self.indexer._by_capability.keys():
                if capability.startswith(cap_prefix):
                    cmds = self.indexer.find_by_capability(capability)
                    for cmd in cmds:
                        if cmd.type == "action":  # Only action commands
                            device = self.indexer.get_device(cmd.device_id)
                            if device:
                                score = self._score_candidate(cmd, device, normalized_text, context)
                                if score > 0:
                                    candidates.append({
                                        "id": cmd.cmd_id,
                                        "score": score,
                                        "data": {
                                            "cmd_id": cmd.cmd_id,
                                            "device_name": device.name,
                                            "room": device.room,
                                            "capability": cmd.capability,
                                            "type": cmd.type,
                                            "subType": cmd.subType,
                                        }
                                    })

        # Sort by score (descending)
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Limit to top 10
        candidates = candidates[:10]

        return {
            "candidates": candidates,
            "debug": {
                "text": normalized_text,
                "matched_capabilities": list(matched_caps),
                "total_candidates": len(candidates),
            }
        }

    def _score_candidate(
        self,
        cmd: Any,
        device: DeviceRecord,
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Score a command candidate.

        Word-level matching with tiered scoring:
        - Exact full name match: highest score
        - Multiple word matches: high score
        - Single word match: medium score
        - Room matches: bonus points
        """
        score = 0.0  # Start from 0

        # Normalize and tokenize
        text_words = set(text.lower().split())
        device_name_lower = device.name.lower()
        device_words = set(device_name_lower.split())
        room_lower = device.room.lower() if device.room else ""

        # 1. Exact device name match (substring in either direction)
        if device_name_lower in text or text in device_name_lower:
            score += 1.0
        else:
            # 2. Word-level matching
            matched_words = text_words & device_words
            if matched_words:
                # Base score for having any match
                score += 0.3

                # Score based on proportion of device words matched
                match_ratio = len(matched_words) / len(device_words) if device_words else 0
                score += match_ratio * 0.4  # Up to +0.4 for matching all device words

                # Bonus for number of matched words (more matches = better)
                word_count_bonus = min(len(matched_words) * 0.1, 0.2)
                score += word_count_bonus
            else:
                # No match at all - very low score
                score = 0.1

        # 3. Room matching bonuses
        if room_lower:
            # Exact room name in query
            if room_lower in text:
                score += 0.2
            # Room name as a word in query
            elif room_lower in text_words:
                score += 0.25

            # Room context match (from find_devices context parameter)
            if context and "room" in context:
                if room_lower == context["room"].lower():
                    score += 0.15

        # Cap at 1.0
        return min(score, 1.0)

    def _start_source(self, source_name: str, source_config: Dict[str, Any]) -> None:
        """Initialize and start a source."""
        source_type = source_config.get("type", "jeedom_mqtt")

        if source_type == "jeedom_mqtt":
            source = JeedomMQTTSource(source_name, source_config)
        else:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.UnknownSourceType", extra={"fields": {
                "source_name": source_name,
                "source_type": source_type
            }})
            return

        # Set callbacks
        source.set_device_callback(self._on_devices_discovered)
        source.set_state_callback(self._on_state_update)

        # Start source
        try:
            source.start()
            self.sources[source_name] = source
            logger.info(f"Started source: {source_name}")
        except Exception as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.SourceStartError", extra={"fields": {
                "source": source_name,
                "error": str(e)
            }})
            raise

    def _on_devices_discovered(self, devices: List[DeviceRecord]) -> None:
        """Handle discovered devices from sources."""
        logger.info(f"Processing {len(devices)} discovered devices")

        for device in devices:
            self.indexer.add_device(device)

        logger.info(
            f"Inventory updated: {self.indexer.device_count} devices, "
            f"{self.indexer.command_count} commands"
        )

    def _on_state_update(self, state: CommandState) -> None:
        """Handle command state update."""
        self._command_states[state.cmd_id] = state
        logger.debug(f"State updated: {state.cmd_id} = {state.value}")

    def _load_from_cache(self) -> None:
        """Load inventory from cache file."""
        if not self.cache_path or not self.cache_path.exists():
            logger.info("No cache file found, starting fresh")
            return

        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)

            # Deserialize devices
            devices = [self._deserialize_device(d) for d in data.get("devices", [])]

            # Rebuild indexes
            self.indexer.rebuild(devices)

            logger.info(
                f"Loaded from cache: {self.indexer.device_count} devices, "
                f"{self.indexer.command_count} commands"
            )

        except Exception as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.CacheLoadError", extra={"fields": {
                "cache_path": str(self.cache_path),
                "error": str(e)
            }})

    def _save_to_cache(self) -> None:
        """Save inventory to cache file."""
        if not self.cache_path:
            return

        try:
            snapshot = self.indexer.get_snapshot()

            data = {
                "devices": [self._serialize_device(d) for d in snapshot.devices.values()],
                "last_update": snapshot.last_update,
            }

            # Ensure parent directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved to cache: {self.cache_path}")

        except Exception as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.CacheSaveError", extra={"fields": {
                "cache_path": str(self.cache_path),
                "error": str(e)
            }})

    def _serialize_device(self, device: DeviceRecord) -> Dict[str, Any]:
        """Serialize DeviceRecord to JSON-compatible dict."""
        return {
            "source": device.source,
            "eq_id": device.eq_id,
            "name": device.name,
            "room": device.room,
            "domain": device.domain,
            "enabled": device.enabled,
            "visible": device.visible,
            "commands": [self._serialize_command(cmd) for cmd in device.commands],
            "tags": list(device.tags),
            "seen_at": device.seen_at,
            "manufacturer": device.manufacturer,
            "model": device.model,
            "firmware": device.firmware,
        }

    def _serialize_command(self, cmd: Any) -> Dict[str, Any]:
        """Serialize CommandRecord to JSON-compatible dict."""
        return {
            "cmd_id": cmd.cmd_id,
            "device_id": cmd.device_id,
            "type": cmd.type,
            "subType": cmd.subType,
            "capability": cmd.capability,
            "unit": cmd.unit,
            "range": list(cmd.range) if cmd.range else None,
            "tags": list(cmd.tags),
            "execution": cmd.execution,
            "name": cmd.name,
            "visible": cmd.visible,
        }

    def _serialize_state(self, state: CommandState) -> Dict[str, Any]:
        """Serialize CommandState to JSON-compatible dict."""
        return {
            "cmd_id": state.cmd_id,
            "value": state.value,
            "unit": state.unit,
            "timestamp": state.timestamp,
        }

    def _deserialize_device(self, data: Dict[str, Any]) -> DeviceRecord:
        """Deserialize dict to DeviceRecord."""
        from app.plugins.inventory.models import CommandRecord

        commands = [self._deserialize_command(cmd_data) for cmd_data in data.get("commands", [])]

        return DeviceRecord(
            source=data["source"],
            eq_id=data["eq_id"],
            name=data["name"],
            room=data["room"],
            domain=data["domain"],
            enabled=data["enabled"],
            visible=data["visible"],
            commands=commands,
            tags=set(data.get("tags", [])),
            seen_at=data.get("seen_at", 0),
            manufacturer=data.get("manufacturer"),
            model=data.get("model"),
            firmware=data.get("firmware"),
        )

    def _deserialize_command(self, data: Dict[str, Any]) -> Any:
        """Deserialize dict to CommandRecord."""
        from app.plugins.inventory.models import CommandRecord

        value_range = tuple(data["range"]) if data.get("range") else None

        return CommandRecord(
            cmd_id=data["cmd_id"],
            device_id=data["device_id"],
            type=data["type"],
            subType=data["subType"],
            capability=data.get("capability"),
            unit=data.get("unit"),
            range=value_range,
            tags=set(data.get("tags", [])),
            execution=data.get("execution"),
            name=data.get("name", ""),
            visible=data.get("visible", True),
        )
