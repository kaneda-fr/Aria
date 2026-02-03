"""
Jeedom → Aria normalization logic.

Transforms Jeedom discovery JSON into DeviceRecord/CommandRecord models.
"""

import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from app.plugins.inventory.models import DeviceRecord, CommandRecord, CommandState

logger = logging.getLogger(__name__)


class JeedomNormalizer:
    """
    Normalizes Jeedom MQTT discovery messages.

    Based on analysis in docs/jeedom_mqtt_analysis.md
    """

    # Capability mapping: Jeedom generic_type → Aria capability
    CAPABILITY_MAP = {
        # Lighting
        "LIGHT_SLIDER": "light_slider",
        "LIGHT_STATE": "light_state",
        "LIGHT_ON": "light_on",
        "LIGHT_OFF": "light_off",

        # Shutters/Blinds
        "FLAP_SLIDER": "flap_slider",
        "FLAP_STATE": "flap_state",
        "FLAP_UP": "flap_up",
        "FLAP_DOWN": "flap_down",
        "FLAP_STOP": "flap_stop",

        # Energy
        "POWER": "power",
        "CONSUMPTION": "consumption",

        # Generic
        "GENERIC_INFO": "generic_info",
    }

    # Domain mapping: Jeedom category → Aria domain
    DOMAIN_MAP = {
        "light": "light",
        "opening": "opening",
        "heating": "heating",
        "energy": "energy",
        "security": "security",
        "multimedia": "multimedia",
    }

    @classmethod
    def normalize_device(cls, raw: Dict[str, Any], source: str = "jeedom_mqtt") -> Optional[DeviceRecord]:
        """
        Normalize a Jeedom eqLogic discovery message.

        Args:
            raw: Raw Jeedom discovery JSON
            source: Source identifier

        Returns:
            DeviceRecord or None if invalid/filtered
        """
        # Debug: Check if raw is actually a dict
        if not isinstance(raw, dict):
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.InvalidPayload", extra={"fields": {
                "expected": "dict",
                "got": type(raw).__name__,
                "preview": str(raw)[:200]
            }})
            return None

        try:
            # Extract basic fields
            eq_id = str(raw.get("id", ""))
            name = raw.get("name", "Unknown")
            room = raw.get("object_name", "")
            enabled = raw.get("isEnable") == "1"
            visible = raw.get("isVisible") == "1"

            # Skip disabled devices
            if not enabled:
                return None

            # Determine primary domain from categories
            categories = raw.get("category", {})
            if not isinstance(categories, dict):
                categories = {}
            domain = cls._extract_domain(categories)

            # Extract last communication timestamp
            status = raw.get("status", {})
            if not isinstance(status, dict):
                status = {}
            last_comm_str = status.get("lastCommunication", "")
            seen_at = cls._parse_timestamp(last_comm_str)

            # Extract metadata
            config = raw.get("configuration", {})
            if not isinstance(config, dict):
                config = {}
            manufacturer_id = config.get("manufacturer_id")
            product_name = config.get("product_name")
            firmware = config.get("firmwareVersion")

            # Normalize commands
            commands = []
            raw_commands = raw.get("cmds", {})

            # Handle both dict and list formats
            if isinstance(raw_commands, dict):
                # Dict format: {cmd_id: cmd_data, ...}
                for cmd_id, cmd_data in raw_commands.items():
                    cmd = cls.normalize_command(cmd_data, eq_id)
                    if cmd:
                        commands.append(cmd)
            elif isinstance(raw_commands, list):
                # List format: [{...}, {...}]
                for cmd_data in raw_commands:
                    if isinstance(cmd_data, dict):
                        cmd = cls.normalize_command(cmd_data, eq_id)
                        if cmd:
                            commands.append(cmd)

            # Build device-level tags
            tags = cls._build_device_tags(raw)

            return DeviceRecord(
                source=source,
                eq_id=eq_id,
                name=name,
                room=room,
                domain=domain,
                enabled=enabled,
                visible=visible,
                commands=commands,
                tags=tags,
                seen_at=seen_at,
                manufacturer=str(manufacturer_id) if manufacturer_id else None,
                model=product_name,
                firmware=firmware,
            )

        except Exception as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")

            logger.error(
                f"Failed to normalize device: {e}",
                extra={
                    "raw_type": type(raw).__name__,
                    "raw_preview": str(raw)[:200] if raw else "None"
                }
            )
            log.error("ARIA.Inventory.NormalizationError", extra={"fields": {
                "error": str(e),
                "raw_type": type(raw).__name__,
                "raw_preview": str(raw)[:200] if raw else "None"
            }})
            return None

    @classmethod
    def normalize_command(cls, raw: Dict[str, Any], device_id: str) -> Optional[CommandRecord]:
        """
        Normalize a Jeedom command.

        Args:
            raw: Raw command JSON
            device_id: Parent device ID

        Returns:
            CommandRecord or None if filtered
        """
        try:
            cmd_id = str(raw.get("id", ""))
            cmd_type = raw.get("type", "")
            sub_type = raw.get("subType", "")
            generic_type = raw.get("generic_type")
            name = raw.get("name", "")
            unit = raw.get("unite") or None
            visible = raw.get("isVisible") == "1"

            # Map capability
            capability = None
            if generic_type and generic_type in cls.CAPABILITY_MAP:
                capability = cls.CAPABILITY_MAP[generic_type]

            # Skip unmapped/technical commands
            if not capability and generic_type in {None, "DONT", "GENERIC_INFO"}:
                # Skip technical commands (pingNode, healNode, etc.)
                logical_id = raw.get("logicalId")
                if logical_id and isinstance(logical_id, str) and logical_id.startswith("0-0-"):
                    return None

            # Extract range
            cmd_config = raw.get("configuration", {})
            if not isinstance(cmd_config, dict):
                cmd_config = {}
            value_range = cls._extract_range(cmd_config)

            # Build tags
            tags = cls._build_command_tags(raw, capability)

            # Build execution spec (for action commands)
            execution = None
            if cmd_type == "action":
                execution = cls._build_execution_spec(raw)

            return CommandRecord(
                cmd_id=cmd_id,
                device_id=device_id,
                type=cmd_type,
                subType=sub_type,
                capability=capability,
                unit=unit,
                range=value_range,
                tags=tags,
                execution=execution,
                name=name,
                visible=visible,
            )

        except Exception as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.CommandNormalizationError", extra={"fields": {
                "error": str(e),
                "raw_preview": str(raw)[:200] if raw else "None"
            }})
            return None

    @classmethod
    def normalize_event(cls, cmd_id: str, raw: Dict[str, Any]) -> Optional[CommandState]:
        """
        Normalize a Jeedom event.

        Args:
            cmd_id: Command ID (from topic)
            raw: Event payload JSON

        Returns:
            CommandState or None if invalid
        """
        try:
            import time

            value = raw.get("value")
            unit = raw.get("unite")
            timestamp = int(time.time())

            return CommandState(
                cmd_id=cmd_id,
                value=value,
                unit=unit,
                timestamp=timestamp,
            )

        except Exception as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.EventNormalizationError", extra={"fields": {
                "cmd_id": cmd_id,
                "error": str(e)
            }})
            return None

    @classmethod
    def _extract_domain(cls, categories: Dict[str, str]) -> str:
        """Extract primary domain from category flags."""
        for cat_name, enabled in categories.items():
            if enabled == "1" and cat_name in cls.DOMAIN_MAP:
                return cls.DOMAIN_MAP[cat_name]
        return "generic"

    @classmethod
    def _extract_range(cls, config: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Extract value range from configuration."""
        min_val = config.get("minValue")
        max_val = config.get("maxValue")

        if min_val is not None and max_val is not None:
            try:
                return (float(min_val), float(max_val))
            except (ValueError, TypeError):
                pass

        return None

    @classmethod
    def _build_device_tags(cls, raw: Dict[str, Any]) -> Set[str]:
        """Build device-level tags."""
        tags = set()

        # Domain tags
        categories = raw.get("category", {})
        if isinstance(categories, dict):
            for cat_name, enabled in categories.items():
                if enabled == "1":
                    tags.add(f"domain:{cat_name}")

        # Backend tag
        eq_type = raw.get("eqType_name")
        if eq_type:
            tags.add(f"backend:{eq_type}")

        # Manufacturer tag
        config = raw.get("configuration", {})
        if isinstance(config, dict):
            manufacturer_id = config.get("manufacturer_id")
            if manufacturer_id:
                tags.add(f"manufacturer:{manufacturer_id}")

        return tags

    @classmethod
    def _build_command_tags(cls, raw: Dict[str, Any], capability: Optional[str]) -> Set[str]:
        """Build command tags."""
        tags = set()

        # Capability tag (mandatory for known capabilities)
        if capability:
            tags.add(f"cap:{capability}")

        # Type tags
        cmd_type = raw.get("type")
        if cmd_type:
            tags.add(f"type:{cmd_type}")

        sub_type = raw.get("subType")
        if sub_type:
            tags.add(f"subtype:{sub_type}")

        return tags

    @classmethod
    def _build_execution_spec(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Build execution spec for action commands."""
        cmd_id = str(raw.get("id"))
        sub_type = raw.get("subType", "other")
        config = raw.get("configuration", {})
        if not isinstance(config, dict):
            config = {}

        # Determine payload template
        payload_template = sub_type if sub_type in {"slider", "color", "message", "select"} else "other"

        # Observer linkage
        observer_cmd_id = raw.get("value") or None

        # Value requirements
        value_required = sub_type == "slider"

        value_type_map = {
            "slider": "int",
            "numeric": "float",
            "binary": "bool",
            "string": "str",
        }
        value_type = value_type_map.get(sub_type)

        # Range
        value_range = cls._extract_range(config)

        return {
            "backend": "jeedom_mqtt",
            "args": {
                "cmd_id": cmd_id,
                "value_required": value_required,
                "value_type": value_type,
                "range": value_range,
                "payload_template": payload_template,
                "observer_cmd_id": observer_cmd_id,
            }
        }

    @classmethod
    def _parse_timestamp(cls, timestamp_str: str) -> int:
        """
        Parse Jeedom timestamp string to UNIX timestamp.

        Format: "2026-01-30 10:15:20"
        """
        if not timestamp_str:
            return 0

        try:
            from datetime import datetime
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            return int(dt.timestamp())
        except Exception:
            return 0
