"""
Actions Provider - command execution plugin.

Executes commands on devices via MQTT.
"""

import json
import logging
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt

from app.plugins.base import AriaPlugin

logger = logging.getLogger(__name__)


class ActionsProvider(AriaPlugin):
    """
    Actions Provider plugin.

    Executes commands on devices via Jeedom MQTT2 protocol.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        self.host = config.get("host", "localhost")
        self.port = config.get("port", 1883)
        self.username = config.get("username")
        self.password = config.get("password")
        self.root_topic = config.get("root_topic", "jeedom")
        self.qos = config.get("qos", 1)

        self._client: mqtt.Client | None = None
        self._connected = False

    def start(self) -> None:
        """Start MQTT client for command execution."""
        from app.aria_logging import get_logger
        log = get_logger("ARIA.Actions")

        logger.info(f"Starting Actions Provider: {self.host}:{self.port}")
        log.info("ARIA.Actions.Connecting", extra={"fields": {"host": self.host, "port": self.port}})

        # Create MQTT client
        self._client = mqtt.Client(client_id=f"aria-actions-{self.name}")

        # Set credentials if provided
        if self.username:
            self._client.username_pw_set(self.username, self.password)

        # Set callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Connect
        try:
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            logger.info("Actions MQTT client started")

        except Exception as e:
            log.error("ARIA.Actions.ConnectionFailed", extra={"fields": {
                "host": self.host,
                "port": self.port,
                "error": str(e)
            }})
            raise

        self._mark_started()

    def stop(self) -> None:
        """Stop MQTT client."""
        logger.info("Stopping Actions Provider")

        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

        self._connected = False
        self._mark_stopped()
        logger.info("Actions Provider stopped")

    def health(self) -> Dict[str, Any]:
        """Report health status."""
        status = "healthy" if self._connected else "degraded"
        return {
            "status": status,
            "message": "Connected to MQTT broker" if self._connected else "Not connected",
            "details": {
                "host": self.host,
                "port": self.port,
                "connected": self._connected,
            }
        }

    def execute_command(
        self,
        cmd_id: str,
        value: Optional[Any] = None,
        execution_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a command.

        Args:
            cmd_id: Command ID to execute
            value: Optional value for the command
            execution_spec: Optional execution spec (from inventory)

        Returns:
            Dict with execution result:
            {
                "success": bool,
                "cmd_id": str,
                "message": str,
                "error": str | None
            }
        """
        from app.aria_logging import get_logger
        log = get_logger("ARIA.Actions")

        if not self._connected:
            log.error("ARIA.Actions.NotConnected", extra={"fields": {"cmd_id": cmd_id}})
            return {
                "success": False,
                "cmd_id": cmd_id,
                "message": "Not connected to MQTT broker",
                "error": "not_connected"
            }

        # Validate parameters if execution spec provided
        if execution_spec:
            validation_error = self._validate_parameters(cmd_id, value, execution_spec)
            if validation_error:
                log.error("ARIA.Actions.ValidationFailed", extra={"fields": {
                    "cmd_id": cmd_id,
                    "error": validation_error
                }})
                return {
                    "success": False,
                    "cmd_id": cmd_id,
                    "message": f"Validation failed: {validation_error}",
                    "error": "validation_failed"
                }

        # Build MQTT payload
        payload = self._build_payload(cmd_id, value, execution_spec)

        # Publish command (Jeedom MQTT2 protocol)
        topic = f"{self.root_topic}/cmd/set/{cmd_id}"

        # ALWAYS log MQTT messages being sent (for debugging and monitoring)
        log.info("ARIA.Actions.Publishing", extra={"fields": {
            "topic": topic,
            "payload": payload,
            "qos": self.qos,
            "cmd_id": cmd_id,
            "value": value
        }})

        try:
            result = self._client.publish(topic, payload, qos=self.qos)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                log.info("ARIA.Actions.Published", extra={"fields": {
                    "cmd_id": cmd_id,
                    "value": value,
                    "topic": topic,
                    "payload": payload,
                    "mid": result.mid
                }})
                return {
                    "success": True,
                    "cmd_id": cmd_id,
                    "message": "Command published successfully",
                    "error": None
                }
            else:
                error_msg = mqtt.error_string(result.rc)
                log.error("ARIA.Actions.PublishFailed", extra={"fields": {
                    "cmd_id": cmd_id,
                    "error": error_msg,
                    "return_code": result.rc,
                    "topic": topic,
                    "payload": payload
                }})
                return {
                    "success": False,
                    "cmd_id": cmd_id,
                    "message": f"Publish failed: {error_msg}",
                    "error": "publish_failed"
                }

        except Exception as e:
            log.error("ARIA.Actions.ExecutionError", extra={"fields": {
                "cmd_id": cmd_id,
                "error": str(e),
                "topic": topic,
                "payload": payload
            }})
            return {
                "success": False,
                "cmd_id": cmd_id,
                "message": f"Execution error: {str(e)}",
                "error": "execution_error"
            }

    def _validate_parameters(
        self,
        cmd_id: str,
        value: Optional[Any],
        execution_spec: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate command parameters.

        Returns:
            Error message if validation fails, None if valid
        """
        args = execution_spec.get("args", {})

        # Check if value is required
        value_required = args.get("value_required", False)
        if value_required and value is None:
            return "Value is required for this command"

        # Check value type if specified
        if value is not None:
            value_type = args.get("value_type")
            if value_type:
                type_map = {
                    "int": int,
                    "float": (int, float),
                    "bool": bool,
                    "str": str,
                }
                expected_type = type_map.get(value_type)
                if expected_type and not isinstance(value, expected_type):
                    return f"Value must be of type {value_type}"

        # Check range if specified
        if value is not None and isinstance(value, (int, float)):
            value_range = args.get("range")
            if value_range and isinstance(value_range, (list, tuple)) and len(value_range) == 2:
                min_val, max_val = value_range
                if value < min_val or value > max_val:
                    return f"Value must be between {min_val} and {max_val}"

        return None

    def _build_payload(
        self,
        cmd_id: str,
        value: Optional[Any],
        execution_spec: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build MQTT payload for command execution.

        Jeedom MQTT2 protocol (from docs/jeedom_mqtt_command_execution.md):
        - Slider (dimmers, shutters): {"slider": X}
        - Select (list commands): {"select": X}
        - Message (notifications): {"title": "...", "message": "..."}
        - Color (RGB lights): {"color": "#rrggbb"}
        - Binary/other (on/off, up/down): {} (empty JSON) or empty string
        """
        # Get payload template from execution spec if available
        payload_template = None
        if execution_spec:
            args = execution_spec.get("args", {})
            payload_template = args.get("payload_template")

        # Build payload based on template
        if payload_template == "slider":
            # Slider commands (dimmers, shutters, etc.)
            if value is not None:
                return json.dumps({"slider": int(value)})
            else:
                return json.dumps({})

        elif payload_template == "select":
            # Select/list commands
            if value is not None:
                return json.dumps({"select": value})
            else:
                return json.dumps({})

        elif payload_template == "message":
            # Message commands (notifications, etc.)
            if isinstance(value, dict):
                return json.dumps(value)
            else:
                return json.dumps({"message": str(value) if value else ""})

        elif payload_template == "color":
            # Color commands (RGB lights)
            if value is not None:
                return json.dumps({"color": str(value)})
            else:
                return json.dumps({})

        elif payload_template == "other" or payload_template is None:
            # Binary/other commands (on/off, up/down, stop)
            # Use empty JSON object
            return json.dumps({})

        else:
            # Unknown template - fallback to generic value
            if value is not None:
                return json.dumps({"value": value})
            else:
                return json.dumps({})

    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        from app.aria_logging import get_logger
        log = get_logger("ARIA.Actions")

        if rc == 0:
            logger.info("Connected to Actions MQTT broker")
            log.info("ARIA.Actions.Connected", extra={"fields": {"host": self.host, "port": self.port}})
            self._connected = True
        else:
            log.error("ARIA.Actions.ConnectionFailed", extra={"fields": {
                "host": self.host,
                "port": self.port,
                "return_code": rc
            }})
            self._connected = False

    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        from app.aria_logging import get_logger
        log = get_logger("ARIA.Actions")

        self._connected = False
        if rc != 0:
            log.warning("ARIA.Actions.UnexpectedDisconnection", extra={"fields": {
                "return_code": rc
            }})
        else:
            logger.info("Disconnected from Actions MQTT broker")
