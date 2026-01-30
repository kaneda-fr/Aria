"""
Jeedom MQTT inventory source.

Subscribes to Jeedom MQTT2 plugin discovery and event topics.
"""

import json
import logging
import threading
from typing import Dict, Any
import paho.mqtt.client as mqtt

from app.plugins.inventory.sources.base import InventorySource
from app.plugins.inventory.normalizer import JeedomNormalizer
from app.plugins.inventory.models import DeviceRecord

logger = logging.getLogger(__name__)


class JeedomMQTTSource(InventorySource):
    """
    Jeedom MQTT2 plugin source.

    Subscribes to:
    - jeedom/discovery/eqLogic/# (device discovery)
    - jeedom/cmd/event/# (state updates)
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
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start MQTT client and subscribe to topics."""
        from app.aria_logging import get_logger
        log = get_logger("ARIA.Inventory")

        logger.info(f"Starting Jeedom MQTT source: {self.host}:{self.port}")
        log.info("ARIA.Inventory.Connecting", extra={"fields": {"host": self.host, "port": self.port}})

        # Create MQTT client
        self._client = mqtt.Client(client_id=f"aria-inventory-{self.name}")

        # Set credentials if provided
        if self.username:
            self._client.username_pw_set(self.username, self.password)

        # Set callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Connect
        try:
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            logger.info("Jeedom MQTT client started")

        except Exception as e:
            log.error("ARIA.Inventory.ConnectionFailed", extra={"fields": {
                "host": self.host,
                "port": self.port,
                "error": str(e)
            }})
            raise

    def stop(self) -> None:
        """Stop MQTT client."""
        logger.info("Stopping Jeedom MQTT source")

        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

        self._connected = False
        logger.info("Jeedom MQTT source stopped")

    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        from app.aria_logging import get_logger
        log = get_logger("ARIA.Inventory")

        if rc == 0:
            logger.info("Connected to Jeedom MQTT broker")
            log.info("ARIA.Inventory.Connected", extra={"fields": {"host": self.host, "port": self.port}})
            self._connected = True

            # Subscribe to discovery
            discovery_topic = f"{self.root_topic}/discovery/eqLogic/#"
            client.subscribe(discovery_topic, qos=self.qos)
            logger.info(f"Subscribed to {discovery_topic}")
            log.info("ARIA.Inventory.Subscribed", extra={"fields": {"topic": discovery_topic}})

            # Subscribe to events
            event_topic = f"{self.root_topic}/cmd/event/#"
            client.subscribe(event_topic, qos=self.qos)
            logger.info(f"Subscribed to {event_topic}")
            log.info("ARIA.Inventory.Subscribed", extra={"fields": {"topic": event_topic}})

        else:
            log.error("ARIA.Inventory.ConnectionFailed", extra={"fields": {
                "host": self.host,
                "port": self.port,
                "return_code": rc
            }})
            self._connected = False

    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        from app.aria_logging import get_logger
        log = get_logger("ARIA.Inventory")

        self._connected = False
        if rc != 0:
            log.warning("ARIA.Inventory.UnexpectedDisconnection", extra={"fields": {
                "return_code": rc
            }})
        else:
            logger.info("Disconnected from Jeedom MQTT broker")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT message."""
        topic = msg.topic

        try:
            # Parse JSON payload
            payload = json.loads(msg.payload.decode("utf-8"))

            # Debug: Log first discovery message
            if "/discovery/eqLogic/" in topic and not hasattr(self, '_first_discovery_logged'):
                from app.aria_logging import get_logger
                log = get_logger("ARIA.Inventory")
                log.info("ARIA.Inventory.FirstDiscovery", extra={"fields": {
                    "topic": topic,
                    "payload_type": type(payload).__name__,
                    "payload_preview": str(payload)[:300]
                }})
                self._first_discovery_logged = True

            # Route by topic pattern
            if "/discovery/eqLogic/" in topic:
                self._handle_discovery(topic, payload)
            elif "/cmd/event/" in topic:
                self._handle_event(topic, payload)
            else:
                logger.debug(f"Ignoring unknown topic: {topic}")

        except json.JSONDecodeError as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.InvalidJSON", extra={"fields": {
                "topic": topic,
                "error": str(e)
            }})
        except Exception as e:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.error("ARIA.Inventory.MessageProcessingError", extra={"fields": {
                "topic": topic,
                "error": str(e)
            }})

    def _handle_discovery(self, topic: str, payload: Dict[str, Any]) -> None:
        """Handle device discovery message."""
        # Extract eq_id from topic: jeedom/discovery/eqLogic/255
        parts = topic.split("/")
        if len(parts) < 4:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.warning("ARIA.Inventory.MalformedTopic", extra={"fields": {
                "topic": topic,
                "type": "discovery"
            }})
            return

        eq_id = parts[-1]

        # Debug: Log payload type and structure
        logger.debug(f"Discovery for {eq_id}: type={type(payload).__name__}, keys={list(payload.keys()) if isinstance(payload, dict) else 'N/A'}")

        # Normalize device
        device = JeedomNormalizer.normalize_device(payload, source=self.name)

        if device:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")

            logger.info(
                f"Discovered device: {device.name} ({eq_id}) "
                f"with {len(device.commands)} commands"
            )
            log.info("ARIA.Inventory.DeviceDiscovered", extra={"fields": {
                "eq_id": eq_id,
                "name": device.name,
                "room": device.room,
                "command_count": len(device.commands)
            }})
            # Notify via callback
            self._on_devices_discovered([device])
        else:
            logger.debug(f"Filtered device {eq_id}")

    def _handle_event(self, topic: str, payload: Dict[str, Any]) -> None:
        """Handle command state event."""
        # Extract cmd_id from topic: jeedom/cmd/event/5454
        parts = topic.split("/")
        if len(parts) < 4:
            from app.aria_logging import get_logger
            log = get_logger("ARIA.Inventory")
            log.warning("ARIA.Inventory.MalformedTopic", extra={"fields": {
                "topic": topic,
                "type": "event"
            }})
            return

        cmd_id = parts[-1]

        # Normalize event
        state = JeedomNormalizer.normalize_event(cmd_id, payload)

        if state:
            # Notify via callback
            self._on_state_update(state)

    @property
    def is_connected(self) -> bool:
        """Check if MQTT client is connected."""
        return self._connected
