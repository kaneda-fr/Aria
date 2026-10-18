#!/usr/bin/env python3
"""
Test MQTT command execution.

This script tests sending commands to Jeedom via MQTT.
Run this from your server (192.168.100.100) where MQTT is accessible.
"""

import time
import paho.mqtt.client as mqtt

# Jeedom MQTT configuration (from start.sh)
MQTT_HOST = "192.168.100.60"
MQTT_PORT = 1883
ROOT_TOPIC = "jeedom"

# Dining room light commands (from inventory test)
CMD_LIGHT_ON = "5631"
CMD_LIGHT_OFF = "5632"
CMD_LIGHT_DIMMER = "5630"


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
    else:
        print(f"❌ Connection failed with code {rc}")


def on_publish(client, userdata, mid, reason_code=None, properties=None):
    print(f"📤 Message published (mid: {mid})")


def test_mqtt_commands():
    """Test MQTT command execution."""

    print("=" * 80)
    print("ARIA MQTT Command Test")
    print("=" * 80)
    print(f"Target: {MQTT_HOST}:{MQTT_PORT}")
    print(f"Device: Lumiere Salle a Manger New")
    print()

    # Create MQTT client
    client = mqtt.Client(client_id="aria-test-client", protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_publish = on_publish

    try:
        print(f"Connecting to {MQTT_HOST}:{MQTT_PORT}...")
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        client.loop_start()

        # Wait for connection
        time.sleep(2)

        # Test 1: Turn ON
        print("\n" + "=" * 80)
        print("TEST 1: Turn ON dining room light")
        print("=" * 80)
        topic = f"{ROOT_TOPIC}/cmd/set/{CMD_LIGHT_ON}"
        payload = '{}'  # Binary command - empty payload (Jeedom MQTT2 protocol)
        print(f"Topic: {topic}")
        print(f"Payload: {payload} (empty = binary on)")

        result = client.publish(topic, payload, qos=1)
        result.wait_for_publish()
        print("✅ Command sent!")

        time.sleep(3)

        # Test 2: Set brightness to 50%
        print("\n" + "=" * 80)
        print("TEST 2: Set brightness to 50%")
        print("=" * 80)
        topic = f"{ROOT_TOPIC}/cmd/set/{CMD_LIGHT_DIMMER}"
        payload = '{"slider": 50}'  # Slider command - use "slider" key (Jeedom MQTT2 protocol)
        print(f"Topic: {topic}")
        print(f"Payload: {payload}")

        result = client.publish(topic, payload, qos=1)
        result.wait_for_publish()
        print("✅ Command sent!")

        time.sleep(3)

        # Test 3: Turn OFF
        print("\n" + "=" * 80)
        print("TEST 3: Turn OFF dining room light")
        print("=" * 80)
        topic = f"{ROOT_TOPIC}/cmd/set/{CMD_LIGHT_OFF}"
        payload = '{}'  # Binary command - empty payload (Jeedom MQTT2 protocol)
        print(f"Topic: {topic}")
        print(f"Payload: {payload} (empty = binary off)")

        result = client.publish(topic, payload, qos=1)
        result.wait_for_publish()
        print("✅ Command sent!")

        time.sleep(1)

        print("\n" + "=" * 80)
        print("✅ All tests complete!")
        print("=" * 80)

    except ConnectionRefusedError:
        print(f"\n❌ Connection refused to {MQTT_HOST}:{MQTT_PORT}")
        print("Make sure:")
        print("  1. You're running this from a machine that can reach the MQTT broker")
        print("  2. The MQTT broker is running on the Jeedom server")
        print("  3. Port 1883 is open and accessible")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        client.loop_stop()
        client.disconnect()
        print("\nDisconnected from MQTT broker")


if __name__ == "__main__":
    print("⚠️  WARNING: This will control your real dining room light!")
    print("Make sure you're ready for the light to turn on/off.")
    print()

    # Uncomment to run:
    # test_mqtt_commands()

    print("To run this test, uncomment the last line in the script")
    print("Or run from your server (192.168.100.100) where MQTT is accessible")
