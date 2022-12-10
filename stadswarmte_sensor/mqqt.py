import json
import random

from paho.mqtt import client as mqtt_client

import stadswarmte_sensor.app_settings as app_settings

# from stadswarmte_sensor import app_settings


def connect_mqtt(settings: app_settings.MQTTSettings):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    # Set Connecting Client ID
    client_id = f"python-mqtt-{random.randint(0, 99999)}"

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(settings.broker, settings.port)
    return client


def format_message(digits: list[int], timestamp: str):
    value = int("".join(map(str, digits))) / 1000
    return json.dumps({"measurement": value, "timestamp": timestamp})


def publish_message(
    digits: list[int], timestamp: str, settings: app_settings.MQTTSettings
) -> None:
    message = format_message(digits, timestamp)

    client = connect_mqtt(settings)
    result = client.publish(settings.topic, message)

    status = result[0]
    if status == 0:
        print(f"Send `{message}` to topic `{settings.topic}`")
        return

    print(f"Failed to send message to topic {settings.topic}")
    return


publish_message([3, 3, 4, 4, 1, 8], "2022_12_09__12_28_37", app_settings.MQTTSettings())
