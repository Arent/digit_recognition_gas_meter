import time

from picamera2 import Picamera2
from PIL import Image

from stadswarmte_sensor.app_settings import CameraSettings


def capture_image(settings: CameraSettings, filename: str) -> Image:

    picam2 = Picamera2()

    capture_config = picam2.create_still_configuration()

    picam2.start()
    time.sleep(2)

    controls = {
        "ExposureTime": settings.shutter_speed_seconds * 1_000_000,
        "AnalogueGain": settings.gain,
    }
    capture_config2 = picam2.create_still_configuration(controls=controls)
    picam2.switch_mode_and_capture_file(capture_config2, filename)

    image = Image.open(filename)
    picam2.close()
    print(f"Captured image at {filename}")
    return image
