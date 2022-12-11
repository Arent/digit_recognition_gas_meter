import tempfile
from fractions import Fraction
from time import sleep

from picamera import PiCamera
from PIL import Image

from stadswarmte_sensor.app_settings import CameraSettings


def capture_image(settings: CameraSettings, filename: str) -> Image:
    print(f"Starting to capture with settings {settings!r}")
    # Force sensor mode 3 (the long exposure mode)
    framerate = Fraction(1, int(settings.shutter_speed_seconds) + 1)
    with PiCamera(
        resolution=(2592, 1944), framerate=framerate, sensor_mode=3
    ) as camera:

        camera.shutter_speed = int(settings.shutter_speed_seconds * 1_000_000)
        camera.iso = settings.iso

        # Give the camera a good long time to set gains and
        # measure AWB (you may wish to use fixed AWB instead)

        sleep(30)
        camera.exposure_mode = "off"
        # Finally, capture an image with a long exposure. Due
        # to mode switching on the still port, this will take
        # longer than the set exposure seconds

        camera.capture(filename, quality=100)
    image = Image.open(filename)

    print("Captured image.")
    return image
