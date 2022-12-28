import datetime
import time
from pathlib import Path

from PIL import Image

from stadswarmte_sensor import app_settings, camera, mqqt, segment_recognition


def log_to_disk(
    image: Image.Image, digits: list[int], timestamp: str, base_path: Path
) -> None:
    digits_string = "".join([str(d) for d in digits])
    name = f"{timestamp}__{digits_string}.png"
    image.save(base_path / name)


def capture_recognise_and_publish(settings: app_settings.AppSettings):
    """This funcion captures an image, recognizes the individual digits
    and publishes a message on a MQQT topic.
    Additionally, the input image and the digits are logged to disk.
    """
    # We want a unqiue timestamp that is the same for the logged images and the mqqt message.
    # We can use this to correlate mqqt messages with logged images.

    timestamp = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")

    image = camera.capture_image(
        settings.camera_settings, settings.images_folder / "input.png"
    )
    digits = segment_recognition.process_and_predict(
        image, settings.digit_recognition_settings
    )
    log_to_disk(image, digits, timestamp, settings.images_folder)
    mqqt.publish_message(digits, timestamp, settings.mqqt_settings)


def main():
    settings = app_settings.AppSettings()
    while True:
        capture_recognise_and_publish(settings)
        time.sleep(settings.time_between_measurements)
