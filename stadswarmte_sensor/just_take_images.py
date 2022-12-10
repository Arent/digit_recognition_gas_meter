import datetime
import time

from stadswarmte_sensor import app_settings, camera

def only_log_to_disk(settings: app_settings.AppSettings) -> None:
    timestamp = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
    image = camera.capture_image(settings.camera_settings)
    name = f"{timestamp}__no_prediction.png"
    image.save(settings.images_folder / name)

def main():
    settings = app_settings.AppSettings()
    while True:
        only_log_to_disk(settings)
        time.sleep(settings.time_between_measurements)