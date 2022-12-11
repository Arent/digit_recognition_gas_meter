from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraSettings:
    shutter_speed_seconds: int = 1
    gain: float = 1.5


@dataclass(frozen=True)
class DigitRecognitionSettings:
    gap_between_digits_percentage: float = 0.01
    corner_points: tuple[tuple[int, int], ...] = (
        (1193, 1684),
        (1279, 1889),
        (2162, 1501),
        (2057, 1292),
    )

    model_location: Path = Path("/home/pi/models/vit-base-mnist-regular")
    # model_location: Path = Path(
    #     "/Users/arentstienstra/Documents/digits/vit-base-mnist-regular"
    # )

    def __post_init__(self):
        assert len(self.corner_points) == 4, "There should be 4 corner locations"
        assert (
            0 <= self.gap_between_digits_percentage <= 1
        ), "The gap should be between 0 and 1"
        assert self.model_location.exists(), "The Model should actually exists"


@dataclass(frozen=True)
class MQTTSettings:
    broker: str = "localhost"
    topic: str = "home/meterkast/stadswarmte"
    port: int = 1883


@dataclass(frozen=True)
class AppSettings:
    time_between_measurements: int = 3600
    images_folder: Path = Path("/home/pi/meterkast_images")
    camera_settings: CameraSettings = CameraSettings()
    digit_recognition_settings: DigitRecognitionSettings = DigitRecognitionSettings()
    mqqt_settings: MQTTSettings = MQTTSettings()
