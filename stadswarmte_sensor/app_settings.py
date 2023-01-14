from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraSettings:
    shutter_speed_seconds: int = 1
    gain: float = 1.5


@dataclass(frozen=True)
class DigitRecognitionSettings:
    top: int = 2
    bottom: int = 2
    right: int = 3
    left: int = 7

    corner_points: tuple[tuple[int, int], ...] = (
        (517, 1461),
        (474, 1696),
        (1547, 1764),
        (1567, 1523),
    )

    gap_between_digits_percentage: float = 0.00
    stripe_height_percentage: float = 0.17
    stripe_width_percentage: float = 0.25
    gap_middle_percentage: float = 0.05

    def __post_init__(self):
        assert len(self.corner_points) == 4, "There should be 4 corner locations"


@dataclass(frozen=True)
class MQTTSettings:
    broker: str = "localhost"
    topic: str = "home/meterkast/stadswarmte"
    port: int = 1883


@dataclass(frozen=True)
class AppSettings:
    time_between_measurements: int = 300
    images_folder: Path = Path("/home/pi/meterkast_images")
    camera_settings: CameraSettings = CameraSettings()
    digit_recognition_settings: DigitRecognitionSettings = DigitRecognitionSettings()
    mqqt_settings: MQTTSettings = MQTTSettings()
