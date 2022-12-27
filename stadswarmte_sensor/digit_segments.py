import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SegmentSettings:
    stripe_height_percentage: float = 0.2
    stripe_width_percentage: float = 0.3
    gap_middle_percentage: float = 0.08


@dataclasses.dataclass(frozen=True)
class Segment:
    x_start: int
    y_start: int
    x_end: int
    y_end: int


@dataclasses.dataclass(frozen=False)
class SegmentDigit:
    top: bool
    top_left: bool
    top_right: bool
    center: bool
    bottom_left: bool
    bottom_right: bool
    bottom: bool

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, field.type):
                continue
            setattr(self, field.name, field.type(value))

    def __iter__(self):
        return iter(dataclasses.astuple(self))

    def __len__(self):
        return len(dataclasses.astuple(self))


def all_segments_in_frame(
    frame_shape: tuple[int, int],
    settings: SegmentSettings = SegmentSettings(),
) -> list[Segment]:
    frame_height, frame_width = frame_shape

    stripe_height = int(settings.stripe_height_percentage * frame_height)
    stripe_width = int(settings.stripe_width_percentage * frame_width)
    gap_middle = int(settings.gap_middle_percentage * frame_height)

    # In the same order as SegmentDigit

    return [
        Segment(0, 0, frame_width, stripe_height),  # top
        Segment(0, 0, stripe_width, frame_height // 2),  # top-left
        Segment(
            frame_width - stripe_width, 0, frame_width, frame_height // 2
        ),  # top-right
        Segment(
            0,
            (frame_height // 2) - gap_middle,
            frame_width,
            (frame_height // 2) + gap_middle,
        ),  # center
        Segment(0, frame_height // 2, stripe_width, frame_height),  # bottom-left
        Segment(
            frame_width - stripe_width, frame_height // 2, frame_width, frame_height
        ),  # bottom-right
        Segment(0, frame_height - stripe_height, frame_width, frame_height),  # bottom
    ]


def segment_to_template(segment: Segment, image_shape: tuple[int, int]) -> np.ndarray:
    image = np.zeros(image_shape)
    image[segment.y_start : segment.y_end, segment.x_start : segment.x_end] = 255
    return image


def image_values_in_segment(image: np.ndarray, segment: Segment) -> np.ndarray:
    return image[segment.y_start : segment.y_end, segment.x_start : segment.x_end]


def image_only_in_segment(image: np.ndarray, segment: Segment) -> np.ndarray:
    result = np.zeros_like(image)

    result[segment.y_start : segment.y_end, segment.x_start : segment.x_end] = image[
        segment.y_start : segment.y_end, segment.x_start : segment.x_end
    ]

    return result


def all_digits() -> dict[int, SegmentDigit]:

    return {
        0: SegmentDigit(
            top=1,
            top_left=1,
            top_right=1,
            center=0,
            bottom_left=1,
            bottom_right=1,
            bottom=1,
        ),
        1: SegmentDigit(
            top=0,
            top_left=0,
            top_right=1,
            center=0,
            bottom_left=0,
            bottom_right=1,
            bottom=0,
        ),
        2: SegmentDigit(
            top=1,
            top_left=0,
            top_right=1,
            center=1,
            bottom_left=1,
            bottom_right=0,
            bottom=1,
        ),
        3: SegmentDigit(
            top=1,
            top_left=0,
            top_right=1,
            center=1,
            bottom_left=0,
            bottom_right=1,
            bottom=1,
        ),
        4: SegmentDigit(
            top=0,
            top_left=1,
            top_right=1,
            center=1,
            bottom_left=0,
            bottom_right=1,
            bottom=0,
        ),
        5: SegmentDigit(
            top=1,
            top_left=1,
            top_right=0,
            center=1,
            bottom_left=0,
            bottom_right=1,
            bottom=1,
        ),
        6: SegmentDigit(
            top=1,
            top_left=1,
            top_right=0,
            center=1,
            bottom_left=1,
            bottom_right=1,
            bottom=1,
        ),
        7: SegmentDigit(
            top=1,
            top_left=0,
            top_right=1,
            center=0,
            bottom_left=0,
            bottom_right=1,
            bottom=0,
        ),
        8: SegmentDigit(
            top=1,
            top_left=1,
            top_right=1,
            center=1,
            bottom_left=1,
            bottom_right=1,
            bottom=1,
        ),
        9: SegmentDigit(
            top=1,
            top_left=1,
            top_right=1,
            center=1,
            bottom_left=0,
            bottom_right=1,
            bottom=0,
        ),
    }


def maximise_segments(digit: SegmentDigit, segment_images: list[np.ndarray]):
    assert len(segment_images) == len(digit)

    total = np.zeros_like(segment_images[0])
    for segment_bool, segment_array in zip(digit, segment_images):
        if not segment_bool:
            continue
        total = np.maximum(total, segment_array)
    return total


def segment_digit_to_image(
    frame_shape: tuple[int, int], digit: SegmentDigit, settings: SegmentSettings
):
    segment_images = [
        segment_to_template(s, frame_shape)
        for s in all_segments_in_frame(frame_shape, settings=settings)
    ]
    return maximise_segments(digit, segment_images)


def all_digit_images(
    frame_shape: tuple[int, int], settings: SegmentSettings
) -> list[np.ndarray]:
    images = []
    for digit in all_digits().values():
        images.append(segment_digit_to_image(frame_shape, digit, settings))
    return images
