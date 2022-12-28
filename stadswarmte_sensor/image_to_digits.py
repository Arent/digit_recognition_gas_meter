import copy

import numpy as np
from PIL import Image, ImageOps


def _crop_and_straigthen(image: Image, corners: list[tuple[int, int]]):
    left_top, left_botton, right_bottom, right_top = corners
    desired_dimensions = (*left_top, *left_botton, *right_bottom, *right_top)
    return image.transform(
        (1500, 250),
        Image.Transform.QUAD,
        data=desired_dimensions,
        resample=Image.Resampling.BILINEAR,
    )


def _split_into_individual_digits(
    cropped_and_straightened: Image, num_digits: int, gap_percentage: float
):
    cropped_and_straightened_array = np.array(cropped_and_straightened)

    total_width = cropped_and_straightened.size[0]
    per_digit_width = total_width / (num_digits + (num_digits - 1) * gap_percentage)
    gap_width = per_digit_width * gap_percentage
    assert np.isclose(
        ((num_digits - 1) * gap_width + num_digits * per_digit_width), total_width
    )

    digits = []
    for i_digit in range(num_digits):
        start = int(i_digit * (per_digit_width + gap_width))
        end = int(start + per_digit_width)
        digit_array = cropped_and_straightened_array[:, start:end]
        digits.append(Image.fromarray(digit_array))

    return digits


def _normalize(image):
    copied = copy.deepcopy(image)
    norm = (copied - np.min(copied)) / (np.max(copied) - np.min(copied))
    norm_im = norm * 255

    return norm_im.astype(np.uint8)


def _invert_image(image):
    return 255 - _normalize(image)


def input_to_individual_and_processed_images(
    original_image: Image, points: list[tuple[int, int]], percentage_space: float
) -> tuple[Image.Image, list[np.ndarray]]:
    area_of_interest = _crop_and_straigthen(original_image, points)
    individual_digits = _split_into_individual_digits(
        area_of_interest, 7, percentage_space
    )
    digits = [
        im.resize((28, 28), resample=Image.Resampling.BILINEAR)
        for im in individual_digits
    ]
    gray_digits = [ImageOps.grayscale(digit) for digit in digits]

    digit_arrays = [np.array(d) for d in gray_digits]
    return area_of_interest, [_invert_image(p) for p in digit_arrays]
