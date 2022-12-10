import copy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from transformers import AutoModelForImageClassification

from stadswarmte_sensor.app_settings import DigitRecognitionSettings


def image_to_inputs(
    image: Image, size=(224, 224), image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)
):
    if not isinstance(image, torch.Tensor):
        image = TF.to_tensor(image)[0, ...]

    resized = TF.resize(image[None, ...], size=size)[0]
    tensor_3_channels = torch.stack([resized, resized, resized], 0)
    normalized = TF.normalize(tensor_3_channels, image_mean, image_std)
    return normalized


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


def _input_to_individual_and_processed_images(
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


def _predict(images, model_name: Path) -> list[int]:
    model = AutoModelForImageClassification.from_pretrained(str(model_name))
    predictions = []
    for d in images:
        d_color = np.repeat(d[..., np.newaxis], 3, axis=2)
        d_pil = Image.fromarray(d_color)
        inputs = image_to_inputs(d_pil)
        with torch.no_grad():
            logits = model(inputs[None, ...]).logits
        predicted_label = logits.argmax(-1).item()
        predictions.append(predicted_label)
    return predictions


def to_individual_digits(image: Image, settings: DigitRecognitionSettings) -> list[int]:
    _, digit_arrays = _input_to_individual_and_processed_images(
        image,
        list(settings.corner_points),
        percentage_space=settings.gap_between_digits_percentage,
    )

    return _predict(digit_arrays, settings.model_location)
