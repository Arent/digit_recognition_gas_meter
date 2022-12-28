import numpy as np
from PIL import Image

from stadswarmte_sensor import app_settings, digit_segments, image_to_digits


def vector_square(vector: np.ndarray) -> float:
    return np.dot(vector, vector.T)


def euclidian_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    distance = (image1 - image2).flatten()
    return np.sqrt(vector_square(distance))


def distances_to_probabillities(
    image: np.ndarray, digit_images: np.ndarray
) -> list[float]:

    distances = [euclidian_distance(image, digit) for digit in digit_images]
    total_distance = np.sum(distances)
    normalized_distances = [d / total_distance for d in distances]

    negative = 1 - np.array(normalized_distances)
    return list(negative / negative.sum())


def normalize(image: np.ndarray) -> np.ndarray:

    cn_image = (image - image.min()) / (image.max() - image.min())
    normalized = (cn_image * 255).astype(np.uint8)
    return normalized


def normalize_and_crop(
    digit_image: np.ndarray, settings: app_settings.DigitRecognitionSettings
) -> np.ndarray:

    height, width = digit_image.shape
    # This works quite well, maybe not for the first image, but that will be zero for a long time anyway
    # We don't really need this 28 by 28 images. This is due to the legacy deep learning solution.
    cropped = digit_image[
        settings.top : height - settings.bottom, settings.left : width - settings.right
    ]
    return normalize(cropped)


def _predict_image(processed: np.ndarray, templates) -> int:
    probabilities = distances_to_probabillities(processed, templates)
    return int(np.argmax(probabilities))


def pre_process_digit_images(
    digit_arrays: list[np.ndarray], settings: app_settings.DigitRecognitionSettings
) -> list[np.ndarray]:

    return [normalize_and_crop(d, settings) for d in digit_arrays]


def predict(processed_digit_arrays, settings) -> list[int]:
    digit_templates = digit_segments.all_digit_images(
        processed_digit_arrays[0].shape, settings
    )
    return [_predict_image(p, digit_templates) for p in processed_digit_arrays]


def process_and_predict(
    original_image: Image, settings: app_settings.DigitRecognitionSettings
) -> list[int]:
    _, digit_arrays = image_to_digits.input_to_individual_and_processed_images(
        original_image,
        settings.corner_points,
        percentage_space=settings.gap_between_digits_percentage,
    )

    processed = pre_process_digit_images(digit_arrays, settings)
    return predict(processed, settings)


def turn_all_axis_off(ax_dict, labels):
    for row in labels:
        for column in row:
            ax_dict[column].axis("off")
