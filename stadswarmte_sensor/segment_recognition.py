import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.widgets import Slider
from PIL import Image

from stadswarmte_sensor import digit_segments, image_to_digits


def vector_square(vector: np.ndarray) -> float:
    return np.dot(vector, vector.T)


def euclidian_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    distance = (image1 - image2).flatten()
    return np.sqrt(vector_square(distance))


@dataclasses.dataclass(frozen=True)
class Settings:
    top: int = 2
    bottom: int = 3
    right: int = 3
    left: int = 6

    corner_points: tuple[tuple[int, int], ...] = ((517, 1461), (474, 1696), (1547, 1764), (1567, 1523))

    percentage_gap: float = 0.02
    stripe_height_percentage: float = 0.17
    stripe_width_percentage: float = 0.25
    gap_middle_percentage: float = 0.10


def normalized_digit_distances(
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


def normalize_and_crop(digit_image: np.ndarray, settings: Settings) -> np.ndarray:

    height, width = digit_image.shape
    # This works quite well, maybe not for the first image, but that will be zero for a long time anyway
    # We don't really need this 28 by 28 images. This is due to the legacy deep learning solution.
    cropped = digit_image[
        settings.top : height - settings.bottom, settings.left : width - settings.right
    ]
    return   normalize(cropped)


def turn_all_axis_off(ax_dict, labels):
    for row in labels:
        for column in row:
            ax_dict[column].axis("off")


def plot(
    original_image: Image,
    settings: Settings,
    ax_dict: dict,
):
    _, digit_arrays = image_to_digits._input_to_individual_and_processed_images(
        original_image, settings.corner_points, percentage_space=settings.percentage_gap
    )

    cropped_shape = normalize_and_crop(digit_arrays[0], settings).shape
    digit_templates = digit_segments.all_digit_images(cropped_shape, settings)

    for i, digit_array in enumerate(digit_arrays):

        cropped = normalize_and_crop(
            digit_array,
            settings,
        )
        distances = normalized_digit_distances(cropped, digit_templates)

        best = np.argmax(distances)
        ax_dict[f"original_{i}"].imshow(digit_array, "gray")
        ax_dict[f"cropped_{i}"].imshow(cropped, "gray")
        ax_dict[f"cropped_{i}"].set_title(best)


    for i, temp in enumerate(digit_templates):
        ax_dict[f"digit_{i}"].imshow(temp)

def predict(original_image: Image, settings: Settings = Settings()) -> list[int]:
    predictions = []
    _, digit_arrays = image_to_digits._input_to_individual_and_processed_images(
        original_image, settings.corner_points, percentage_space=settings.percentage_gap
    )
    cropped_shape = normalize_and_crop(digit_arrays[0], settings).shape
    digit_templates = digit_segments.all_digit_images(cropped_shape, settings)

    for digit_array in digit_arrays:
        cropped = normalize_and_crop(
            digit_array,
            settings,
        )
        distances = normalized_digit_distances(cropped, digit_templates)
        best = np.argmax(distances)
        predictions.append(best)

    return predictions


def main():

    original_image = Image.open(
        "ground_truth/2022_12_28_08_45_05__gt_0336139.jpg"
    )
    # original_image = Image.open("ground_truth/2022_12_27_06_43_22__gt_0335948.jpg")
    all_axis = [
        [f"original_{i}" for i in range(10)],
        [f"cropped_{i}" for i in range(10)],
        ["gap"] * 4
        + [
            "stripe_height",
            "stripe_height",
            "stripe_width",
            "stripe_width",
            "middle_gap",
            "middle_gap",
        ],
        [f"digit_{i}" for i in range(10)],

    ]

    settings = Settings()

    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(all_axis)

    stripe_height_slider = Slider(
        ax=ax_dict["stripe_height"],
        label="stripe_height",
        valstep=0.05,
        valmin=0,
        valmax=1,
        valinit=settings.stripe_height_percentage,
    )
    stripe_width_slider = Slider(
        ax=ax_dict["stripe_width"],
        label="stripe_width",
        valstep=0.05,
        valmin=0,
        valmax=1,
        valinit=settings.stripe_width_percentage,
    )

    middle_gap_slider = Slider(
        ax=ax_dict["middle_gap"],
        label="middle_gap",
        valstep=0.05,
        valmin=0,
        valmax=1,
        valinit=settings.gap_middle_percentage,
    )

    turn_all_axis_off(ax_dict, all_axis)

    def update(*args, **kwargs):
        updated_settings = dataclasses.replace(
            settings,
            stripe_width_percentage=stripe_width_slider.val,
            stripe_height_percentage=stripe_height_slider.val,
            gap_middle_percentage=middle_gap_slider.val,
        )

        plot(original_image, updated_settings, ax_dict)

    stripe_height_slider.on_changed(update)
    stripe_width_slider.on_changed(update)
    middle_gap_slider.on_changed(update)

    plot(original_image, settings, ax_dict)
    plt.show()


if __name__ == "__main__":
    main()
