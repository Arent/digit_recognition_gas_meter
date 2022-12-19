import matplotlib.pyplot as plt
from matplotlib.widgets import Button, PolygonSelector, Slider
from PIL import Image
import cv2
from stadswarmte_sensor import image_to_digits, digit_segments
import numpy as np


def recognize_segments(
    image: np.ndarray, threshold: int
) -> digit_segments.SegmentDigit:

    booleans = []
    for segment in digit_segments.all_segments_in_frame(image.shape):
        values_for_segment = digit_segments.image_values_in_segment(image, segment)
        booleans.append(np.mean(values_for_segment) > threshold)

    return digit_segments.SegmentDigit(*booleans)


def recognized_segments_images(
    potential_digit: digit_segments.SegmentDigit, image_shape: tuple[int, int]
):

    all_segments = digit_segments.all_segments_in_frame(image_shape)

    segment_arrays = [
        digit_segments.image_only_in_segment(np.ones(image_shape), segment)
        for segment in all_segments
    ]

    yes = digit_segments.maximise_segments(potential_digit, segment_arrays)
    no = digit_segments.maximise_segments(potential_digit.reversed(), segment_arrays)
    return yes, no


def crop_to_digit(digit_image: np.ndarray) -> np.ndarray:
    # This works quite well, maybe not for the first image, but that will be zero for a long time anyway
    # We don't really need this 28 by 28 images. This is due to the legacy deep learning solution.
    return digit_image[1:28, 10:27]


def turn_all_axis_off(ax_dict, labels):
    for row in labels:
        for column in row:
            ax_dict[column].axis("off")


def main():

    corner_points: list[tuple[int, int]] = [
        (501, 1440),
        (491, 1655),
        (1518, 1695),
        (1537, 1492),
    ]

    original_image = Image.open(
        "meterkast_images/2022_12_18_15_40_37__no_prediction.jpg"
    )
    _, digit_arrays = image_to_digits._input_to_individual_and_processed_images(
        original_image, corner_points, percentage_space=0.02
    )

    all_axis = [
        [f"original_{i}" for i in range(10)],
        [f"cropped_{i}" for i in range(10)],
        [f"recognized_segments_{i}" for i in range(10)],
        [f"digit_{i}" for i in range(10)],
    ]
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(all_axis)
    turn_all_axis_off(ax_dict, all_axis)
    # The first one is too difficult, way too noisy
    for i, digit_array in enumerate(digit_arrays):
        cropped = crop_to_digit(digit_array)
        ax_dict[f"original_{i}"].imshow(digit_array, "gray")
        ax_dict[f"cropped_{i}"].imshow(cropped, "gray")

        bools = recognize_segments(cropped, 140)
        yes, no = recognized_segments_images(
            bools,
            cropped.shape,
        )

        ax_dict[f"recognized_segments_{i}"].imshow(
            yes,
        )

    digit_array_temps = digit_segments.all_digit_images(
        crop_to_digit(digit_arrays[0]).shape
    )
    for i, array in enumerate(digit_array_temps):
        ax_dict[f"digit_{i}"].imshow(array)

    plt.show()


main()
