import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, PolygonSelector, Slider
from PIL import Image, ImageOps
from transformers import AutoModelForImageClassification

from fine_tune_image_net import image_to_inputs

PATH_TO_IMAGE = "meterkast_data/new_pose.jpg"
MODEL_NAME = "farleyknight-org-username/vit-base-mnist"
MODEL_NAME = "/Users/arentstienstra/Documents/digits/vit-base-mnist-regular"


INITIAL_POINTS = [
    (2110, 1419),
    (2021, 1190),
    (1148, 1550),
    (1203, 1787),
]
INITIAL_PERCENTAGE = 0.05


def crop_and_straigthen(image: Image, corners: list[tuple[int, int]]):
    left_top, left_botton, right_bottom, right_top = corners
    desired_dimensions = (*left_top, *left_botton, *right_bottom, *right_top)

    # TODO Do something about the space between digits.
    return image.transform(
        (1500, 250),
        Image.Transform.QUAD,
        data=desired_dimensions,
        resample=Image.Resampling.BILINEAR,
    )


def split_into_individual_digits(
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


def normalize(image):
    copied = copy.deepcopy(image)
    norm = (copied - np.min(copied)) / (np.max(copied) - np.min(copied))
    norm_im = norm * 255

    return norm_im.astype(np.uint8)


def plot(
    ax_dict, original_image, area_of_interest, predictions, digit_arrays, row_digits
):

    ax_dict["original"].imshow(
        original_image,
    )
    ax_dict["cropped"].imshow(area_of_interest, "gray")
    ax_dict["cropped"].axis("off")
    ax_dict["original"].axis("off")

    for name, digit, pred in zip(row_digits, digit_arrays, predictions):
        ax_dict[name].imshow(digit, "gray")
        ax_dict[name].axis("off")
        ax_dict[name].set_title(f"Pred: {pred}")


def plot_results(
    original_image,
) -> None:

    row_original = ["original"] * 7

    row_cropped = ["cropped"] * 7
    row_digits = list("0123456")
    row_button = ["button", "button"] + ["nothing", "nothing"] + ["slider"] * 3

    all_axis = [
        row_original,
        row_original,
        row_original,
        row_original,
        row_cropped,
        row_digits,
        row_button,
    ]

    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(all_axis)

    selector = PolygonSelector(ax_dict["original"], lambda *args: None)

    # Add three vertices
    ax_dict["nothing"].axis("off")
    selector.verts = INITIAL_POINTS
    bprev = Button(ax_dict["button"], "REPREDICT")

    freq_slider = Slider(
        ax=ax_dict["slider"],
        label="Percentage gap",
        valmin=0.0,
        valmax=1.0,
        valinit=INITIAL_PERCENTAGE,
    )

    def repredict(event):
        print("Repredicting .... ")
        points = [(int(x), int(y)) for x, y in selector.verts]
        points_right_order = [points[i] for i in [2, 3, 0, 1]]
        area_of_interest, digit_arrays = input_to_individual_and_processed_images(
            original_image, points_right_order, percentage_space=freq_slider.val
        )
        predictions = predict(digit_arrays)
        plot(
            ax_dict,
            original_image,
            area_of_interest,
            predictions,
            digit_arrays,
            row_digits,
        )
        print("Done :)")

    bprev.on_clicked(repredict)
    repredict(None)

    plt.show()


def invert_image(image):
    return 255 - normalize(image)


def input_to_individual_and_processed_images(
    original_image: Image, points: list[tuple[int, int]], percentage_space: float
) -> list[np.ndarray]:
    area_of_interest = crop_and_straigthen(original_image, points)
    individual_digits = split_into_individual_digits(
        area_of_interest, 7, percentage_space
    )
    digits = [
        im.resize((28, 28), resample=Image.Resampling.BILINEAR)
        for im in individual_digits
    ]
    gray_digits = [ImageOps.grayscale(digit) for digit in digits]

    digit_arrays = [np.array(d) for d in gray_digits]
    return area_of_interest, [invert_image(p) for p in digit_arrays]


def predict(images) -> list[int]:
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
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


def main():
    original_image = Image.open(PATH_TO_IMAGE)
    plot_results(original_image)


main()
