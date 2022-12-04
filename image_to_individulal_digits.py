import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import copy
from PIL import Image, ImageDraw, ImageTransform, ImageOps

import torch
from transformers import AutoModelForImageClassification
from fine_tune_image_net import image_to_inputs

PATH_TO_IMAGE = "meterkast_data/input.jpg"
MODEL_NAME = "farleyknight-org-username/vit-base-mnist"
MODEL_NAME = "/Users/arentstienstra/Documents/digits/vit-base-mnist-regular"
POINTS = [ (272, 1250), (199,1488), (1616, 1634),(1607, 1417),]

def crop_and_straigthen(image : Image, corners: list[tuple[int, int]] ):
    left_top, left_botton, right_bottom, right_top = corners
    desired_dimensions = (*left_top, *left_botton, *right_bottom, * right_top)

    # TODO Do something about the space between digits. 
    return image.transform(
        (7*(28), 28),
        Image.Transform.QUAD,
        data=desired_dimensions,
        resample=Image.Resampling.BILINEAR,
    )

def split_into_individual_digits(cropped_and_straightened: Image, num_digits: int = 7):
    cropped_and_straightened_array = np.array(cropped_and_straightened)
    per_digit_width = int(cropped_and_straightened.size[0] / num_digits)
    digits = []
    for i_digit in range(num_digits):
        start = i_digit  * per_digit_width
        end = start + per_digit_width
        digit_array = cropped_and_straightened_array[:, start:end]
        digits.append(Image.fromarray(digit_array))

    return digits

def normalize(image):
    copied = copy.deepcopy(image)
    norm = (copied - np.min(copied)) / (np.max(copied) - np.min(copied))
    norm_im = norm * 255

    return norm_im.astype(np.uint8)


def plot_results(
    original_image,
    cropped_image: np.ndarray,
    padded,
    digit_images: list[np.ndarray],
    predictions: list[int],
) -> None:

    assert len(digit_images) == 7

    row_original = ["original"] * 7

    row_cropped = ["cropped"] * 7
    row_digits = list("0123456")
    row_digits_raw = [f"p{i}" for i in range(7)]

    all_axis = [
        row_original,
        row_original,
        row_original,
        row_original,
        row_cropped,
        row_digits_raw,
        row_digits,
    ]

    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(all_axis)

    ax_dict["original"].imshow(
        original_image,
    )
    ax_dict["cropped"].imshow(cropped_image, "gray")
    ax_dict["cropped"].axis("off")
    ax_dict["original"].axis("off")

    for name, digit, pred in zip(row_digits, digit_images, predictions):
        ax_dict[name].imshow(digit, "gray")
        ax_dict[name].axis("off")
        ax_dict[name].set_title(f"Pred: {pred}")

    for name, digit in zip(row_digits_raw, padded):
        ax_dict[name].imshow(digit, "gray")
        ax_dict[name].axis("off")

    plt.show()


def invert_image(image):
    return  255 - normalize(image)

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

    image = cv2.imread(PATH_TO_IMAGE)
   
    original_image = Image.fromarray(image)
   
    area_of_interest = crop_and_straigthen(original_image, POINTS)
    digits = split_into_individual_digits(area_of_interest, 7)
    gray_digits = [ImageOps.grayscale(digit) for digit in digits]
    
    digit_arrays = [np.array(d) for d in gray_digits]
    print(digit_arrays[0].shape)
    inverted = [invert_image(p) for p in digit_arrays]

    predictions = predict(inverted)
    plot_results(image, np.array(area_of_interest), digit_arrays, inverted, predictions)


main()
