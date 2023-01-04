import dataclasses

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, PolygonSelector, Slider
from PIL import Image

from stadswarmte_sensor import (app_settings, image_to_digits,
                                segment_recognition)


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
    original_image, settings: app_settings.DigitRecognitionSettings
) -> None:

    row_original = ["original"] * 7

    row_cropped = ["cropped"] * 7
    row_digits = list("0123456")

    row_button = (
        ["button", "button"]
        + ["stripe_width", "stripe_height", "middle_gap"]
        + ["nothing"] * 2
    )

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

    vertices_order = [settings.corner_points[i] for i in [2, 3, 0, 1]]
    selector.verts = vertices_order
    bprev = Button(ax_dict["button"], "REPREDICT")

    stripe_height_slider = Slider(
        ax=ax_dict["stripe_height"],
        label="stripe_height",
        valstep=0.05,
        valmin=0,
        valmax=0.5,
        valinit=settings.stripe_height_percentage,
    )
    stripe_width_slider = Slider(
        ax=ax_dict["stripe_width"],
        label="stripe_width",
        valstep=0.05,
        valmin=0,
        valmax=0.5,
        valinit=settings.stripe_width_percentage,
    )

    middle_gap_slider = Slider(
        ax=ax_dict["middle_gap"],
        label="middle_gap",
        valstep=0.05,
        valmin=0,
        valmax=0.5,
        valinit=settings.gap_middle_percentage,
    )

    def repredict(event):
        print("Repredicting .... ")
        points = [(int(x), int(y)) for x, y in selector.verts]
        points_right_order = [points[i] for i in [2, 3, 0, 1]]
        (
            area_of_interest,
            digit_arrays,
        ) = image_to_digits.input_to_individual_and_processed_images(
            original_image,
            points_right_order,
            percentage_space=settings.gap_between_digits_percentage,
        )

        updated_settings = dataclasses.replace(
            settings,
            stripe_width_percentage=stripe_width_slider.val,
            stripe_height_percentage=stripe_height_slider.val,
            gap_middle_percentage=middle_gap_slider.val,
        )

        processed = segment_recognition.pre_process_digit_images(
            digit_arrays, updated_settings
        )
        predictions = segment_recognition.predict(processed, updated_settings)
        plot(
            ax_dict,
            original_image,
            area_of_interest,
            predictions,
            processed,
            row_digits,
        )
        print("Done :)")
        print(
            f"Corner points = {points_right_order}\nstripe_height={stripe_height_slider.val}\nstripe_width={stripe_width_slider.val}\nmiddle_hap={middle_gap_slider.val}"
        )

    bprev.on_clicked(repredict)
    repredict(None)

    plt.show()


def main():
    # meterkast_images/2022_12_31_19_55_07__0736520.png

    # meterkast_images/2023_01_04_11_05_42__0336747.jpg
    # meterkast_images/2023_01_04_09_05_08__0336744.jpg
    # meterkast_images/2023_01_04_14_06_32__0336751.jpg

    # meterkast_images/2022_12_31_02_08_25__0336463.png
    # meterkast_images/2022_12_31_01_37_56__0336463.png
    # ground_truth/2022_12_26_23_42_54__gt_0335923.jpg
    # ground_truth/2022_12_27_00_42_58__gt_0335923.jpg

    original_image = Image.open("meterkast_images/2022_12_31_01_37_56__0336463.png")

    plot_results(original_image, app_settings.DigitRecognitionSettings())


main()
