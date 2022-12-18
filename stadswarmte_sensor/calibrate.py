import matplotlib.pyplot as plt
from matplotlib.widgets import Button, PolygonSelector, Slider
from PIL import Image

from stadswarmte_sensor import app_settings, image_to_digits


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
    selector.verts = settings.corner_points
    bprev = Button(ax_dict["button"], "REPREDICT")

    freq_slider = Slider(
        ax=ax_dict["slider"],
        label="Percentage gap",
        valmin=0.0,
        valmax=1.0,
        valinit=settings.gap_between_digits_percentage,
    )

    def repredict(event):
        print("Repredicting .... ")
        points = [(int(x), int(y)) for x, y in selector.verts]
        points_right_order = [points[i] for i in [2, 3, 0, 1]]
        (
            area_of_interest,
            digit_arrays,
        ) = image_to_digits._input_to_individual_and_processed_images(
            original_image, points_right_order, percentage_space=freq_slider.val
        )
        predictions = image_to_digits._predict(digit_arrays, settings.model_location)
        plot(
            ax_dict,
            original_image,
            area_of_interest,
            predictions,
            digit_arrays,
            row_digits,
        )
        print("Done :)")
        print(
            f"Corner points = {points_right_order}\ngap_between_digits_percentage={freq_slider.val}"
        )

    bprev.on_clicked(repredict)
    repredict(None)

    plt.show()


def main():
    original_image = Image.open(
        "meterkast_images/2022_12_11_18_19_20__no_prediction.jpg"
    )
    plot_results(original_image, app_settings.DigitRecognitionSettings())


main()
