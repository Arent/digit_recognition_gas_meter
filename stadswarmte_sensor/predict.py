from PIL import Image

from stadswarmte_sensor import app_settings, image_to_digits


def main():
    name = "meterkast_data/example_image__0334630.png"

    original_image = Image.open("meterkast_data/example_image__0334630.png")
    predictions = image_to_digits.to_individual_digits(
        original_image, app_settings.DigitRecognitionSettings()
    )

    labels = [int(p) for p in name.split("__")[-1].replace(".png", "")]

    print(f"Predicted digits: {predictions} ")

    for pred, label in zip(predictions, labels):
        print(f"Predicted: {pred}, label: {label} correct: {label==pred}")


main()
