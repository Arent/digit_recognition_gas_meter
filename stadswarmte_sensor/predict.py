from pathlib import Path

from PIL import Image

from stadswarmte_sensor import app_settings, image_to_digits


def main():

    for file in Path("meterkast_images").iterdir():
        if not str(file).endswith("jpg"):
            continue
        original_image = Image.open("meterkast_data/example_image__0334630.png")
        predictions = image_to_digits.to_individual_digits(
            original_image, app_settings.DigitRecognitionSettings()
        )
        print(f"Predicted digits: {predictions} ")


main()
