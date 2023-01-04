from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from stadswarmte_sensor import app_settings, segment_recognition


def labels_from_name(file: str) -> list[int]:
    if "gt" not in file:
        nrs = file.split("__")[1].split(".")[0]
    else:
        nrs = file.split("__gt_")[1].split(".")[0]

    return list(map(int, list(nrs)))


def _score_file(file: str) -> int:
    image = Image.open(file)
    prediction = segment_recognition.process_and_predict(
        image, app_settings.DigitRecognitionSettings()
    )
    labels = labels_from_name(str(file))
    print(labels)
    return prediction, labels


def score():

    files = [file for file in Path("meterkast_images").iterdir() if file.is_file()]
    files += [file for file in Path("ground_truth").iterdir() if file.is_file()]

    all_predictions = []
    all_labels = []
    file_correct = []
    for file in files:
        prediction, label = _score_file(file)
        file_correct.append(prediction == label)
        all_predictions.extend(prediction)
        all_labels.extend(label)
    file_acc = np.array(file_correct).mean()
    digit_acc = (np.array(all_predictions) == np.array(all_labels)).mean()
    print(f"Overal file accuracy {file_acc:.3f}")
    print(f"Overal digit accuracy {digit_acc:.3f}")

    print("The following files are wrong! ")
    wrong_files = list(np.array(files)[np.invert(np.array(file_correct))])
    for w in wrong_files:
        print(w)
    cm = confusion_matrix(
        all_predictions,
        all_labels,
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot()
    plt.show()


def main():

    Path("predictions").mkdir(exist_ok=True)
    Path("ground_truth").mkdir(exist_ok=True)

    for file in Path("meterkast_images").iterdir():
        if not str(file).endswith("jpg"):
            continue
        original_image = Image.open(file)

        predictions = segment_recognition.predict(original_image)

        digits_string = "__pred_" + "".join([str(d) for d in predictions])
        prediction_name = file.name.replace("__no_prediction", digits_string)

        gt_nn = "__gt_" + "".join([str(d) for d in predictions])
        gt_nn_name = file.name.replace("__no_prediction", gt_nn)

        original_image.save(Path("predictions") / prediction_name)

        original_image.save(Path("ground_truth") / gt_nn_name)


score()
