import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import copy
from PIL import Image

import PIL
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


PATH_TO_IMAGE = "input.jpg"
MODEL_NAME = "farleyknight-org-username/vit-base-mnist"
MODEL_NAME = "Karelito00/beit-base-patch16-224-pt22k-ft22k-finetuned-mnist"


def split_individual_digets(cropped_image: np.ndarray) -> list[np.ndarray]:
    digit_width = 218
    number_of_digets = 7
    space_between_digits = 0

    digits = []
    for i in range(number_of_digets -1 ):
        start_digit =  (digit_width + space_between_digits) * i
        end_digit = start_digit + digit_width
        digits.append( cropped_image[:, start_digit:end_digit])

    # Very hacky. Due to the angle, the digit width isn't constant. 
    # It's only visible for the last digit so this hack is needed
    
    start_last_diget =  (digit_width + space_between_digits) * 6 - 80
    end_last_diget =  start_last_diget + digit_width
    digits.append( cropped_image[:, start_last_diget:end_last_diget])

    return digits

def get_relevant_area(image: np.ndarray,) -> np.ndarray:
    xstart, xend = 1225, 1675
    ystart, yend = 200, 1620
    angle = 8

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped = gray[xstart:xend, ystart:yend]
    rotated =  ndimage.rotate(cropped, angle, reshape=False, mode='nearest')

    cropped_again =  rotated[90:340, :]   
    return cropped_again

def normalize(image):
    copied = copy.deepcopy(image)
    norm =  (copied - np.min(copied)) / (np.max(copied) - np.min(copied))
    norm_im = norm * 255

    return norm_im.astype(np.uint8)

def plot_results(original_image, cropped_image: np.ndarray, padded, digit_images: list[np.ndarray], predictions:list[int]) -> None:


    assert len(digit_images) == 7

    row_original = ['original'] * 7


    row_cropped = ['cropped'] * 7
    row_digits = list('0123456')
    row_digits_raw = [f"p{i}" for i in range(7)]


    all_axis = [ row_original,row_original, row_original, row_original, row_cropped, row_digits_raw, row_digits]


    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(all_axis)
    
    ax_dict["original"].imshow(original_image,)
    ax_dict["cropped"].imshow(cropped_image, 'gray')
    ax_dict["cropped"].axis('off')
    ax_dict["original"].axis('off')

    for name, digit, pred in zip(row_digits, digit_images, predictions):
        ax_dict[name].imshow(digit, 'gray', vmin=0, vmax=255)
        ax_dict[name].axis('off')
        ax_dict[name].set_title(f"Pred: {pred}")


    for name, digit in zip(row_digits_raw, padded):
        ax_dict[name].imshow(digit, 'gray', vmin=0, vmax=255)
        ax_dict[name].axis('off')

    plt.show()


def diff_to_tuple(diff: int) -> tuple[int, int]:
    if diff % 2 == 0: 
        return int(diff/2), int(diff/2)
    
    lower = int((diff - 1)/2)
    higher = int((diff + 1)/2)
    return lower,higher

def make_square(image):

    desired_shape = 28
    x_diff = desired_shape - image.shape[0]
    y_diff = desired_shape - image.shape[1]

    
    value = image.max()
    return np.pad(image, (diff_to_tuple(x_diff), diff_to_tuple(y_diff) ), mode='constant', constant_values=((value,value), (value,value)))

def downsample(image):
    return ndimage.interpolation.zoom(image,.1) #decimate resolution
    
def invert_image(image):
    inverted = 255 - image


    image_pil = Image.fromarray(inverted)
    new_image = PIL.ImageEnhance.Contrast(image_pil).enhance(2.5)
    # new_image = PIL.ImageEnhance.Brightness(new_image).enhance(0.9)

    arr = np.array(new_image)

    arr[arr < 100] = 0
    return  arr


def predict(images) -> list[int]:
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    predictions = []
    for d in images:
        d_color = np.repeat(d[..., np.newaxis], 3, axis=2)
        d_pil = Image.fromarray(d_color)

        inputs = extractor(d_pil, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        predictions.append(predicted_label)
    return predictions

def main():

    
   
    image = cv2.imread(PATH_TO_IMAGE)
    aoi = get_relevant_area(image)
    digits = split_individual_digets(aoi)
    digit_normalized = [normalize(d) for d in digits]

    downsampled = [downsample(d) for d in digit_normalized]

    padded = [make_square(i) for i in downsampled ]
    inverted = [invert_image(p) for p in padded]
    predictions = predict(inverted)
    plot_results(image, aoi, padded, inverted, predictions)

    
    
main()