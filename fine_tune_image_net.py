import functools
import datasets
import random
from PIL import Image
from transformers import (
    ViTFeatureExtractor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch

import numpy as np
from datasets import load_metric
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


from collections import Counter
MODEL_NAME = "google/vit-base-patch16-224-in21k"


metric = load_metric("accuracy")


def image_to_inputs(image: Image, size = (224,224), image_mean = (0.5, 0.5, 0.5), image_std =(0.5, 0.5, 0.5)):
    if not isinstance(image, torch.Tensor):
        image = TF.to_tensor(image)[0, ...]


    resized = TF.resize(image[None, ...], size=size)[0]
    tensor_3_channels = torch.stack([resized,resized,resized],0)
    normalized = TF.normalize(tensor_3_channels, image_mean, image_std)
    return  normalized



def augment_image(image, pixel_noise = 0.25):
    if not isinstance(image, torch.Tensor):
        image = TF.to_tensor(image)[0, ...]

    less_contrast = get_contrast_adjuster()(image[None, ...])
    added_noise= add_noise(pixel_noise)(less_contrast)
    return added_noise[0, ...]

def add_noise(noise):
    def add_noise_to_image(img):
        std = np.abs(np.random.uniform(0, noise))
        return torch.min(img + torch.max(torch.randn(img.size()), torch.tensor(0.0)) * std, torch.tensor(1.0))

    return add_noise_to_image

def get_contrast_adjuster():

    def adjust_contrast(image):
        factor = 1 - np.random.uniform(0, 0.85)
        return TF.adjust_contrast(image, contrast_factor=factor)
    return adjust_contrast

def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

    # with Image.open("mnist_example.jpg") as im:

    w, h = size
    labels = ds["train"].features["labels"].names
    grid_original = Image.new("RGB", size=(examples_per_class * w, len(labels) * h))
    grid_original_input = Image.new("RGB", size=(examples_per_class * w, len(labels) * h))

    grid_augmented_input = Image.new("RGB", size=(examples_per_class * w, len(labels) * h))
    grid_augmented = Image.new("RGB", size=(examples_per_class * w, len(labels) * h))
    for label_id, _ in enumerate(labels):

        # Filter the dataset by a single label, shuffle it, and grab a few samples
        ds_slice = (
            ds["train"]
            .filter(lambda ex: ex["labels"] == label_id)
            .shuffle(seed)
            .select(range(examples_per_class))
        )

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):

            orignal_image = example['image']
            original_image_inputs = TF.to_pil_image(image_to_inputs(example['image']))
            augmented_ = augment_image(example['image'])
            augmented_inputs_ = image_to_inputs(augmented_)
            augmented, augmented_inputs = TF.to_pil_image(augmented_), TF.to_pil_image(augmented_inputs_) 
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)


            grid_original.paste(orignal_image.resize(size), box=box)
            grid_original_input.paste(original_image_inputs.resize(size), box=box)

            grid_augmented.paste(augmented.resize(size), box=box)
            grid_augmented_input.paste(augmented_inputs.resize(size), box=box)

            
    grid_original.show()
    grid_original_input.show()
    grid_augmented.show()
    grid_augmented_input.show()


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[..., np.newaxis], 3, axis=2)



def transform_without_augmentation(example_batch):
    images = [image_to_inputs(im) for im in example_batch["image"]]

    
    return {"pixel_values": images, "labels": example_batch["labels"]}


def transform_with_augmentation(example_batch):
    

    images = [image_to_inputs(augment_image(im)) for im in example_batch["image"]]

    
    return {"pixel_values": images, "labels": example_batch["labels"]}

def transform(example_batch, feature_extractor):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(
        [gray_to_rgb(np.array(x)) for x in example_batch["image"]], return_tensors="pt"
    )

    # Don't forget to include the labels!
    inputs["labels"] = example_batch["labels"]
    return inputs


def get_prepared_dataset(dataset, feature_extractor):
    train = dataset["train"].with_transform(transform_with_augmentation)
    val_with = dataset["val"].with_transform(transform_with_augmentation)


    test_with = dataset["test"].with_transform(transform_with_augmentation)
    test_without = dataset["test"].with_transform(transform_without_augmentation)

    return train, val_with, test_with, test_without


def _get_model(train):
    labels = train.features["labels"].names

    return ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
    )


def _training_args():
    return TrainingArguments(
        output_dir="./vit-base-mnist-regular",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=1, 
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )


def _trainer(model, training_args, feature_extractor, train, val):
    return Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=feature_extractor,
    )


def load_dataset():

    train, val, test = datasets.load_dataset("mnist", split=['train[:1%]' , 'test[:5%]', 'test[5%:10%]'])

    ds = datasets.DatasetDict(train=train, val=val, test=test, ).rename_column("label", "labels") 

    print('test', Counter(ds['test']['labels']))
    print('train', Counter(ds['train']['labels']))

    return ds

def main():

    ds = load_dataset()
    

    show_examples(ds, seed=random.randint(0, 1337), examples_per_class=3)
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    train, val_with, test_with, test_without = get_prepared_dataset(ds, feature_extractor)

    # assure that the dataset is prepared correclty
    _ = train[0:2]["pixel_values"]

    model = _get_model(train)
    trainer = _trainer(model, _training_args(), feature_extractor, train, val_with)

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    print("Test with augmentation")
    metrics = trainer.evaluate(test_with)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print("Test without augmentation")
    metrics = trainer.evaluate(test_without)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
