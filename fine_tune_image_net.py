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

from collections import Counter
MODEL_NAME = "google/vit-base-patch16-224-in21k"


metric = load_metric("accuracy")


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
    grid = Image.new("RGB", size=(examples_per_class * w, len(labels) * h))
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
            image = example["image"]
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
    grid.show()


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[..., np.newaxis], 3, axis=2)


def transform(example_batch, feature_extractor):

    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(
        [gray_to_rgb(np.array(x)) for x in example_batch["image"]], return_tensors="pt"
    )

    # Don't forget to include the labels!
    inputs["labels"] = example_batch["labels"]
    return inputs


def get_prepared_dataset(dataset, feature_extractor):

    part_transform = functools.partial(transform, feature_extractor=feature_extractor)
    return dataset.with_transform(part_transform)


def _get_model(prepared_dataset):
    labels = prepared_dataset["train"].features["labels"].names

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


def _trainer(model, training_args, feature_extractor, prepared_dataset):
    return Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["test"],
        tokenizer=feature_extractor,
    )


def load_dataset():

    train, test = datasets.load_dataset("mnist", split=['train[:60]' , 'test[:500]'])

    ds = datasets.DatasetDict(train=train, test=test).rename_column("label", "labels") 

    print('test', Counter(ds['test']['labels']))
    print('train', Counter(ds['train']['labels']))

    return ds

def main():

    ds = load_dataset()
    

    show_examples(ds, seed=random.randint(0, 1337), examples_per_class=3)
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)

    prepared_dataset = get_prepared_dataset(ds, feature_extractor)

    # assure that the dataset is prepared correclty
    _ = prepared_dataset["train"][0:2]["pixel_values"]

    model = _get_model(prepared_dataset)
    trainer = _trainer(model, _training_args(), feature_extractor, prepared_dataset)

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_dataset["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
