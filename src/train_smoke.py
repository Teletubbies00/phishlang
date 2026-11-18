
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    MobileBertForSequenceClassification,
    MobileBertTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
SAMPLES_PER_CLASS = 300
MODEL_NAME = "google/mobilebert-uncased"
DATA_PATH = Path("/data")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_data():
    rng = random.Random(SEED)
    records = []

    for folder, label in [
        ("benign_samples", 0),
        ("phish_samples", 1),
    ]:
        files = sorted((DATA_PATH / folder).glob("*.txt"))
        selected = rng.sample(files, SAMPLES_PER_CLASS)

        for path in selected:
            records.append({
                "text": path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ),
                "label": label,
            })

    return records


def make_dataset(records, tokenizer):
    dataset = Dataset.from_list(records)

    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        ),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(prediction):
    labels = prediction.label_ids
    preds = prediction.predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(
            labels, preds, average="weighted"
        ),
    }


def main():
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this experiment")

    records = load_data()

    train_records, temp_records = train_test_split(
        records,
        test_size=0.2,
        stratify=[r["label"] for r in records],
        random_state=SEED,
    )

    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.5,
        stratify=[r["label"] for r in temp_records],
        random_state=SEED,
    )

    print("Train:", len(train_records))
    print("Validation:", len(val_records))
    print("Test:", len(test_records))

    tokenizer = MobileBertTokenizer.from_pretrained(MODEL_NAME)

    train_ds = make_dataset(train_records, tokenizer)
    val_ds = make_dataset(val_records, tokenizer)
    test_ds = make_dataset(test_records, tokenizer)

    model = MobileBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    args = TrainingArguments(
        output_dir="/outputs/smoke",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        fp16=False,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        logging_steps=10,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    trainer.train()

    print("\n===== Independent Test =====")
    results = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    )

    print(results)
    print("Total time:", round(time.perf_counter() - start, 2))


if __name__ == "__main__":
    main()
