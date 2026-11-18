
import random
import time
from pathlib import Path

import torch
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ===== 實驗設定 =====
MODEL_PATH = "./model"
DATA_PATH = Path("/data")
SAMPLES_PER_CLASS = 1000
BATCH_SIZE = 16
SEED = 42
MAX_LENGTH = 128


def load_samples():
    rng = random.Random(SEED)
    texts = []
    labels = []

    for folder, label in [
        ("benign_samples", 0),
        ("phish_samples", 1),
    ]:
        files = sorted((DATA_PATH / folder).glob("*.txt"))
        selected = rng.sample(files, min(SAMPLES_PER_CLASS, len(files)))

        for path in selected:
            text = path.read_text(encoding="utf-8", errors="ignore")
            texts.append(text)
            labels.append(label)

        print(f"{folder}: {len(selected)} samples")

    return texts, labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = MobileBertTokenizer.from_pretrained(MODEL_PATH)
    model = MobileBertForSequenceClassification.from_pretrained(
        MODEL_PATH
    ).to(device)
    model.eval()

    texts, labels = load_samples()
    predictions = []

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().tolist())

            if (i // BATCH_SIZE + 1) % 20 == 0:
                print(f"Processed: {len(predictions)}/{len(texts)}")

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n===== PhishLang Evaluation =====")
    print(f"Total samples: {len(labels)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Time:      {elapsed:.2f} seconds")
    print(f"Throughput: {len(labels) / elapsed:.2f} samples/sec")

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Phishing"],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
