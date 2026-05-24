# ============================================================
# NLP Model Training Script — Engine 3 (RoBERTa Fine-tuning)
# ============================================================
# Fine-tunes a RoBERTa-base model for BINARY phishing
# classification on labeled email text data.
#
# Dataset Format (CSV):
#   text,label
#   "Your account has been suspended...",phishing
#   "Meeting at 3pm tomorrow",legitimate
#
# Usage:
#   python training/train_nlp_model.py \
#       --data_path data/phishing_labeled.csv \
#       --output_dir models/phishing_roberta \
#       --epochs 4 \
#       --batch_size 8
# ============================================================

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# Label mapping — BINARY classification
LABEL_MAP = {
    "legitimate": 0,
    "phishing": 1,
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def compute_metrics(eval_pred):
    """Compute evaluation metrics for the Trainer."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


def load_and_prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    test_size: float = 0.2,
):
    """
    Load CSV dataset and prepare for transformer training.

    Expected CSV columns:
        - text: Raw email body text
        - label: 'phishing' or 'legitimate'

    Returns:
        train_dataset, eval_dataset (HuggingFace Dataset objects)
    """
    from datasets import Dataset

    df = pd.read_csv(data_path)

    # Validate columns
    assert "text" in df.columns, "CSV must have a 'text' column"
    assert "label" in df.columns, "CSV must have a 'label' column"

    # Map string labels to integers
    df["label"] = df["label"].map(LABEL_MAP)
    unknown_labels = df[df["label"].isna()]
    if len(unknown_labels) > 0:
        logger.warning(
            f"Dropping {len(unknown_labels)} rows with unknown labels"
        )
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Log class distribution
    logger.info("Class distribution:")
    for label_name, label_id in LABEL_MAP.items():
        count = (df["label"] == label_id).sum()
        logger.info(f"  {label_name}: {count} ({count / len(df):.1%})")

    # Split into train/eval
    from sklearn.model_selection import train_test_split

    train_df, eval_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    # Tokenize
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    eval_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    logger.info(
        f"Dataset loaded: {len(train_dataset)} train, "
        f"{len(eval_dataset)} eval samples"
    )

    return train_dataset, eval_dataset


def compute_class_weights(train_dataset) -> list[float]:
    """
    Compute class weights for imbalanced datasets.
    
    Higher weights for underrepresented classes to prevent
    the model from being biased toward majority classes.
    """
    from collections import Counter

    labels = [item["label"].item() for item in train_dataset]
    counter = Counter(labels)
    total = len(labels)
    n_classes = len(LABEL_MAP)

    weights = []
    for i in range(n_classes):
        count = counter.get(i, 1)
        weight = total / (n_classes * count)
        weights.append(weight)

    logger.info(f"Class weights: {weights}")
    return weights


def train(
    data_path: str,
    output_dir: str,
    model_name: str = "roberta-base",
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    fp16: bool = True,
):
    """
    Fine-tune a RoBERTa model for phishing intent classification.

    Architecture:
        RoBERTa-base (125M params)
          └─ Classification Head (768 → 2)  # Binary: legitimate vs phishing

    Training pipeline:
        1. Load and tokenize labeled email dataset
        2. Compute class weights for imbalance handling
        3. Fine-tune with HuggingFace Trainer API
        4. Save best model (based on eval F1)
        5. Generate classification report
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )

    logger.info(f"Starting training: model={model_name}, epochs={epochs}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Compute class weights
    class_weights = compute_class_weights(train_dataset)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        id2label=REVERSE_LABEL_MAP,
        label2id=LABEL_MAP,
    )

    # Custom Trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fn(logits, labels)

            return (loss, outputs) if return_outputs else loss

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=50,
        fp16=fp16 and device == "cuda",
        dataloader_num_workers=2,
        report_to="none",  # Disable WandB
    )

    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Model saved to {output_dir}")

    # Final evaluation
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation: {eval_results}")

    # Detailed classification report
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    from sklearn.metrics import classification_report

    report = classification_report(
        labels,
        preds,
        target_names=list(LABEL_MAP.keys()),
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Save report
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Training Results\n{'=' * 50}\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Train Samples: {len(train_dataset)}\n")
        f.write(f"Eval Samples: {len(eval_dataset)}\n\n")
        f.write(f"Evaluation Metrics:\n")
        for k, v in eval_results.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nClassification Report:\n{report}")

    logger.info(f"Training report saved to {report_path}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Train NLP model for phishing intent classification"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CSV dataset (columns: text, label)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/phishing_roberta",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="Base model from HuggingFace (default: roberta-base)",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=500)

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
