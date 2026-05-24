# ============================================================
# Dataset Preparation Script — Kaggle Phishing Email Dataset
# ============================================================
# Converts the raw Kaggle CSV into the format expected by
# the NLP training script (train_nlp_model.py).
#
# Input:  data/Phishing_Email.csv (downloaded from Kaggle)
# Output: data/phishing_labeled.csv (pipeline-ready format)
#
# Kaggle Dataset: https://www.kaggle.com/datasets/subhajournal/phishingemails
#
# Usage:
#   python training/prepare_dataset.py \
#       --input_path data/Phishing_Email.csv \
#       --output_path data/phishing_labeled.csv
# ============================================================

from __future__ import annotations

import argparse
import os

import pandas as pd
from loguru import logger


# Kaggle dataset label mapping → our pipeline format
KAGGLE_LABEL_MAP = {
    "Safe Email": "legitimate",
    "Phishing Email": "phishing",
    # Alternative column values (some versions use different naming)
    "safe email": "legitimate",
    "phishing email": "phishing",
    "legitimate": "legitimate",
    "phishing": "phishing",
    "ham": "legitimate",
    "spam": "phishing",
    "0": "legitimate",
    "1": "phishing",
    0: "legitimate",
    1: "phishing",
}


def prepare_dataset(
    input_path: str,
    output_path: str,
    text_column: str = None,
    label_column: str = None,
    max_samples: int = None,
) -> None:
    """
    Convert a raw email dataset CSV into the format expected
    by the NLP training pipeline.

    Auto-detects common column naming conventions:
    - text / text_combined / email_text / body / message / content
    - label / labels / class / category / target

    Output format:
        text,label
        "email body text...",phishing
        "meeting tomorrow at 3pm",legitimate

    Args:
        input_path:   Path to raw CSV file
        output_path:  Path to save prepared CSV
        text_column:  Override auto-detection for text column name
        label_column: Override auto-detection for label column name
        max_samples:  Optional limit on total samples (balanced)
    """
    logger.info(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Columns found: {list(df.columns)}")

    # ── Auto-detect text column ──────────────────────────────
    TEXT_CANDIDATES = [
        "text_combined", "text", "email_text", "body",
        "message", "content", "email_body", "email",
        "Text", "Email_Text", "Body", "Message",
    ]

    if text_column:
        assert text_column in df.columns, f"Column '{text_column}' not found"
        text_col = text_column
    else:
        text_col = None
        for candidate in TEXT_CANDIDATES:
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            raise ValueError(
                f"Could not auto-detect text column. "
                f"Available columns: {list(df.columns)}. "
                f"Use --text_column to specify manually."
            )

    logger.info(f"Using text column: '{text_col}'")

    # ── Auto-detect label column ─────────────────────────────
    LABEL_CANDIDATES = [
        "label", "labels", "class", "category", "target",
        "Label", "Class", "Category", "Target", "is_phishing",
    ]

    if label_column:
        assert label_column in df.columns, f"Column '{label_column}' not found"
        label_col = label_column
    else:
        label_col = None
        for candidate in LABEL_CANDIDATES:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError(
                f"Could not auto-detect label column. "
                f"Available columns: {list(df.columns)}. "
                f"Use --label_column to specify manually."
            )

    logger.info(f"Using label column: '{label_col}'")

    # ── Rename and standardize ───────────────────────────────
    result = pd.DataFrame()
    result["text"] = df[text_col]
    result["label"] = df[label_col]

    # Map labels to standard format
    result["label"] = result["label"].map(KAGGLE_LABEL_MAP)

    # Check for unmapped labels
    unmapped = result[result["label"].isna()]
    if len(unmapped) > 0:
        unique_unmapped = df.loc[unmapped.index, label_col].unique()
        logger.warning(
            f"Found {len(unmapped)} rows with unmapped labels: {unique_unmapped}"
        )
        logger.warning("Dropping unmapped rows")
        result = result.dropna(subset=["label"])

    # Drop rows with empty text
    result = result[result["text"].notna() & (result["text"].str.strip() != "")]

    # ── Balancing (optional) ─────────────────────────────────
    if max_samples:
        per_class = max_samples // 2
        phishing = result[result["label"] == "phishing"].sample(
            n=min(per_class, len(result[result["label"] == "phishing"])),
            random_state=42,
        )
        legitimate = result[result["label"] == "legitimate"].sample(
            n=min(per_class, len(result[result["label"] == "legitimate"])),
            random_state=42,
        )
        result = pd.concat([phishing, legitimate]).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        logger.info(f"Balanced to {len(result)} samples ({per_class} per class)")

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)

    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Dataset prepared successfully!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total samples: {len(result)}")
    logger.info(f"Class distribution:")
    for label, count in result["label"].value_counts().items():
        pct = count / len(result) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    logger.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare email dataset for NLP training pipeline"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to raw CSV dataset (e.g., Kaggle download)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/phishing_labeled.csv",
        help="Path to save the prepared dataset",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Override: name of the text/email body column",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Override: name of the label/class column",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit total samples (balanced between classes)",
    )

    args = parser.parse_args()

    prepare_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        text_column=args.text_column,
        label_column=args.label_column,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
