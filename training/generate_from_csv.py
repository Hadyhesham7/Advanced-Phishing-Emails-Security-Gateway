# ============================================================
# Generate Aggregator Training Dataset from CSV
# ============================================================
# Runs the 4-engine pipeline on labeled emails from a CSV file
# and records the 24-feature vectors for XGBoost training.
#
# Usage:
#   python training/generate_from_csv.py \
#       --csv_path data/phishing_labeled.csv \
#       --output_path data/aggregator_training.csv \
#       --max_samples 1000
# ============================================================

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.analyzer import PhishingAnalyzer
from pipeline.models import AggregatorFeatureVector


def wrap_as_email(text: str, label: str) -> str:
    """Wrap raw text into a minimal RFC 5322 email format."""
    if label == "phishing":
        from_addr = "sender@suspicious-domain.com"
        subject = "Important Notice"
    else:
        from_addr = "colleague@company.com"
        subject = "Regular message"

    return (
        f"From: {from_addr}\n"
        f"To: user@company.com\n"
        f"Subject: {subject}\n"
        f"MIME-Version: 1.0\n"
        f"Content-Type: text/plain; charset=\"UTF-8\"\n"
        f"\n"
        f"{text}\n"
    )


def generate_dataset(
    csv_path: str,
    output_path: str,
    max_samples: int = None,
) -> None:
    """
    Process labeled emails from CSV through the full pipeline
    and record feature vectors for XGBoost training.
    """
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Balance and limit samples
    if max_samples:
        per_class = max_samples // 2
        phishing = df[df["label"] == "phishing"].sample(
            n=min(per_class, len(df[df["label"] == "phishing"])),
            random_state=42,
        )
        legitimate = df[df["label"] == "legitimate"].sample(
            n=min(per_class, len(df[df["label"] == "legitimate"])),
            random_state=42,
        )
        df = pd.concat([phishing, legitimate]).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        logger.info(f"Using {len(df)} samples ({per_class} per class)")

    # Initialize pipeline
    analyzer = PhishingAnalyzer()
    rows = []
    errors = 0
    total = len(df)

    logger.info(f"Processing {total} emails through pipeline...")
    start_time = time.time()

    for idx, row in df.iterrows():
        try:
            text = str(row["text"])
            label = row["label"]

            # Wrap text as email
            raw_email = wrap_as_email(text, label)

            # Run through pipeline
            verdict = analyzer.analyze(raw_email)

            # Extract feature vector + label
            features = verdict.feature_vector.to_feature_array()
            features.append(1 if label == "phishing" else 0)
            rows.append(features)

            # Progress logging
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (total - idx - 1) / rate
                logger.info(
                    f"  Progress: {idx + 1}/{total} "
                    f"({(idx + 1) / total:.0%}) | "
                    f"Speed: {rate:.1f} emails/sec | "
                    f"ETA: {remaining / 60:.1f} min"
                )

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.debug(f"Error on row {idx}: {e}")

    # Create DataFrame
    columns = AggregatorFeatureVector.feature_names() + ["label"]
    result_df = pd.DataFrame(rows, columns=columns)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result_df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Dataset generated successfully!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total samples: {len(result_df)}")
    logger.info(f"  Phishing: {(result_df['label'] == 1).sum()}")
    logger.info(f"  Legitimate: {(result_df['label'] == 0).sum()}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Time: {elapsed / 60:.1f} minutes")
    logger.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate XGBoost training data from CSV"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/phishing_labeled.csv",
        help="Path to labeled CSV (columns: text, label)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/aggregator_training.csv",
        help="Path to save the feature dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max samples to process (balanced between classes)",
    )

    args = parser.parse_args()
    generate_dataset(args.csv_path, args.output_path, args.max_samples)


if __name__ == "__main__":
    main()
