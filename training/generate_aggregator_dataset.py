# ============================================================
# Dataset Generation Script
# ============================================================
# Runs the 4-engine pipeline on a labeled email corpus and
# generates the tabular training dataset for the ML Aggregator.
#
# Input:  Directory of .eml files organized as:
#           data/emails/phishing/*.eml
#           data/emails/legitimate/*.eml
#
# Output: CSV with 24 features + label column
#
# Usage:
#   python training/generate_aggregator_dataset.py \
#       --email_dir data/emails \
#       --output_path data/aggregator_training.csv
# ============================================================

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.analyzer import PhishingAnalyzer
from pipeline.models import AggregatorFeatureVector


def generate_dataset(
    email_dir: str,
    output_path: str,
    config_path: str = None,
) -> None:
    """
    Process all labeled emails through the pipeline and
    record their feature vectors for aggregator training.

    Expected directory structure:
        email_dir/
        ├── phishing/       # .eml files labeled as phishing
        │   ├── sample1.eml
        │   └── sample2.eml
        └── legitimate/     # .eml files labeled as legitimate
            ├── sample3.eml
            └── sample4.eml
    """
    analyzer = PhishingAnalyzer(config_path)
    rows = []
    errors = 0

    # Process phishing emails
    phishing_dir = os.path.join(email_dir, "phishing")
    phishing_files = glob.glob(os.path.join(phishing_dir, "*.eml"))
    logger.info(f"Found {len(phishing_files)} phishing emails")

    for filepath in phishing_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_email = f.read()
            verdict = analyzer.analyze(raw_email)
            features = verdict.feature_vector.to_feature_array()
            features.append(1)  # label = phishing
            rows.append(features)
        except Exception as e:
            errors += 1
            logger.debug(f"Error processing {filepath}: {e}")

    # Process legitimate emails
    legit_dir = os.path.join(email_dir, "legitimate")
    legit_files = glob.glob(os.path.join(legit_dir, "*.eml"))
    logger.info(f"Found {len(legit_files)} legitimate emails")

    for filepath in legit_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_email = f.read()
            verdict = analyzer.analyze(raw_email)
            features = verdict.feature_vector.to_feature_array()
            features.append(0)  # label = legitimate
            rows.append(features)
        except Exception as e:
            errors += 1
            logger.debug(f"Error processing {filepath}: {e}")

    # Create DataFrame
    columns = AggregatorFeatureVector.feature_names() + ["label"]
    df = pd.DataFrame(rows, columns=columns)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        f"\nDataset generated: {len(df)} samples "
        f"({(df['label'] == 1).sum()} phishing, "
        f"{(df['label'] == 0).sum()} legitimate)"
    )
    logger.info(f"Errors: {errors}")
    logger.info(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregator training dataset from labeled emails"
    )
    parser.add_argument(
        "--email_dir",
        type=str,
        required=True,
        help="Directory with phishing/ and legitimate/ subdirs",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/aggregator_training.csv",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional pipeline config YAML",
    )

    args = parser.parse_args()
    generate_dataset(args.email_dir, args.output_path, args.config_path)


if __name__ == "__main__":
    main()
