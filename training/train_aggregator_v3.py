"""
Train XGBoost Aggregator v3 — 31 Features
==========================================
Final training script for the hardened pipeline.
Uses 31-feature vector from all patched engines.
Includes macro attachment detection.

Sources:
  - Synthetic Phishing (label 1)
  - Synthetic Clean (label 0)
  - Enron Legitimate (label 0)
"""

import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, ".")
from pipeline.analyzer import PhishingAnalyzer
from pipeline.models import AggregatorFeatureVector


def extract_features(analyzer, raw_email):
    """Run all 4 engines and extract the 30-feature vector."""
    parsed = analyzer._parse_email(raw_email)

    header_result = analyzer.engine_header.analyze(raw_email)

    structure_result = analyzer.engine_structure.analyze(
        html_body=parsed.get("html_body"),
        plain_text=parsed.get("plain_text"),
        attachments=parsed.get("attachments", []),
    )

    nlp_input = structure_result.cleaned_text or parsed.get("plain_text") or ""
    if structure_result.ocr_extracted_text:
        nlp_input += " " + structure_result.ocr_extracted_text
    nlp_result = analyzer.engine_nlp.analyze(nlp_input)

    link_result = analyzer.engine_links.analyze(
        html_body=parsed.get("html_body"),
        plain_text=parsed.get("plain_text"),
    )

    vec = analyzer.aggregator.build_feature_vector(
        header_result, structure_result, nlp_result, link_result
    )
    return vec.to_feature_array()


def main():
    start_time = time.time()

    print("=" * 60)
    print("  Training XGBoost Aggregator v3 (30 Features)")
    print("=" * 60)

    # Verify feature vector size
    n_features = len(AggregatorFeatureVector().to_feature_array())
    feature_names = AggregatorFeatureVector.feature_names()
    print(f"\n  Feature vector size: {n_features}")
    print(f"  Feature names: {len(feature_names)}")
    assert n_features == 31, f"Expected 31 features, got {n_features}"

    print("\nInitializing PhishingAnalyzer...")
    analyzer = PhishingAnalyzer("config/settings.yaml")

    # ── Discover datasets ──
    phish_dir = os.path.join("data", "synthetic_phishing")
    clean_dir = os.path.join("data", "synthetic_clean")
    enron_dir = os.path.join("data", "enron_legitimate")

    phish_files = sorted(glob.glob(os.path.join(phish_dir, "*.eml")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.eml")))
    enron_files = sorted(glob.glob(os.path.join(enron_dir, "*.eml")))

    print(f"\nDatasets found:")
    print(f"  Phishing:  {len(phish_files)} files")
    print(f"  Clean:     {len(clean_files)} files")
    print(f"  Enron:     {len(enron_files)} files")
    print(f"  Total:     {len(phish_files) + len(clean_files) + len(enron_files)} files")

    if not phish_files:
        print("\n❌ No phishing files found. Run generate_synthetic_phishing.py first.")
        return

    # ── Feature extraction ──
    features_list = []
    labels = []
    errors = 0

    def process_files(files, label, desc):
        nonlocal errors
        for fpath in tqdm(files, desc=desc, ncols=80):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    raw_email = f.read()
                feats = extract_features(analyzer, raw_email)
                features_list.append(feats)
                labels.append(label)
            except Exception as e:
                errors += 1

    process_files(phish_files, 1, "Processing Phishing")
    process_files(clean_files, 0, "Processing Clean")
    process_files(enron_files, 0, "Processing Enron")

    if not features_list:
        print("\n❌ No feature vectors extracted. Check dataset paths.")
        return

    X = np.array(features_list)
    y = np.array(labels)

    extraction_time = time.time() - start_time

    print(f"\n  Feature extraction complete:")
    print(f"  Samples: {len(X)} ({sum(y)} phishing, {len(y) - sum(y)} legitimate)")
    print(f"  Errors:  {errors} files skipped")
    print(f"  Time:    {extraction_time:.0f}s")

    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # ── XGBoost Training ──
    print("\n  Training XGBoost (Deep)...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ──
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  XGBoost Aggregator v3 — Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # ── Feature Importance ──
    importance = model.feature_importances_
    importance_pairs = sorted(
        zip(feature_names, importance), key=lambda x: x[1], reverse=True
    )
    print("  Top 10 Features by Importance:")
    for name, imp in importance_pairs[:10]:
        bar = "█" * int(imp * 50)
        print(f"    {name:30s} {imp:.4f} {bar}")

    # ── Save Model ──
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "aggregator_xgb.json")
    model.save_model(model_path)

    # Save metadata
    metadata = {
        "version": "v3",
        "n_features": n_features,
        "feature_names": feature_names,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time_seconds": round(total_time, 1),
        "confusion_matrix": cm.tolist(),
    }

    meta_path = os.path.join("models", "aggregator_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved to: {model_path}")
    print(f"  Metadata saved to: {meta_path}")
    print(f"  Total training time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
