"""
Train URL Risk Model on LegitPhish Dataset
==========================================
Trains an XGBoost model on 101K URLs to score phishing risk.
This model powers Engine 4's ML-based URL analysis.

Usage:
    python training/improve_engine4.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def main():
    print("=" * 60)
    print("  Training URL Risk Model (LegitPhish Dataset)")
    print("=" * 60)

    # Load dataset
    csv_path = os.path.join(
        "LegitPhish Dataset", "LegitPhish Dataset", "url_features_extracted1.csv"
    )
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} URLs")
    print(f"  Phishing: {(df['ClassLabel'] == 0).sum()}")
    print(f"  Legitimate: {(df['ClassLabel'] == 1).sum()}")

    # Drop URL column and NaN rows
    feature_cols = [
        "url_length", "has_ip_address", "dot_count", "https_flag",
        "url_entropy", "token_count", "subdomain_count", "query_param_count",
        "tld_length", "path_length", "has_hyphen_in_domain", "number_of_digits",
        "tld_popularity", "suspicious_file_extension", "domain_name_length",
        "percentage_numeric_chars",
    ]

    df = df.dropna(subset=feature_cols + ["ClassLabel"])
    X = df[feature_cols].values
    y = df["ClassLabel"].values.astype(int)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples after cleanup: {len(X)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  URL Risk Model Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()
    print(classification_report(y_test, y_pred, target_names=["Phishing", "Legitimate"]))

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("  Top Features:")
    for feat, imp in feat_imp[:8]:
        print(f"    {feat:30s} {imp:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "url_risk_model.json")
    model.save_model(model_path)
    print(f"\n  Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
