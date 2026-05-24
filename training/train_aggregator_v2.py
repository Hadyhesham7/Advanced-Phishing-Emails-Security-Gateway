"""
Train XGBoost Aggregator v2
===========================
Trains the final 24-feature XGBoost aggregator using:
- Synthetic Phishing (label 1)
- Synthetic Clean (label 0)
- Enron Legitimate (label 0)
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from pipeline.analyzer import PhishingAnalyzer

def main():
    print("=" * 60)
    print("  Training XGBoost Aggregator (24 Features)")
    print("=" * 60)
    
    # Initialize analyzer (which loads the 4 engines)
    # We will use this to generate the feature vectors
    print("Initializing PhishingAnalyzer...")
    analyzer = PhishingAnalyzer("config/settings.yaml")
    
    # Paths
    phish_dir = os.path.join("data", "synthetic_phishing")
    clean_dir = os.path.join("data", "synthetic_clean")
    enron_dir = os.path.join("data", "enron_legitimate")
    
    phish_files = glob.glob(os.path.join(phish_dir, "*.eml"))
    clean_files = glob.glob(os.path.join(clean_dir, "*.eml"))
    enron_files = glob.glob(os.path.join(enron_dir, "*.eml"))
    
    print(f"Found datasets:")
    print(f"  - Phishing: {len(phish_files)} files")
    print(f"  - Clean:    {len(clean_files)} files")
    print(f"  - Enron:    {len(enron_files)} files")
    
    # Balance datasets (Optional: we can use all)
    # Let's use all to get maximum variety
    
    features_list = []
    labels = []
    
    def process_files(files, label, desc):
        for fpath in tqdm(files, desc=desc):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    raw_email = f.read()
                
                # Run the 4 engines
                parsed = analyzer._parse_email(raw_email)
                
                header_result = analyzer.engine_header.analyze(raw_email)
                
                structure_result = analyzer.engine_structure.analyze(
                    html_body=parsed.get("html_body"),
                    plain_text=parsed.get("plain_text"),
                    attachments=parsed.get("attachments", [])
                )
                
                nlp_input = structure_result.cleaned_text or parsed.get("plain_text", "")
                nlp_result = analyzer.engine_nlp.analyze(nlp_input)
                
                link_result = analyzer.engine_links.analyze(
                    html_body=parsed.get("html_body"),
                    plain_text=parsed.get("plain_text")
                )
                
                # Generate 24-feature vector
                vec = analyzer.aggregator.build_feature_vector(
                    header_result, structure_result, nlp_result, link_result
                )
                
                features_list.append(vec.to_feature_array())
                labels.append(label)
            except Exception as e:
                # print(f"Error processing {fpath}: {e}")
                pass

    process_files(phish_files, 1, "Processing Phishing")
    process_files(clean_files, 0, "Processing Clean")
    process_files(enron_files, 0, "Processing Enron")
    
    if not features_list:
        print("Error: No feature vectors generated.")
        return
        
    X = np.array(features_list)
    y = np.array(labels)
    
    print(f"\nExtracted {len(X)} feature vectors.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
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
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'=' * 60}")
    print(f"  XGBoost Aggregator Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "aggregator_xgb.json")
    model.save_model(model_path)
    
    print(f"  Model saved to: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
