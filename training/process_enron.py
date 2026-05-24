"""
Process Enron Legitimate Dataset
================================
Extracts legitimate emails from the Enron dataset to use as 
negative samples for the XGBoost aggregator training.
"""

import os
import pandas as pd
import email
from email import policy
from tqdm import tqdm

INPUT_CSV = os.path.join("emails.csv", "emails.csv")
OUTPUT_DIR = os.path.join("data", "enron_legitimate")
NUM_SAMPLES = 3000

def main():
    print("=" * 60)
    print("  Extracting Enron Legitimate Samples")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Reading first 10000 rows from {INPUT_CSV}...")
    try:
        # We only need a few thousand, so don't load the whole 1.4GB file
        df = pd.read_csv(INPUT_CSV, nrows=10000)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    print(f"Loaded {len(df)} emails.")
    
    # We want emails that look like standard user emails, not automated stuff
    count = 0
    
    # Use tqdm for progress bar
    for i, row in tqdm(df.iterrows(), total=min(NUM_SAMPLES, len(df))):
        if count >= NUM_SAMPLES:
            break
            
        raw_msg = row.get("message", "")
        if not isinstance(raw_msg, str) or not raw_msg.strip():
            continue
            
        try:
            # Parse it to make sure it's valid
            msg = email.message_from_string(raw_msg, policy=policy.default)
            
            # Simple filter for decent emails
            if not msg.get("From") or not msg.get("To") or not msg.get("Subject"):
                continue
                
            filepath = os.path.join(OUTPUT_DIR, f"enron_{count:04d}.eml")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(raw_msg)
                
            count += 1
            
        except Exception as e:
            continue
            
    print(f"\nSuccessfully extracted {count} legitimate Enron emails.")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
