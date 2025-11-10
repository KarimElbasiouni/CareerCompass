"""Majority-class baseline: always predict the most frequent training label."""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from src.config import PROCESSED_PARQUET, SPLITS_JSON, RUNS_DIR

def load_splits() -> dict:
    with open(SPLITS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def majority_label(series: pd.Series) -> str:
    return Counter(series).most_common(1)[0][0]

def main():
    df = pd.read_parquet(PROCESSED_PARQUET)
    splits = load_splits()
    
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    
    train_df = df[df["resume_id"].astype(str).isin(train_ids)]
    val_df = df[df["resume_id"].astype(str).isin(val_ids)]
    test_df = df[df["resume_id"].astype(str).isin(test_ids)]

    target = "title_raw"
    maj = majority_label(train_df[target])
    
    # Val predictions
    val_preds = [maj] * len(val_df)
    val_acc = accuracy_score(val_df[target].values, val_preds)
    val_f1 = f1_score(val_df[target].values, val_preds, average="macro", zero_division=0)
    
    # Test predictions
    test_preds = [maj] * len(test_df)
    test_acc = accuracy_score(test_df[target].values, test_preds)
    test_f1 = f1_score(test_df[target].values, test_preds, average="macro", zero_division=0)

    # Clean format matching SVM
    metrics = {
        "val": {
            "accuracy": val_acc,
            "macro_f1": val_f1
        },
        "test": {
            "accuracy": test_acc,
            "macro_f1": test_f1
        }
    }
    
    out_dir = RUNS_DIR / "majority_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[majority] val accuracy: {val_acc:.4f}, macro-F1: {val_f1:.4f}")
    print(f"[majority] test accuracy: {test_acc:.4f}, macro-F1: {test_f1:.4f}")
    print(f"[majority] metrics -> {out_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
