# src/data_filter.py

import pandas as pd
from src.load_dataset import load_jsonl_dataset

IMPORTANT_FIELDS = ["Summary", "Experience", "Education", "Skills", "Text"]

def load_and_filter() -> pd.DataFrame:
    df = load_jsonl_dataset()

    keep_mask = df[IMPORTANT_FIELDS].notna().any(axis=1)
    df = df[keep_mask].copy()

    print(f"[data_filter] after basic filter: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_and_filter()
    print(df.head(5))
