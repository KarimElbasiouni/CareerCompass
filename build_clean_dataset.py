# build_clean_dataset.py

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROCESSED_PARQUET
from src.data_filter import load_and_filter
from src.label_creation import add_label_columns
from src.text_processing import add_text_clean, scrub_columns

CSV_FALLBACK = PROCESSED_DATA_DIR / "resumes_clean.csv"


def _assign_resume_ids(df: pd.DataFrame) -> pd.Series:
    ids: list[str] = []
    raw_series = df.get("ResumeID")
    for idx, raw in enumerate(raw_series if raw_series is not None else [None] * len(df), start=1):
        if isinstance(raw, str) and raw.strip():
            ids.append(raw.strip())
        elif raw is not None and not pd.isna(raw):
            ids.append(str(raw))
        else:
            ids.append(f"resume_{idx:05d}")
    return pd.Series(ids, index=df.index, name="resume_id")


def main() -> None:
    df = load_and_filter()
    df = scrub_columns(df)
    df = add_text_clean(df)

    df["text_norm"] = (
        df["text_clean"].fillna("").astype(str).str.strip()
    )
    df = df[df["text_norm"].str.len() > 0].copy()

    df = add_label_columns(df)
    df["title_raw"] = df["y_title"]
    df["resume_id"] = _assign_resume_ids(df)

    df.to_parquet(PROCESSED_PARQUET, index=False)
    df.to_csv(CSV_FALLBACK, index=False)
    print(f"[build_clean_dataset] saved canonical parquet to {PROCESSED_PARQUET}")
    print(f"[build_clean_dataset] saved CSV fallback to {CSV_FALLBACK}")
    print(f"[build_clean_dataset] final shape: {df.shape}")


if __name__ == "__main__":
    main()
