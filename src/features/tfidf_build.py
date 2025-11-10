"""Build TF-IDF features for the cleaned resume dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    FEATURES_DIR,
    PROCESSED_PARQUET,
    RANDOM_SEED,
    SPLITS_JSON,
)
from src.utils.splits import ensure_splits


@dataclass
class TfidfArtifacts:
    vectorizer_path: Path
    matrix_path: Path
    index_path: Path
    splits_path: Path


def _normalize_column(
    df: pd.DataFrame,
    column: str,
    fallback: str | None = None,
) -> str:
    if column in df.columns:
        return column
    if fallback and fallback in df.columns:
        return fallback
    raise ValueError(f"Column '{column}' not found and no fallback available.")


def _filter_splits(splits: Dict[str, List[str]], ids: Iterable[str]) -> Dict[str, List[str]]:
    id_set = set(ids)
    filtered: Dict[str, List[str]] = {}
    for split, values in splits.items():
        filtered_values = [rid for rid in values if rid in id_set]
        filtered[split] = filtered_values
    return filtered


def build_tfidf_features(
    parquet_path: Path = PROCESSED_PARQUET,
    text_col: str = "text_norm",
    id_col: str = "resume_id",
    label_col: str = "title_raw",
    splits_path: Path = SPLITS_JSON,
    out_dir: Path = FEATURES_DIR,
    seed: int = RANDOM_SEED,
) -> TfidfArtifacts:
    """Fit a TF-IDF vectorizer (train-only) and save aligned artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)

    id_col = _normalize_column(df, id_col, fallback="ResumeID")
    text_col = _normalize_column(df, text_col, fallback="text_clean")
    label_col = _normalize_column(df, label_col, fallback="y_title")

    df = df[[id_col, text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].copy()
    df = df.sort_values(id_col).reset_index(drop=True)

    splits = ensure_splits(df, label_col=label_col, id_col=id_col, splits_path=splits_path, seed=seed)
    splits = _filter_splits(splits, df[id_col])

    split_lookup = {rid: split for split, values in splits.items() for rid in values}
    df["split"] = df[id_col].map(split_lookup).fillna("unknown")

    train_mask = df["split"] == "train"
    if not train_mask.any():
        raise RuntimeError("No training rows found in splits; cannot fit TF-IDF.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        strip_accents="unicode",
    )
    vectorizer.fit(df.loc[train_mask, text_col].tolist())
    matrix = vectorizer.transform(df[text_col].tolist())

    matrix_path = out_dir / "tfidf_X.npz"
    index_path = out_dir / "tfidf_index.parquet"
    vectorizer_path = out_dir / "tfidf_vectorizer.joblib"

    sp.save_npz(matrix_path, matrix)
    joblib.dump(vectorizer, vectorizer_path)
    index_df = pd.DataFrame(
        {
            "resume_id": df[id_col],
            "row_ix": range(len(df)),
            "split": df["split"],
        }
    )
    index_df.to_parquet(index_path, index=False)

    return TfidfArtifacts(
        vectorizer_path=vectorizer_path,
        matrix_path=matrix_path,
        index_path=index_path,
        splits_path=Path(splits_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TF-IDF features for resumes.")
    parser.add_argument("--in", dest="parquet_path", default=str(PROCESSED_PARQUET), help="Input parquet path.")
    parser.add_argument("--text-col", default="text_norm", help="Name of text column.")
    parser.add_argument("--id-col", default="resume_id", help="Unique ID column.")
    parser.add_argument(
        "--label-col",
        default="title_raw",
        help="Label column for stratified splits.",
    )
    parser.add_argument(
        "--splits",
        dest="splits_path",
        default=str(SPLITS_JSON),
        help="JSON file with train/val/test resume IDs.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(FEATURES_DIR),
        help="Directory where TF-IDF artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_tfidf_features(
        parquet_path=Path(args.parquet_path),
        text_col=args.text_col,
        id_col=args.id_col,
        label_col=args.label_col,
        splits_path=Path(args.splits_path),
        out_dir=Path(args.out_dir),
    )
    print(f"[tfidf] vectorizer -> {artifacts.vectorizer_path}")
    print(f"[tfidf] matrix     -> {artifacts.matrix_path}")
    print(f"[tfidf] index      -> {artifacts.index_path}")
    print(f"[tfidf] splits     -> {artifacts.splits_path}")


if __name__ == "__main__":
    main()
