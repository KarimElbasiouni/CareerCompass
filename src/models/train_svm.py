"""Train a LinearSVC baseline on resume TF-IDF features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    PROCESSED_PARQUET,
    RANDOM_SEED,
    RUNS_DIR,
    SPLITS_JSON,
)
from src.eval.metrics import accuracy, confusion, macro_f1, topk_from_scores
from src.features.tfidf_build import build_tfidf_features

DEFAULT_RUN_DIR = RUNS_DIR / "svm_tfidf"


def _maybe_build_tfidf(args) -> None:
    tfidf_dir = Path(args.tfidf_dir)
    matrix_path = tfidf_dir / "tfidf_X.npz"
    index_path = tfidf_dir / "tfidf_index.parquet"
    vectorizer_path = tfidf_dir / "tfidf_vectorizer.joblib"
    if matrix_path.exists() and index_path.exists() and vectorizer_path.exists():
        return
    print("[train_svm] TF-IDF artifacts missing; building them now...")
    build_tfidf_features(
        parquet_path=Path(args.parquet),
        text_col=args.text_col,
        id_col=args.id_col,
        label_col=args.label_col,
        splits_path=Path(args.splits),
        out_dir=tfidf_dir,
    )


def _load_artifacts(tfidf_dir: Path) -> Tuple[sp.csr_matrix, pd.DataFrame]:
    matrix = sp.load_npz(tfidf_dir / "tfidf_X.npz").tocsr()
    index_df = pd.read_parquet(tfidf_dir / "tfidf_index.parquet")
    return matrix, index_df


def _prepare_labels(
    parquet_path: Path,
    id_col: str,
    label_col: str,
    order_df: pd.DataFrame,
) -> pd.Series:
    df = pd.read_parquet(parquet_path, columns=[id_col, label_col])
    if id_col not in df.columns:
        fallback = "ResumeID"
        if fallback not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found and no fallback available.")
        df[fallback] = df[fallback].astype(str).str.strip()
        df = df.rename(columns={fallback: id_col})
    df[id_col] = df[id_col].astype(str).str.strip()
    if label_col not in df.columns:
        fallback = "y_title"
        if fallback not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found. Tried fallback '{fallback}' but missing too.")
        df = df.rename(columns={fallback: label_col})
    df = df[[id_col, label_col]]
    label_lookup = df.set_index(id_col)[label_col].to_dict()
    ordered = order_df["resume_id"].map(label_lookup)
    ordered.index = order_df.index
    return ordered


def _evaluate_split(
    model: LinearSVC,
    X_split: sp.csr_matrix,
    y_split: np.ndarray,
    class_labels: np.ndarray,
    out_dir: Path,
    split_name: str,
) -> Dict[str, float]:
    if X_split.shape[0] == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "top1": 0.0, "top3": 0.0}
    preds = model.predict(X_split)
    scores = model.decision_function(X_split)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    metrics = {
        "accuracy": accuracy(y_split, preds),
        "macro_f1": macro_f1(y_split, preds),
        "top1": topk_from_scores(y_split, scores, 1),
        "top3": topk_from_scores(y_split, scores, min(3, len(class_labels))),
    }
    cm, fig = confusion(
        y_split,
        preds,
        labels=range(len(class_labels)),
        display_labels=class_labels,
    )
    fig_path = out_dir / f"confusion_{split_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics


def _train_model(X_train, y_train, C: float) -> LinearSVC:
    clf = LinearSVC(
        C=C,
        class_weight="balanced",
        dual=True,
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LinearSVC on TF-IDF features.")
    parser.add_argument("--parquet", default=str(PROCESSED_PARQUET), help="Input parquet with cleaned data.")
    parser.add_argument("--label-col", default="title_raw", help="Label column to use.")
    parser.add_argument("--id-col", default="resume_id", help="Resume ID column.")
    parser.add_argument("--text-col", default="text_norm", help="Text column used for TF-IDF (if rebuilding).")
    parser.add_argument("--splits", default=str(SPLITS_JSON), help="Splits JSON path.")
    parser.add_argument("--tfidf-dir", default=str(FEATURES_DIR), help="Directory holding TF-IDF artifacts.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Where to save trained SVM + label encoder.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUN_DIR), help="Where to write metrics/plots.")
    parser.add_argument("--tune", action="store_true", help="Enable quick grid-search on C over the val split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tfidf_dir = str(Path(args.tfidf_dir))
    args.models_dir = str(Path(args.models_dir))
    args.runs_dir = str(Path(args.runs_dir))

    _maybe_build_tfidf(args)

    tfidf_dir = Path(args.tfidf_dir)
    runs_dir = Path(args.runs_dir)
    models_dir = Path(args.models_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    matrix, index_df = _load_artifacts(tfidf_dir)
    label_series = _prepare_labels(
        parquet_path=Path(args.parquet),
        id_col=args.id_col,
        label_col=args.label_col,
        order_df=index_df,
    )

    valid_mask = label_series.notna() & label_series.astype(str).str.len().gt(0)
    matrix = matrix[valid_mask.to_numpy()]
    index_df = index_df[valid_mask].reset_index(drop=True)
    label_series = label_series[valid_mask].reset_index(drop=True)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(label_series.astype(str))

    split_indices = {
        split: np.where(index_df["split"] == split)[0]
        for split in ("train", "val", "test")
    }

    def subset(split_name: str) -> Tuple[sp.csr_matrix, np.ndarray]:
        rows = split_indices[split_name]
        return matrix[rows], y_all[rows]

    X_train, y_train = subset("train")
    X_val, y_val = subset("val")
    X_test, y_test = subset("test")

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise RuntimeError("Not enough data in train/val splits to train LinearSVC.")

    grid = [1.0]
    if args.tune:
        grid = [0.25, 0.5, 1.0, 2.0]

    best_C = grid[0]
    best_score = -np.inf
    best_model = None
    for C in grid:
        model = _train_model(X_train, y_train, C=C)
        val_preds = model.predict(X_val)
        score = macro_f1(y_val, val_preds)
        if score > best_score:
            best_score = score
            best_C = C
            best_model = model

    assert best_model is not None

    metrics = {
        "best_C": best_C,
        "val": _evaluate_split(best_model, X_val, y_val, encoder.classes_, runs_dir, "val"),
        "test": _evaluate_split(best_model, X_test, y_test, encoder.classes_, runs_dir, "test"),
    }

    model_path = models_dir / "svm_title.joblib"
    label_path = models_dir / "label_encoder.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(encoder, label_path)

    metrics_path = runs_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_svm] train shape={X_train.shape}, val shape={X_val.shape}, test shape={X_test.shape}")
    print(f"[train_svm] best C={best_C}")
    for split in ("val", "test"):
        m = metrics[split]
        print(
            f"[train_svm][{split}] acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
            f"top1={m['top1']:.3f} top3={m['top3']:.3f}"
        )
    print(f"[train_svm] model -> {model_path}")
    print(f"[train_svm] label encoder -> {label_path}")
    print(f"[train_svm] metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
