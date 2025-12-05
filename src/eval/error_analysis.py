"""Error analysis: compare predictions across models and identify error patterns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    PROCESSED_PARQUET,
    RUNS_DIR,
    SPLITS_JSON,
)
from src.utils.splits import load_splits


def load_svm_model() -> Tuple[any, any, any]:

    model_path = MODELS_DIR / "svm_title.joblib"
    vectorizer_path = FEATURES_DIR / "tfidf_vectorizer.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found: {model_path}. Train SVM first.")
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {vectorizer_path}. Build TF-IDF features first."
        )
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {encoder_path}. Train a model first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(encoder_path)

    return model, vectorizer, encoder


def load_bert_model():
    try:
        from src.models.inference_bert import load_bert_model

        model_dir = MODELS_DIR / "bert_title"
        if not model_dir.exists():
            return None
        if not any(model_dir.glob("*.bin")) and not any(model_dir.glob("*.safetensors")):
            return None
        return load_bert_model(model_dir)
    except (ImportError, FileNotFoundError, OSError):
        return None


def get_majority_label(df_train: pd.DataFrame, label_col: str) -> str:
    """Get the majority class label from training data."""
    return Counter(df_train[label_col].astype(str)).most_common(1)[0][0]


def predict_svm(
    texts: List[str], model: any, vectorizer: any, label_encoder: any
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate SVM predictions for texts.

    Returns:
        Tuple of (predicted_indices, predicted_labels)
    """
    X = vectorizer.transform(texts)
    pred_indices = model.predict(X)
    pred_labels = label_encoder.inverse_transform(pred_indices).tolist()
    return pred_indices, pred_labels


def predict_bert(texts: List[str], bert_artifacts: Tuple) -> Tuple[np.ndarray, List[str]]:

    from src.models.inference_bert import predict_bert

    model, tokenizer, label_encoder, device = bert_artifacts
    results = predict_bert(texts, model, tokenizer, label_encoder, device)
    return results["pred_indices"], results["pred_labels"]


def analyze_errors(
    split: str = "test",
    output_dir: Path | None = None,
    include_bert: bool = True,
) -> pd.DataFrame:

    if output_dir is None:
        output_dir = RUNS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[error_analysis] Loading data and splits...")
    df = pd.read_parquet(PROCESSED_PARQUET)
    splits = load_splits()

    split_ids = set(splits[split])
    df_split = df[df["resume_id"].astype(str).isin(split_ids)].copy()
    df_split = df_split.reset_index(drop=True)

    label_col = "title_raw"
    true_labels = df_split[label_col].astype(str).values
    resume_ids = df_split["resume_id"].astype(str).values
    texts = df_split.get("text_clean", df_split.get("text", "")).fillna("").astype(str).tolist()

    print(f"[error_analysis] Analyzing {len(df_split)} samples from {split} set...")

    print(f"[error_analysis] Loading SVM model...")
    svm_model, svm_vectorizer, label_encoder = load_svm_model()

    train_ids = set(splits["train"])
    df_train = df[df["resume_id"].astype(str).isin(train_ids)]
    majority_label_str = get_majority_label(df_train, label_col)

    print(f"[error_analysis] Generating SVM predictions...")
    svm_pred_indices, svm_pred_labels = predict_svm(texts, svm_model, svm_vectorizer, label_encoder)

    print(f"[error_analysis] Generating Majority predictions...")
    majority_pred_labels = [majority_label_str] * len(df_split)

    true_indices = label_encoder.transform(true_labels)
    majority_pred_indices = label_encoder.transform(majority_pred_labels)

    bert_available = False
    bert_pred_indices = None
    bert_pred_labels = None

    if include_bert:
        print(f"[error_analysis] Attempting to load BERT model...")
        bert_artifacts = load_bert_model()
        if bert_artifacts is not None:
            print(f"[error_analysis] Generating BERT predictions...")
            bert_pred_indices, bert_pred_labels = predict_bert(texts, bert_artifacts)
            bert_available = True
        else:
            print(f"[error_analysis] BERT model not available, skipping...")

    error_data = {
        "resume_id": resume_ids,
        "true_label": true_labels,
        "svm_pred": svm_pred_labels,
        "majority_pred": majority_pred_labels,
        "svm_correct": (svm_pred_indices == true_indices),
        "majority_correct": (majority_pred_indices == true_indices),
    }

    if bert_available:
        error_data["bert_pred"] = bert_pred_labels
        error_data["bert_correct"] = (bert_pred_indices == true_indices)

    error_df = pd.DataFrame(error_data)

    print(f"[error_analysis] Categorizing errors...")

    error_df["both_wrong"] = ~error_df["svm_correct"] & ~error_df["majority_correct"]

    if bert_available:
        error_df["svm_wrong_bert_right"] = ~error_df["svm_correct"] & error_df["bert_correct"]
        error_df["svm_right_bert_wrong"] = error_df["svm_correct"] & ~error_df["bert_correct"]
        error_df["svm_bert_both_wrong"] = ~error_df["svm_correct"] & ~error_df["bert_correct"]

    error_df["text_snippet"] = [text[:200] + "..." if len(text) > 200 else text for text in texts]

    output_path = output_dir / f"error_cases_{split}.csv"
    error_df.to_csv(output_path, index=False)
    print(f"[error_analysis] Error analysis saved to: {output_path}")

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nSplit: {split}")
    print(f"Total samples: {len(error_df)}")
    print(f"\nSVM Accuracy: {error_df['svm_correct'].mean():.4f}")
    print(f"Majority Accuracy: {error_df['majority_correct'].mean():.4f}")
    if bert_available:
        print(f"BERT Accuracy: {error_df['bert_correct'].mean():.4f}")

    print(f"\nHard cases (both SVM and Majority wrong): {error_df['both_wrong'].sum()}")
    if bert_available:
        print(f"SVM wrong / BERT right (improvements): {error_df['svm_wrong_bert_right'].sum()}")
        print(f"SVM right / BERT wrong (regressions): {error_df['svm_right_bert_wrong'].sum()}")
        print(f"Both SVM and BERT wrong: {error_df['svm_bert_both_wrong'].sum()}")

    print("\n" + "=" * 80)

    return error_df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze prediction errors across models")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for error analysis CSV",
    )
    parser.add_argument(
        "--no-bert",
        action="store_true",
        help="Skip BERT analysis even if model is available",
    )

    args = parser.parse_args()
    analyze_errors(
        split=args.split,
        output_dir=args.output_dir,
        include_bert=not args.no_bert,
    )


if __name__ == "__main__":
    main()

