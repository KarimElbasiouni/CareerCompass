"""Top-k Recall Curve visualization comparing SVM and BERT models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    PROCESSED_PARQUET,
    RUNS_DIR,
)
from src.utils.splits import load_splits


def load_svm_model() -> Tuple[any, any, LabelEncoder]:
    """
    Load SVM model, vectorizer, and label encoder.

    Returns:
        Tuple of (model, vectorizer, label_encoder)
    """
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


def load_bert_model() -> Optional[Tuple[BertForSequenceClassification, BertTokenizer, LabelEncoder, torch.device]]:
    """
    Load BERT model, tokenizer, and label encoder if available.
    Checks both bert_title and bert_finetuned directories.

    Returns:
        Tuple of (model, tokenizer, label_encoder, device) or None if not available
    """
    # Try bert_finetuned first (actual location from training)
    model_dir = MODELS_DIR / "bert_finetuned"
    if not model_dir.exists():
        # Fallback to bert_title
        model_dir = MODELS_DIR / "bert_title"
    
    if not model_dir.exists():
        return None
    
    # Check if directory is empty or missing model files
    if not any(model_dir.glob("*.bin")) and not any(model_dir.glob("*.safetensors")):
        return None
    
    try:
        print(f"[plot_topk_curve] Loading BERT model from {model_dir}")
        
        # Load model and tokenizer
        model = BertForSequenceClassification.from_pretrained(str(model_dir))
        tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        
        # Load label encoder
        encoder_path = MODELS_DIR / "label_encoder.joblib"
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found: {encoder_path}. Train a model first."
            )
        label_encoder = joblib.load(encoder_path)
        
        # Move model to device and set to eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"[plot_topk_curve] BERT model loaded on {device}")
        return model, tokenizer, label_encoder, device
    except Exception as e:
        print(f"[plot_topk_curve] Failed to load BERT model: {e}")
        return None


def compute_svm_topk_scores(
    texts: list[str],
    model: any,
    vectorizer: any,
    k_max: int = 5,
) -> np.ndarray:
    """
    Compute top-k scores for SVM using decision_function.

    Args:
        texts: List of text strings
        model: Trained SVM model
        vectorizer: TF-IDF vectorizer
        k_max: Maximum k value

    Returns:
        Array of shape (n_samples, k_max) with top-k class indices for each sample
    """
    # Transform texts to TF-IDF features
    X = vectorizer.transform(texts)
    
    # Get decision function scores (shape: n_samples x n_classes)
    scores = model.decision_function(X)
    
    # Handle binary case (scores might be 1D)
    if scores.ndim == 1:
        # Binary classification: convert to 2D
        scores = np.column_stack([-scores, scores])
    
    # Get top-k indices for each sample
    topk_indices = np.argsort(-scores, axis=1)[:, :k_max]
    
    return topk_indices


def compute_bert_topk_scores(
    texts: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    k_max: int = 5,
    max_length: int = 512,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute top-k scores for BERT using logits and softmax.

    Args:
        texts: List of text strings
        model: Fine-tuned BERT model
        tokenizer: BERT tokenizer
        device: Torch device
        k_max: Maximum k value
        max_length: Maximum sequence length
        batch_size: Batch size for inference

    Returns:
        Array of shape (n_samples, k_max) with top-k class indices for each sample
    """
    model.eval()
    all_logits = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Move to device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
        
        # Store logits (will convert to probabilities later)
        all_logits.append(logits.cpu().numpy())
    
    # Concatenate all logits
    logits = np.vstack(all_logits)
    
    # Apply softmax to get probabilities
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    
    # Get top-k indices for each sample
    topk_indices = np.argsort(-probs, axis=1)[:, :k_max]
    
    return topk_indices


def compute_recall_at_k(
    true_labels: np.ndarray,
    topk_indices: np.ndarray,
    k: int,
) -> float:
    """
    Compute recall@k: proportion of samples where true label is in top-k predictions.

    Args:
        true_labels: Array of true label indices (n_samples,)
        topk_indices: Array of top-k class indices (n_samples, k_max)
        k: Value of k (must be <= k_max)

    Returns:
        Recall@k as a float between 0 and 1
    """
    if k > topk_indices.shape[1]:
        raise ValueError(f"k={k} exceeds k_max={topk_indices.shape[1]}")
    
    # Get top-k predictions for each sample
    topk_preds = topk_indices[:, :k]
    
    # Check if true label is in top-k for each sample
    hits = np.array([
        true_labels[i] in topk_preds[i] for i in range(len(true_labels))
    ])
    
    return hits.mean()


def plot_topk_recall_curve(
    svm_model: any,
    svm_vectorizer: any,
    bert_model: Optional[Tuple[BertForSequenceClassification, BertTokenizer, LabelEncoder, torch.device]],
    label_encoder: LabelEncoder,
    df_test: pd.DataFrame,
    output_path: Path,
    k_max: int = 5,
) -> None:
    """
    Plot Top-k Recall Curve comparing SVM and BERT.

    Args:
        svm_model: Trained SVM model
        svm_vectorizer: TF-IDF vectorizer
        bert_model: BERT model artifacts tuple or None
        label_encoder: Label encoder for class mapping
        df_test: Test dataframe with 'text_norm' and 'title_raw' columns
        output_path: Path to save the plot
        k_max: Maximum k value to compute
    """
    print(f"[plot_topk_curve] Computing top-k scores for {len(df_test)} test samples...")
    
    # Get texts and true labels
    texts = df_test["text_norm"].fillna("").astype(str).tolist()
    true_labels_str = df_test["title_raw"].astype(str).tolist()
    
    # Convert true labels to indices
    true_labels = label_encoder.transform(true_labels_str)
    
    # Compute SVM top-k scores
    print("[plot_topk_curve] Computing SVM top-k scores...")
    svm_topk = compute_svm_topk_scores(texts, svm_model, svm_vectorizer, k_max=k_max)
    
    # Compute BERT top-k scores (if available)
    bert_topk = None
    if bert_model is not None:
        print("[plot_topk_curve] Computing BERT top-k scores...")
        model, tokenizer, _, device = bert_model
        bert_topk = compute_bert_topk_scores(
            texts, model, tokenizer, device, k_max=k_max
        )
    
    # Compute recall@k for k = 1..k_max
    k_values = list(range(1, k_max + 1))
    svm_recalls = []
    bert_recalls = []
    
    print("[plot_topk_curve] Computing recall@k values...")
    for k in k_values:
        svm_recall = compute_recall_at_k(true_labels, svm_topk, k)
        svm_recalls.append(svm_recall)
        print(f"  SVM Recall@{k}: {svm_recall:.4f}")
        
        if bert_topk is not None:
            bert_recall = compute_recall_at_k(true_labels, bert_topk, k)
            bert_recalls.append(bert_recall)
            print(f"  BERT Recall@{k}: {bert_recall:.4f}")
    
    # Create plot
    print(f"[plot_topk_curve] Creating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot SVM curve
    ax.plot(
        k_values,
        svm_recalls,
        marker="o",
        label="SVM",
        linewidth=2,
        markersize=8,
    )
    
    # Plot BERT curve (if available)
    if bert_topk is not None:
        ax.plot(
            k_values,
            bert_recalls,
            marker="o",
            label="BERT",
            linewidth=2,
            markersize=8,
        )
    
    # Formatting
    ax.set_xlabel("k (Top-k)", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Top-k Recall Curve: SVM vs BERT", fontsize=14, fontweight="bold")
    ax.set_xticks(k_values)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    
    # Add value annotations on points
    for k, svm_r in zip(k_values, svm_recalls):
        ax.annotate(
            f"{svm_r:.3f}",
            (k, svm_r),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    
    if bert_topk is not None:
        for k, bert_r in zip(k_values, bert_recalls):
            ax.annotate(
                f"{bert_r:.3f}",
                (k, bert_r),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
            )
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[plot_topk_curve] Plot saved to {output_path}")
    
    # Print summary
    print("\n[plot_topk_curve] Summary:")
    print("=" * 50)
    print(f"{'k':<5} {'SVM Recall@k':<15} {'BERT Recall@k':<15}")
    print("-" * 50)
    for k, svm_r in zip(k_values, svm_recalls):
        bert_str = f"{bert_recalls[k-1]:.4f}" if bert_topk is not None else "N/A"
        print(f"{k:<5} {svm_r:<15.4f} {bert_str:<15}")
    print("=" * 50)


def main() -> None:
    """Main entry point for Top-k Recall Curve visualization."""
    print("[plot_topk_curve] Starting Top-k Recall Curve generation...")
    
    # Load SVM model
    print("[plot_topk_curve] Loading SVM model...")
    svm_model, svm_vectorizer, label_encoder = load_svm_model()
    
    # Load BERT model (gracefully handle if missing)
    print("[plot_topk_curve] Attempting to load BERT model...")
    bert_model = load_bert_model()
    if bert_model is None:
        print("[plot_topk_curve] WARNING: BERT model not found. Plot will only show SVM.")
    else:
        print("[plot_topk_curve] BERT model loaded successfully.")
    
    # Load data and splits
    print("[plot_topk_curve] Loading data and splits...")
    df = pd.read_parquet(PROCESSED_PARQUET)
    splits = load_splits()
    
    # Filter to test split
    test_ids = set(splits["test"])
    df_test = df[df["resume_id"].astype(str).isin(test_ids)].copy()
    df_test = df_test.reset_index(drop=True)
    
    print(f"[plot_topk_curve] Test set size: {len(df_test)} samples")
    
    # Ensure required columns exist
    if "text_norm" not in df_test.columns:
        raise ValueError("Column 'text_norm' not found in test data. Run preprocessing first.")
    if "title_raw" not in df_test.columns:
        raise ValueError("Column 'title_raw' not found in test data. Run preprocessing first.")
    
    # Generate plot
    output_path = RUNS_DIR / "evaluation" / "topk_curve.png"
    plot_topk_recall_curve(
        svm_model=svm_model,
        svm_vectorizer=svm_vectorizer,
        bert_model=bert_model,
        label_encoder=label_encoder,
        df_test=df_test,
        output_path=output_path,
        k_max=5,
    )
    
    print(f"[plot_topk_curve] Complete! Output saved to {output_path}")


if __name__ == "__main__":
    main()

