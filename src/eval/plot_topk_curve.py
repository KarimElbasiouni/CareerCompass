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
    model_dir = MODELS_DIR / "bert_finetuned"
    if not model_dir.exists():
        model_dir = MODELS_DIR / "bert_title"
    
    if not model_dir.exists():
        return None
    
    if not any(model_dir.glob("*.bin")) and not any(model_dir.glob("*.safetensors")):
        return None
    
    try:
        print(f"[plot_topk_curve] Loading BERT model from {model_dir}")
        
        model = BertForSequenceClassification.from_pretrained(str(model_dir))
        tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        
        encoder_path = MODELS_DIR / "label_encoder.joblib"
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found: {encoder_path}. Train a model first."
            )
        label_encoder = joblib.load(encoder_path)
        
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
    X = vectorizer.transform(texts)
    
    scores = model.decision_function(X)
    
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    
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
    model.eval()
    all_logits = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
        
        all_logits.append(logits.cpu().numpy())
    
    logits = np.vstack(all_logits)
    
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    
    topk_indices = np.argsort(-probs, axis=1)[:, :k_max]
    
    return topk_indices


def compute_recall_at_k(
    true_labels: np.ndarray,
    topk_indices: np.ndarray,
    k: int,
) -> float:
    if k > topk_indices.shape[1]:
        raise ValueError(f"k={k} exceeds k_max={topk_indices.shape[1]}")
    
    topk_preds = topk_indices[:, :k]
    
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
    print(f"[plot_topk_curve] Computing top-k scores for {len(df_test)} test samples...")
    
    texts = df_test["text_norm"].fillna("").astype(str).tolist()
    true_labels_str = df_test["title_raw"].astype(str).tolist()
    
    true_labels = label_encoder.transform(true_labels_str)

    print("[plot_topk_curve] Computing SVM top-k scores...")
    svm_topk = compute_svm_topk_scores(texts, svm_model, svm_vectorizer, k_max=k_max)
    
    bert_topk = None
    if bert_model is not None:
        print("[plot_topk_curve] Computing BERT top-k scores...")
        model, tokenizer, _, device = bert_model
        bert_topk = compute_bert_topk_scores(
            texts, model, tokenizer, device, k_max=k_max
        )
    
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
    
    print(f"[plot_topk_curve] Creating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(
        k_values,
        svm_recalls,
        marker="o",
        label="SVM",
        linewidth=2,
        markersize=8,
    )
    
    if bert_topk is not None:
        ax.plot(
            k_values,
            bert_recalls,
            marker="o",
            label="BERT",
            linewidth=2,
            markersize=8,
        )
    
    ax.set_xlabel("k (Top-k)", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Top-k Recall Curve: SVM vs BERT", fontsize=14, fontweight="bold")
    ax.set_xticks(k_values)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    
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
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[plot_topk_curve] Plot saved to {output_path}")
    
    print("\n[plot_topk_curve] Summary:")
    print("=" * 50)
    print(f"{'k':<5} {'SVM Recall@k':<15} {'BERT Recall@k':<15}")
    print("-" * 50)
    for k, svm_r in zip(k_values, svm_recalls):
        bert_str = f"{bert_recalls[k-1]:.4f}" if bert_topk is not None else "N/A"
        print(f"{k:<5} {svm_r:<15.4f} {bert_str:<15}")
    print("=" * 50)


def main() -> None:
    print("[plot_topk_curve] Starting Top-k Recall Curve generation...")
    
    print("[plot_topk_curve] Loading SVM model...")
    svm_model, svm_vectorizer, label_encoder = load_svm_model()
    
    print("[plot_topk_curve] Attempting to load BERT model...")
    bert_model = load_bert_model()
    if bert_model is None:
        print("[plot_topk_curve] WARNING: BERT model not found. Plot will only show SVM.")
    else:
        print("[plot_topk_curve] BERT model loaded successfully.")
    
    print("[plot_topk_curve] Loading data and splits...")
    df = pd.read_parquet(PROCESSED_PARQUET)
    splits = load_splits()
    
    test_ids = set(splits["test"])
    df_test = df[df["resume_id"].astype(str).isin(test_ids)].copy()
    df_test = df_test.reset_index(drop=True)
    
    print(f"[plot_topk_curve] Test set size: {len(df_test)} samples")
    
    if "text_norm" not in df_test.columns:
        raise ValueError("Column 'text_norm' not found in test data. Run preprocessing first.")
    if "title_raw" not in df_test.columns:
        raise ValueError("Column 'title_raw' not found in test data. Run preprocessing first.")
    
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