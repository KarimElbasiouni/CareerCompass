"""Inference helper for fine-tuned BERT resume classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

import src.config as config

DEFAULT_MODEL_DIR = config.MODELS_DIR / "bert_title"


def load_bert_model(
    model_dir: Path | None = None,
) -> Tuple[BertForSequenceClassification, BertTokenizer, LabelEncoder, torch.device]:
    """
    Load the fine-tuned BERT model and tokenizer from disk.

    Args:
        model_dir: Path to model directory. Defaults to models/bert_title.

    Returns:
        Tuple of (model, tokenizer, label_encoder, device)
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Train a model first."
        )

    print(f"[inference_bert] Loading model from {model_dir}")

    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))

    # Load label encoder
    encoder_path = config.MODELS_DIR / "label_encoder.joblib"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {encoder_path}. Train a model first."
        )
    label_encoder = joblib.load(encoder_path)

    # Move model to device and set to eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[inference_bert] Model loaded on {device}")
    print(f"[inference_bert] Number of classes: {len(label_encoder.classes_)}")

    return model, tokenizer, label_encoder, device


def predict_bert(
    texts: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    label_encoder: LabelEncoder,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
) -> Dict[str, np.ndarray | list[str]]:
    """
    Given a list of raw texts, returns predictions and probabilities.

    Args:
        texts: List of resume text strings
        model: Fine-tuned BERT model
        tokenizer: BERT tokenizer
        label_encoder: Label encoder for class mapping
        device: Torch device
        max_length: Maximum sequence length
        batch_size: Batch size for inference

    Returns:
        Dictionary with:
        - 'probs': np.ndarray of shape (n_samples, n_classes) with class probabilities
        - 'pred_indices': np.ndarray of predicted class indices
        - 'pred_labels': list[str] of predicted label names
    """
    model.eval()
    all_probs = []
    all_pred_indices = []

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

        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        pred_indices = np.argmax(probs, axis=-1)

        all_probs.append(probs)
        all_pred_indices.append(pred_indices)

    # Concatenate results
    probs = np.vstack(all_probs)
    pred_indices = np.concatenate(all_pred_indices)

    # Map indices to label names
    pred_labels = label_encoder.inverse_transform(pred_indices).tolist()

    return {
        "probs": probs,
        "pred_indices": pred_indices,
        "pred_labels": pred_labels,
    }


def main() -> None:
    """Simple CLI for testing inference."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.models.inference_bert <text1> [text2] ...")
        print("Or pipe text: echo 'resume text' | python -m src.models.inference_bert")
        sys.exit(1)

    # Load model
    model, tokenizer, label_encoder, device = load_bert_model()

    # Get input texts
    if sys.stdin.isatty():
        # Command line arguments
        texts = sys.argv[1:]
    else:
        # Read from stdin
        texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("No input text provided.")
        sys.exit(1)

    # Predict
    results = predict_bert(texts, model, tokenizer, label_encoder, device)

    # Print results
    for i, (text, pred_label, probs) in enumerate(
        zip(texts, results["pred_labels"], results["probs"])
    ):
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        print(f"\nText {i+1}: {text[:100]}...")
        print(f"Predicted: {pred_label} (confidence: {top_prob:.4f})")
        print(f"Top 3 classes:")
        top3_indices = np.argsort(probs)[-3:][::-1]
        for idx in top3_indices:
            label = label_encoder.classes_[idx]
            prob = probs[idx]
            print(f"  - {label}: {prob:.4f}")


if __name__ == "__main__":
    main()

