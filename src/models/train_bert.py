"""Train a BERT-based classifier on resume text for job title classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

import src.config as config
from src.eval import metrics as eval_metrics

DEFAULT_RUN_DIR = config.RUNS_DIR / "bert_title"
DEFAULT_MODEL_DIR = config.MODELS_DIR / "bert_title"


class ResumeDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: BertTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_or_fit_label_encoder(
    df_train: pd.DataFrame, label_col: str
) -> LabelEncoder:
    encoder_path = config.MODELS_DIR / "label_encoder.joblib"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if encoder_path.exists():
        print(f"[train_bert] Loading existing label encoder from {encoder_path}")
        encoder = joblib.load(encoder_path)
    else:
        print(f"[train_bert] Fitting new label encoder on training labels")
        encoder = LabelEncoder()
        encoder.fit(df_train[label_col].astype(str))
        joblib.dump(encoder, encoder_path)
        print(f"[train_bert] Saved label encoder to {encoder_path}")

    return encoder


def compute_metrics(pred):
    predictions = pred.predictions
    labels = pred.label_ids

    if predictions.ndim > 1:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions

    accuracy = eval_metrics.accuracy(labels, preds)
    macro_f1 = eval_metrics.macro_f1(labels, preds)

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT on resume text.")
    parser.add_argument(
        "--parquet",
        default=str(config.PROCESSED_PARQUET),
        help="Input parquet with cleaned data.",
    )
    parser.add_argument(
        "--label-col", default="title_raw", help="Label column to use."
    )
    parser.add_argument(
        "--text-col",
        default="text_norm",
        help="Text column to use (fallback to text_clean if missing).",
    )
    parser.add_argument(
        "--splits",
        default=str(config.SPLITS_JSON),
        help="Splits JSON path.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Where to save trained BERT model and tokenizer.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Where to write metrics/plots.",
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="Hugging Face model name (used only if --local-model-dir is not provided).",
    )
    parser.add_argument(
        "--local-model-dir",
        default=None,
        type=str,
        help="Optional local directory containing a pretrained BERT model/tokenizer (offline mode).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Evaluation batch size per device.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    runs_dir = Path(args.runs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_bert] Loading data from {args.parquet}")
    df = pd.read_parquet(args.parquet)

    text_col = args.text_col
    if text_col not in df.columns:
        if "text_clean" in df.columns:
            text_col = "text_clean"
            print(f"[train_bert] {args.text_col} not found, using text_clean")
        else:
            raise ValueError(
                f"Text column '{args.text_col}' not found and no text_clean fallback."
            )

    label_col = args.label_col
    if label_col not in df.columns:
        if "y_title" in df.columns:
            label_col = "y_title"
            print(f"[train_bert] {args.label_col} not found, using y_title")
        else:
            raise ValueError(
                f"Label column '{args.label_col}' not found and no y_title fallback."
            )

    print(f"[train_bert] Loading splits from {args.splits}")
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    df["resume_id"] = df["resume_id"].astype(str)
    df_train = df[df["resume_id"].isin(train_ids)].copy()
    df_val = df[df["resume_id"].isin(val_ids)].copy()
    df_test = df[df["resume_id"].isin(test_ids)].copy()

    print(
        f"[train_bert] Splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )

    encoder = load_or_fit_label_encoder(df_train, label_col)
    num_labels = len(encoder.classes_)
    print(f"[train_bert] Number of classes: {num_labels}")

    y_train = encoder.transform(df_train[label_col].astype(str))
    y_val = encoder.transform(df_val[label_col].astype(str))
    y_test = encoder.transform(df_test[label_col].astype(str))

    if args.local_model_dir:
        local_dir = Path(args.local_model_dir)
        if not local_dir.exists():
            raise FileNotFoundError(
                f"Local model directory not found: {local_dir}. "
                "Please provide a valid path to a directory containing a pretrained BERT model."
            )
        print(f"[train_bert] Loading tokenizer and model from local directory: {local_dir}")
        try:
            tokenizer = BertTokenizer.from_pretrained(str(local_dir), local_files_only=True)
            model = BertForSequenceClassification.from_pretrained(
                str(local_dir), num_labels=num_labels, local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from local directory {local_dir}. "
                f"Ensure it contains valid tokenizer and model files (pytorch_model.bin or model.safetensors). "
                f"Error: {e}"
            ) from e
    else:
        print(f"[train_bert] Attempting to load model from cache (offline mode): {args.model_name}")
        try:
            tokenizer = BertTokenizer.from_pretrained(args.model_name, local_files_only=True)
            model = BertForSequenceClassification.from_pretrained(
                args.model_name, num_labels=num_labels, local_files_only=True
            )
            print(f"[train_bert] Successfully loaded from cache")
        except Exception as cache_error:
            print(f"[train_bert] Cache load failed, attempting online download: {args.model_name}")
            print(f"[train_bert] Downloading tokenizer and model from Hugging Face Hub...")
            try:
                tokenizer = BertTokenizer.from_pretrained(args.model_name)
                print(f"[train_bert] Tokenizer downloaded successfully")
                model = BertForSequenceClassification.from_pretrained(
                    args.model_name, num_labels=num_labels
                )
                print(f"[train_bert] Model downloaded successfully")
            except Exception as download_error:
                raise RuntimeError(
                    f"Failed to load model '{args.model_name}':\n"
                    f"  - Cache load failed: {cache_error}\n"
                    f"  - Online download failed: {download_error}\n\n"
                    f"This may be due to network connectivity issues or incomplete cache. "
                    f"To use offline mode with a local model directory, run with:\n"
                    f"  --local-model-dir /path/to/downloaded/model"
                ) from download_error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[train_bert] Using device: {device}")

    train_texts = df_train[text_col].fillna("").astype(str).tolist()
    val_texts = df_val[text_col].fillna("").astype(str).tolist()
    test_texts = df_test[text_col].fillna("").astype(str).tolist()

    train_dataset = ResumeDataset(train_texts, y_train, tokenizer, args.max_length)
    val_dataset = ResumeDataset(val_texts, y_val, tokenizer, args.max_length)
    test_dataset = ResumeDataset(test_texts, y_test, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(runs_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=config.RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("[train_bert] Starting training...")
    trainer.train()

    print("[train_bert] Evaluating on validation set...")
    val_results = trainer.evaluate(eval_dataset=val_dataset)

    print("[train_bert] Evaluating on test set...")
    test_predictions = trainer.predict(test_dataset)
    test_logits = test_predictions.predictions
    test_preds = np.argmax(test_logits, axis=-1)

    test_accuracy = eval_metrics.accuracy(y_test, test_preds)
    test_macro_f1 = eval_metrics.macro_f1(y_test, test_preds)

    print("[train_bert] Generating confusion matrices...")

    val_predictions = trainer.predict(val_dataset)
    val_logits = val_predictions.predictions
    val_preds = np.argmax(val_logits, axis=-1)

    val_cm, val_fig = eval_metrics.confusion(
        y_val,
        val_preds,
        labels=np.arange(num_labels),
        display_labels=encoder.classes_,
    )
    val_fig_path = runs_dir / "confusion_val.png"
    val_fig.savefig(val_fig_path, dpi=150, bbox_inches="tight")
    plt.close(val_fig)

    test_cm, test_fig = eval_metrics.confusion(
        y_test,
        test_preds,
        labels=np.arange(num_labels),
        display_labels=encoder.classes_,
    )
    test_fig_path = runs_dir / "confusion_test.png"
    test_fig.savefig(test_fig_path, dpi=150, bbox_inches="tight")
    plt.close(test_fig)

    metrics = {
        "val": {
            "accuracy": val_results["eval_accuracy"],
            "macro_f1": val_results["eval_macro_f1"],
        },
        "test": {
            "accuracy": float(test_accuracy),
            "macro_f1": float(test_macro_f1),
        },
    }

    metrics_path = runs_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_bert] Saving model and tokenizer to {models_dir}")
    model.save_pretrained(models_dir)
    tokenizer.save_pretrained(models_dir)

    print(f"[train_bert] Training complete!")
    print(f"[train_bert] Validation: acc={metrics['val']['accuracy']:.4f}, macro_f1={metrics['val']['macro_f1']:.4f}")
    print(f"[train_bert] Test: acc={metrics['test']['accuracy']:.4f}, macro_f1={metrics['test']['macro_f1']:.4f}")
    print(f"[train_bert] Model -> {models_dir}")
    print(f"[train_bert] Metrics -> {metrics_path}")
    print(f"[train_bert] Confusion matrices -> {runs_dir}")


if __name__ == "__main__":
    main()s