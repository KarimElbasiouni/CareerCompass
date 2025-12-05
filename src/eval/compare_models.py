"""Compare metrics across different models (Majority, SVM, BERT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import RUNS_DIR


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[compare_models] Warning: Failed to load {path}: {e}")
        return None


def format_metric(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def compare_models(
    majority_path: Path | None = None,
    svm_path: Path | None = None,
    bert_path: Path | None = None,
    output_path: Path | None = None,
    print_table: bool = True,
) -> Dict[str, Any]:

    if majority_path is None:
        majority_path = RUNS_DIR / "majority_baseline" / "metrics.json"
    if svm_path is None:
        svm_path = RUNS_DIR / "svm_tfidf" / "metrics.json"
    if bert_path is None:
        bert_path = RUNS_DIR / "bert_title" / "metrics.json"
    if output_path is None:
        output_path = RUNS_DIR / "comparison" / "model_comparison.json"

    majority_metrics = load_metrics(majority_path)
    svm_metrics = load_metrics(svm_path)
    bert_metrics = load_metrics(bert_path)

    comparison = {
        "models": {
            "majority": {
                "present": majority_metrics is not None,
                "metrics_path": str(majority_path),
            },
            "svm": {
                "present": svm_metrics is not None,
                "metrics_path": str(svm_path),
            },
            "bert": {
                "present": bert_metrics is not None,
                "metrics_path": str(bert_path),
            },
        },
        "validation": {},
        "test": {},
    }

    if majority_metrics and "val" in majority_metrics:
        comparison["validation"]["majority"] = {
            "accuracy": majority_metrics["val"].get("accuracy"),
            "macro_f1": majority_metrics["val"].get("macro_f1"),
        }
    else:
        comparison["validation"]["majority"] = {"accuracy": None, "macro_f1": None}

    if svm_metrics and "val" in svm_metrics:
        comparison["validation"]["svm"] = {
            "accuracy": svm_metrics["val"].get("accuracy"),
            "macro_f1": svm_metrics["val"].get("macro_f1"),
            "top1": svm_metrics["val"].get("top1"),
            "top3": svm_metrics["val"].get("top3"),
        }
    else:
        comparison["validation"]["svm"] = {
            "accuracy": None,
            "macro_f1": None,
            "top1": None,
            "top3": None,
        }

    if bert_metrics and "val" in bert_metrics:
        comparison["validation"]["bert"] = {
            "accuracy": bert_metrics["val"].get("accuracy"),
            "macro_f1": bert_metrics["val"].get("macro_f1"),
        }
    else:
        comparison["validation"]["bert"] = {"accuracy": None, "macro_f1": None}

    if majority_metrics and "test" in majority_metrics:
        comparison["test"]["majority"] = {
            "accuracy": majority_metrics["test"].get("accuracy"),
            "macro_f1": majority_metrics["test"].get("macro_f1"),
        }
    else:
        comparison["test"]["majority"] = {"accuracy": None, "macro_f1": None}

    if svm_metrics and "test" in svm_metrics:
        comparison["test"]["svm"] = {
            "accuracy": svm_metrics["test"].get("accuracy"),
            "macro_f1": svm_metrics["test"].get("macro_f1"),
            "top1": svm_metrics["test"].get("top1"),
            "top3": svm_metrics["test"].get("top3"),
        }
    else:
        comparison["test"]["svm"] = {
            "accuracy": None,
            "macro_f1": None,
            "top1": None,
            "top3": None,
        }

    if bert_metrics and "test" in bert_metrics:
        comparison["test"]["bert"] = {
            "accuracy": bert_metrics["test"].get("accuracy"),
            "macro_f1": bert_metrics["test"].get("macro_f1"),
        }
    else:
        comparison["test"]["bert"] = {"accuracy": None, "macro_f1": None}

    if svm_metrics and "best_C" in svm_metrics:
        comparison["svm_hyperparameters"] = {"best_C": svm_metrics["best_C"]}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    if print_table:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        print("\nVALIDATION SET:")
        print("-" * 80)
        print(f"{'Model':<15} {'Accuracy':<12} {'Macro-F1':<12} {'Top-1':<12} {'Top-3':<12}")
        print("-" * 80)

        maj_val = comparison["validation"]["majority"]
        print(
            f"{'Majority':<15} {format_metric(maj_val['accuracy']):<12} "
            f"{format_metric(maj_val['macro_f1']):<12} {'N/A':<12} {'N/A':<12}"
        )

        svm_val = comparison["validation"]["svm"]
        print(
            f"{'SVM':<15} {format_metric(svm_val['accuracy']):<12} "
            f"{format_metric(svm_val['macro_f1']):<12} "
            f"{format_metric(svm_val.get('top1')):<12} {format_metric(svm_val.get('top3')):<12}"
        )

        bert_val = comparison["validation"]["bert"]
        print(
            f"{'BERT':<15} {format_metric(bert_val['accuracy']):<12} "
            f"{format_metric(bert_val['macro_f1']):<12} {'N/A':<12} {'N/A':<12}"
        )

        print("\nTEST SET:")
        print("-" * 80)
        print(f"{'Model':<15} {'Accuracy':<12} {'Macro-F1':<12} {'Top-1':<12} {'Top-3':<12}")
        print("-" * 80)

        maj_test = comparison["test"]["majority"]
        print(
            f"{'Majority':<15} {format_metric(maj_test['accuracy']):<12} "
            f"{format_metric(maj_test['macro_f1']):<12} {'N/A':<12} {'N/A':<12}"
        )

        svm_test = comparison["test"]["svm"]
        print(
            f"{'SVM':<15} {format_metric(svm_test['accuracy']):<12} "
            f"{format_metric(svm_test['macro_f1']):<12} "
            f"{format_metric(svm_test.get('top1')):<12} {format_metric(svm_test.get('top3')):<12}"
        )

        bert_test = comparison["test"]["bert"]
        print(
            f"{'BERT':<15} {format_metric(bert_test['accuracy']):<12} "
            f"{format_metric(bert_test['macro_f1']):<12} {'N/A':<12} {'N/A':<12}"
        )

        print("\n" + "=" * 80)
        print(f"Comparison saved to: {output_path}")
        print("=" * 80 + "\n")

    return comparison


def main() -> None:
    """CLI entrypoint for model comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare metrics across models")
    parser.add_argument(
        "--majority-path",
        type=Path,
        default=None,
        help="Path to majority baseline metrics.json",
    )
    parser.add_argument(
        "--svm-path",
        type=Path,
        default=None,
        help="Path to SVM metrics.json",
    )
    parser.add_argument(
        "--bert-path",
        type=Path,
        default=None,
        help="Path to BERT metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for comparison JSON",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip printing the comparison table",
    )

    args = parser.parse_args()
    compare_models(
        majority_path=args.majority_path,
        svm_path=args.svm_path,
        bert_path=args.bert_path,
        output_path=args.output,
        print_table=not args.no_print,
    )

if __name__ == "__main__":
    main()