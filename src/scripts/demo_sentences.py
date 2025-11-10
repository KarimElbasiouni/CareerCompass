"""Quick demo to highlight TF-IDF vs transformer sentence representations."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from src.features.vectorize import FeatureBuilder, Mode

SENT_A = "I created a Python app for administrative use."
SENT_B = "I used a Python app for administrative use."
DEMO_SENTENCES = [SENT_A, SENT_B]


def _top_tfidf_tokens(vector, feature_names: np.ndarray, top_k: int = 5):
    dense = vector.toarray().ravel()
    if dense.sum() == 0:
        return []
    top_indices = np.argsort(dense)[::-1][:top_k]
    return [(feature_names[idx], float(dense[idx])) for idx in top_indices if dense[idx] > 0]


def run_tfidf_demo() -> None:
    with TemporaryDirectory(prefix="demo_tfidf_") as tmpdir:
        builder = FeatureBuilder(mode="tfidf", output_dir=Path(tmpdir))
        matrix, _ = builder.fit_transform(DEMO_SENTENCES)
        vectorizer = builder.vectorizer
        if vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer missing after fitting.")
        feature_names = np.array(vectorizer.get_feature_names_out())
        print("[demo][tfidf] Top n-grams per sentence:")
        for sentence, row in zip(DEMO_SENTENCES, matrix):
            tokens = _top_tfidf_tokens(row, feature_names)
            readable = ", ".join(f"{tok}:{weight:.3f}" for tok, weight in tokens)
            print(f"  - '{sentence}' -> {readable}")


def run_transformer_demo() -> None:
    with TemporaryDirectory(prefix="demo_transformer_") as tmpdir:
        builder = FeatureBuilder(mode="transformer", output_dir=Path(tmpdir))
        embeddings, _ = builder.fit_transform(DEMO_SENTENCES)
    vec_a, vec_b = embeddings
    similarity = float(np.dot(vec_a, vec_b))
    print(
        "[demo][transformer] Cosine similarity "
        f"between '{SENT_A}' and '{SENT_B}': {similarity:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-sentence feature demo.")
    parser.add_argument(
        "--mode",
        choices=("tfidf", "transformer"),
        required=True,
        help="Feature mode to preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode: Mode = args.mode
    if mode == "tfidf":
        run_tfidf_demo()
    else:
        run_transformer_demo()


if __name__ == "__main__":
    main()
