"""Switchable feature builders for TF-IDF or transformer embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DEFAULT_FEATURE_MODE,
    FEATURES_DIR,
    PROCESSED_PARQUET,
    TRANSFORMER_MODEL_NAME,
)

Mode = Literal["tfidf", "transformer"]


@dataclass
class FeatureArtifacts:
    mode: Mode
    x_path: str
    vectorizer_path: str | None
    model_name: str | None


class FeatureBuilder:
    """Single entry point for fitting TF-IDF or transformer features."""

    def __init__(
        self,
        mode: Mode = DEFAULT_FEATURE_MODE,
        output_dir: Path | None = None,
        batch_size: int = 32,
    ) -> None:
        if mode not in ("tfidf", "transformer"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode: Mode = mode
        self.output_dir = Path(output_dir or FEATURES_DIR)
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._vectorizer: TfidfVectorizer | None = None
        self._model = None
        self._artifacts: FeatureArtifacts | None = None

    @property
    def artifacts(self) -> FeatureArtifacts | None:
        return self._artifacts

    @property
    def vectorizer(self) -> TfidfVectorizer | None:
        return self._vectorizer

    def fit_transform(
        self, texts: Sequence[str]
    ) -> tuple[sp.csr_matrix | np.ndarray, FeatureArtifacts]:
        cleaned = _prepare_texts(texts)
        if not cleaned:
            raise ValueError("No texts available for feature extraction.")
        if self.mode == "tfidf":
            matrix = self._fit_transform_tfidf(cleaned)
        else:
            matrix = self._fit_transform_transformer(cleaned)
        if self._artifacts is None:
            raise RuntimeError("Artifacts were not recorded after fitting.")
        return matrix, self._artifacts

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix | np.ndarray:
        cleaned = _prepare_texts(texts)
        if not cleaned:
            raise ValueError("No texts provided for transform.")
        if self.mode == "tfidf":
            if self._vectorizer is None:
                raise ValueError("TF-IDF vectorizer is not fitted.")
            return self._vectorizer.transform(cleaned)
        model = self._load_transformer_model()
        return model.encode(
            cleaned,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _fit_transform_tfidf(self, texts: Sequence[str]) -> sp.csr_matrix:
        min_df = 3 if len(texts) >= 3 else 1  # keep defaults usable for tiny smoke runs
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.9,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(texts)
        self._vectorizer = vectorizer
        x_path = self.output_dir / "tfidf_features.npz"
        vectorizer_path = self.output_dir / "tfidf_vectorizer.joblib"
        sp.save_npz(x_path, matrix)
        joblib.dump(vectorizer, vectorizer_path)
        self._artifacts = FeatureArtifacts(
            mode="tfidf",
            x_path=str(x_path),
            vectorizer_path=str(vectorizer_path),
            model_name=None,
        )
        return matrix

    def _fit_transform_transformer(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_transformer_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        x_path = self.output_dir / "transformer_embeddings.npy"
        np.save(x_path, embeddings)
        self._artifacts = FeatureArtifacts(
            mode="transformer",
            x_path=str(x_path),
            vectorizer_path=None,
            model_name=TRANSFORMER_MODEL_NAME,
        )
        return embeddings

    def _load_transformer_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "sentence-transformers is required for transformer mode."
                ) from exc

            self._model = SentenceTransformer(TRANSFORMER_MODEL_NAME)
        return self._model


def _prepare_texts(texts: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for text in texts:
        if text is None:
            continue
        normalized = str(text).strip()
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _load_texts_from_parquet(parquet_path: Path) -> list[str]:
    df = pd.read_parquet(parquet_path, columns=["text_norm"])
    return df["text_norm"].fillna("").astype(str).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resume features.")
    parser.add_argument(
        "--mode",
        choices=("tfidf", "transformer"),
        default=DEFAULT_FEATURE_MODE,
        help="Feature mode to build.",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(PROCESSED_PARQUET),
        help="Input parquet file with text_norm column.",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default=str(FEATURES_DIR),
        help="Directory where artifacts will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.input_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {parquet_path}")
    texts = _load_texts_from_parquet(parquet_path)
    builder = FeatureBuilder(mode=args.mode, output_dir=Path(args.output_dir))
    matrix, artifacts = builder.fit_transform(texts)
    shape = matrix.shape
    print(f"[features] mode={args.mode} shape={shape}")
    print(f"[features] saved features -> {artifacts.x_path}")
    if artifacts.vectorizer_path:
        print(f"[features] saved vectorizer -> {artifacts.vectorizer_path}")
    if artifacts.model_name:
        print(f"[features] model name: {artifacts.model_name}")


if __name__ == "__main__":
    main()
