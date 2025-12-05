"""Helpers for loading or creating deterministic dataset splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, SPLITS_JSON

SplitDict = Dict[str, List[str]]


def load_splits(path: Path | None = None) -> SplitDict | None:
    splits_path = Path(path or SPLITS_JSON)
    if not splits_path.exists():
        return None
    with splits_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_splits(splits: Mapping[str, Iterable[str]], path: Path | None = None) -> Path:
    splits_path = Path(path or SPLITS_JSON)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: list(v) for k, v in splits.items()}
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return splits_path


def ensure_splits(
    df: pd.DataFrame,
    label_col: str,
    id_col: str,
    splits_path: Path | None = None,
    seed: int = RANDOM_SEED,
) -> SplitDict:
    existing = load_splits(splits_path)
    if existing is not None:
        return existing

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in dataframe.")

    helper = df[[id_col, label_col]].copy()
    helper[label_col] = helper[label_col].fillna("Unknown")

    test_size = 0.1
    val_size = 0.1
    try:
        train_val_ids, test_ids = train_test_split(
            helper[id_col],
            test_size=test_size,
            random_state=seed,
            stratify=helper[label_col],
        )
    except ValueError:
        train_val_ids, test_ids = train_test_split(
            helper[id_col],
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    relative_val = val_size / (1 - test_size)
    helper_train_val = helper.set_index(id_col).loc[train_val_ids]
    try:
        train_ids, val_ids = train_test_split(
            helper_train_val.index,
            test_size=relative_val,
            random_state=seed,
            stratify=helper_train_val[label_col],
        )
    except ValueError:
        train_ids, val_ids = train_test_split(
            helper_train_val.index,
            test_size=relative_val,
            random_state=seed,
            stratify=None,
        )

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    save_splits(splits, splits_path)
    return splits
