"""Lightweight evaluation helpers for classification baselines."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0


def confusion(y_true, y_pred, labels, display_labels=None, scale: float = 1.25) -> tuple[np.ndarray, plt.Figure]:
    """
    Bigger confusion matrix:
    - scale: multiplies base side length
    - shows count + row %
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_classes = len(labels)
    tick_labels = display_labels if display_labels is not None else labels

    base_side = max(6, min(1.0 * n_classes, 14))
    side = base_side * scale
    fig, ax = plt.subplots(figsize=(side, side), dpi=150)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=15, fontsize=12)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (count / row%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    row_sums = cm.sum(axis=1, keepdims=True)
    max_val = cm.max()
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            pct = (100.0 * count / row_sums[i, 0]) if row_sums[i, 0] else 0.0
            txt = f"{count}\n{pct:.1f}%"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if count > 0.55 * max_val else "black",
                fontsize=9 + (0.3 * scale),
            )

    ax.tick_params(length=0)
    ax.grid(False)
    fig.tight_layout()
    return cm, fig


def topk_from_scores(y_true, scores: np.ndarray, k: int) -> float:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, scores.shape[1])
    partition = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    y_true = np.asarray(y_true)
    hits = sum(y_true[i] in partition[i] for i in range(len(y_true)))
    return hits / len(y_true) if len(y_true) else 0.0
