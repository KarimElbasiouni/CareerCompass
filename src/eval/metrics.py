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


def confusion(y_true, y_pred, labels, display_labels=None) -> tuple[np.ndarray, plt.Figure]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = display_labels if display_labels is not None else labels
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
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
