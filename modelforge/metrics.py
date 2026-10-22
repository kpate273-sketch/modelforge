"""Evaluating classifiers honestly (Module 3: Evaluating Classifiers Honestly;
Module 2: Loss Functions).

Accuracy alone can lie — a model that always predicts the majority class looks
good on imbalanced data. So every model here is scored on the held-out test set
with a confusion matrix plus per-class precision, recall and F1, and a macro
average that weights every class equally.

Also the loss functions: cross-entropy is what the models minimize during
training, so it lives next to the metrics that judge them afterward.
"""

from __future__ import annotations

from math import fsum

import math


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    labels = labels or sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return {"labels": labels, "matrix": m}


def precision_recall_f1(y_true: list, y_pred: list) -> dict:
    """Per-class precision/recall/F1 plus their macro averages."""
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    for l in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[l] = {"precision": prec, "recall": rec, "f1": f1}
    n = len(labels) or 1
    return {
        "per_class": per_class,
        "macro_precision": fsum(c["precision"] for c in per_class.values()) / n,
        "macro_recall": fsum(c["recall"] for c in per_class.values()) / n,
        "macro_f1": fsum(c["f1"] for c in per_class.values()) / n,
    }


def cross_entropy(probs: list, y: list) -> float:
    """Mean negative log-likelihood of the true class. The training loss."""
    eps = 1e-12
    return -fsum(math.log(max(p[t], eps)) for p, t in zip(probs, y)) / max(len(y), 1)


def evaluate(y_true: list, y_pred: list) -> dict:
    """Everything a report needs from one held-out evaluation."""
    prf = precision_recall_f1(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_precision": prf["macro_precision"],
        "macro_recall": prf["macro_recall"],
        "macro_f1": prf["macro_f1"],
        "per_class": prf["per_class"],
        "confusion": confusion_matrix(y_true, y_pred),
    }


def format_confusion(cm: dict) -> str:
    labels, m = cm["labels"], cm["matrix"]
    head = "        " + "".join(f"pred {l:<3}" for l in labels)
    rows = [head]
    for l, row in zip(labels, m):
        rows.append(f"true {l:<3}" + "".join(f"{v:>8}" for v in row))
    return "\n".join(rows)
