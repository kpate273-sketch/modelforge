"""The end-to-end ML workflow (Module 1: The ML Workflow End to End;
Module 2: Bias, Variance, and Overfitting; The Limits: No Free Lunch).

    data -> split -> scale (fit on train only) -> train -> evaluate on held-out test

compare() runs that workflow for every model on one dataset and returns a
leaderboard. Run it across blobs, moons and circles and the No Free Lunch lesson
shows up in the numbers: the ranking changes with the shape of the data, and the
linear model that is perfect on blobs is near chance on circles.

depth_sweep() trains decision trees of increasing depth and records train vs
test accuracy. Train accuracy climbs toward 100% while test accuracy stalls or
falls — the overfitting curve from the bias-variance lesson, measured rather
than described.
"""

from __future__ import annotations

import time

from .datasets import DATASETS, train_test_split, StandardScaler
from .linear import SoftmaxRegression
from .metrics import accuracy, evaluate
from .neighbors import KNearestNeighbors
from .neural import MLP
from .tree import DecisionTree, RandomForest


def default_models(fast: bool = False) -> list:
    """The five models, from simplest to most flexible."""
    epochs = 60 if fast else 150
    return [
        SoftmaxRegression(lr=0.5, epochs=300),
        KNearestNeighbors(k=5),
        DecisionTree(max_depth=5),
        RandomForest(n_trees=15, max_depth=6),
        MLP(hidden=(16, 16), activation="tanh", lr=0.1, epochs=epochs),
    ]


def prepare(dataset: str, n: int = 300, seed: int = 7, test_size: float = 0.25):
    """Generate, split and scale a dataset. Scaler is fit on training data only."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    X, y = DATASETS[dataset](n=n, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size)
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte, scaler


_NET_CACHE: dict = {}


def trained_network(dataset: str = "moons") -> MLP:
    """The standard neural network (the lab's settings) trained on `dataset`.

    Training is fully seeded, so every experiment that needs "the moons network"
    would get an identical model anyway; caching it just avoids retraining the
    same weights several times in one run. Callers must not modify it.
    """
    if dataset not in _NET_CACHE:
        Xtr, _, ytr, _, _ = prepare(dataset)
        _NET_CACHE[dataset] = MLP(hidden=(16, 16), activation="tanh", lr=0.1,
                                  epochs=150).fit(Xtr, ytr)
    return _NET_CACHE[dataset]


def run_one(model, Xtr, Xte, ytr, yte) -> dict:
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_s = time.time() - t0
    train_acc = accuracy(ytr, model.predict(Xtr))
    test_pred = model.predict(Xte)
    ev = evaluate(yte, test_pred)
    return {
        "model": model.name,
        "train_accuracy": train_acc,
        "test_accuracy": ev["accuracy"],
        "macro_f1": ev["macro_f1"],
        "overfit_gap": train_acc - ev["accuracy"],
        "fit_seconds": fit_s,
        "confusion": ev["confusion"],
        "loss_history": list(getattr(model, "loss_history", []) or []),
        "_model": model,
    }


def compare(dataset: str, fast: bool = False) -> dict:
    """Train every model on one dataset; return a leaderboard sorted by test accuracy."""
    Xtr, Xte, ytr, yte, scaler = prepare(dataset)
    rows = [run_one(m, Xtr, Xte, ytr, yte) for m in default_models(fast=fast)]
    rows.sort(key=lambda r: (-r["test_accuracy"], -r["macro_f1"]))
    return {"dataset": dataset, "n_train": len(Xtr), "n_test": len(Xte),
            "rows": rows, "scaler": scaler, "data": (Xtr, Xte, ytr, yte)}


def depth_sweep(dataset: str = "moons", depths: tuple = (1, 2, 3, 4, 6, 8, 12)) -> list:
    """Train vs test accuracy as tree depth grows — the overfitting curve."""
    Xtr, Xte, ytr, yte, _ = prepare(dataset)
    out = []
    for d in depths:
        t = DecisionTree(max_depth=d, min_samples_split=2).fit(Xtr, ytr)
        out.append({"depth": d,
                    "train_accuracy": accuracy(ytr, t.predict(Xtr)),
                    "test_accuracy": accuracy(yte, t.predict(Xte)),
                    "leaves": t.n_leaves()})
    return out


def knn_sweep(dataset: str = "moons", ks: tuple = (1, 3, 5, 9, 15, 31, 61)) -> list:
    """Bias-variance from the other side: small k overfits, large k underfits."""
    Xtr, Xte, ytr, yte, _ = prepare(dataset)
    out = []
    for k in ks:
        m = KNearestNeighbors(k=k).fit(Xtr, ytr)
        out.append({"k": k,
                    "train_accuracy": accuracy(ytr, m.predict(Xtr)),
                    "test_accuracy": accuracy(yte, m.predict(Xte))})
    return out


def format_leaderboard(result: dict) -> str:
    lines = [f"LEADERBOARD — {result['dataset']}  "
             f"(train {result['n_train']}, test {result['n_test']})",
             f"  {'model':20s}{'train':>8}{'test':>8}{'F1':>8}{'gap':>8}{'fit s':>8}"]
    for r in result["rows"]:
        lines.append(f"  {r['model']:20s}{r['train_accuracy']:>8.3f}"
                     f"{r['test_accuracy']:>8.3f}{r['macro_f1']:>8.3f}"
                     f"{r['overfit_gap']:>+8.3f}{r['fit_seconds']:>8.2f}")
    return "\n".join(lines)
