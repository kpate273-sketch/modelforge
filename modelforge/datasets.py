"""Datasets and data preparation (Module 1: The ML Workflow End to End).

No downloads, no data files. The classic teaching datasets are generated from
scratch with a fixed seed, so every run is reproducible and the project costs
nothing to run:

  make_blobs   gaussian clusters  -> linearly separable; a linear model wins
  make_moons   two interleaving half-circles -> NOT linearly separable;
               a linear model fails and a neural network succeeds
  make_circles one ring inside another -> also non-linear

Generating the data yourself is a deliberate choice: you know the true
structure, so you can tell whether a model learned it or got lucky. That
pairing (blobs vs moons) is the backbone of the demo — it shows *when* each
model works, not just that it runs.

Also here: train/test splitting and standardization, the two preparation
steps the workflow lesson says you never skip.
"""

from __future__ import annotations

from math import fsum

import math
import random


def _gauss(rng: random.Random) -> float:
    """Box-Muller transform: two uniforms -> one standard normal."""
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_blobs(n: int = 300, centers: int = 3, spread: float = 0.9,
               seed: int = 7) -> tuple:
    """n points in `centers` gaussian clusters placed around a circle."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        c = i % centers
        angle = 2 * math.pi * c / centers
        cx, cy = 4 * math.cos(angle), 4 * math.sin(angle)
        X.append([cx + spread * _gauss(rng), cy + spread * _gauss(rng)])
        y.append(c)
    return X, y


def make_moons(n: int = 300, noise: float = 0.15, seed: int = 7) -> tuple:
    """Two interleaving half-circles. Classic non-linear benchmark."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        if i % 2 == 0:
            t = math.pi * rng.random()
            X.append([math.cos(t) + noise * _gauss(rng),
                      math.sin(t) + noise * _gauss(rng)])
            y.append(0)
        else:
            t = math.pi * rng.random()
            X.append([1 - math.cos(t) + noise * _gauss(rng),
                      0.5 - math.sin(t) + noise * _gauss(rng)])
            y.append(1)
    return X, y


def make_circles(n: int = 300, noise: float = 0.08, factor: float = 0.45,
                 seed: int = 7) -> tuple:
    """An inner ring (class 1) inside an outer ring (class 0)."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        t = 2 * math.pi * rng.random()
        r = 1.0 if i % 2 == 0 else factor
        X.append([r * math.cos(t) + noise * _gauss(rng),
                  r * math.sin(t) + noise * _gauss(rng)])
        y.append(0 if i % 2 == 0 else 1)
    return X, y


def make_imbalanced(n: int = 2000, positive_rate: float = 0.05, separation: float = 2.2,
                    seed: int = 7) -> tuple:
    """A rare-event dataset, like fraud: about 5% positives (class 1) whose
    feature distribution overlaps the negatives. Accuracy looks great on this
    data even for a useless model, which is the point of the lesson."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        pos = rng.random() < positive_rate
        c = separation if pos else 0.0
        X.append([c + _gauss(rng), c + _gauss(rng)])
        y.append(1 if pos else 0)
    return X, y


DATASETS = {"blobs": make_blobs, "moons": make_moons, "circles": make_circles}


def train_test_split(X: list, y: list, test_size: float = 0.25,
                     seed: int = 42) -> tuple:
    """Shuffle, then hold out a test set the model never trains on.

    Stratified: each class is split in the same proportion, so a small test set
    still contains every class.
    """
    rng = random.Random(seed)
    by_class: dict = {}
    for i, label in enumerate(y):
        by_class.setdefault(label, []).append(i)
    train_idx, test_idx = [], []
    for label in sorted(by_class):
        idx = by_class[label][:]
        rng.shuffle(idx)
        k = max(1, round(len(idx) * test_size))
        test_idx += idx[:k]
        train_idx += idx[k:]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return ([X[i] for i in train_idx], [X[i] for i in test_idx],
            [y[i] for i in train_idx], [y[i] for i in test_idx])


class StandardScaler:
    """Rescale each feature to mean 0, std 1.

    Fit on the TRAINING data only, then apply to test data — fitting on the full
    dataset would leak test information into training.
    """

    def __init__(self):
        self.mean: list = []
        self.std: list = []

    def fit(self, X: list) -> "StandardScaler":
        cols = list(zip(*X))
        self.mean = [fsum(c) / len(c) for c in cols]
        self.std = []
        for c, m in zip(cols, self.mean):
            var = fsum((v - m) ** 2 for v in c) / len(c)
            self.std.append(math.sqrt(var) or 1.0)
        return self

    def transform(self, X: list) -> list:
        return [[(v - m) / s for v, m, s in zip(row, self.mean, self.std)]
                for row in X]

    def fit_transform(self, X: list) -> list:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        s = cls()
        s.mean, s.std = d["mean"], d["std"]
        return s
