"""k-Nearest Neighbors, from scratch (Module 3: Trees, Forests, and Neighbors).

The simplest model that works: there is no training at all. To classify a new
point, find the k closest training points and take a majority vote. "Learning"
is just remembering the data — which is why k-NN is a useful contrast with the
models that actually fit parameters.

It also illustrates the bias-variance trade-off from Module 2 directly: k=1
memorizes noise (high variance); a large k smooths everything out (high bias).
"""

from __future__ import annotations

from collections import Counter

from .linalg import euclidean


class KNearestNeighbors:
    name = "knn"

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.X: list = []
        self.y: list = []

    def fit(self, X: list, y: list) -> "KNearestNeighbors":
        self.X, self.y = [list(r) for r in X], list(y)
        return self

    def _predict_one(self, x: list):
        dists = sorted(((euclidean(x, xi), yi) for xi, yi in zip(self.X, self.y)),
                       key=lambda t: t[0])
        votes = Counter(label for _, label in dists[: self.k])
        top = max(votes.values())
        # Tie-break deterministically: smallest label among the tied winners.
        return min(l for l, c in votes.items() if c == top)

    def predict(self, X: list) -> list:
        return [self._predict_one(x) for x in X]

    def to_dict(self) -> dict:
        return {"type": self.name, "k": self.k, "X": self.X, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "KNearestNeighbors":
        m = cls(k=d["k"])
        m.X, m.y = d["X"], d["y"]
        return m
