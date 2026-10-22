"""Decision trees and random forests, from scratch
(Module 3: Trees, Forests, and Neighbors; Module 2: Bias, Variance, Overfitting).

DecisionTree is a CART classifier. At each node it tries every feature and every
threshold between adjacent values, and keeps the split that most reduces Gini
impurity (or, with criterion="entropy", information entropy):

    gini(node)    = 1 - sum_k p_k^2           (0 = pure, higher = mixed)
    entropy(node) = -sum_k p_k log2 p_k

Left alone, a tree keeps splitting until every leaf is pure — it memorizes the
training set. max_depth and min_samples_split are the brakes, and they are the
clearest hands-on demonstration of overfitting in the project.

RandomForest trains many trees on bootstrap samples, each looking at a random
subset of features at every split, and takes a majority vote. Averaging many
high-variance trees lowers the variance — bias-variance in practice.
"""

from __future__ import annotations

from math import fsum

import math
import random
from collections import Counter


def gini(labels: list) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - fsum((c / n) ** 2 for c in counts.values())


def entropy(labels: list) -> float:
    """Information entropy in bits: 0 = pure, 1 = a 50/50 two-class mix.
    The lesson's alternative to Gini. Both reward purer children; entropy
    penalizes mixed nodes a little more strongly, so the chosen splits can differ."""
    n = len(labels)
    if n == 0:
        return 0.0
    return -fsum((c / n) * math.log2(c / n) for c in Counter(labels).values())


IMPURITY = {"gini": gini, "entropy": entropy}


def _majority(labels: list):
    counts = Counter(labels)
    top = max(counts.values())
    return min(l for l, c in counts.items() if c == top)


class DecisionTree:
    name = "decision_tree"

    def __init__(self, max_depth: int = 5, min_samples_split: int = 4,
                 max_features: int | None = None, seed: int = 0,
                 criterion: str = "gini"):
        if criterion not in IMPURITY:
            raise ValueError(f"criterion must be one of {sorted(IMPURITY)}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.root: dict | None = None
        self._rng = random.Random(seed)

    # -- training ---------------------------------------------------------- #
    def _best_split(self, X: list, y: list):
        n, d = len(X), len(X[0])
        impurity = IMPURITY[self.criterion]
        parent = impurity(y)
        best = None  # (gain, feature, threshold)
        features = list(range(d))
        if self.max_features and self.max_features < d:
            features = self._rng.sample(features, self.max_features)
        for f in features:
            values = sorted(set(row[f] for row in X))
            for a, b in zip(values, values[1:]):
                thr = (a + b) / 2
                left = [y[i] for i in range(n) if X[i][f] <= thr]
                right = [y[i] for i in range(n) if X[i][f] > thr]
                if not left or not right:
                    continue
                w = (len(left) * impurity(left) + len(right) * impurity(right)) / n
                gain = parent - w
                if best is None or gain > best[0]:
                    best = (gain, f, thr)
        return best

    def _build(self, X: list, y: list, depth: int) -> dict:
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(set(y)) == 1):
            return {"leaf": True, "label": _majority(y), "n": len(y)}
        split = self._best_split(X, y)
        if split is None or split[0] <= 1e-12:
            return {"leaf": True, "label": _majority(y), "n": len(y)}
        _, f, thr = split
        li = [i for i, row in enumerate(X) if row[f] <= thr]
        ri = [i for i, row in enumerate(X) if row[f] > thr]
        return {
            "leaf": False, "feature": f, "threshold": thr,
            "left": self._build([X[i] for i in li], [y[i] for i in li], depth + 1),
            "right": self._build([X[i] for i in ri], [y[i] for i in ri], depth + 1),
        }

    def fit(self, X: list, y: list) -> "DecisionTree":
        self._rng = random.Random(self.seed)
        self.root = self._build(X, y, 0)
        return self

    # -- inference --------------------------------------------------------- #
    def _predict_one(self, x: list):
        node = self.root
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["label"]

    def predict(self, X: list) -> list:
        return [self._predict_one(x) for x in X]

    def depth(self, node: dict | None = None) -> int:
        node = node or self.root
        if node["leaf"]:
            return 0
        return 1 + max(self.depth(node["left"]), self.depth(node["right"]))

    def n_leaves(self, node: dict | None = None) -> int:
        node = node or self.root
        if node["leaf"]:
            return 1
        return self.n_leaves(node["left"]) + self.n_leaves(node["right"])

    def to_dict(self) -> dict:
        return {"type": self.name, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "max_features": self.max_features, "seed": self.seed,
                "criterion": self.criterion, "root": self.root}

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionTree":
        m = cls(max_depth=d["max_depth"], min_samples_split=d["min_samples_split"],
                max_features=d["max_features"], seed=d["seed"],
                criterion=d.get("criterion", "gini"))
        m.root = d["root"]
        return m


class RandomForest:
    name = "random_forest"

    def __init__(self, n_trees: int = 15, max_depth: int = 6,
                 max_features: int | None = 1, seed: int = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees: list = []

    def fit(self, X: list, y: list) -> "RandomForest":
        rng = random.Random(self.seed)
        n = len(X)
        self.trees = []
        for t in range(self.n_trees):
            idx = [rng.randrange(n) for _ in range(n)]          # bootstrap sample
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=2,
                                max_features=self.max_features, seed=self.seed + t)
            tree.fit([X[i] for i in idx], [y[i] for i in idx])
            self.trees.append(tree)
        return self

    def predict(self, X: list) -> list:
        all_preds = [t.predict(X) for t in self.trees]
        return [_majority([p[i] for p in all_preds]) for i in range(len(X))]

    def to_dict(self) -> dict:
        return {"type": self.name, "n_trees": self.n_trees,
                "max_depth": self.max_depth, "max_features": self.max_features,
                "seed": self.seed, "trees": [t.to_dict() for t in self.trees]}

    @classmethod
    def from_dict(cls, d: dict) -> "RandomForest":
        m = cls(n_trees=d["n_trees"], max_depth=d["max_depth"],
                max_features=d["max_features"], seed=d["seed"])
        m.trees = [DecisionTree.from_dict(t) for t in d["trees"]]
        return m
