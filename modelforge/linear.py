"""Logistic (softmax) regression trained by gradient descent, from scratch.
(Module 2: Loss Functions, Gradient Descent; Module 3: Logistic Regression and Beyond.)

The model is one weight vector per class. For an input x, each class gets a
score w_k . x + b_k; softmax turns the scores into probabilities. Training
minimizes cross-entropy loss by gradient descent:

    gradient for class k  =  mean over samples of (p_k - [y == k]) * x
    w_k  <-  w_k - lr * (gradient + l2 * w_k)

That one line of calculus is the whole learning rule. The loss is recorded every
epoch so you can watch it fall — the clearest evidence that the model is
learning rather than memorizing a lookup.

It draws a straight-line (linear) decision boundary, which is exactly why it
succeeds on blobs and fails on moons — the honest limit the course asks you to
notice.
"""

from __future__ import annotations

from .linalg import dot, softmax, argmax
from .metrics import cross_entropy


class SoftmaxRegression:
    name = "logistic_regression"

    def __init__(self, lr: float = 0.5, epochs: int = 300, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.W: list = []          # [n_classes][n_features]
        self.b: list = []          # [n_classes]
        self.classes: list = []
        self.loss_history: list = []

    def _scores(self, x: list) -> list:
        return [dot(w, x) + b for w, b in zip(self.W, self.b)]

    def predict_proba(self, X: list) -> list:
        return [softmax(self._scores(x)) for x in X]

    def predict(self, X: list) -> list:
        return [self.classes[argmax(p)] for p in self.predict_proba(X)]

    def fit(self, X: list, y: list) -> "SoftmaxRegression":
        self.classes = sorted(set(y))
        k, d, n = len(self.classes), len(X[0]), len(X)
        cidx = {c: i for i, c in enumerate(self.classes)}
        yi = [cidx[v] for v in y]
        self.W = [[0.0] * d for _ in range(k)]
        self.b = [0.0] * k
        self.loss_history = []

        for _ in range(self.epochs):
            probs = self.predict_proba(X)
            gW = [[0.0] * d for _ in range(k)]
            gb = [0.0] * k
            for x, p, t in zip(X, probs, yi):
                for c in range(k):
                    err = p[c] - (1.0 if c == t else 0.0)
                    gb[c] += err
                    row = gW[c]
                    for j in range(d):
                        row[j] += err * x[j]
            for c in range(k):
                for j in range(d):
                    self.W[c][j] -= self.lr * (gW[c][j] / n + self.l2 * self.W[c][j])
                self.b[c] -= self.lr * gb[c] / n
            self.loss_history.append(cross_entropy(probs, yi))
        return self

    def to_dict(self) -> dict:
        return {"type": self.name, "lr": self.lr, "epochs": self.epochs,
                "l2": self.l2, "W": self.W, "b": self.b, "classes": self.classes}

    @classmethod
    def from_dict(cls, d: dict) -> "SoftmaxRegression":
        m = cls(lr=d["lr"], epochs=d["epochs"], l2=d["l2"])
        m.W, m.b, m.classes = d["W"], d["b"], d["classes"]
        return m
