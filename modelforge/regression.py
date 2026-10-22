"""Linear regression with mean squared error, from scratch
(Module 1: The Three Learning Paradigms — regression; Module 2: Vectors,
Matrices, and What Models Compute; Loss Functions; Gradient Descent).

Classification predicts a category; regression predicts a number. This module is
the course's own running example made real: a house is a vector
[square feet, bedrooms, year built], the model is a weight vector, and the
prediction is their dot product.

    house   = [1500, 3, 1990]
    weights = [200, 10000, -50]
    price   = 1500*200 + 3*10000 + 1990*(-50) = 230,500

make_houses() generates sales from exactly those weights plus noise, and
LinearRegression has to recover them by gradient descent on mean squared error:

    MSE        = mean((prediction - truth)^2)
    gradient   = 2/n * X^T (prediction - truth)
    w  <-  w - lr * gradient

batch_size picks the flavour of gradient descent from the Gradient Descent
lesson: None = batch (whole dataset per step), 1 = stochastic, anything else =
mini-batch.
"""

from __future__ import annotations

from math import fsum

import math
import random

from .datasets import _gauss

TRUE_HOUSE_WEIGHTS = [200.0, 10000.0, -50.0]      # $ per sq ft, per bedroom, per year
HOUSE_FEATURES = ["square_feet", "bedrooms", "year_built"]


def mse(y_true: list, y_pred: list) -> float:
    """Mean squared error: average of squared differences. Lower is better."""
    return fsum((p - t) ** 2 for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)


def make_houses(n: int = 200, noise: float = 15000.0, seed: int = 7) -> tuple:
    """House sales priced by the course's weight vector plus gaussian noise."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        sqft = rng.uniform(900, 3200)
        beds = rng.randint(1, 5)
        year = rng.randint(1950, 2022)
        x = [round(sqft), beds, year]
        X.append([float(v) for v in x])
        y.append(fsum(a * w for a, w in zip(x, TRUE_HOUSE_WEIGHTS)) + noise * _gauss(rng))
    return X, y


class LinearRegression:
    name = "linear_regression"

    def __init__(self, lr: float = 0.1, epochs: int = 200, batch_size: int | None = None,
                 seed: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.w: list = []
        self.b: float = 0.0
        self.loss_history: list = []      # full-data MSE after every epoch
        self.step_history: list = []      # full-data MSE after every single update
        self.updates = 0
        self.diverged = False

    def predict(self, X: list) -> list:
        return [fsum(wi * xi for wi, xi in zip(self.w, x)) + self.b for x in X]

    def fit(self, X: list, y: list, track_steps: bool = False) -> "LinearRegression":
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        self.loss_history, self.step_history = [], []
        self.updates, self.diverged = 0, False
        rng = random.Random(self.seed)
        order = list(range(n))
        bs = n if self.batch_size is None else self.batch_size
        for _ in range(self.epochs):
            if bs < n:
                rng.shuffle(order)
            for start in range(0, n, bs):
                batch = order[start:start + bs]
                m = len(batch)
                errs = [fsum(wi * xi for wi, xi in zip(self.w, X[i])) + self.b - y[i]
                        for i in batch]
                gw = [2.0 * fsum(e * X[i][j] for e, i in zip(errs, batch)) / m
                      for j in range(d)]
                gb = 2.0 * fsum(errs) / m
                self.w = [wi - self.lr * g for wi, g in zip(self.w, gw)]
                self.b -= self.lr * gb
                self.updates += 1
                if track_steps:
                    self.step_history.append(mse(y, self.predict(X)))
            loss = mse(y, self.predict(X))
            self.loss_history.append(loss)
            if not math.isfinite(loss) or loss > 1e12:
                self.diverged = True       # the "too large a learning rate" failure
                break
        return self


class HousePriceModel:
    """Standardize features and target, fit by gradient descent, then convert the
    weights back to dollars so they can be compared with the true ones."""

    def __init__(self, **kw):
        self.reg = LinearRegression(**kw)

    def fit(self, X: list, y: list) -> "HousePriceModel":
        cols = list(zip(*X))
        self.mx = [fsum(c) / len(c) for c in cols]
        self.sx = [math.sqrt(fsum((v - m) ** 2 for v in c) / len(c)) or 1.0
                   for c, m in zip(cols, self.mx)]
        self.my = fsum(y) / len(y)
        self.sy = math.sqrt(fsum((v - self.my) ** 2 for v in y) / len(y)) or 1.0
        Xs = [[(v - m) / s for v, m, s in zip(r, self.mx, self.sx)] for r in X]
        ys = [(v - self.my) / self.sy for v in y]
        self.reg.fit(Xs, ys)
        return self

    def weights_in_dollars(self) -> list:
        return [w * self.sy / s for w, s in zip(self.reg.w, self.sx)]

    def predict(self, X: list) -> list:
        Xs = [[(v - m) / s for v, m, s in zip(r, self.mx, self.sx)] for r in X]
        return [p * self.sy + self.my for p in self.reg.predict(Xs)]
