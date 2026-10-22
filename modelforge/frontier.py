"""The Open Frontier: What ML Still Can't Do (Module 6), measured on this code.

  fgsm_attack            adversarial robustness: a small step in exactly the
                         wrong direction flips predictions that random noise of
                         the same size leaves alone
  overconfidence         epistemic uncertainty: the network is near-certain about
                         points nothing like its training data
  permutation_importance interpretability: which inputs mattered, measured by
                         shuffling each one — and how explanations differ by model

Catastrophic forgetting lives in deep_experiments.py; the data-scaling limit
(more data is not automatically better) in experiments.learning_curve.
"""

from __future__ import annotations

from math import fsum

import random

from .datasets import StandardScaler, make_moons, train_test_split
from .linear import SoftmaxRegression
from .metrics import accuracy
from .mlops import OODGuard
from .neural import MLP
from .tree import DecisionTree


def _sign(v: float) -> float:
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


def fgsm_attack(model: MLP, X: list, y: list, eps: float) -> list:
    """Fast Gradient Sign Method: move every feature by eps in the direction that
    increases the loss the most."""
    cidx = {c: i for i, c in enumerate(model.classes)}
    out = []
    for x, t in zip(X, y):
        g = model.input_gradient(x, cidx[t])
        out.append([xi + eps * _sign(gi) for xi, gi in zip(x, g)])
    return out


def random_sign_noise(X: list, eps: float, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [[xi + eps * rng.choice((-1.0, 1.0)) for xi in x] for x in X]


def adversarial_robustness(model: MLP, X: list, y: list,
                           eps_values: tuple = (0.0, 0.1, 0.2, 0.3)) -> list:
    rows = []
    for eps in eps_values:
        rows.append({"eps": eps,
                     "adversarial_accuracy": accuracy(y, model.predict(fgsm_attack(model, X, y, eps))),
                     "random_noise_accuracy": accuracy(y, model.predict(random_sign_noise(X, eps)))})
    return rows


def flipped_example(model: MLP, X: list, y: list, eps: float = 0.3) -> dict | None:
    """A test point the model got right with high confidence that a small FGSM step
    pushes into the wrong class — the panda-to-gibbon effect in two dimensions."""
    cidx = {c: i for i, c in enumerate(model.classes)}
    adv = fgsm_attack(model, X, y, eps)
    best = None
    for x, xa, t in zip(X, adv, y):
        p = model.predict_proba([x])[0][cidx[t]]
        pa = model.predict_proba([xa])[0]
        if model.predict([x])[0] == t and model.predict([xa])[0] != t:
            if best is None or p > best["confidence_before"]:
                best = {"x": x, "x_adv": xa, "label": t, "confidence_before": p,
                        "confidence_after_true_class": pa[cidx[t]], "eps": eps}
    return best


def overconfidence(model: MLP, guard: OODGuard, probes: list) -> list:
    """Confidence of the network vs the out-of-distribution guard, on points
    inside and far outside the training data (scaled feature space)."""
    rows = []
    for x in probes:
        p = model.predict_proba([x])[0]
        chk = guard.check(x)
        rows.append({"x": x, "confidence": max(p), "distance": chk["distance"],
                     "flagged_unknown": chk["out_of_distribution"]})
    return rows


def permutation_importance(model, X: list, y: list, repeats: int = 5, seed: int = 0) -> list:
    """Accuracy drop when one feature's column is shuffled (breaking its link to
    the label). A drop near zero means the model does not rely on that feature."""
    base = accuracy(y, model.predict(X))
    rng = random.Random(seed)
    out = []
    for j in range(len(X[0])):
        drops = []
        for _ in range(repeats):
            col = [r[j] for r in X]
            rng.shuffle(col)
            Xp = [r[:j] + [v] + r[j + 1:] for r, v in zip(X, col)]
            drops.append(base - accuracy(y, model.predict(Xp)))
        out.append(fsum(drops) / repeats)
    return out


def interpretability_demo() -> dict:
    """Moons plus a third column of pure noise. Every model should give the noise
    near-zero importance; how they split credit between x0 and x1 differs."""
    X, y = make_moons()
    rng = random.Random(4)
    X = [r + [rng.gauss(0, 1)] for r in X]
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    for m in (SoftmaxRegression(), DecisionTree(max_depth=5), MLP(hidden=(16, 16), epochs=60)):
        m.fit(Xtr, ytr)
        out[m.name] = {"test_accuracy": accuracy(yte, m.predict(Xte)),
                       "importance": permutation_importance(m, Xte, yte)}
    return {"features": ["x0", "x1", "noise"], "models": out}
