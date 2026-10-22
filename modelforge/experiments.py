"""Experiments for Modules 1-3, one per idea the course teaches.

Each function answers one lesson's claim with a measurement on this code:

  course_examples          the worked examples in the lesson text, recomputed
  rule_vs_learned          ML vs. AI: a hand-written rule against learned models
  paradigms_demo           The Three Learning Paradigms: k-means and Q-learning
  label_noise              The ML Workflow: garbage in, garbage out
  learning_curve           more data is not automatically better
  house_regression         Vectors & Loss Functions: dot products and MSE
  learning_rate_sweep      Gradient Descent: too small, about right, too large
  gd_flavours              Gradient Descent: batch vs. mini-batch vs. stochastic
  ridge_sweep              Bias & Variance: L2 regularization as the dial
  early_stopping           Bias & Variance: the validation set
  vc_shattering            No Free Lunch and VC Dimension: a line shatters 3, not 4
  decision_boundary        The Decision Boundary Mindset: the boundary is the weights
  threshold_tradeoff       Evaluating Classifiers Honestly: accuracy lies; thresholds
  gini_vs_entropy          Trees: the two impurity measures (Gini vs entropy)
  curse_of_dimensionality  Neighbors: distance stops meaning much in high dimensions
"""

from __future__ import annotations

from math import fsum

import math
import random
from itertools import combinations_with_replacement, product

from .datasets import (StandardScaler, _gauss, make_blobs, make_circles, make_imbalanced,
                       make_moons, train_test_split)
from .linalg import dot, euclidean, sigmoid, softmax, solve
from .linear import SoftmaxRegression
from .metrics import accuracy, cross_entropy, precision_recall_f1
from .neighbors import KNearestNeighbors
from .neural import MLP
from .paradigms import GridWorld, KMeans, purity, q_learning, render_policy
from .pipeline import prepare, trained_network
from .regression import (HOUSE_FEATURES, TRUE_HOUSE_WEIGHTS, HousePriceModel,
                         LinearRegression, make_houses, mse)
from .tree import DecisionTree


# --------------------------------------------------------------------------- #
# The lesson text's own worked examples
# --------------------------------------------------------------------------- #
def course_examples() -> list:
    """(module, lesson, what the lesson says, what the code computes)."""
    sm = softmax([2.0, 1.0, 0.1])
    p_fraud, r_fraud = 80 / 400, 80 / 100
    rows = [
        (2, "Vectors, Matrices, and What Models Compute", "house [1500, 3, 1990] . weights [200, 10000, -50] = 230500",
         f"{dot([1500, 3, 1990], [200, 10000, -50]):,.0f}"),
        (2, "Vectors, Matrices, and What Models Compute", "[2000, 4, 2000] . [100, 5000, -20] = 180000",
         f"{dot([2000, 4, 2000], [100, 5000, -20]):,.0f}"),
        (2, "Loss Functions: Measuring Wrongness", "true 250,000, predicted 230,500: squared error about 380 million",
         f"{mse([250000], [230500]):,.0f}"),
        (2, "Loss Functions: Measuring Wrongness", "off by 2x hurts 4x as much",
         f"{mse([0], [2]) / mse([0], [1]):.0f}x"),
        (2, "Loss Functions: Measuring Wrongness", "confident wrong answer (p = 0.01 for the true class): "
         "cross-entropy explodes, squared error stays gentle",
         f"cross-entropy {-math.log(0.01):.2f} vs squared error {(1 - 0.01) ** 2:.2f}"),
        (3, "Logistic Regression and Beyond", "sigmoid(2.2) is about 0.90", f"{sigmoid(2.2):.3f}"),
        (3, "Logistic Regression and Beyond", "softmax(2.0, 1.0, 0.1) is roughly (0.63, 0.23, 0.14)",
         "(" + ", ".join(f"{v:.3f}" for v in sm) + ")"),
        (3, "Logistic Regression and Beyond", "softmax over two classes reduces to the sigmoid",
         f"softmax(0, 2.2)[1] = {softmax([0.0, 2.2])[1]:.6f}, sigmoid(2.2) = {sigmoid(2.2):.6f}"),
        (3, "Evaluating Classifiers Honestly", "fraud: recall 0.80, precision 0.20, F1 about 0.32",
         f"F1 = {2 * p_fraud * r_fraud / (p_fraud + r_fraud):.3f}"),
        (3, "Evaluating Classifiers Honestly", "always 'not fraud' at 1 in 1,000 is 99.9% accurate",
         f"{accuracy([1] + [0] * 999, [0] * 1000):.1%}"),
        (5, "RNNs and the Vanishing Gradient Problem", "repeatedly multiply by 0.5 vs 2 for 20 steps: tiny vs over a million",
         f"0.5^20 = {0.5 ** 20:.2e}, 2^20 = {2 ** 20:,}"),
        (5, "Transformers and Self-Attention", "a 10x longer input costs about 100x more",
         f"{(10 * 50) ** 2 // 50 ** 2}x the attention scores"),
    ]
    return [{"module": m, "lesson": a, "claim": b, "computed": c} for m, a, b, c in rows]


# --------------------------------------------------------------------------- #
# Module 1
# --------------------------------------------------------------------------- #
def rule_vs_learned() -> dict:
    """A rule a person would write after looking at the moons plot, versus models
    that learn the boundary. The rule is perfectly readable; the learned models
    are more accurate — the transparency/accuracy trade-off from lesson 1."""
    X, y = make_moons()
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    rule = lambda p: 1 if p[1] < 0.25 else 0          # "below the middle line = lower moon"
    rule_acc = accuracy(yte, [rule(p) for p in Xte])
    sc = StandardScaler().fit(Xtr)
    learned = {}
    for m in (SoftmaxRegression().fit(sc.transform(Xtr), ytr),
              DecisionTree(max_depth=5).fit(sc.transform(Xtr), ytr),
              trained_network("moons")):
        learned[m.name] = accuracy(yte, m.predict(sc.transform(Xte)))
    return {"rule": "IF x1 < 0.25 THEN class 1 ELSE class 0", "rule_accuracy": rule_acc,
            "learned": learned}


def paradigms_demo() -> dict:
    X, y = make_blobs()
    elbow = []
    for k in range(1, 7):
        km = KMeans(k=k).fit(X)
        elbow.append({"k": k, "inertia": km.inertia, "purity": purity(km.labels_, y)})
    env = GridWorld()
    rl = q_learning(env)
    s = rl["steps_per_episode"]
    return {"kmeans_elbow": elbow,
            "rl_first_episode_steps": s[0],
            "rl_first20_mean": fsum(s[:20]) / 20,
            "rl_last20_mean": fsum(s[-20:]) / 20,
            "rl_greedy_steps": len(rl["greedy_path"]) - 1,
            "rl_reached_goal": rl["reached_goal"],
            "rl_optimal_steps": 2 * (env.size - 1),
            "rl_policy": render_policy(env, rl["policy"]),
            "rl_steps_per_episode": s}


def _flip(y: list, rate: float, seed: int) -> list:
    rng = random.Random(seed)
    return [1 - v if rng.random() < rate else v for v in y]


def label_noise() -> dict:
    """15% of training labels wrong (the lesson's defective-parts example).

    random   errors scattered at random: extra data partly averages them out.
    biased   errors concentrated in one region (one labeller always wrong on one
             kind of part): the model learns the mistake, and 4x more data
             with the same bias does not fix it. Cleaning the labels does.
    Test labels are always clean, because that is the real world.
    """
    out = {}
    X_big, y_big = make_moons(n=1200, seed=21)
    X_test, y_test = make_moons(n=400, seed=22)
    sc = StandardScaler().fit(X_big[:300])
    Xb, Xt = sc.transform(X_big), sc.transform(X_test)

    def score(X, y):
        return accuracy(y_test, DecisionTree(max_depth=6).fit(X, y).predict(Xt))

    def biased(X, y):
        # Every class-1 point with x0 > 1.5 (the right tip of the lower moon) is
        # mislabeled — 15% of all labels, all in one place.
        return [0 if (t == 1 and r[0] > 1.5) else t for r, t in zip(X, y)]

    small_X, small_y = Xb[:300], y_big[:300]
    out["clean_300"] = score(small_X, small_y)
    out["random15_300"] = score(small_X, _flip(small_y, 0.15, 1))
    out["random15_1200"] = score(Xb, _flip(y_big, 0.15, 1))
    raw_small = X_big[:300]
    out["biased_rate"] = sum(1 for a, b in zip(small_y, biased(raw_small, small_y)) if a != b) / 300
    out["biased_300"] = score(small_X, biased(raw_small, small_y))
    out["biased_1200"] = score(Xb, biased(X_big, y_big))
    return out


def learning_curve(sizes: tuple = (25, 50, 100, 200, 400, 800)) -> list:
    """Test accuracy as the training set grows. A model that cannot represent the
    problem (a straight line on moons) stops improving early; more data only
    helps a model with the capacity to use it — and with diminishing returns."""
    X, y = make_moons(n=max(sizes), seed=31)
    Xt, yt = make_moons(n=400, seed=32)
    out = []
    for n in sizes:
        sc = StandardScaler().fit(X[:n])
        Xs, Xts = sc.transform(X[:n]), sc.transform(Xt)
        lin = SoftmaxRegression().fit(Xs, y[:n])
        knn = KNearestNeighbors(k=5).fit(Xs, y[:n])
        out.append({"n": n, "logistic_regression": accuracy(yt, lin.predict(Xts)),
                    "knn": accuracy(yt, knn.predict(Xts))})
    return out


# --------------------------------------------------------------------------- #
# Module 2
# --------------------------------------------------------------------------- #
def least_squares(X: list, y: list) -> list:
    """Exact minimiser of MSE via the normal equations (X^T X) w = X^T y,
    with a bias column appended. Returns [w..., b]."""
    Xb = [list(r) + [1.0] for r in X]
    d = len(Xb[0])
    A = [[fsum(r[i] * r[j] for r in Xb) for j in range(d)] for i in range(d)]
    rhs = [fsum(r[i] * t for r, t in zip(Xb, y)) for i in range(d)]
    return solve(A, rhs)


def house_regression() -> dict:
    X, y = make_houses(n=400, noise=3000.0)
    Xtr, Xte, ytr, yte = X[:300], X[300:], y[:300], y[300:]
    model = HousePriceModel(lr=0.1, epochs=300).fit(Xtr, ytr)
    learned = model.weights_in_dollars()
    exact = least_squares(Xtr, ytr)[:3]
    rmse = math.sqrt(mse(yte, model.predict(Xte)))
    return {"features": HOUSE_FEATURES, "true": TRUE_HOUSE_WEIGHTS, "learned": learned,
            "exact_least_squares": exact,
            "max_gap_to_exact": max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(learned, exact)),
            "test_rmse": rmse, "course_house_prediction": model.predict([[1500, 3, 1990]])[0],
            "loss_history": model.reg.loss_history, "n_train": len(Xtr), "n_test": len(Xte)}


def _std_houses():
    X, y = make_houses(n=400, noise=3000.0)
    X, y = X[:300], y[:300]
    m = HousePriceModel()
    m.fit(X, y)                                   # only to get the standardization
    Xs = [[(v - a) / s for v, a, s in zip(r, m.mx, m.sx)] for r in X]
    ys = [(v - m.my) / m.sy for v in y]
    return Xs, ys


def learning_rate_sweep(rates: tuple = (0.001, 0.1, 1.2), epochs: int = 100) -> list:
    """Same model, same data, three step sizes (loss is MSE on standardized prices)."""
    Xs, ys = _std_houses()
    out = []
    for lr in rates:
        r = LinearRegression(lr=lr, epochs=epochs).fit(Xs, ys)
        h = r.loss_history
        out.append({"lr": lr, "epochs_run": len(h), "loss_epoch1": h[0],
                    "loss_epoch10": h[min(9, len(h) - 1)], "final_loss": h[-1],
                    "diverged": r.diverged, "loss_history": h})
    return out


def gd_flavours(epochs: int = 5, lr: float = 0.02) -> list:
    """Batch, mini-batch (32) and stochastic gradient descent for the same number of
    passes over the data. 'Uphill steps' counts updates that made the full-data
    loss worse — the zig-zag the lesson describes."""
    Xs, ys = _std_houses()
    out = []
    for label, bs in (("batch", None), ("mini-batch (32)", 32), ("stochastic (1)", 1)):
        r = LinearRegression(lr=lr, epochs=epochs, batch_size=bs).fit(Xs, ys, track_steps=True)
        h = r.step_history
        uphill = sum(1 for a, b in zip(h, h[1:]) if b > a)
        out.append({"method": label, "updates": r.updates, "final_loss": h[-1],
                    "uphill_steps": uphill, "uphill_fraction": uphill / max(len(h) - 1, 1)})
    return out


def _poly(x: float, degree: int) -> list:
    return [x ** d for d in range(1, degree + 1)]


def ridge_sweep(degree: int = 9, lambdas: tuple = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> dict:
    """15 noisy points from sin(pi x), fit with a degree-9 polynomial.

        total loss = sum of squared errors + lambda * (sum of weights squared)

    solved exactly for each lambda. lambda = 0 threads the curve through the
    noise (high variance); a large lambda flattens it (high bias)."""
    rng = random.Random(5)
    f = lambda x: math.sin(math.pi * x)
    xs = [rng.uniform(-1, 1) for _ in range(15)]
    ys = [f(x) + 0.25 * _gauss(rng) for x in xs]
    xt = [-1 + 2 * i / 199 for i in range(200)]
    yt = [f(x) for x in xt]
    rows = []
    for lam in lambdas:
        F = [_poly(x, degree) + [1.0] for x in xs]
        d = len(F[0])
        A = [[fsum(r[i] * r[j] for r in F) + (lam if i == j and i < d - 1 else 0.0)
              for j in range(d)] for i in range(d)]
        rhs = [fsum(r[i] * t for r, t in zip(F, ys)) for i in range(d)]
        w = solve(A, rhs)
        pred = lambda x: fsum(a * b for a, b in zip(w, _poly(x, degree) + [1.0]))
        rows.append({"lambda": lam,
                     "train_mse": fsum((pred(x) - t) ** 2 for x, t in zip(xs, ys)) / len(xs),
                     "test_mse": fsum((pred(x) - t) ** 2 for x, t in zip(xt, yt)) / len(xt),
                     "max_abs_weight": max(abs(v) for v in w[:-1]),
                     "curve": [pred(x) for x in xt[::4]]})
    best = min(rows, key=lambda r: r["test_mse"])
    return {"degree": degree, "n_points": len(xs), "rows": rows, "best_lambda": best["lambda"],
            "points": list(zip(xs, ys)), "grid": xt[::4], "truth": [f(x) for x in xt[::4]]}


def early_stopping(epochs: int = 400, patience: int = 50) -> dict:
    """Train / validation / test. The validation loss is watched every epoch; when it
    starts rising while training loss keeps falling, the network is overfitting."""
    X, y = make_moons(n=160, noise=0.35, seed=11)
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.25, seed=1)
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.33, seed=2)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    net = MLP(hidden=(16, 16), activation="tanh", lr=0.1, epochs=epochs, batch_size=8)
    net.fit(Xtr, ytr, Xva, yva)
    cidx = {c: i for i, c in enumerate(net.classes)}
    full = {"test_accuracy": accuracy(yte, net.predict(Xte)),
            "test_loss": cross_entropy(net.predict_proba(Xte), [cidx[v] for v in yte])}
    best_epoch = net.best_epoch
    net.restore_best()
    stopped = {"test_accuracy": accuracy(yte, net.predict(Xte)),
               "test_loss": cross_entropy(net.predict_proba(Xte), [cidx[v] for v in yte])}
    vl, tl = net.val_loss_history, net.loss_history
    return {"sizes": (len(Xtr), len(Xva), len(Xte)), "best_epoch": best_epoch,
            "stop_epoch": min(best_epoch + patience, epochs - 1), "patience": patience,
            "val_loss_at_best": vl[best_epoch], "val_loss_final": vl[-1],
            "train_loss_at_best": tl[best_epoch], "train_loss_final": tl[-1],
            "full_run": full, "early_stopped": stopped,
            "val_loss_history": vl, "train_loss_history": tl}


def depth_by_validation(depths: tuple = (1, 2, 3, 4, 6, 8, 12)) -> dict:
    """Pick the tree depth on the validation set, then touch the test set once."""
    X, y = make_moons(n=400, noise=0.25, seed=12)
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.25, seed=1)
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.33, seed=2)
    rows = []
    for d in depths:
        t = DecisionTree(max_depth=d, min_samples_split=2).fit(Xtr, ytr)
        rows.append({"depth": d, "train": accuracy(ytr, t.predict(Xtr)),
                     "validation": accuracy(yva, t.predict(Xva))})
    best = max(rows, key=lambda r: (r["validation"], -r["depth"]))
    final = DecisionTree(max_depth=best["depth"], min_samples_split=2).fit(Xtr, ytr)
    return {"rows": rows, "chosen_depth": best["depth"],
            "test_accuracy": accuracy(yte, final.predict(Xte))}


def _perceptron_separable(points: list, labels: list, max_epochs: int = 1000) -> bool:
    """A perceptron converges (zero mistakes) if and only if the labels are linearly
    separable; capped epochs make 'never converged' the non-separable answer."""
    w, b = [0.0, 0.0], 0.0
    for _ in range(max_epochs):
        mistakes = 0
        for p, t in zip(points, labels):
            s = 1 if t else -1
            if s * (w[0] * p[0] + w[1] * p[1] + b) <= 0:
                w = [w[0] + s * p[0], w[1] + s * p[1]]
                b += s
                mistakes += 1
        if mistakes == 0:
            return True
    return False


def vc_shattering() -> dict:
    """Can a straight line realise every labeling of these points?"""
    sets = {
        "3 points (triangle)": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        "3 points (in a line)": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        "4 points (square)": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    }
    out = {}
    for name, pts in sets.items():
        labelings = list(product((0, 1), repeat=len(pts)))
        failed = [lab for lab in labelings if not _perceptron_separable(pts, lab)]
        out[name] = {"labelings": len(labelings), "separable": len(labelings) - len(failed),
                     "failed": ["".join(map(str, f)) for f in failed]}
    return out


def decision_boundary() -> dict:
    """Logistic regression on moons: the boundary is where the two class scores tie,
    (w1 - w0) . x + (b1 - b0) = 0 — a line written entirely in learned weights."""
    Xtr, Xte, ytr, yte, _ = prepare("moons")
    m = SoftmaxRegression().fit(Xtr, ytr)
    a = [w1 - w0 for w0, w1 in zip(m.W[0], m.W[1])]
    c = m.b[1] - m.b[0]
    return {"a": a, "c": c,
            "equation": f"{a[0]:+.3f}*x0 {a[1]:+.3f}*x1 {c:+.3f} = 0  (scaled features)"}


def threshold_tradeoff(thresholds: tuple = (0.9, 0.5, 0.3, 0.1)) -> dict:
    """Rare positives (about 5%). Accuracy flatters a useless model; precision and
    recall trade off as the probability threshold moves."""
    X, y = make_imbalanced()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)
    sc = StandardScaler().fit(Xtr)
    m = SoftmaxRegression().fit(sc.transform(Xtr), ytr)
    p1 = [p[1] for p in m.predict_proba(sc.transform(Xte))]

    def scores(pred):
        prf = precision_recall_f1(yte, pred)["per_class"].get(1, {"precision": 0.0,
                                                                  "recall": 0.0, "f1": 0.0})
        tp = sum(1 for t, p in zip(yte, pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(yte, pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yte, pred) if t == 1 and p == 0)
        return {"accuracy": accuracy(yte, pred), "precision": prf["precision"],
                "recall": prf["recall"], "f1": prf["f1"], "tp": tp, "fp": fp, "fn": fn}

    rows = [{"threshold": "always 'no'", **scores([0] * len(yte))}]
    for t in thresholds:
        rows.append({"threshold": t, **scores([1 if p >= t else 0 for p in p1])})
    return {"n_test": len(yte), "positives_in_test": sum(yte),
            "positive_rate": sum(y) / len(y), "rows": rows}


def gini_vs_entropy() -> list:
    out = []
    for d in ("moons", "circles"):
        Xtr, Xte, ytr, yte, _ = prepare(d)
        row = {"dataset": d}
        for crit in ("gini", "entropy"):
            t = DecisionTree(max_depth=5, criterion=crit).fit(Xtr, ytr)
            row[crit] = accuracy(yte, t.predict(Xte))
        out.append(row)
    return out


def curse_of_dimensionality(dims: tuple = (2, 10, 100, 1000), n: int = 200) -> list:
    """For random points in a unit cube, nearest / farthest distance from a query.
    Near 0 = 'nearest' is meaningfully near; near 1 = everything is about equally far."""
    rng = random.Random(3)
    out = []
    for d in dims:
        pts = [[rng.random() for _ in range(d)] for _ in range(n)]
        q = [rng.random() for _ in range(d)]
        ds = [euclidean(q, p) for p in pts]
        out.append({"dims": d, "nearest_over_farthest": min(ds) / max(ds)})
    return out


def run_module_1_to_3() -> dict:
    return {
        "course_examples": course_examples(),
        "rule_vs_learned": rule_vs_learned(),
        "paradigms": paradigms_demo(),
        "label_noise": label_noise(),
        "learning_curve": learning_curve(),
        "house_regression": house_regression(),
        "learning_rate": learning_rate_sweep(),
        "gd_flavours": gd_flavours(),
        "ridge": ridge_sweep(),
        "early_stopping": early_stopping(),
        "depth_by_validation": depth_by_validation(),
        "vc": vc_shattering(),
        "decision_boundary": decision_boundary(),
        "threshold": threshold_tradeoff(),
        "gini_vs_entropy": gini_vs_entropy(),
        "curse": curse_of_dimensionality(),
    }
