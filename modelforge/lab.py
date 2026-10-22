"""Run every experiment in the project in one go — the full lab.

Each experiment answers one question from the course with a measurement:

  leaderboards      does the best model depend on the data?        Modules 1-4
  ablation          is nonlinearity what lets a network bend?     Module 4
  depth/k sweeps    what does overfitting look like?               Module 2
  gradient checks   is my backprop actually correct?               Module 4
  vanishing grads   why do deep sigmoid nets / plain RNNs forget?  Modules 4-5
  architectures     what do conv, LSTM and attention compute?      Module 5
  registry + gate   how does a model ship safely?                  Module 6
  drift             how do you notice the world has moved?         Module 6
  modules 1-6       every remaining lesson, one experiment each    see course_modules.py
"""

from __future__ import annotations

from pathlib import Path

from .architectures import (SOBEL_X, conv2d, demo_image, demo_tokens, lstm_step,
                            max_pool, self_attention, tiny_lstm_params,
                            vanishing_gradient)
from .datasets import StandardScaler, make_moons
from .metrics import accuracy
from .neural import ACTIVATIONS, MLP
from .pipeline import compare, depth_sweep, knn_sweep, prepare
from .registry import ModelRegistry, data_fingerprint, drift_report, shift
from .tree import DecisionTree

DATASET_ORDER = ("blobs", "moons", "circles")


def gradient_checks() -> list:
    X, y = make_moons()
    X = StandardScaler().fit_transform(X)
    out = []
    for act in ACTIVATIONS:
        net = MLP(hidden=(8, 6), activation=act, epochs=3, seed=1).fit(X, y)
        cidx = {c: i for i, c in enumerate(net.classes)}
        err = net.gradient_check(X[5], cidx[y[5]], samples=30)
        out.append({"activation": act, "max_relative_error": err, "passed": err < 1e-4})
    return out


def architecture_demos() -> dict:
    img = demo_image(8)
    edges = conv2d(img, SOBEL_X)
    pooled = max_pool(edges, 2)
    tokens, E = demo_tokens()
    _, weights = self_attention(E, E, E)
    params = tiny_lstm_params()
    h, c = [0.0] * 3, [0.0] * 3
    trace = []
    for t, x in enumerate([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]):
        h, c, gates = lstm_step(x, h, c, params)
        trace.append({"t": t, "x": x, "h": [round(v, 3) for v in h],
                      "c": [round(v, 3) for v in c],
                      "forget": [round(v, 3) for v in gates["forget"]]})
    vg = {act: vanishing_gradient(20, act) for act in ("sigmoid", "tanh", "relu")}
    return {"image": img, "edges": edges, "pooled": pooled, "tokens": tokens,
            "attention": weights, "lstm_trace": trace, "vanishing": vg}


def fresh_registry(path: str) -> ModelRegistry:
    """A registry emptied of earlier lab artifacts, so every lab run is identical.

    Without this, a second run would find the first run's champion already in
    place and make different promotion decisions. Only the registry's own .json
    files are removed.
    """
    p = Path(path)
    if p.exists():
        for f in p.glob("*.json"):
            f.unlink()
    return ModelRegistry(path)


def nonlinearity_ablation() -> dict:
    """Same network, with and without a hidden layer, on circles.

    With no hidden layer the network is just logistic regression and cannot
    separate one ring from another; one hidden layer of 16 tanh units can.
    """
    Xtr, Xte, ytr, yte, _ = prepare("circles")
    flat = MLP(hidden=(), epochs=40).fit(Xtr, ytr)
    deep = MLP(hidden=(16,), epochs=80).fit(Xtr, ytr)
    return {"no_hidden_layer": accuracy(yte, flat.predict(Xte)),
            "one_hidden_layer": accuracy(yte, deep.predict(Xte))}


def production_demo(board: dict, registry_dir: str) -> dict:
    """Ship the moons models through the registry with a champion/challenger gate."""
    reg = fresh_registry(registry_dir)
    rows_by_name = {r["model"]: r for r in board["rows"]}
    data_version = data_fingerprint(board["data"][0], board["data"][2])
    events = []
    # Deliberately submit a weak model first, then stronger ones, then a worse one,
    # so the gate is seen both promoting and refusing.
    for name in ("logistic_regression", "decision_tree", "neural_network", "knn"):
        r = rows_by_name[name]
        meta = reg.save(r["_model"], board["scaler"],
                        {"test_accuracy": r["test_accuracy"],
                         "train_accuracy": r["train_accuracy"],
                         "macro_f1": r["macro_f1"]},
                        dataset=board["dataset"], notes=f"lab run: {name}",
                        data_version=data_version)
        decision = reg.promote(meta)
        events.append({"model": name, "version": meta["version"],
                       "test_accuracy": r["test_accuracy"], **decision})
    model, scaler, champ = reg.load()
    Xtr = board["data"][0]
    # Fresh live traffic from the same world (new seed) vs. a world that moved.
    fresh_raw, fresh_y = make_moons(n=300, seed=99)
    moved_raw = shift(fresh_raw, dx=0.9, dy=-0.6, scale=1.3)
    fresh = scaler.transform(fresh_raw)
    moved = scaler.transform(moved_raw)
    acc = lambda Xs: sum(p == t for p, t in zip(model.predict(Xs), fresh_y)) / len(fresh_y)
    return {
        "events": events,
        "champion": champ,
        "versions": reg.versions(),
        "drift_stable": drift_report(Xtr, fresh),
        "drift_shifted": drift_report(Xtr, moved),
        "champion_acc_fresh": acc(fresh),
        "champion_acc_shifted": acc(moved),
    }


def run_lab(fast: bool = False, registry_dir: str = "lab_registry",
            modules: bool = True) -> dict:
    from .course_modules import run_module
    boards = {d: compare(d, fast=fast) for d in DATASET_ORDER}
    return {
        "modules": {n: run_module(n) for n in range(1, 7)} if modules else {},
        "boards": boards,
        "ablation": nonlinearity_ablation(),
        "depth_sweep": depth_sweep("moons"),
        "knn_sweep": knn_sweep("moons"),
        "gradient_checks": gradient_checks(),
        "architectures": architecture_demos(),
        "production": production_demo(boards["moons"], registry_dir),
    }
