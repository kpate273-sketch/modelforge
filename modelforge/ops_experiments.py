"""Experiments for Module 6 — production practice and the open frontier.

  versioning           code, data and weights all versioned; one changed row
                       changes the data version
  data_validation      the CI check that rejects a bad batch before training
  lifecycle            quality floor, staging -> production -> archived, rollback
  canary_and_shadow    a good model rolls out; a model with a hidden live-traffic
                       flaw is caught on 5% of traffic and rolled back
  feature_store_skew   the lesson's "average purchase over 30 days" bug, and the
                       feature store that removes it
  drift_scenarios      no drift / data drift / concept drift -> what the monitor
                       does about each, and retraining after data drift
  frontier             adversarial examples, overconfidence far from the data,
                       and permutation importance
"""

from __future__ import annotations

from math import fsum

import math
import random
import shutil
import tempfile

from .datasets import StandardScaler, make_circles, make_moons, train_test_split
from .frontier import (adversarial_robustness, flipped_example, interpretability_demo,
                       overconfidence)
from .linear import SoftmaxRegression
from .metrics import accuracy
from .mlops import (FeatureStore, OODGuard, canary_rollout, infer_schema, monitor_decision,
                    shadow_deploy, validate_data)
from .neural import MLP
from .pipeline import prepare, trained_network
from .registry import ModelRegistry, data_fingerprint, drift_report, shift
from .tree import DecisionTree


def versioning() -> dict:
    Xtr, _, ytr, _, _ = prepare("moons")
    v1 = data_fingerprint(Xtr, ytr)
    edited = [list(r) for r in Xtr]
    edited[0][0] += 0.001                       # one value in one row, nudged
    v2 = data_fingerprint(edited, ytr)
    return {"data_version": v1, "after_one_edit": v2, "changed": v1 != v2}


def data_validation() -> dict:
    X, y = make_moons()
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    schema = infer_schema(Xtr, ytr)
    nulls = [[r[0], None] if i % 10 == 0 else list(r) for i, r in enumerate(Xte)]
    wild = [[r[0] * 100, r[1]] if i < 5 else list(r) for i, r in enumerate(Xte)]
    return {"schema_ranges": schema["ranges"],
            "clean_batch": validate_data(Xte, yte, schema),
            "column_with_nulls": validate_data(nulls, yte, schema),
            "out_of_range": validate_data(wild, yte, schema),
            "unknown_label": validate_data(Xte, yte[:-1] + [7], schema)}


def lifecycle() -> dict:
    tmp = tempfile.mkdtemp()
    try:
        reg = ModelRegistry(tmp)
        events = []
        Xc, Xct, yc, yct, scc = prepare("circles")
        weak = SoftmaxRegression().fit(Xc, yc)
        m = reg.save(weak, scc, {"test_accuracy": accuracy(yct, weak.predict(Xct))}, "circles")
        events.append(("logistic_regression on circles", m["metrics"]["test_accuracy"], reg.promote(m)))
        Xtr, Xte, ytr, yte, sc = prepare("moons")
        dv = data_fingerprint(Xtr, ytr)
        for model in (DecisionTree(max_depth=5).fit(Xtr, ytr), trained_network("moons")):
            m = reg.save(model, sc, {"test_accuracy": accuracy(yte, model.predict(Xte))},
                         "moons", data_version=dv)
            events.append((model.name + " on moons", m["metrics"]["test_accuracy"], reg.promote(m)))
        before = [(v["model_type"], v["stage"]) for v in reg.versions()]
        rb = reg.rollback()
        after = [(v["model_type"], v["stage"]) for v in reg.versions()]
        champ = reg.champion()
        return {"events": [{"submitted": n, "test_accuracy": a, "promoted": d["promoted"],
                            "reason": d["reason"]} for n, a, d in events],
                "stages_before_rollback": before, "stages_after_rollback": after,
                "champion_after_rollback": champ["model_type"],
                "champion_meta": {k: champ[k] for k in ("data_version", "code_version", "stage")}}
    finally:
        shutil.rmtree(tmp)


def canary_and_shadow() -> dict:
    """Champion: decision tree (90.8% offline). Challenger: neural network (97.4%).

    'skewed' serves the same challenger through a re-implemented serving path that
    feeds the features in the wrong order — a bug no offline test of the model
    can see, because the model itself is fine."""
    Xtr, Xte, ytr, yte, sc = prepare("moons")
    champ = DecisionTree(max_depth=5).fit(Xtr, ytr)
    chall = trained_network("moons")
    live_raw, live_y = make_moons(n=2000, seed=77)
    live = sc.transform(live_raw)
    good = canary_rollout(champ.predict, chall.predict, live, live_y)
    skewed_predict = lambda X: chall.predict([[r[1], r[0]] for r in X])
    bad = canary_rollout(champ.predict, skewed_predict, live, live_y)
    shadow = shadow_deploy(champ.predict, chall.predict, live, live_y)
    return {"offline": {"champion": accuracy(yte, champ.predict(Xte)),
                        "challenger": accuracy(yte, chall.predict(Xte))},
            "good": good, "skewed": bad, "shadow": shadow}


# --------------------------------------------------------------------------- #
# Training-serving skew: the lesson's own example
# --------------------------------------------------------------------------- #
def make_customers(n: int = 600, days: int = 61, seed: int = 8) -> tuple:
    """Daily purchase amounts for n customers over `days` days; day days-1 is
    'today'. Label 1 = a high-value customer (true spending rate above $40/day)."""
    rng = random.Random(seed)
    customers, labels = [], []
    for _ in range(n):
        rate = rng.uniform(10, 70)
        history = []
        for d in range(days):
            buys = rng.random() < 0.6
            amount = max(0.0, rate * (1 + 0.6 * rng.gauss(0, 1))) if buys else 0.0
            if d == days - 1:
                amount *= 2.5             # today is a sale day
            history.append(round(amount, 2))
        customers.append(history)
        labels.append(1 if rate > 40 else 0)
    return customers, labels


def avg_purchase_30d(history: list, as_of: int) -> float:
    """THE definition: average daily purchase over the 30 full days before as_of."""
    window = history[as_of - 30:as_of]
    return fsum(window) / 30


def avg_purchase_30d_reimplemented(history: list, as_of: int) -> float:
    """A serving-side rewrite in 'another codebase': the window ends at as_of
    inclusive, so it silently includes today's (sale-day) purchases."""
    window = history[as_of - 29:as_of + 1]
    return fsum(window) / 30


def feature_store_skew() -> dict:
    customers, labels = make_customers()
    today = len(customers[0]) - 1
    train_c, test_c = customers[:400], customers[400:]
    train_y, test_y = labels[:400], labels[400:]
    store = FeatureStore()
    store.register("avg_purchase_30d", 1, avg_purchase_30d)
    feats = [("avg_purchase_30d", 1)]
    Xtr = store.offline(feats, train_c, as_of=today)
    sc = StandardScaler().fit(Xtr)
    model = SoftmaxRegression().fit(sc.transform(Xtr), train_y)
    via_store = [store.online(feats, c, as_of=today) for c in test_c]
    via_rewrite = [[avg_purchase_30d_reimplemented(c, today)] for c in test_c]
    offline_rows = store.offline(feats, test_c, as_of=today)
    p_store = model.predict(sc.transform(via_store))
    p_rewrite = model.predict(sc.transform(via_rewrite))
    p_offline = model.predict(sc.transform(offline_rows))
    gaps = [abs(a[0] - b[0]) for a, b in zip(via_rewrite, offline_rows)]
    return {"features": store.names(), "n_test": len(test_c),
            "store_rows_matching_training_code": sum(a == b for a, b in zip(via_store, offline_rows)),
            "rewrite_rows_matching_training_code": sum(a == b for a, b in zip(via_rewrite, offline_rows)),
            "mean_feature_gap": fsum(gaps) / len(gaps),
            "predictions_changed_by_rewrite": sum(a != b for a, b in zip(p_rewrite, p_offline)),
            "accuracy_feature_store": accuracy(test_y, p_store),
            "accuracy_rewrite": accuracy(test_y, p_rewrite)}


# --------------------------------------------------------------------------- #
# Drift: data drift vs concept drift, and the response
# --------------------------------------------------------------------------- #
def drift_scenarios() -> dict:
    Xtr, Xte, ytr, yte, sc = prepare("moons")
    model = trained_network("moons")
    fresh_raw, fresh_y = make_moons(n=300, seed=99)
    scen = {}

    def run(name, raw, labels):
        live = sc.transform(raw)
        rep = drift_report(Xtr, live)
        acc = accuracy(labels, model.predict(live))
        scen[name] = {"max_psi": rep["max_psi"], "ks_alert": rep["ks_alert"],
                      "min_ks_p": min(f["ks_p"] for f in rep["features"]),
                      "live_accuracy": acc, **monitor_decision(rep, acc)}

    run("same world", fresh_raw, fresh_y)
    moved = shift(fresh_raw, dx=0.9, dy=-0.6, scale=1.3)
    run("data drift (inputs moved)", moved, fresh_y)
    # Concept drift: identical inputs, but the rule for the label changed for the
    # right-hand third of the space (what counts as class 1 moved).
    concept_y = [1 - t if r[0] > 1.0 else t for r, t in zip(fresh_raw, fresh_y)]
    run("concept drift (labels' meaning changed)", fresh_raw, concept_y)

    # Retraining trigger for data drift: train on newly labelled live data from the
    # moved world, check on a separate moved sample.
    new_raw, new_y = make_moons(n=300, seed=97)
    new_raw = shift(new_raw, dx=0.9, dy=-0.6, scale=1.3)
    sc2 = StandardScaler().fit(new_raw)
    retrained = MLP(hidden=(16, 16), epochs=150).fit(sc2.transform(new_raw), new_y)
    return {"scenarios": scen,
            "old_model_on_moved_world": scen["data drift (inputs moved)"]["live_accuracy"],
            "retrained_on_moved_world": accuracy(fresh_y, retrained.predict(sc2.transform(moved)))}


def frontier() -> dict:
    Xtr, Xte, ytr, yte, sc = prepare("moons")
    net = trained_network("moons")
    guard = OODGuard().fit(Xtr)
    probes = [Xte[0], Xte[1], [6.0, 6.0], [-8.0, 3.0], [12.0, -12.0]]
    flip = flipped_example(net, Xte, yte, eps=0.3)
    return {"adversarial": adversarial_robustness(net, Xte, yte),
            "flipped": flip,
            "overconfidence": overconfidence(net, guard, probes),
            "guard_threshold": guard.threshold,
            "interpretability": interpretability_demo()}


def run_module_6() -> dict:
    return {
        "versioning": versioning(),
        "validation": data_validation(),
        "lifecycle": lifecycle(),
        "canary": canary_and_shadow(),
        "feature_store": feature_store_skew(),
        "drift": drift_scenarios(),
        "frontier": frontier(),
    }
