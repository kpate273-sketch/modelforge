"""Operating a model safely (Module 6: MLOps Foundations; Deploying Models Safely;
Monitoring Drift).

  infer_schema / validate_data   data validation, the CI check that catches "a
                                 column of nulls" before it reaches training
  FeatureStore                   one definition of each feature, served to both
                                 training and production — no training-serving skew
  canary_rollout                 send a small, growing slice of live traffic to a
                                 new model; roll back the moment it does worse
  shadow_deploy                  run a new model on live traffic without showing
                                 its answers to anyone
  monitor_decision               turn drift and accuracy signals into an action:
                                 keep serving, watch, retrain, or ask a human
  OODGuard                       flag inputs far from anything seen in training,
                                 so the system can say "I don't know"
"""

from __future__ import annotations

from math import fsum

import hashlib
import math

from .linalg import euclidean


# --------------------------------------------------------------------------- #
# Data validation
# --------------------------------------------------------------------------- #
def infer_schema(X: list, y: list, margin: float = 0.5) -> dict:
    """Expected shape, ranges and labels, learned from data you trust."""
    cols = list(zip(*X))
    ranges = []
    for c in cols:
        lo, hi = min(c), max(c)
        pad = (hi - lo) * margin
        ranges.append((lo - pad, hi + pad))
    return {"n_features": len(cols), "ranges": ranges, "labels": sorted(set(y))}


def validate_data(X: list, y: list, schema: dict) -> list:
    """Return a list of problems; empty means the batch is safe to train on."""
    problems = []
    n_bad_width = sum(1 for r in X if len(r) != schema["n_features"])
    if n_bad_width:
        problems.append(f"{n_bad_width} rows do not have {schema['n_features']} features")
    for j in range(schema["n_features"]):
        col = [r[j] for r in X if len(r) == schema["n_features"]]
        missing = sum(1 for v in col if v is None or (isinstance(v, float) and math.isnan(v)))
        if missing:
            problems.append(f"feature x{j}: {missing} missing values")
            continue
        wrong_type = sum(1 for v in col if not isinstance(v, (int, float)))
        if wrong_type:
            problems.append(f"feature x{j}: {wrong_type} non-numeric values")
            continue
        lo, hi = schema["ranges"][j]
        out = sum(1 for v in col if not lo <= v <= hi)
        if out:
            problems.append(f"feature x{j}: {out} values outside the expected range "
                            f"[{lo:.2f}, {hi:.2f}]")
    unknown = sorted(set(y) - set(schema["labels"]), key=str)
    if unknown:
        problems.append(f"unknown labels {unknown}")
    if len(X) != len(y):
        problems.append(f"{len(X)} rows but {len(y)} labels")
    return problems


# --------------------------------------------------------------------------- #
# Feature store
# --------------------------------------------------------------------------- #
class FeatureStore:
    """Each feature is defined once, by name and version. Training (offline) and
    serving (online) both call the same definition, so they cannot disagree."""

    def __init__(self):
        self._defs: dict = {}

    def register(self, name: str, version: int, fn) -> None:
        self._defs[(name, version)] = fn

    def names(self) -> list:
        return sorted(f"{n}@v{v}" for n, v in self._defs)

    def compute(self, name: str, version: int, entity, as_of: int) -> float:
        return self._defs[(name, version)](entity, as_of)

    def offline(self, features: list, entities: list, as_of: int) -> list:
        """A training table: one row per entity."""
        return [[self.compute(n, v, e, as_of) for n, v in features] for e in entities]

    def online(self, features: list, entity, as_of: int) -> list:
        """One live feature vector, computed by the same code."""
        return [self.compute(n, v, entity, as_of) for n, v in features]


# --------------------------------------------------------------------------- #
# Safe rollout
# --------------------------------------------------------------------------- #
def _bucket(request_id: int) -> float:
    """Deterministic 0-1 bucket per request, so routing is stable and reproducible."""
    h = hashlib.sha256(str(request_id).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def canary_rollout(champion_predict, challenger_predict, X: list, y: list,
                   stages: tuple = (0.05, 0.25, 0.50, 1.00), tolerance: float = 0.02,
                   min_requests: int = 20) -> dict:
    """Route a growing share of live requests to the challenger.

    The stream is cut into equal chunks, one per stage. Within a chunk, requests
    whose bucket is below the stage's share go to the challenger. After each
    chunk the challenger's error rate is compared with the champion's; if it is
    worse by more than `tolerance`, everyone goes back to the champion at once.
    """
    chunk = len(X) // len(stages)
    log, served_by_challenger = [], 0
    for k, share in enumerate(stages):
        idx = range(k * chunk, (k + 1) * chunk)
        to_new = [i for i in idx if _bucket(i) < share]
        to_old = [i for i in idx if _bucket(i) >= share]
        served_by_challenger += len(to_new)
        new_pred = challenger_predict([X[i] for i in to_new]) if to_new else []
        old_pred = champion_predict([X[i] for i in to_old]) if to_old else []
        new_err = (sum(p != y[i] for p, i in zip(new_pred, to_new)) / len(to_new)) if to_new else 0.0
        if old_pred:
            old_err = sum(p != y[i] for p, i in zip(old_pred, to_old)) / len(to_old)
        else:                                   # 100% stage: compare on the same requests
            old_err = sum(p != y[i] for p, i in zip(champion_predict([X[i] for i in to_new]), to_new)) / len(to_new)
        ok = len(to_new) < min_requests or new_err <= old_err + tolerance
        log.append({"share": share, "challenger_requests": len(to_new),
                    "challenger_error": new_err, "champion_error": old_err, "ok": ok})
        if not ok:
            return {"result": "rolled back", "stopped_at": share, "stages": log,
                    "requests_exposed": served_by_challenger, "total_requests": chunk * len(stages)}
    return {"result": "fully rolled out", "stopped_at": None, "stages": log,
            "requests_exposed": served_by_challenger, "total_requests": chunk * len(stages)}


def shadow_deploy(champion_predict, challenger_predict, X: list, y: list) -> dict:
    """Users only ever see the champion's answers; the challenger's are logged and
    compared afterwards. Zero user-facing risk."""
    served = champion_predict(X)
    shadow = challenger_predict(X)
    return {"requests": len(X),
            "agreement": sum(a == b for a, b in zip(served, shadow)) / len(X),
            "champion_accuracy": sum(a == t for a, t in zip(served, y)) / len(X),
            "shadow_accuracy": sum(a == t for a, t in zip(shadow, y)) / len(X),
            "users_shown_shadow_answers": 0}


# --------------------------------------------------------------------------- #
# Monitoring -> action
# --------------------------------------------------------------------------- #
def monitor_decision(drift: dict, live_accuracy: float | None = None,
                     accuracy_floor: float = 0.90) -> dict:
    """The lesson's three layers: alert thresholds, retraining triggers, and human
    review for possible concept drift."""
    if drift["alert"]:
        return {"action": "RETRAIN", "alert": True,
                "why": f"data drift: max PSI {drift['max_psi']:.2f} > 0.25 — retrain on fresh "
                       "labelled data, then send the new model through the gate and a canary"}
    if live_accuracy is not None and live_accuracy < accuracy_floor:
        return {"action": "HUMAN REVIEW", "alert": True,
                "why": f"inputs look unchanged (max PSI {drift['max_psi']:.2f}) but live accuracy "
                       f"{live_accuracy:.3f} is below {accuracy_floor:.2f} — possible concept drift; "
                       "the meaning of the label may have changed, so retraining alone may not fix it"}
    if drift["max_psi"] >= 0.1:
        return {"action": "WATCH", "alert": False, "why": "moderate shift — watch closely"}
    return {"action": "OK", "alert": False, "why": "inputs and accuracy look normal"}


# --------------------------------------------------------------------------- #
# Knowing when you don't know
# --------------------------------------------------------------------------- #
class OODGuard:
    """Flags inputs that are far from every training point.

    Threshold = the largest nearest-neighbour distance seen inside the training
    set, times a safety factor. A neural network gives a confident answer
    everywhere; this guard is how the system admits an input is unlike anything
    it has seen."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.X: list = []
        self.threshold = 0.0

    def fit(self, X: list) -> "OODGuard":
        self.X = [list(r) for r in X]
        nn = []
        for i, a in enumerate(self.X):
            nn.append(min(euclidean(a, b) for j, b in enumerate(self.X) if j != i))
        self.threshold = max(nn) * self.factor
        return self

    def distance(self, x: list) -> float:
        return min(euclidean(x, b) for b in self.X)

    def check(self, x: list) -> dict:
        d = self.distance(x)
        return {"distance": d, "threshold": self.threshold, "out_of_distribution": d > self.threshold}
