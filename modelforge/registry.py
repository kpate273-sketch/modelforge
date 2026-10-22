"""From notebook to production (Module 6: MLOps Foundations; Deploying Models
Safely; Monitoring Drift: Why Models Decay).

  ModelRegistry   saves every trained model as a versioned JSON artifact with a
                  content hash, its metrics, the dataset it was trained on and
                  the scaler it needs — so any prediction can be traced back to
                  exactly which model produced it.

  promote()       a champion/challenger gate. A newly trained model becomes the
                  production model only if it beats the current champion on the
                  held-out test set by a margin. Deploying safely means a worse
                  model can never silently replace a better one.

  drift_report()  the Population Stability Index and the Kolmogorov-Smirnov test
                  per feature, comparing live data against the training data. A
                  model can be perfect on the day it ships and wrong a month
                  later because the world moved; PSI and KS are how you notice
                  before your users do.

Versioning covers all three things the MLOps lesson lists: the code (the
package version), the data (a fingerprint of the exact training rows) and the
weights (the content hash that names each version). Every version also has a
lifecycle stage — staging, production or archived — and the previous champion
can be restored with rollback().
"""

from __future__ import annotations

from math import fsum

import hashlib
import json
import math
import time
from pathlib import Path

from . import __version__ as CODE_VERSION
from .datasets import StandardScaler
from .linear import SoftmaxRegression
from .neighbors import KNearestNeighbors
from .neural import MLP
from .tree import DecisionTree, RandomForest

MODEL_TYPES = {
    "logistic_regression": SoftmaxRegression,
    "knn": KNearestNeighbors,
    "decision_tree": DecisionTree,
    "random_forest": RandomForest,
    "neural_network": MLP,
}


def model_from_dict(d: dict):
    cls = MODEL_TYPES.get(d.get("type"))
    if cls is None:
        raise ValueError(f"unknown model type {d.get('type')!r}")
    return cls.from_dict(d)


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def data_fingerprint(X: list, y: list) -> str:
    """A short hash of the exact training rows and labels — the 'data version'.
    Change one row and the fingerprint changes, so a model always records which
    snapshot of the data produced it."""
    return "d-" + _hash({"X": X, "y": y})[:10]


class ModelRegistry:
    """A folder of versioned model artifacts plus a pointer to the champion."""

    def __init__(self, root: str = "model_registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write_index({"versions": [], "champion": None})

    # -- index -------------------------------------------------------------- #
    def _read_index(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, idx: dict) -> None:
        self.index_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")

    # -- save / load -------------------------------------------------------- #
    def save(self, model, scaler: StandardScaler, metrics: dict,
             dataset: str, notes: str = "", data_version: str | None = None) -> dict:
        body = {"model": model.to_dict(), "scaler": scaler.to_dict()}
        version_id = _hash(body)
        meta = {
            "version": version_id,
            "model_type": model.name,
            "dataset": dataset,
            "data_version": data_version or "unrecorded",
            "code_version": CODE_VERSION,
            "stage": "staging",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": {k: round(v, 4) for k, v in metrics.items()
                        if isinstance(v, (int, float))},
            "notes": notes,
        }
        artifact = {"meta": meta, **body}
        (self.root / f"{version_id}.json").write_text(
            json.dumps(artifact), encoding="utf-8")
        idx = self._read_index()
        existing = next((v for v in idx["versions"] if v["version"] == version_id), None)
        if existing is None:
            idx["versions"].append(meta)
        else:
            meta = existing                 # identical model: keep its stage and history
        self._write_index(idx)
        return dict(meta)

    def load(self, version: str | None = None) -> tuple:
        """Return (model, scaler, meta). Defaults to the current champion."""
        idx = self._read_index()
        version = version or idx.get("champion")
        if not version:
            raise LookupError("no champion model yet — train and promote one first")
        art = json.loads((self.root / f"{version}.json").read_text(encoding="utf-8"))
        return model_from_dict(art["model"]), StandardScaler.from_dict(art["scaler"]), art["meta"]

    def versions(self) -> list:
        return self._read_index()["versions"]

    def champion(self) -> dict | None:
        idx = self._read_index()
        for v in idx["versions"]:
            if v["version"] == idx.get("champion"):
                return v
        return None

    def _set_stage(self, idx: dict, version: str, stage: str) -> None:
        for v in idx["versions"]:
            if v["version"] == version:
                v["stage"] = stage

    # -- safe deployment ---------------------------------------------------- #
    def promote(self, meta: dict, metric: str = "test_accuracy",
                min_improvement: float = 0.005, min_score: float = 0.80) -> dict:
        """Two quality gates, as in the MLOps lesson's CI pipeline:

        1. an absolute floor — a model below min_score is never promoted, even
           when there is no champion yet;
        2. champion/challenger — it must beat the current champion by a margin.
        A promoted model moves to 'production' and the old champion to 'archived'.
        """
        idx = self._read_index()
        current = self.champion()
        challenger_score = meta["metrics"].get(metric, 0.0)
        if challenger_score < min_score:
            decision = False
            reason = (f"challenger {challenger_score:.3f} is below the quality floor "
                      f"{min_score:.2f} — not eligible")
        elif current is None:
            decision, reason = True, "no champion yet — first model becomes champion"
        else:
            champ_score = current["metrics"].get(metric, 0.0)
            if challenger_score >= champ_score + min_improvement:
                decision = True
                reason = (f"challenger {challenger_score:.3f} beats champion "
                          f"{champ_score:.3f} by >= {min_improvement}")
            else:
                decision = False
                reason = (f"challenger {challenger_score:.3f} does not beat champion "
                          f"{champ_score:.3f} by {min_improvement} — keeping champion")
        if decision and meta["version"] != idx.get("champion"):
            if idx.get("champion"):
                self._set_stage(idx, idx["champion"], "archived")
                idx.setdefault("history", []).append(idx["champion"])
            idx["champion"] = meta["version"]
            self._set_stage(idx, meta["version"], "production")
            self._write_index(idx)
        return {"promoted": decision, "reason": reason,
                "champion": idx.get("champion")}

    def rollback(self) -> dict:
        """Put the previous champion back into production (the 'instantly route
        everyone back to the old model' step of a failed rollout)."""
        idx = self._read_index()
        history = idx.get("history", [])
        if not history:
            raise LookupError("nothing to roll back to — only one model has been in production")
        bad, previous = idx.get("champion"), history.pop()
        self._set_stage(idx, bad, "archived")
        self._set_stage(idx, previous, "production")
        idx["champion"] = previous
        self._write_index(idx)
        return {"rolled_back": bad, "champion": previous}


# --------------------------------------------------------------------------- #
# Drift monitoring
# --------------------------------------------------------------------------- #
def psi(expected: list, actual: list, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant drift.
    """
    lo, hi = min(expected), max(expected)
    if hi == lo:
        hi = lo + 1e-9
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    edges[0], edges[-1] = -math.inf, math.inf

    def frac(values):
        counts = [0] * bins
        for v in values:
            for b in range(bins):
                if edges[b] <= v < edges[b + 1]:
                    counts[b] += 1
                    break
        n = max(len(values), 1)
        return [max(c / n, 1e-4) for c in counts]   # floor avoids log(0)

    e, a = frac(expected), frac(actual)
    return fsum((ai - ei) * math.log(ai / ei) for ei, ai in zip(e, a))


def ks_test(a: list, b: list) -> tuple:
    """Two-sample Kolmogorov-Smirnov test: (D, p_value).

    D is the largest vertical gap between the two empirical CDFs. The p-value
    uses the asymptotic Kolmogorov distribution; a very small p-value means the
    two samples almost certainly come from different distributions.
    """
    xa, xb = sorted(a), sorted(b)
    na, nb = len(xa), len(xb)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        v = min(xa[i], xb[j])
        while i < na and xa[i] <= v:
            i += 1
        while j < nb and xb[j] <= v:
            j += 1
        d = max(d, abs(i / na - j / nb))
    ne = na * nb / (na + nb)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 1e-3:
        return d, 1.0
    p = 2.0 * fsum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, min(max(p, 0.0), 1.0)


def drift_report(train_X: list, live_X: list, ks_alpha: float = 0.01) -> dict:
    """Per-feature PSI (alert above 0.25) and KS test (alert below p = 0.01).

    The verdict and alert follow PSI, the lesson's main rule of thumb; KS is
    reported alongside as a second, independent test.
    """
    features = []
    for j in range(len(train_X[0])):
        ref, cur = [r[j] for r in train_X], [r[j] for r in live_X]
        value = psi(ref, cur)
        d, p = ks_test(ref, cur)
        status = "stable" if value < 0.1 else ("moderate" if value < 0.25 else "DRIFT")
        features.append({"feature": j, "psi": round(value, 4), "status": status,
                         "ks": round(d, 4), "ks_p": p, "ks_drift": p < ks_alpha})
    worst = max(f["psi"] for f in features)
    return {"features": features, "max_psi": round(worst, 4),
            "alert": worst >= 0.25,
            "ks_alert": any(f["ks_drift"] for f in features),
            "verdict": ("Significant drift — retrain or investigate before trusting predictions."
                        if worst >= 0.25 else
                        "Moderate shift — watch closely." if worst >= 0.1 else
                        "Live data looks like training data.")}


def shift(X: list, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> list:
    """Simulate a changing world: move and stretch the live data."""
    return [[r[0] * scale + dx, r[1] * scale + dy] + list(r[2:]) for r in X]
