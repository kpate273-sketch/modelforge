"""Command-line interface.

    python -m modelforge lab                     run every experiment, write report.html
    python -m modelforge compare --dataset moons leaderboard for one dataset
    python -m modelforge sweep                   overfitting curves (tree depth, k)
    python -m modelforge gradcheck               verify backprop numerically
    python -m modelforge train --dataset moons --model neural_network
    python -m modelforge registry                list saved versions and the champion
    python -m modelforge predict 0.5 -0.3        classify a point with the champion
    python -m modelforge drift --dx 0.9 --dy -0.6 --scale 1.3
    python -m modelforge rollback                put the previous champion back
    python -m modelforge module 3                every experiment for one course module
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .datasets import DATASETS
from .metrics import format_confusion
from .pipeline import compare, default_models, depth_sweep, format_leaderboard, knn_sweep, prepare, run_one
from .mlops import OODGuard, monitor_decision
from .registry import ModelRegistry, data_fingerprint, drift_report, shift

MODEL_NAMES = [m.name for m in default_models(fast=True)]


def cmd_lab(a) -> int:
    from .lab import run_lab
    from .report_html import render
    t0 = time.time()
    print("Running the full lab (5 models x 3 datasets, sweeps, gradient checks, architectures, "
          "registry, drift, and every module experiment — about 2-3 minutes)...", file=sys.stderr)
    lab = run_lab(fast=a.fast, registry_dir=a.lab_registry)
    for b in lab["boards"].values():
        print(format_leaderboard(b), "\n")
    ab = lab["ablation"]
    print(f"ablation (circles): no hidden layer {ab['no_hidden_layer']:.3f}  ->  "
          f"one hidden layer {ab['one_hidden_layer']:.3f}\n")
    for g in lab["gradient_checks"]:
        print(f"gradient check {g['activation']:10s} max rel error {g['max_relative_error']:.2e} "
              f"{'PASS' if g['passed'] else 'FAIL'}")
    p = lab["production"]
    print(f"\nchampion: {p['champion']['model_type']} {p['champion']['version']}  "
          f"fresh acc {p['champion_acc_fresh']:.3f} -> shifted acc {p['champion_acc_shifted']:.3f}")
    print(f"drift (shifted world): {p['drift_shifted']['verdict']}")
    Path(a.out).write_text(render(lab), encoding="utf-8")
    print(f"\nreport written to {a.out}  ({time.time() - t0:.1f}s)")
    return 0


def cmd_compare(a) -> int:
    res = compare(a.dataset, fast=a.fast)
    print(format_leaderboard(res))
    best = res["rows"][0]
    print(f"\nbest: {best['model']} — confusion matrix on the test set:")
    print(format_confusion(best["confusion"]))
    return 0


def cmd_sweep(a) -> int:
    print("Decision tree depth sweep (moons) — overfitting curve")
    print(f"  {'depth':>5} {'leaves':>7} {'train':>7} {'test':>7}")
    for r in depth_sweep("moons"):
        print(f"  {r['depth']:>5} {r['leaves']:>7} {r['train_accuracy']:>7.3f} {r['test_accuracy']:>7.3f}")
    print("\nk-NN sweep (moons) — small k overfits, large k underfits")
    print(f"  {'k':>5} {'train':>7} {'test':>7}")
    for r in knn_sweep("moons"):
        print(f"  {r['k']:>5} {r['train_accuracy']:>7.3f} {r['test_accuracy']:>7.3f}")
    return 0


def cmd_gradcheck(a) -> int:
    from .lab import gradient_checks
    ok = True
    for g in gradient_checks():
        status = "PASS" if g["passed"] else "FAIL"
        ok &= g["passed"]
        print(f"{g['activation']:10s} max relative error {g['max_relative_error']:.2e}  {status}")
    return 0 if ok else 1


def cmd_train(a) -> int:
    Xtr, Xte, ytr, yte, scaler = prepare(a.dataset)
    model = next(m for m in default_models(fast=a.fast) if m.name == a.model)
    r = run_one(model, Xtr, Xte, ytr, yte)
    reg = ModelRegistry(a.registry)
    meta = reg.save(model, scaler, {"test_accuracy": r["test_accuracy"],
                                    "train_accuracy": r["train_accuracy"],
                                    "macro_f1": r["macro_f1"]}, dataset=a.dataset,
                    data_version=data_fingerprint(Xtr, ytr))
    print(f"trained {a.model} on {a.dataset}: train {r['train_accuracy']:.3f}  "
          f"test {r['test_accuracy']:.3f}  -> saved version {meta['version']} "
          f"(data {meta['data_version']}, code {meta['code_version']})")
    decision = reg.promote(meta)
    print(("PROMOTED: " if decision["promoted"] else "NOT PROMOTED: ") + decision["reason"])
    return 0


def cmd_registry(a) -> int:
    reg = ModelRegistry(a.registry)
    champ = reg.champion()
    vs = reg.versions()
    if not vs:
        print("registry is empty — run `python -m modelforge train` or `lab` first")
        return 0
    print(f"{'version':14s}{'model':22s}{'dataset':9s}{'test acc':>9}  {'stage':11s}{'data':14s}created")
    for v in vs:
        star = " *" if champ and v["version"] == champ["version"] else ""
        print(f"{v['version']:14s}{v['model_type']:22s}{v['dataset']:9s}"
              f"{v['metrics'].get('test_accuracy', 0):>9.3f}  {v.get('stage', '-'):11s}"
              f"{v.get('data_version', '-'):14s}{v['created']}{star}")
    print("\n* = current champion (serving predictions)")
    return 0


def cmd_predict(a) -> int:
    reg = ModelRegistry(a.registry)
    model, scaler, meta = reg.load()
    x = scaler.transform([[a.x, a.y]])
    pred = model.predict(x)[0]
    Xtr, _, _, _, _ = prepare(meta.get("dataset", "moons"))
    guard = OODGuard().fit(Xtr).check(x[0])
    proba = ""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[0]
        proba = "  probabilities " + ", ".join(f"class {c}: {v:.3f}" for c, v in zip(model.classes, p))
    print(f"champion {meta['model_type']} {meta['version']} predicts class {pred} for "
          f"({a.x}, {a.y}){proba}")
    if guard["out_of_distribution"]:
        print(f"WARNING: this input is {guard['distance']:.2f} from the nearest training point "
              f"(threshold {guard['threshold']:.2f}) — unlike anything the model was trained on, "
              "so treat the prediction as unreliable")
    else:
        print(f"input is within the training data (distance {guard['distance']:.2f} <= {guard['threshold']:.2f})")
    return 0


def cmd_drift(a) -> int:
    """Compare simulated live traffic with the champion's training data.

    Reference = the exact training split the models were fit on; live = a fresh
    sample from the same generator (new seed), optionally shifted. Same inputs
    as the lab, so the numbers match the HTML report.
    """
    reg = ModelRegistry(a.registry)
    model, scaler, meta = reg.load()
    Xtr, _, _, _, _ = prepare(meta.get("dataset", "moons"))
    gen = DATASETS[meta.get("dataset", "moons")]
    live_raw, live_y = gen(n=300, seed=99)
    live_raw = shift(live_raw, dx=a.dx, dy=a.dy, scale=a.scale)
    live = scaler.transform(live_raw)
    rep = drift_report(Xtr, live)
    print(f"champion {meta['model_type']} {meta['version']} · shift dx={a.dx} dy={a.dy} scale={a.scale}")
    for f in rep["features"]:
        print(f"feature x{f['feature']}: PSI {f['psi']:.4f}  {f['status']:8s} "
              f"KS D {f['ks']:.3f}  p {f['ks_p']:.2e}{'  (KS: different)' if f['ks_drift'] else ''}")
    acc = sum(p == t for p, t in zip(model.predict(live), live_y)) / len(live_y)
    print(f"verdict: {rep['verdict']}")
    print(f"champion accuracy on this live data: {acc:.3f}")
    action = monitor_decision(rep, acc)
    print(f"action: {action['action']} — {action['why']}")
    return 2 if rep["alert"] else 0


def cmd_rollback(a) -> int:
    reg = ModelRegistry(a.registry)
    res = reg.rollback()
    champ = reg.champion()
    print(f"rolled back {res['rolled_back']} -> champion is now {champ['model_type']} "
          f"{champ['version']} (stage {champ['stage']})")
    return 0


def cmd_module(a) -> int:
    from .course_modules import format_module, run_module
    print(format_module(a.n, run_module(a.n)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="modelforge",
                                 description="Machine learning from scratch, from data to deployed model.")
    ap.add_argument("--registry", default="model_registry", help="model registry folder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lab", help="run every experiment and write the HTML report")
    p.add_argument("--out", default="report.html")
    p.add_argument("--lab-registry", default="lab_registry",
                   help="folder the lab rebuilds from scratch on every run")
    p.add_argument("--fast", action="store_true", help="fewer neural-net epochs")
    p.set_defaults(fn=cmd_lab)

    p = sub.add_parser("compare", help="leaderboard of all models on one dataset")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="moons")
    p.add_argument("--fast", action="store_true")
    p.set_defaults(fn=cmd_compare)

    sub.add_parser("sweep", help="overfitting curves").set_defaults(fn=cmd_sweep)
    sub.add_parser("gradcheck", help="verify backprop numerically").set_defaults(fn=cmd_gradcheck)

    p = sub.add_parser("train", help="train one model, save it, try to promote it")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="moons")
    p.add_argument("--model", choices=MODEL_NAMES, default="neural_network")
    p.add_argument("--fast", action="store_true")
    p.set_defaults(fn=cmd_train)

    sub.add_parser("registry", help="list saved model versions").set_defaults(fn=cmd_registry)

    p = sub.add_parser("predict", help="classify one point with the champion model")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("drift", help="check live data for drift against training data")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.set_defaults(fn=cmd_drift)

    sub.add_parser("rollback", help="restore the previous champion").set_defaults(fn=cmd_rollback)

    p = sub.add_parser("module", help="run every experiment for one course module (1-6)")
    p.add_argument("n", type=int, choices=range(1, 7), metavar="N")
    p.set_defaults(fn=cmd_module)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except BrokenPipeError:          # e.g. `modelforge sweep | head` — not an error
        sys.stderr.close()
        return 0
    except LookupError as e:         # e.g. predict before any model is promoted
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
