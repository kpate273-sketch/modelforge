"""A guided, module-by-module walkthrough of ModelForge. No downloads, no GPU.

    python demo/run_demo.py

Exits non-zero if any check fails, so it doubles as a smoke test. For every
lesson's experiment run  python -m modelforge module N  (N = 1-6); for all of it
plus the HTML report, run  python -m modelforge lab
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelforge.architectures import (SOBEL_X, conv2d, demo_image, demo_tokens,
                                      self_attention, vanishing_gradient)
from modelforge.datasets import StandardScaler, make_moons
from modelforge.lab import gradient_checks, nonlinearity_ablation
from modelforge.metrics import format_confusion
from modelforge.neural import MLP
from modelforge.pipeline import compare, depth_sweep, format_leaderboard, prepare
from modelforge.registry import ModelRegistry, drift_report, shift
from modelforge.linear import SoftmaxRegression


def hr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> int:
    fails = 0

    hr("MODULE 1 — the ML workflow: generate -> split -> scale -> train -> evaluate")
    Xtr, Xte, ytr, yte, scaler = prepare("moons")
    print(f"  moons dataset: {len(Xtr)} train / {len(Xte)} test, stratified split")
    print(f"  scaler fit on training data only: mean {[round(m, 3) for m in scaler.mean]}")

    hr("MODULE 2 — gradient descent: watch the loss fall")
    lr = SoftmaxRegression(epochs=300).fit(Xtr, ytr)
    h = lr.loss_history
    for e in (0, 10, 50, 150, 299):
        print(f"  epoch {e:>3}: cross-entropy {h[e]:.4f}")
    print("  -> it plateaus around 0.27: a straight line is the best this model can do on moons")
    if h[-1] >= h[0]:
        fails += 1

    hr("MODULE 2 — bias, variance and overfitting (decision-tree depth sweep)")
    print(f"  {'depth':>5} {'train':>7} {'test':>7}")
    for r in depth_sweep("moons", depths=(1, 2, 4, 8, 12)):
        print(f"  {r['depth']:>5} {r['train_accuracy']:>7.3f} {r['test_accuracy']:>7.3f}")

    hr("MODULES 3-4 — five from-scratch models, three datasets (No Free Lunch)")
    for d in ("blobs", "moons", "circles"):
        res = compare(d, fast=True)
        print(format_leaderboard(res), "\n")
        if d == "circles":
            lin = next(r for r in res["rows"] if r["model"] == "logistic_regression")
            net = next(r for r in res["rows"] if r["model"] == "neural_network")
            print(f"  logistic regression {lin['test_accuracy']:.3f} vs neural network "
                  f"{net['test_accuracy']:.3f} — nonlinearity is the difference")
            print("\n  neural network confusion matrix on circles (test set):")
            print("  " + format_confusion(net["confusion"]).replace("\n", "\n  "))
            if net["test_accuracy"] <= lin["test_accuracy"]:
                fails += 1

    hr("MODULE 4 — ablation: is nonlinearity what lets a network bend?")
    ab = nonlinearity_ablation()
    print(f"  circles, no hidden layer (= logistic regression): {ab['no_hidden_layer']:.3f}")
    print(f"  circles, one hidden layer of 16 tanh units:        {ab['one_hidden_layer']:.3f}")
    if ab["one_hidden_layer"] <= ab["no_hidden_layer"]:
        fails += 1

    hr("MODULE 4 — backpropagation, verified against a numerical derivative")
    for g in gradient_checks():
        print(f"  {g['activation']:10s} max relative error {g['max_relative_error']:.2e}  "
              f"{'PASS' if g['passed'] else 'FAIL'}")
        fails += 0 if g["passed"] else 1

    hr("MODULE 5 — what CNNs, Transformers and RNNs actually compute")
    fmap = conv2d(demo_image(8), SOBEL_X)
    print("  convolution: 8x8 image with one vertical edge, Sobel kernel ->")
    for row in fmap[:3]:
        print("   ", " ".join(f"{v:>3.0f}" for v in row))
    tokens, E = demo_tokens()
    _, w = self_attention(E, E, E)
    print("\n  self-attention weights (row = query token):")
    print("         " + "".join(f"{t:>7}" for t in tokens))
    for t, row in zip(tokens, w):
        print(f"    {t:>4} " + "".join(f"{v:>7.2f}" for v in row))
    s, r = vanishing_gradient(20, "sigmoid"), vanishing_gradient(20, "relu")
    print(f"\n  gradient after 20 layers: sigmoid {s[-1]:.1e}   relu {r[-1]:.1f}  (vanishing gradient)")

    hr("MODULE 6 — ship it safely: registry, champion/challenger gate, drift monitor")
    tmp = tempfile.mkdtemp()
    try:
        reg = ModelRegistry(tmp)
        # Same settings as the lab's neural network, so the numbers match the HTML report.
        net = MLP(hidden=(16, 16), activation="tanh", lr=0.1, epochs=150).fit(Xtr, ytr)
        from modelforge.metrics import accuracy
        for model in (lr, net):
            acc = accuracy(yte, model.predict(Xte))
            meta = reg.save(model, scaler, {"test_accuracy": acc}, "moons")
            d = reg.promote(meta)
            print(f"  {model.name:20s} v{meta['version']}  test {acc:.3f}  -> "
                  f"{'PROMOTED' if d['promoted'] else 'kept champion'}")
        champ, sc, meta = reg.load()
        fresh, fy = make_moons(n=300, seed=99)
        moved = shift(fresh, dx=0.9, dy=-0.6, scale=1.3)
        acc_f = accuracy(fy, champ.predict(sc.transform(fresh)))
        acc_m = accuracy(fy, champ.predict(sc.transform(moved)))
        stable = drift_report(Xtr, sc.transform(fresh))
        rep = drift_report(Xtr, sc.transform(moved))
        print(f"\n  champion on fresh data: {acc_f:.3f}   after the world shifts: {acc_m:.3f}")
        print(f"  drift monitor, same world:  max PSI {stable['max_psi']:.2f} -> {stable['verdict']}")
        print(f"  drift monitor, world moved: max PSI {rep['max_psi']:.2f} -> {rep['verdict']}")
        if not rep["alert"] or acc_m >= acc_f:
            fails += 1
    finally:
        shutil.rmtree(tmp)

    hr("RESULT")
    print("All checks passed." if not fails else f"{fails} check(s) failed.")
    print("Every lesson, module by module:  python -m modelforge module 1   (... 6)")
    print("Full lab + HTML report:          python -m modelforge lab")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
