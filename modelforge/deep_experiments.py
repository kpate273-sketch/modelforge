"""Experiments for Modules 4-5 — one per idea the lessons teach.

  neuron_is_logistic_regression  From Neuron to Network: one softmax/sigmoid layer
                                 is logistic regression
  activation_comparison          Activation Functions: value and slope of each,
                                 gradient check, accuracy after a short budget
  dying_relu                     Activation Functions: ReLU units that never recover
  transfer_learning              Why Deep Networks Work: fine-tuning vs from scratch,
                                 and domain mismatch
  catastrophic_forgetting        The Open Frontier: learning B overwrites A
  cnn_properties                 CNNs: weight sharing, translation, pooling, rotation
  rnn_gradients                  RNNs: vanishing / exploding through time, clipping
  gating                         LSTMs and GRUs: the cell-state gradient path, GRU size
  attention_order_and_cost       Transformers: positional encoding, quadratic cost
"""

from __future__ import annotations

from math import fsum

import copy
import math
import random

from .architectures import (COURSE_EDGE_KERNEL, SOBEL_Y, attention_cost, bptt_gradient,
                            clip_by_norm, conv2d, conv_param_counts, count_gate_params,
                            edge_image, global_max, gru_step, lstm_cell_gradient, max_pool,
                            positional_encoding, rotate90, self_attention, strongest_column,
                            tanh_rnn_gradient, tiny_gru_params, word_vectors)
from .datasets import StandardScaler, make_circles, make_moons
from .linear import SoftmaxRegression
from .metrics import accuracy
from .neural import ACTIVATIONS, MLP, activation_value_and_slope
from .pipeline import prepare
from .registry import shift


# --------------------------------------------------------------------------- #
# Module 4
# --------------------------------------------------------------------------- #
def neuron_is_logistic_regression() -> dict:
    """A network with no hidden layer computes softmax(W x + b) — the same function
    family as logistic regression. Trained on the same data, they agree."""
    Xtr, Xte, ytr, yte, _ = prepare("moons")
    lr = SoftmaxRegression(lr=0.5, epochs=300, l2=0.0).fit(Xtr, ytr)
    net = MLP(hidden=(), lr=0.5, epochs=300, batch_size=len(Xtr)).fit(Xtr, ytr)
    p_lr, p_net = lr.predict(Xte), net.predict(Xte)
    return {"logistic_regression": accuracy(yte, p_lr), "no_hidden_layer_net": accuracy(yte, p_net),
            "same_predictions": sum(a == b for a, b in zip(p_lr, p_net)) / len(p_lr)}


def activation_comparison(points: tuple = (-3.0, -1.0, 0.5, 3.0), epochs: int = 60) -> dict:
    Xtr, Xte, ytr, yte, _ = prepare("moons")
    X, y = make_moons()
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for act in ACTIVATIONS:
        vals = {z: activation_value_and_slope(act, z) for z in points}
        net = MLP(hidden=(16, 16), activation=act, epochs=epochs).fit(Xtr, ytr)
        chk = MLP(hidden=(8, 6), activation=act, epochs=3, seed=1).fit(Xs, y)
        cidx = {c: i for i, c in enumerate(chk.classes)}
        rows.append({"activation": act,
                     "slope": {z: vals[z][1] for z in points},
                     "test_accuracy": accuracy(yte, net.predict(Xte)),
                     "final_loss": net.loss_history[-1],
                     "gradcheck": chk.gradient_check(Xs[5], cidx[y[5]], samples=30)})
    return {"points": points, "epochs": epochs, "rows": rows}


def dying_relu(bias_shift: float = 6.0, epochs: int = 60) -> dict:
    """Simulate one overly large update that pushes every first-layer unit's input
    below zero, then keep training. Plain ReLU units now output 0 with slope 0 on
    every sample, so no gradient can ever revive them; Leaky ReLU's 0.01 slope
    still lets the error signal through."""
    Xtr, Xte, ytr, yte, _ = prepare("moons")
    out = {}
    for act in ("relu", "leaky_relu"):
        net = MLP(hidden=(16,), activation=act, epochs=epochs)
        net._init(len(Xtr[0]), 2)
        net.classes = [0, 1]
        net.b[0] = [v - bias_shift for v in net.b[0]]
        before = net.inactive_units(Xtr)
        net.fit(Xtr, ytr, warm_start=True)
        out[act] = {"inactive_before": before, "dead_after": net.dead_units(Xtr),
                    "test_accuracy": accuracy(yte, net.predict(Xte))}
    return out


def _embed(X: list, M: list, seed: int, noise: float = 0.3) -> list:
    rng = random.Random(seed)
    return [[fsum(M[i][j] * x[j] for j in range(2)) + noise * rng.gauss(0, 1)
             for i in range(len(M))] for x in X]


def _rotate(X: list, degrees: float) -> list:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return [[c * x - s * y, s * x + c * y] for x, y in X]


def transfer_learning(sizes: tuple = (8, 16, 32), dims: int = 10) -> dict:
    """10-dimensional 'sensor readings' of a hidden 2-D moons pattern.

    source    400 labelled examples of the original task, seen through sensor P
    target    the same kind of pattern rotated 20 degrees (related, not identical),
              through sensor P, with only 8 / 16 / 32 labels
    mismatch  a source network trained on sensor Q — same task, different
              'domain', so its learned features do not fit sensor P

    fine-tuning keeps the pretrained hidden layer frozen and trains only a new
    output layer, exactly as the lesson describes.
    """
    rng_p, rng_q = random.Random(1), random.Random(2)
    P = [[rng_p.gauss(0, 1) for _ in range(2)] for _ in range(dims)]
    Q = [[rng_q.gauss(0, 1) for _ in range(2)] for _ in range(dims)]
    Xs, ys = make_moons(n=400, seed=41)
    Es = _embed(Xs, P, 1)
    sc = StandardScaler().fit(Es)
    source = MLP(hidden=(16,), epochs=60).fit(sc.transform(Es), ys)
    mismatched = MLP(hidden=(16,), epochs=60).fit(sc.transform(_embed(Xs, Q, 1)), ys)
    Xt, yt = make_moons(n=400, noise=0.2, seed=43)
    Et = sc.transform(_embed(_rotate(Xt, 20), P, 3))
    rows = []
    for n in sizes:
        Xn, yn = make_moons(n=n, noise=0.2, seed=44)
        En = sc.transform(_embed(_rotate(Xn, 20), P, 4))
        scratch = MLP(hidden=(16,), epochs=150).fit(En, yn)
        tuned = copy.deepcopy(source).fine_tune(En, yn, epochs=150)
        wrong = copy.deepcopy(mismatched).fine_tune(En, yn, epochs=150)
        rows.append({"n_labels": n, "from_scratch": accuracy(yt, scratch.predict(Et)),
                     "fine_tuned": accuracy(yt, tuned.predict(Et)),
                     "mismatched_source": accuracy(yt, wrong.predict(Et))})
    return {"dims": dims, "n_test": len(yt), "rows": rows}


def catastrophic_forgetting(epochs: int = 80) -> dict:
    """Task A (moons, left side) and task B (circles, right side) never overlap, so
    one network can learn both — if it sees both. Trained on A, then only on B,
    it overwrites what it knew about A."""
    XA, yA = make_moons(n=300, seed=51)
    XB, yB = make_circles(n=300, seed=52)
    TA, tA = make_moons(n=300, seed=53)
    TB, tB = make_circles(n=300, seed=54)
    XA, TA = shift(XA, dx=-2.5), shift(TA, dx=-2.5)
    XB, TB = shift(XB, dx=2.5), shift(TB, dx=2.5)
    sc = StandardScaler().fit(XA + XB)
    A, B, TA, TB = map(sc.transform, (XA, XB, TA, TB))
    net = MLP(hidden=(16, 16), epochs=epochs).fit(A, yA)
    after_a = (accuracy(tA, net.predict(TA)), accuracy(tB, net.predict(TB)))
    net.fit(B, yB, warm_start=True)
    after_b = (accuracy(tA, net.predict(TA)), accuracy(tB, net.predict(TB)))
    joint = MLP(hidden=(16, 16), epochs=epochs).fit(A + B, yA + yB)
    both = (accuracy(tA, joint.predict(TA)), accuracy(tB, joint.predict(TB)))
    return {"after_A": {"task_A": after_a[0], "task_B": after_a[1]},
            "after_B": {"task_A": after_b[0], "task_B": after_b[1]},
            "joint": {"task_A": both[0], "task_B": both[1]}}


# --------------------------------------------------------------------------- #
# Module 5
# --------------------------------------------------------------------------- #
def cnn_properties() -> dict:
    flat = conv2d(edge_image(8, 4), COURSE_EDGE_KERNEL)[0]
    shifted = {}
    for col in (3, 4, 5):
        fmap = conv2d(edge_image(8, col), COURSE_EDGE_KERNEL)
        shifted[col] = {"feature_row": fmap[0], "pooled_row": max_pool(fmap)[0],
                        "strongest_column": strongest_column(fmap),
                        "global_max": global_max(fmap)}
    rotated = rotate90(edge_image(8, 4))
    return {"params": conv_param_counts(),
            "flat_region_response": flat[0],
            "edge_response": max(flat),
            "shifted": shifted,
            "rotated_vertical_kernel": global_max(conv2d(rotated, COURSE_EDGE_KERNEL)),
            "rotated_horizontal_kernel": global_max(conv2d(rotated, SOBEL_Y))}


def rnn_gradients(steps: int = 20) -> dict:
    exploding = [3.0e5, 4.0e5]
    clipped = clip_by_norm(exploding, 5.0)
    cos = (fsum(a * b for a, b in zip(exploding, clipped))
           / (math.sqrt(fsum(a * a for a in exploding)) * math.sqrt(fsum(b * b for b in clipped))))
    tanh_g = tanh_rnn_gradient(w_h=0.9, steps=50)
    return {"factor_0.5": bptt_gradient(0.5, steps)[-1],
            "factor_1.0": bptt_gradient(1.0, steps)[-1],
            "factor_2.0": bptt_gradient(2.0, steps)[-1],
            "tanh_rnn_10": tanh_g[9], "tanh_rnn_50": tanh_g[49],
            "tanh_rnn_curve": tanh_g,
            "clip": {"before": exploding, "after": clipped, "max_norm": 5.0, "cosine": cos}}


def gating() -> dict:
    cell = {fb: lstm_cell_gradient(fb, 50) for fb in (0.0, 1.0, 3.0)}
    h, gates = gru_step([1.0, 0.0], [0.0, 0.0, 0.0], tiny_gru_params())
    small, large = count_gate_params(2, 3), count_gate_params(256, 256)
    return {"lstm_cell_50": {fb: c[-1] for fb, c in cell.items()},
            "lstm_forget_gate": {fb: _forget_value(fb) for fb in cell},
            "lstm_cell_curves": cell,
            "tanh_rnn_50": tanh_rnn_gradient(w_h=0.9, steps=50)[-1],
            "gru_h": h, "gru_gates": gates,
            "params_small": small, "params_256": large,
            "gru_saving": 1 - large["gru"] / large["lstm"]}


def _forget_value(forget_bias: float) -> float:
    from .architectures import lstm_step, tiny_lstm_params
    _, _, g = lstm_step([0.0, 0.0], [0.0] * 3, [0.0] * 3, tiny_lstm_params(forget_bias=forget_bias))
    return g["forget"][0]


def attention_order_and_cost(d: int = 8) -> dict:
    """'The dog bit the man' vs 'the man bit the dog': same words, opposite meaning.

    Self-attention on its own treats the input as a set, so 'dog' gets exactly the
    same output vector in both sentences. Adding positional encodings makes the
    two differ — order becomes visible to the model."""
    s1 = ["the", "dog", "bit", "the", "man"]
    s2 = ["the", "man", "bit", "the", "dog"]
    vec = word_vectors(s1 + s2, d)
    pe = positional_encoding(len(s1), d)

    def outputs(sentence, with_pe):
        E = [list(vec[w]) for w in sentence]
        if with_pe:
            E = [[a + b for a, b in zip(e, p)] for e, p in zip(E, pe)]
        return self_attention(E, E, E)[0]

    def dog_gap(with_pe):
        o1, o2 = outputs(s1, with_pe), outputs(s2, with_pe)
        a, b = o1[s1.index("dog")], o2[s2.index("dog")]
        return max(abs(x - y) for x, y in zip(a, b))

    cost = [{"n": n, "full": attention_cost(n), "local_w8": attention_cost(n, 8)}
            for n in (10, 100, 1000)]
    return {"sentences": (" ".join(s1), " ".join(s2)),
            "dog_difference_without_pe": dog_gap(False),
            "dog_difference_with_pe": dog_gap(True),
            "cost": cost}


def run_module_4_to_5() -> dict:
    return {
        "neuron_is_lr": neuron_is_logistic_regression(),
        "activations": activation_comparison(),
        "dying_relu": dying_relu(),
        "transfer": transfer_learning(),
        "forgetting": catastrophic_forgetting(),
        "cnn": cnn_properties(),
        "rnn": rnn_gradients(),
        "gating": gating(),
        "attention": attention_order_and_cost(),
    }
