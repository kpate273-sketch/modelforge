"""A multilayer perceptron with backpropagation, from scratch.
(Module 4: From Neuron to Network; Activation Functions and Nonlinearity;
Backpropagation: Learning Across Layers; Why Deep Networks Work.)

Activations: tanh, sigmoid, ReLU, and the refinements the course names —
Leaky ReLU, GELU and Swish. Also here: validation tracking with early stopping
(Module 2), fine-tuning for transfer learning, and the input gradient used for
adversarial examples (Module 6: The Open Frontier).

Architecture: input -> one or more hidden layers -> softmax output.

  forward    z_l = W_l a_{l-1} + b_l,   a_l = activation(z_l)
             output probabilities p = softmax(z_L)

  backward   with cross-entropy loss, the output error is simply
                 delta_L = p - onehot(y)
             and it flows backward through the chain rule:
                 delta_l = (W_{l+1}^T delta_{l+1}) * activation'(z_l)
             each layer's gradient is   dW_l = delta_l (outer) a_{l-1}

That backward pass is the entire "learning across layers" lesson in about
twenty lines. The nonlinearity is what makes it more than logistic regression:
without an activation function, stacked layers collapse into one linear map and
the network can no longer bend its decision boundary around the moons.

gradient_check() verifies the backward pass against a numerical derivative.
Backprop is easy to get subtly wrong, and a gradient check is how you prove it
is right instead of hoping.
"""

from __future__ import annotations

from math import fsum

import math
import random

from .linalg import softmax, argmax
from .metrics import cross_entropy


LEAKY_SLOPE = 0.01
ACTIVATIONS = ("tanh", "sigmoid", "relu", "leaky_relu", "gelu", "swish")


def _sig(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _act(name: str, z: float) -> float:
    if name == "tanh":
        return math.tanh(z)
    if name == "relu":
        return z if z > 0 else 0.0
    if name == "sigmoid":
        return _sig(z)
    if name == "leaky_relu":          # a small slope for z < 0, so a unit can't die
        return z if z > 0 else LEAKY_SLOPE * z
    if name == "gelu":                # z * Phi(z), Phi = standard normal CDF
        return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))
    if name == "swish":               # z * sigmoid(z)
        return z * _sig(z)
    raise ValueError(f"unknown activation {name!r}")


def _act_grad(name: str, z: float, a: float) -> float:
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return 1.0 if z > 0 else 0.0
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "leaky_relu":
        return 1.0 if z > 0 else LEAKY_SLOPE
    if name == "gelu":
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return cdf + z * pdf
    if name == "swish":
        s = _sig(z)
        return s + z * s * (1.0 - s)
    raise ValueError(f"unknown activation {name!r}")


def activation_value_and_slope(name: str, z: float) -> tuple:
    """(f(z), f'(z)) — used to compare how each activation passes gradient."""
    a = _act(name, z)
    return a, _act_grad(name, z, a)


class MLP:
    name = "neural_network"

    def __init__(self, hidden: tuple = (16, 16), activation: str = "tanh",
                 lr: float = 0.1, epochs: int = 300, batch_size: int = 16,
                 seed: int = 0):
        self.hidden = tuple(hidden)
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.W: list = []      # per layer: [out][in]
        self.b: list = []      # per layer: [out]
        self.classes: list = []
        self.loss_history: list = []
        self.val_loss_history: list = []
        self.best_epoch = None
        self.stopped_epoch = None

    # -- setup -------------------------------------------------------------- #
    def _init(self, n_in: int, n_out: int) -> None:
        rng = random.Random(self.seed)
        sizes = [n_in, *self.hidden, n_out]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            # Xavier/Glorot uniform init keeps activations from exploding or vanishing.
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.W.append([[rng.uniform(-limit, limit) for _ in range(fan_in)]
                           for _ in range(fan_out)])
            self.b.append([0.0] * fan_out)

    # -- forward ------------------------------------------------------------ #
    def _forward(self, x: list) -> tuple:
        """Return (zs, activations) for every layer; last activation = probs."""
        a = list(x)
        zs, acts = [], [a]
        L = len(self.W)
        for l in range(L):
            W, b = self.W[l], self.b[l]
            z = [fsum(wi * ai for wi, ai in zip(row, a)) + bi for row, bi in zip(W, b)]
            zs.append(z)
            if l == L - 1:
                a = softmax(z)
            else:
                a = [_act(self.activation, v) for v in z]
            acts.append(a)
        return zs, acts

    def predict_proba(self, X: list) -> list:
        return [self._forward(x)[1][-1] for x in X]

    def predict(self, X: list) -> list:
        return [self.classes[argmax(p)] for p in self.predict_proba(X)]

    # -- backward ----------------------------------------------------------- #
    def _backward(self, x: list, t: int) -> tuple:
        """Gradients of cross-entropy for one sample, via the chain rule."""
        zs, acts = self._forward(x)
        L = len(self.W)
        gW = [None] * L
        gb = [None] * L
        # Output layer: softmax + cross-entropy => delta = p - onehot
        delta = [p - (1.0 if k == t else 0.0) for k, p in enumerate(acts[-1])]
        for l in range(L - 1, -1, -1):
            a_prev = acts[l]
            gW[l] = [[d * ap for ap in a_prev] for d in delta]
            gb[l] = list(delta)
            if l > 0:
                W = self.W[l]
                back = [fsum(W[k][j] * delta[k] for k in range(len(delta)))
                        for j in range(len(a_prev))]
                delta = [back[j] * _act_grad(self.activation, zs[l - 1][j], a_prev[j])
                         for j in range(len(a_prev))]
        return gW, gb

    # -- training ----------------------------------------------------------- #
    def fit(self, X: list, y: list, X_val: list | None = None, y_val: list | None = None,
            patience: int | None = None, warm_start: bool = False,
            frozen_layers: int = 0) -> "MLP":
        """Mini-batch gradient descent.

        X_val / y_val    track validation loss every epoch (Module 2: the validation set).
        patience         early stopping: stop once validation loss has not improved
                         for this many epochs, and restore the best weights.
        warm_start       keep the current weights instead of re-initializing
                         (used for fine-tuning and the catastrophic-forgetting demo).
        frozen_layers    leave the first N weight layers untouched (transfer learning).
        """
        if not (warm_start and self.W):
            self.classes = sorted(set(y))
        cidx = {c: i for i, c in enumerate(self.classes)}
        yi = [cidx[v] for v in y]
        if not (warm_start and self.W):
            self._init(len(X[0]), len(self.classes))
        rng = random.Random(self.seed)
        order = list(range(len(X)))
        self.loss_history = []
        self.val_loss_history = []
        yv = [cidx[v] for v in y_val] if y_val is not None else None
        best = (math.inf, -1, None)
        self.best_epoch, self.stopped_epoch = None, None

        for epoch in range(self.epochs):
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                accW = [[[0.0] * len(r) for r in W] for W in self.W]
                accb = [[0.0] * len(b) for b in self.b]
                for i in batch:
                    gW, gb = self._backward(X[i], yi[i])
                    for l in range(len(self.W)):
                        for r in range(len(gW[l])):
                            row_acc, row_g = accW[l][r], gW[l][r]
                            for c in range(len(row_g)):
                                row_acc[c] += row_g[c]
                            accb[l][r] += gb[l][r]
                m = len(batch)
                for l in range(frozen_layers, len(self.W)):
                    for r in range(len(self.W[l])):
                        Wr, Ar = self.W[l][r], accW[l][r]
                        for c in range(len(Wr)):
                            Wr[c] -= self.lr * Ar[c] / m
                        self.b[l][r] -= self.lr * accb[l][r] / m
            self.loss_history.append(cross_entropy(self.predict_proba(X), yi))
            if yv is not None:
                vl = cross_entropy(self.predict_proba(X_val), yv)
                self.val_loss_history.append(vl)
                if vl < best[0]:
                    best = (vl, epoch, ([[r[:] for r in W] for W in self.W],
                                        [b[:] for b in self.b]))
                elif patience is not None and epoch - best[1] >= patience:
                    self.stopped_epoch = epoch
                    break
        if yv is not None:
            self.best_epoch = best[1]
            self.best_weights = best[2]
            if patience is not None and best[2] is not None:
                self.W, self.b = best[2]
        return self

    def restore_best(self) -> "MLP":
        """Load the weights from the epoch with the lowest validation loss."""
        if getattr(self, "best_weights", None):
            self.W = [[r[:] for r in W] for W in self.best_weights[0]]
            self.b = [b[:] for b in self.best_weights[1]]
        return self

    def fine_tune(self, X: list, y: list, epochs: int | None = None,
                  freeze: bool = True) -> "MLP":
        """Transfer learning: keep the learned hidden layers, re-initialize only the
        output layer, and train it (and optionally the rest) on the new task."""
        rng = random.Random(self.seed + 7)
        fan_in, fan_out = len(self.W[-1][0]), len(sorted(set(y)))
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        self.W[-1] = [[rng.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]
        self.b[-1] = [0.0] * fan_out
        self.classes = sorted(set(y))
        if epochs is not None:
            self.epochs = epochs
        return self.fit(X, y, warm_start=True,
                        frozen_layers=len(self.W) - 1 if freeze else 0)

    def input_gradient(self, x: list, t: int) -> list:
        """d loss / d input — the direction an adversarial perturbation follows."""
        zs, acts = self._forward(x)
        delta = [p - (1.0 if k == t else 0.0) for k, p in enumerate(acts[-1])]
        for l in range(len(self.W) - 1, -1, -1):
            W = self.W[l]
            back = [fsum(W[k][j] * delta[k] for k in range(len(delta)))
                    for j in range(len(acts[l]))]
            if l == 0:
                return back
            delta = [back[j] * _act_grad(self.activation, zs[l - 1][j], acts[l][j])
                     for j in range(len(acts[l]))]
        return []

    def inactive_units(self, X: list) -> int:
        """First-layer hidden units whose input is <= 0 for every sample."""
        if len(self.W) < 2:
            return 0
        W, b = self.W[0], self.b[0]
        return sum(1 for row, bias in zip(W, b)
                   if all(fsum(w * v for w, v in zip(row, x)) + bias <= 0 for x in X))

    def dead_units(self, X: list) -> int:
        """Units that get exactly zero gradient on every sample, so they can never
        recover — the dying ReLU problem. Only plain ReLU has a zero slope for
        z <= 0; Leaky ReLU keeps a small slope, so its inactive units still learn."""
        return self.inactive_units(X) if self.activation == "relu" else 0

    # -- verification ------------------------------------------------------- #
    def gradient_check(self, x: list, t: int, eps: float = 1e-5,
                       samples: int = 20) -> float:
        """Max relative error between backprop and numerical gradients.

        Perturbs a sample of weights by +/-eps and compares the finite-difference
        slope of the loss with what _backward() computed. A value below ~1e-4
        means the backward pass is correct.
        """
        rng = random.Random(self.seed + 99)
        gW, _ = self._backward(x, t)

        def loss() -> float:
            p = self._forward(x)[1][-1]
            return -math.log(max(p[t], 1e-12))

        worst = 0.0
        for _ in range(samples):
            l = rng.randrange(len(self.W))
            r = rng.randrange(len(self.W[l]))
            c = rng.randrange(len(self.W[l][r]))
            orig = self.W[l][r][c]
            self.W[l][r][c] = orig + eps
            up = loss()
            self.W[l][r][c] = orig - eps
            down = loss()
            self.W[l][r][c] = orig
            numeric = (up - down) / (2 * eps)
            analytic = gW[l][r][c]
            denom = max(abs(numeric) + abs(analytic), 1e-12)
            worst = max(worst, abs(numeric - analytic) / denom)
        return worst

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"type": self.name, "hidden": list(self.hidden),
                "activation": self.activation, "lr": self.lr,
                "epochs": self.epochs, "batch_size": self.batch_size,
                "seed": self.seed, "W": self.W, "b": self.b,
                "classes": self.classes}

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        m = cls(hidden=tuple(d["hidden"]), activation=d["activation"], lr=d["lr"],
                epochs=d["epochs"], batch_size=d["batch_size"], seed=d["seed"])
        m.W, m.b, m.classes = d["W"], d["b"], d["classes"]
        return m
