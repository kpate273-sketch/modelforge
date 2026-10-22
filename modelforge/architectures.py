"""The core computation of each modern architecture, from scratch
(Module 5: Architectures for Vision and Sequences).

Training a CNN, an LSTM or a Transformer properly needs a GPU and a real
framework, which this project deliberately does without. So instead of claiming
to train them, this module implements the one operation that defines each
architecture and runs it on a small example you can read by hand:

  conv2d             what a CNN layer computes — a small kernel slid over an
                     image. An edge-detection kernel lights up exactly where the
                     image changes.
  lstm_step          one step of an LSTM cell — the forget/input/output gates
                     that decide what memory to keep.
  self_attention     scaled dot-product attention — every token scores every
                     other token, softmaxes the scores, and mixes their values.
  vanishing_gradient multiply the per-layer derivative through N layers and
                     watch it shrink toward zero for sigmoid but not for ReLU —
                     the reason plain RNNs forget and LSTMs/ReLUs were invented.

Added to cover the rest of Module 5:

  conv_param_counts / rotate90   weight sharing, translation vs rotation
  rnn_forward / bptt_gradient    a plain RNN, and the gradient that has to
                                 travel back through every time step
  clip_by_norm                   gradient clipping for exploding gradients
  gru_step / count_gate_params   the leaner two-gate GRU vs the LSTM
  lstm_cell_gradient             why the cell state is a conveyor belt for
                                 gradients (the forget-gate path)
  positional_encoding            why attention alone cannot tell word order
  attention_cost                 the quadratic cost, and local attention
"""

from __future__ import annotations

from math import fsum

import math
import random

from .linalg import dot, softmax, sigmoid


# --------------------------------------------------------------------------- #
# CNNs: convolution
# --------------------------------------------------------------------------- #
def conv2d(image: list, kernel: list, stride: int = 1) -> list:
    """Valid (no padding) 2-D convolution of `image` by `kernel`."""
    kh, kw = len(kernel), len(kernel[0])
    ih, iw = len(image), len(image[0])
    out = []
    for r in range(0, ih - kh + 1, stride):
        row = []
        for c in range(0, iw - kw + 1, stride):
            s = 0.0
            for i in range(kh):
                for j in range(kw):
                    s += image[r + i][c + j] * kernel[i][j]
            row.append(s)
        out.append(row)
    return out


SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]   # responds to vertical edges


def demo_image(size: int = 8) -> list:
    """A tiny image: dark on the left half, bright on the right — one vertical edge."""
    return [[0.0 if c < size // 2 else 1.0 for c in range(size)] for _ in range(size)]


def max_pool(fmap: list, size: int = 2) -> list:
    """2x2 max pooling: keep the strongest response in each window."""
    return [[max(fmap[r + i][c + j] for i in range(size) for j in range(size))
             for c in range(0, len(fmap[0]) - size + 1, size)]
            for r in range(0, len(fmap) - size + 1, size)]


# --------------------------------------------------------------------------- #
# RNNs / LSTMs: one gated memory step
# --------------------------------------------------------------------------- #
def lstm_step(x: list, h_prev: list, c_prev: list, params: dict) -> tuple:
    """One LSTM time step. params holds W_* (on [x; h]) and b_* for f, i, o, g."""
    xh = list(x) + list(h_prev)

    def gate(name, fn):
        return [fn(dot(w, xh) + b) for w, b in zip(params["W_" + name], params["b_" + name])]

    f = gate("f", sigmoid)            # forget gate: how much old memory to keep
    i = gate("i", sigmoid)            # input gate: how much new info to write
    o = gate("o", sigmoid)            # output gate: how much memory to expose
    g = gate("g", math.tanh)          # candidate memory
    c = [fk * ck + ik * gk for fk, ck, ik, gk in zip(f, c_prev, i, g)]
    h = [ok * math.tanh(ck) for ok, ck in zip(o, c)]
    return h, c, {"forget": f, "input": i, "output": o}


def tiny_lstm_params(n_in: int = 2, n_hidden: int = 3, forget_bias: float = 1.0) -> dict:
    """Deterministic small weights; a positive forget bias is the standard trick
    that makes an untrained LSTM remember by default."""
    p = {}
    for k, gate in enumerate("fiog"):
        p["W_" + gate] = [[0.1 * ((r + c + k) % 5 - 2) for c in range(n_in + n_hidden)]
                          for r in range(n_hidden)]
        p["b_" + gate] = [forget_bias if gate == "f" else 0.0] * n_hidden
    return p


# --------------------------------------------------------------------------- #
# Transformers: scaled dot-product self-attention
# --------------------------------------------------------------------------- #
def self_attention(Q: list, K: list, V: list) -> tuple:
    """Return (outputs, attention_weights) for scaled dot-product attention."""
    d_k = len(K[0])
    scale = 1.0 / math.sqrt(d_k)
    weights, outputs = [], []
    for q in Q:
        scores = [dot(q, k) * scale for k in K]
        a = softmax(scores)
        weights.append(a)
        outputs.append([fsum(a[t] * V[t][j] for t in range(len(V)))
                        for j in range(len(V[0]))])
    return outputs, weights


def demo_tokens() -> tuple:
    """Four toy 'word' embeddings. Tokens 0 and 2 are similar, so they should
    attend to each other more than to the rest."""
    tokens = ["the", "cat", "a", "sat"]
    E = [[1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.9, 0.1, 1.0, 0.0],
         [0.0, 0.8, 0.2, 1.0]]
    return tokens, E


# --------------------------------------------------------------------------- #
# The vanishing gradient problem
# --------------------------------------------------------------------------- #
def vanishing_gradient(depth: int = 20, activation: str = "sigmoid",
                       pre_activation: float = 0.5) -> list:
    """Gradient magnitude reaching each earlier layer, assuming unit weights.

    Backprop multiplies one derivative per layer. sigmoid'(z) is at most 0.25,
    so the product shrinks geometrically; ReLU's derivative is 1 for positive
    inputs, so the signal survives.
    """
    if activation == "sigmoid":
        s = sigmoid(pre_activation)
        d = s * (1 - s)
    elif activation == "tanh":
        d = 1 - math.tanh(pre_activation) ** 2
    elif activation == "relu":
        d = 1.0 if pre_activation > 0 else 0.0
    else:
        raise ValueError(activation)
    grads, g = [], 1.0
    for _ in range(depth):
        g *= d
        grads.append(g)
    return grads


# --------------------------------------------------------------------------- #
# CNNs: weight sharing, translation, rotation
# --------------------------------------------------------------------------- #
COURSE_EDGE_KERNEL = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]   # positive left, negative right
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]               # responds to horizontal edges


def conv_param_counts(height: int = 224, width: int = 224, channels: int = 3,
                      neurons: int = 64, kernel: int = 3) -> dict:
    """Weights needed by one fully connected layer vs one convolutional layer with
    the same number of outputs per position (filters)."""
    inputs = height * width * channels
    return {"inputs": inputs,
            "dense_weights": inputs * neurons,
            "conv_weights": kernel * kernel * channels * neurons}


def edge_image(size: int = 8, edge_col: int = 4) -> list:
    """Bright on the left, dark from edge_col on — the lesson's vertical edge, which
    COURSE_EDGE_KERNEL (positive left, negative right) answers with a large sum."""
    return [[1.0 if c < edge_col else 0.0 for c in range(size)] for _ in range(size)]


def rotate90(image: list) -> list:
    """Rotate an image 90 degrees clockwise (a vertical edge becomes horizontal)."""
    return [list(row) for row in zip(*image[::-1])]


def global_max(fmap: list) -> float:
    return max(abs(v) for row in fmap for v in row)


def strongest_column(fmap: list) -> int:
    col_strength = [fsum(abs(row[c]) for row in fmap) for c in range(len(fmap[0]))]
    return max(range(len(col_strength)), key=lambda c: col_strength[c])


# --------------------------------------------------------------------------- #
# RNNs: backpropagation through time, exploding gradients, clipping
# --------------------------------------------------------------------------- #
def rnn_forward(xs: list, w_h: float, w_x: float, h0: float = 0.0) -> list:
    """A one-unit tanh RNN: h_t = tanh(w_h * h_{t-1} + w_x * x_t). The same two
    weights are reused at every step — weight sharing across time."""
    hs, h = [], h0
    for x in xs:
        h = math.tanh(w_h * h + w_x * x)
        hs.append(h)
    return hs


def bptt_gradient(factor: float, steps: int = 50) -> list:
    """How strongly the last step's error reaches each earlier step when every
    step multiplies the gradient by the same recurrent factor (the eigenvalue the
    lesson talks about). Entry t is the gradient after travelling t steps back."""
    out, g = [], 1.0
    for _ in range(steps):
        g *= factor
        out.append(g)
    return out


def tanh_rnn_gradient(w_h: float = 0.9, steps: int = 50, x_scale: float = 0.5) -> list:
    """BPTT on a real tanh RNN: d h_T / d h_t = product of w_h * tanh'(.) along the
    chain. Inputs are a fixed alternating signal, so the number is reproducible."""
    xs = [x_scale if t % 2 == 0 else -x_scale for t in range(steps)]
    hs = rnn_forward(xs, w_h, 1.0)
    out, g = [], 1.0
    for t in range(steps - 1, -1, -1):
        g *= w_h * (1.0 - hs[t] ** 2)
        out.append(g)
    return out


def clip_by_norm(grad: list, max_norm: float) -> list:
    """If the gradient's length exceeds max_norm, scale it down to max_norm.
    The direction is unchanged — only the size of the step is capped."""
    n = math.sqrt(fsum(g * g for g in grad))
    if n <= max_norm or n == 0.0:
        return list(grad)
    return [g * max_norm / n for g in grad]


# --------------------------------------------------------------------------- #
# GRUs and the LSTM gradient path
# --------------------------------------------------------------------------- #
def gru_step(x: list, h_prev: list, params: dict) -> tuple:
    """One GRU step: an update gate z and a reset gate r, no separate cell state."""
    xh = list(x) + list(h_prev)
    z = [sigmoid(dot(w, xh) + b) for w, b in zip(params["W_z"], params["b_z"])]
    r = [sigmoid(dot(w, xh) + b) for w, b in zip(params["W_r"], params["b_r"])]
    xrh = list(x) + [ri * hi for ri, hi in zip(r, h_prev)]
    cand = [math.tanh(dot(w, xrh) + b) for w, b in zip(params["W_h"], params["b_h"])]
    h = [(1 - zi) * hp + zi * ci for zi, hp, ci in zip(z, h_prev, cand)]
    return h, {"update": z, "reset": r}


def tiny_gru_params(n_in: int = 2, n_hidden: int = 3) -> dict:
    p = {}
    for k, gate in enumerate("zrh"):
        p["W_" + gate] = [[0.1 * ((r + c + k) % 5 - 2) for c in range(n_in + n_hidden)]
                          for r in range(n_hidden)]
        p["b_" + gate] = [0.0] * n_hidden
    return p


def count_gate_params(n_in: int, n_hidden: int) -> dict:
    """Each gate is a layer on [x; h]: n_hidden * (n_in + n_hidden) weights + n_hidden biases.
    LSTM has 4 such blocks (forget, input, output, candidate); GRU has 3."""
    block = n_hidden * (n_in + n_hidden) + n_hidden
    return {"lstm": 4 * block, "gru": 3 * block}


def lstm_cell_gradient(forget_bias: float, steps: int = 50) -> list:
    """Gradient along the LSTM cell state: d c_T / d c_t is the product of the
    forget-gate values in between (the cell is only scaled by f, never pushed
    through a squashing weight matrix). Uses the real lstm_step gate values."""
    params = tiny_lstm_params(forget_bias=forget_bias)
    h, c = [0.0] * 3, [0.0] * 3
    forget = []
    for t in range(steps):
        x = [1.0, 0.0] if t == 0 else [0.0, 0.0]
        h, c, gates = lstm_step(x, h, c, params)
        forget.append(gates["forget"][0])
    out, g = [], 1.0
    for f in reversed(forget):
        g *= f
        out.append(g)
    return out


# --------------------------------------------------------------------------- #
# Transformers: positional encoding and the quadratic cost
# --------------------------------------------------------------------------- #
def positional_encoding(n_positions: int, d_model: int) -> list:
    """Sinusoidal positional encoding from "Attention Is All You Need"."""
    pe = []
    for pos in range(n_positions):
        row = []
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        pe.append(row)
    return pe


def word_vectors(words: list, d: int = 8, seed: int = 0) -> dict:
    """A fixed pseudo-random embedding per distinct word (deterministic)."""
    out = {}
    for w in sorted(set(words)):
        rng = random.Random(f"{seed}:{w}")
        out[w] = [rng.uniform(-1, 1) for _ in range(d)]
    return out


def attention_cost(n: int, window: int | None = None) -> int:
    """Number of query-key scores computed. Full attention: n^2. Local attention
    (each position sees `window` neighbours on each side): about n * (2*window + 1)."""
    if window is None:
        return n * n
    return sum(min(n, i + window + 1) - max(0, i - window) for i in range(n))
