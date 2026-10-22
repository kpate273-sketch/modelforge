"""Vectors and matrices from scratch (Module 2: Vectors, Matrices, and What Models Compute).

Every model in this project reduces to a handful of operations on lists of
numbers: dot products, matrix-vector products, elementwise maps. Writing them by
hand, instead of importing numpy, is the point of a fundamentals course — it
makes it visible that a "model" is arithmetic on vectors and nothing more.
"""

from __future__ import annotations

from math import fsum

import math

Vector = list
Matrix = list


def dot(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(f"dot: length mismatch {len(a)} vs {len(b)}")
    return fsum(x * y for x, y in zip(a, b))


def matvec(m: Matrix, v: Vector) -> Vector:
    """Matrix (rows x cols) times vector (cols) -> vector (rows)."""
    return [dot(row, v) for row in m]


def transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def scale(v: Vector, k: float) -> Vector:
    return [x * k for x in v]


def zeros(n: int) -> Vector:
    return [0.0] * n


def zeros_matrix(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def norm(v: Vector) -> float:
    return math.sqrt(dot(v, v))


def euclidean(a: Vector, b: Vector) -> float:
    return math.sqrt(fsum((x - y) ** 2 for x, y in zip(a, b)))


def argmax(v: Vector) -> int:
    best, idx = v[0], 0
    for i, x in enumerate(v):
        if x > best:
            best, idx = x, i
    return idx


def softmax(z: Vector) -> Vector:
    """Numerically stable softmax: subtract the max before exponentiating."""
    m = max(z)
    exps = [math.exp(x - m) for x in z]
    s = fsum(exps)
    return [e / s for e in exps]


def sigmoid(x: float) -> float:
    # Stable for large negative inputs.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def solve(A: Matrix, b: Vector) -> Vector:
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Used to compute exact least-squares answers (the normal equations), so the
    gradient-descent results can be checked against the true optimum.
    """
    n = len(A)
    M = [list(map(float, row)) + [float(v)] for row, v in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ValueError("matrix is singular")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - fsum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x
