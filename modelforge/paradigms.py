"""The other two learning paradigms (Module 1: The Three Learning Paradigms).

Everything else in ModelForge is supervised learning — every example comes with
its label. The course names two more paradigms, and each is defined by a
different feedback signal. This module implements one small, honest example of
each, from scratch:

  KMeans      unsupervised: no labels at all. It groups points by finding k
              centres and assigning every point to its nearest one. The true
              labels are used only afterwards, to score how well the discovered
              groups line up with reality (purity).

  GridWorld + q_learning
              reinforcement: an agent acts in an environment and only gets a
              reward when it reaches the goal — a delayed reward, like winning
              a game of chess. Q-learning discovers, by trial and error, which
              action is worth taking in each square. Nobody ever tells it the
              right move.
"""

from __future__ import annotations

from math import fsum

import random
from collections import Counter

from .linalg import euclidean


# --------------------------------------------------------------------------- #
# Unsupervised: k-means clustering
# --------------------------------------------------------------------------- #
class KMeans:
    name = "kmeans"

    def __init__(self, k: int = 3, max_iter: int = 100, seed: int = 0):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centers: list = []
        self.iterations = 0
        self.inertia = 0.0

    def _init_centers(self, X: list) -> list:
        """k-means++ seeding: spread the starting centres out, deterministically."""
        rng = random.Random(self.seed)
        centers = [list(X[rng.randrange(len(X))])]
        while len(centers) < self.k:
            d2 = [min(euclidean(x, c) ** 2 for c in centers) for x in X]
            total = fsum(d2)
            r, acc = rng.random() * total, 0.0
            for x, w in zip(X, d2):
                acc += w
                if acc >= r:
                    centers.append(list(x))
                    break
            else:
                centers.append(list(X[-1]))
        return centers

    def _assign(self, X: list) -> list:
        return [min(range(self.k), key=lambda j: euclidean(x, self.centers[j])) for x in X]

    def fit(self, X: list) -> "KMeans":
        self.centers = self._init_centers(X)
        labels = None
        for it in range(self.max_iter):
            new = self._assign(X)
            self.iterations = it + 1
            if new == labels:
                break
            labels = new
            for j in range(self.k):
                members = [x for x, l in zip(X, labels) if l == j]
                if members:
                    self.centers[j] = [fsum(col) / len(members) for col in zip(*members)]
        self.labels_ = labels
        self.inertia = fsum(euclidean(x, self.centers[l]) ** 2 for x, l in zip(X, labels))
        return self

    def predict(self, X: list) -> list:
        return self._assign(X)


def purity(clusters: list, y_true: list) -> float:
    """Fraction of points whose cluster's majority label matches their own label."""
    by: dict = {}
    for c, t in zip(clusters, y_true):
        by.setdefault(c, []).append(t)
    return sum(Counter(v).most_common(1)[0][1] for v in by.values()) / max(len(y_true), 1)


# --------------------------------------------------------------------------- #
# Reinforcement: Q-learning in a small grid world
# --------------------------------------------------------------------------- #
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">"}


class GridWorld:
    """A size x size grid. Start top-left, goal bottom-right, a few walls.
    Reward +1 only on reaching the goal; every other step costs 0.01."""

    def __init__(self, size: int = 5, walls: tuple = ((1, 1), (1, 2), (3, 2), (3, 3))):
        self.size = size
        self.walls = set(walls)
        self.start, self.goal = (0, 0), (size - 1, size - 1)

    def step(self, state: tuple, action: str) -> tuple:
        dr, dc = ACTIONS[action]
        r, c = state[0] + dr, state[1] + dc
        if not (0 <= r < self.size and 0 <= c < self.size) or (r, c) in self.walls:
            r, c = state                      # bumped into an edge or a wall
        if (r, c) == self.goal:
            return (r, c), 1.0, True
        return (r, c), -0.01, False

    def states(self) -> list:
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.walls]


def q_learning(env: GridWorld, episodes: int = 300, alpha: float = 0.5,
               gamma: float = 0.95, epsilon: float = 0.2, max_steps: int = 100,
               seed: int = 0) -> dict:
    """Tabular Q-learning with an epsilon-greedy policy.

    Q[s][a] estimates the total future reward of taking action a in state s.
    After each step:  Q[s][a] += alpha * (reward + gamma * max_a' Q[s'][a'] - Q[s][a])
    The reward only arrives at the goal; gamma passes its value back, step by
    step, to the squares that led there.
    """
    rng = random.Random(seed)
    names = list(ACTIONS)
    Q = {s: {a: 0.0 for a in names} for s in env.states()}
    steps_per_episode = []

    def greedy(s):
        best = max(Q[s].values())
        return next(a for a in names if Q[s][a] == best)

    for _ in range(episodes):
        s, steps = env.start, 0
        for steps in range(1, max_steps + 1):
            a = rng.choice(names) if rng.random() < epsilon else greedy(s)
            s2, r, done = env.step(s, a)
            target = r if done else r + gamma * max(Q[s2].values())
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
            if done:
                break
        steps_per_episode.append(steps)

    # Follow the learned policy greedily from the start.
    path, s = [env.start], env.start
    for _ in range(env.size * env.size):
        s, _, done = env.step(s, greedy(s))
        path.append(s)
        if done:
            break
    policy = {s: greedy(s) for s in env.states() if s != env.goal}
    return {"Q": Q, "steps_per_episode": steps_per_episode, "greedy_path": path,
            "reached_goal": path[-1] == env.goal, "policy": policy}


def render_policy(env: GridWorld, policy: dict) -> list:
    rows = []
    for r in range(env.size):
        row = []
        for c in range(env.size):
            if (r, c) in env.walls:
                row.append("#")
            elif (r, c) == env.goal:
                row.append("G")
            else:
                row.append(ARROWS[policy[(r, c)]])
        rows.append(" ".join(row))
    return rows
