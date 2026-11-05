"""Tiny MLP with explicit forward/backward passes and an Adam optimiser."""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator, out: str = "linear", slope: float = 0.2):
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.out, self.slope = out, slope
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self) -> list[np.ndarray]:
        return [p for pair in zip(self.W, self.b) for p in pair]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.acts, self.pre = [x], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.acts[-1] @ W + b
            self.pre.append(z)
            last = i == len(self.W) - 1
            if last and self.out == "sigmoid":
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            elif last:
                a = z
            else:
                a = np.where(z > 0, z, self.slope * z)
            self.acts.append(a)
        return self.acts[-1]

    def backward(self, grad_out: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """grad_out is dLoss/d(pre-activation of the last layer) for sigmoid heads, else dLoss/dOutput."""
        grads = [None] * len(self.params())
        g = grad_out
        for i in reversed(range(len(self.W))):
            grads[2 * i] = self.acts[i].T @ g
            grads[2 * i + 1] = g.sum(0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * np.where(self.pre[i - 1] > 0, 1.0, self.slope)
        return grads, g  # g = gradient wrt the network input

    def step(self, grads: list[np.ndarray], lr: float, b1: float = 0.5, b2: float = 0.999) -> None:
        self.t += 1
        for k, (p, g) in enumerate(zip(self.params(), grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
