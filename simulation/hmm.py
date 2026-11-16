"""Two-state Gaussian hidden Markov model of bar returns, used as a baseline regime filter."""
from __future__ import annotations

from math import exp, pi, sqrt

import numpy as np


def _pdf(x, mu, var):
    return exp(-0.5 * (x - mu) ** 2 / var) / sqrt(2.0 * pi * var)


def fit_hmm(returns: np.ndarray, iters: int = 30) -> dict:
    """Baum-Welch for two Gaussian states. Returns means, variances, transition matrix, start."""
    x = np.asarray(returns, dtype=float)
    n = len(x)
    s = x.std()
    mu = [x.mean(), x.mean()]
    var = [(0.6 * s) ** 2, (1.8 * s) ** 2]
    a = [[0.98, 0.02], [0.05, 0.95]]
    start = [0.7, 0.3]
    for _ in range(iters):
        b = np.column_stack([np.exp(-0.5 * (x - mu[k]) ** 2 / var[k]) / np.sqrt(2 * np.pi * var[k]) for k in (0, 1)])
        b = np.maximum(b, 1e-300)
        alpha = np.empty((n, 2)); scale = np.empty(n)
        a0, a1 = start[0] * b[0, 0], start[1] * b[0, 1]
        c = a0 + a1; alpha[0] = (a0 / c, a1 / c); scale[0] = c
        for t in range(1, n):
            p0, p1 = alpha[t - 1]
            a0 = (p0 * a[0][0] + p1 * a[1][0]) * b[t, 0]
            a1 = (p0 * a[0][1] + p1 * a[1][1]) * b[t, 1]
            c = a0 + a1
            alpha[t, 0] = a0 / c; alpha[t, 1] = a1 / c; scale[t] = c
        beta = np.empty((n, 2)); beta[-1] = (1.0, 1.0)
        for t in range(n - 2, -1, -1):
            n0, n1 = beta[t + 1, 0] * b[t + 1, 0], beta[t + 1, 1] * b[t + 1, 1]
            c = scale[t + 1]
            beta[t, 0] = (a[0][0] * n0 + a[0][1] * n1) / c
            beta[t, 1] = (a[1][0] * n0 + a[1][1] * n1) / c
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        nb = beta[1:] * b[1:] / scale[1:, None]
        xi = np.empty((2, 2))
        for i in (0, 1):
            for j in (0, 1):
                xi[i, j] = np.sum(alpha[:-1, i] * a[i][j] * nb[:, j])
        a = (xi / xi.sum(axis=1, keepdims=True)).tolist()
        start = gamma[0].tolist()
        for k in (0, 1):
            w = gamma[:, k]
            mu[k] = float(np.sum(w * x) / w.sum())
            var[k] = float(np.sum(w * (x - mu[k]) ** 2) / w.sum())
    return {"mu": mu, "var": var, "a": a, "start": start}


def filter_low_vol(returns: np.ndarray, model: dict) -> np.ndarray:
    """Causal (forward-filtered) probability of the low-variance state at each bar."""
    mu, var, a = model["mu"], model["var"], model["a"]
    low = 0 if var[0] <= var[1] else 1
    p0, p1 = model["start"]
    out = np.empty(len(returns))
    for t, x in enumerate(returns):
        q0 = (p0 * a[0][0] + p1 * a[1][0]) * _pdf(x, mu[0], var[0])
        q1 = (p0 * a[0][1] + p1 * a[1][1]) * _pdf(x, mu[1], var[1])
        c = q0 + q1
        if c <= 0.0:
            q0, q1, c = 0.5, 0.5, 1.0
        p0, p1 = q0 / c, q1 / c
        out[t] = p0 if low == 0 else p1
    return out
