"""Market state engine: weighted evidence voting with threshold, margin and persistence.

Implements Section IV of the paper:
  A_t(s) = sum_l w_l * v_t^l(s)                         (eq. 15)
  m_t    = A_t(s1) - A_t(s2)                            (eq. 16)
  c_t    = s1 if A_t(s1) >= theta and m_t >= delta, else undetermined   (eq. 17)
  S_t    = c_t if c_t has held for h bars and differs from S_{t-1}, else S_{t-1}   (eq. 18)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .layers import STATES

UNDETERMINED = "s0"


@dataclass(frozen=True)
class EngineConfig:
    """Engine parameters. ``weights`` maps layer name to weight; weights must sum to 1."""

    weights: Mapping[str, float]
    theta: float = 0.55
    delta: float = 0.05
    h: int = 3
    states: tuple = field(default_factory=lambda: tuple(STATES))

    def __post_init__(self):
        w = np.array(list(self.weights.values()), dtype=float)
        if (w < 0).any() or not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must be non-negative and sum to 1")
        if not 0 < self.theta <= 1:
            raise ValueError("theta must lie in (0, 1]")
        if self.delta < 0:
            raise ValueError("delta must be >= 0")
        if self.h < 1:
            raise ValueError("h must be >= 1")


def aggregate(votes: Mapping[str, pd.DataFrame], weights: Mapping[str, float], states=STATES) -> pd.DataFrame:
    """Aggregate evidence scores A_t(s) (eq. 15)."""
    names = list(weights)
    index = votes[names[0]].index
    total = np.zeros((len(index), len(states)))
    for name in names:
        v = votes[name].reindex(columns=list(states)).to_numpy(dtype=float)
        if (v < -1e-12).any() or (v > 1 + 1e-12).any() or (v.sum(axis=1) > 1 + 1e-9).any():
            raise ValueError(f"layer '{name}' violates the vote constraint")
        total += weights[name] * v
    return pd.DataFrame(total, index=index, columns=list(states))


def candidate(scores: pd.DataFrame, theta: float, delta: float):
    """Candidate state c_t and margin m_t (eqs. 16 and 17)."""
    a = scores.to_numpy()
    order = np.argsort(-a, axis=1, kind="stable")
    rows = np.arange(len(a))
    top = a[rows, order[:, 0]]
    second = a[rows, order[:, 1]] if a.shape[1] > 1 else np.zeros(len(a))
    margin = top - second
    ok = (top >= theta) & (margin >= delta)
    if delta == 0:                       # a tie has no unique leader
        ok &= margin > 0
    names = np.array(list(scores.columns), dtype=object)
    cand = np.where(ok, names[order[:, 0]], UNDETERMINED)
    return pd.Series(cand, index=scores.index), pd.Series(margin, index=scores.index)


def persist(cand: pd.Series, h: int) -> pd.Series:
    """Persistence filter (eq. 18). The declared state starts as undetermined."""
    values = cand.to_numpy()
    out = np.empty(len(values), dtype=object)
    current = UNDETERMINED
    run_value, run_len = None, 0
    for t, c in enumerate(values):
        run_len = run_len + 1 if c == run_value else 1
        run_value = c
        if run_len >= h and c != current:
            current = c
        out[t] = current
    return pd.Series(out, index=cand.index)


def switch_count(declared: pd.Series) -> int:
    """Number of bars at which the declared state changes (N_T of Proposition 5)."""
    values = np.concatenate([[UNDETERMINED], declared.to_numpy()])
    return int((values[1:] != values[:-1]).sum())


def run_engine(votes: Mapping[str, pd.DataFrame], config: EngineConfig) -> pd.DataFrame:
    """Run the engine. Returns the scores, margin, candidate and declared state per bar."""
    scores = aggregate(votes, config.weights, config.states)
    cand, margin = candidate(scores, config.theta, config.delta)
    out = scores.copy()
    out["abstention"] = 1.0 - scores.sum(axis=1)
    out["margin"] = margin
    out["candidate"] = cand
    out["state"] = persist(cand, config.h)
    return out
