"""Synthetic NIFTY-like market with known regimes.

A four-state Markov chain (U bullish trend, D bearish trend, R range-bound,
V high volatility) drives the drift and volatility of a one-minute log-price
process. Minutes are aggregated to bars. A put-call ratio series is generated
as a noisy AR(1) process around a regime-dependent mean.

Every parameter is a stylised assumption, not an estimate from market data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

REGIMES = ["U", "D", "R", "V"]


@dataclass(frozen=True)
class MarketParams:
    s0: float = 25000.0
    bars_per_day: int = 25              # 15-minute bars, 09:15 to 15:30
    minutes_per_bar: int = 15
    days_per_year: int = 252
    # per regime, in the order U, D, R, V
    drift_daily: tuple = (0.008, -0.009, 0.0, 0.0)       # expected log return per day
    vol_annual: tuple = (0.11, 0.14, 0.09, 0.26)
    mean_duration_days: tuple = (1.5, 1.2, 3.0, 1.0)
    # where the chain goes when it leaves a regime (rows U, D, R, V)
    exit_probs: tuple = (
        (0.0, 0.2, 0.6, 0.2),
        (0.2, 0.0, 0.5, 0.3),
        (0.4, 0.35, 0.0, 0.25),
        (0.2, 0.3, 0.5, 0.0),
    )
    range_half_life_days: float = 1.0   # mean reversion of the range-bound regime
    pcr_mean: tuple = (1.25, 0.75, 1.00, 0.90)
    pcr_phi: float = 0.95
    pcr_sigma: float = 0.05

    @property
    def minutes_per_year(self) -> int:
        return self.days_per_year * self.bars_per_day * self.minutes_per_bar

    @property
    def bars_per_year(self) -> int:
        return self.days_per_year * self.bars_per_day

    def transition_matrix(self) -> np.ndarray:
        p = np.zeros((4, 4))
        for i in range(4):
            stay = 1.0 - 1.0 / (self.mean_duration_days[i] * self.bars_per_day)
            p[i] = (1.0 - stay) * np.array(self.exit_probs[i])
            p[i, i] = stay
        return p

    def stationary(self) -> np.ndarray:
        vals, vecs = np.linalg.eig(self.transition_matrix().T)
        v = np.real(vecs[:, np.argmax(np.real(vals))])
        return v / v.sum()


def generate_path(seed: int, days: int, params: MarketParams = MarketParams()) -> pd.DataFrame:
    """One simulated path. Returns one row per bar with columns
    open, high, low, close, fill (price one minute into the bar), regime, pcr."""
    rng = np.random.default_rng(seed)
    p = params
    n_bars = days * p.bars_per_day
    m = p.minutes_per_bar
    trans = p.transition_matrix()
    cum = np.cumsum(trans, axis=1)

    regime = np.empty(n_bars, dtype=np.int8)
    state = rng.choice(4, p=p.stationary())
    u = rng.random(n_bars)
    for t in range(n_bars):
        regime[t] = state
        state = int(np.searchsorted(cum[state], u[t]))
        state = min(state, 3)

    dt = 1.0 / p.minutes_per_year
    eps = rng.standard_normal(n_bars * m)
    logp = np.empty(n_bars * m)
    level = np.log(p.s0)
    # process each run of constant regime
    change = np.flatnonzero(np.diff(regime)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n_bars]])
    kappa = np.log(2.0) / (p.range_half_life_days / p.days_per_year)
    for b0, b1 in zip(starts, ends):
        r = regime[b0]
        sl = slice(b0 * m, b1 * m)
        sigma = p.vol_annual[r]
        if REGIMES[r] == "R":
            a = np.exp(-kappa * dt)
            s = sigma * np.sqrt((1.0 - a * a) / (2.0 * kappa))
            x = lfilter([s], [1.0, -a], eps[sl])
            logp[sl] = level + x
        else:
            mu = p.drift_daily[r] * p.days_per_year
            inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps[sl]
            logp[sl] = level + np.cumsum(inc)
        level = logp[b1 * m - 1]

    px = np.exp(logp).reshape(n_bars, m)
    close = px[:, -1]
    open_ = np.concatenate([[p.s0], close[:-1]])
    high = np.maximum(px.max(axis=1), open_)
    low = np.minimum(px.min(axis=1), open_)

    pcr = np.empty(n_bars)
    means = np.array(p.pcr_mean)[regime]
    shocks = rng.standard_normal(n_bars) * p.pcr_sigma
    level_pcr = means[0]
    for t in range(n_bars):
        level_pcr = p.pcr_phi * level_pcr + (1.0 - p.pcr_phi) * means[t] + shocks[t]
        pcr[t] = level_pcr
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "fill": px[:, 0],
         "regime": regime, "pcr": np.maximum(pcr, 0.05)}
    )
