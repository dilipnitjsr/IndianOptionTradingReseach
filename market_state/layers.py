"""Evidence layers: each converts indicators into a vote vector over the states.

A layer returns a DataFrame with one column per state in ``STATES``. Every row
satisfies the vote constraint of the paper (eq. 12): each vote lies in [0, 1]
and the votes of a row sum to at most 1. Mass that is not voted is abstention.

The thresholds below are the reference configuration of the paper. They are
placeholders until they are matched to the production engine.

Known limit of the reference configuration: only the volatility layer votes for
V, so the score of V can never exceed that layer's weight. With an acceptance
threshold above that weight the engine never declares V. See README.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["U", "D", "R", "V"]  # bullish trend, bearish trend, range-bound, high volatility


def ramp(x, a: float, b: float):
    """rho(x; a, b) = min(1, max(0, (x - a) / (b - a))), for a < b (eq. 13)."""
    if not a < b:
        raise ValueError("ramp needs a < b")
    return np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)


def _frame(index, u, d, r, v) -> pd.DataFrame:
    out = pd.DataFrame({"U": u, "D": d, "R": r, "V": v}, index=index)
    return out.fillna(0.0)  # an undefined indicator abstains


def trend_layer(ema_fast: pd.Series, ema_slow: pd.Series, atr: pd.Series,
                a: float = 0.2, b: float = 1.0) -> pd.DataFrame:
    """Layer 1: z = (EMA_fast - EMA_slow) / ATR. Needs a >= 0."""
    if a < 0:
        raise ValueError("trend layer needs a >= 0")
    z = (ema_fast - ema_slow) / atr.where(atr > 0)
    u, d = ramp(z, a, b), ramp(-z, a, b)
    return _frame(z.index, u, d, np.where(z.notna(), 1.0 - u - d, 0.0), 0.0)


def momentum_layer(rsi: pd.Series, a: float = 55.0, b: float = 70.0) -> pd.DataFrame:
    """Layer 2: RSI. Needs a >= 50."""
    if a < 50:
        raise ValueError("momentum layer needs a >= 50")
    u, d = ramp(rsi, a, b), ramp(100.0 - rsi, a, b)
    return _frame(rsi.index, u, d, np.where(rsi.notna(), 1.0 - u - d, 0.0), 0.0)


def strength_layer(adx: pd.Series, di_plus: pd.Series, di_minus: pd.Series,
                   a: float = 18.0, b: float = 30.0) -> pd.DataFrame:
    """Layer 3: g = rho(ADX); direction from the sign of DI+ - DI-."""
    g = ramp(adx, a, b)
    u = np.where(di_plus > di_minus, g, 0.0)
    d = np.where(di_plus < di_minus, g, 0.0)
    return _frame(adx.index, u, d, np.where(adx.notna(), 1.0 - g, 0.0), 0.0)


def volatility_layer(vol_rank: pd.Series, a: float = 0.75, b: float = 0.95) -> pd.DataFrame:
    """Layer 4: percentile rank of ATR/C. Votes only for V."""
    return _frame(vol_rank.index, 0.0, 0.0, 0.0, ramp(vol_rank, a, b))


def derivatives_layer(pcr: pd.Series, a_bull: float = 1.05, b_bull: float = 1.25,
                      a_bear: float = 0.75, b_bear: float = 0.95) -> pd.DataFrame:
    """Layer 5: put-call ratio of open interest; a high ratio is read as bullish.

    Needs b_bear <= a_bull so the layer never votes for U and D at once.
    """
    if b_bear > a_bull:
        raise ValueError("derivatives layer needs b_bear <= a_bull")
    u = ramp(pcr, a_bull, b_bull)
    d = np.where(pcr.notna(), 1.0 - ramp(pcr, a_bear, b_bear), 0.0)
    return _frame(pcr.index, u, d, 0.0, 0.0)
