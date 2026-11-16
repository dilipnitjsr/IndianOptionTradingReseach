"""Technical indicators used as evidence by the market state engine.

All functions are causal: the value at bar t uses data up to and including t.
Inputs are pandas Series indexed by bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, E_t = a*C_t + (1-a)*E_{t-1}, a = 2/(span+1)."""
    return close.ewm(span=span, adjust=False).mean()


def wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing, X_t = ((n-1)*X_{t-1} + x_t) / n."""
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's average true range."""
    return wilder(true_range(high, low, close), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative strength index, 100 * G / (G + B) with Wilder averages of gains and losses."""
    diff = close.diff()
    gain = wilder(diff.clip(lower=0.0).fillna(0.0), n)
    loss = wilder((-diff).clip(lower=0.0).fillna(0.0), n)
    total = gain + loss
    out = 100.0 * gain / total.where(total > 0)
    return out.fillna(50.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Average directional index with the directional indicators.

    Returns a DataFrame with columns ``adx``, ``di_plus``, ``di_minus``.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).fillna(0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0).fillna(0.0)
    tr_s = wilder(true_range(high, low, close), n)
    di_plus = 100.0 * wilder(plus_dm, n) / tr_s.where(tr_s > 0)
    di_minus = 100.0 * wilder(minus_dm, n) / tr_s.where(tr_s > 0)
    di_sum = di_plus + di_minus
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum.where(di_sum > 0)
    return pd.DataFrame(
        {
            "adx": wilder(dx.fillna(0.0), n),
            "di_plus": di_plus.fillna(0.0),
            "di_minus": di_minus.fillna(0.0),
        }
    )


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Share of the previous ``window`` values that are <= the current value."""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(window, len(values)):
        out[t] = np.mean(values[t - window : t] <= values[t])
    return pd.Series(out, index=series.index)


def put_call_ratio(put_oi: pd.Series, call_oi: pd.Series) -> pd.Series:
    """Put-call ratio of open interest (each input already summed over strikes)."""
    return put_oi / call_oi.where(call_oi > 0)
