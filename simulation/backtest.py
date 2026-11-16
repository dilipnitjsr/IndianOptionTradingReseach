"""Bar-by-bar backtest of a long-option strategy driven by a desired-position series."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np

from .options import CostParams, bs_price, round_trip_cost, select_strike


@dataclass(frozen=True)
class PolicyParams:
    equity0: float = 1_000_000.0
    risk_fraction: float = 0.01      # f: share of equity risked per trade
    stop: float = 0.30               # lambda: stop-loss as a share of premium
    target: float = 0.60             # gamma: profit target as a share of premium
    target_delta: float = 0.5
    lot_size: int = 65
    strike_step: float = 50.0
    rate: float = 0.065
    expiry_every_days: int = 5       # weekly expiry
    min_days_to_expiry: int = 1
    last_entry_bar: int = 19         # no new entries after this bar of the day
    squareoff_bar: int = 23          # exit decision at the close of this bar


def run_backtest(close, fill, iv, desired, label, bars_per_day, bars_per_year, start_bar,
                 policy: PolicyParams = PolicyParams(), costs: CostParams = CostParams()):
    """Simulate the strategy.

    desired[t] is +1 (hold a call), -1 (hold a put) or 0 (flat), known at the close of bar t.
    Orders decided at the close of bar t are filled one minute into bar t+1.
    label[t] is recorded with each trade (the declared state at the entry decision).

    Returns (daily_equity, trades) where trades is a list of dicts.
    """
    n = len(close)
    bar_years = 1.0 / bars_per_year
    minute_years = bar_years / 15.0
    equity = policy.equity0
    daily = []
    trades = []
    pos = None
    for t in range(start_bar, n - 1):
        day, bib = divmod(t, bars_per_day)
        if pos is not None:
            direction, strike, exp_bar, p_entry, lots, t_in, lab = pos
            tau = (exp_bar - t) * bar_years
            premium = bs_price(close[t], strike, tau, iv[t], policy.rate, direction)
            reason = None
            if premium <= (1.0 - policy.stop) * p_entry:
                reason = "stop"
            elif premium >= (1.0 + policy.target) * p_entry:
                reason = "target"
            elif desired[t] != direction:
                reason = "state"
            elif bib >= policy.squareoff_bar:
                reason = "squareoff"
            if reason is not None:
                p_exit = bs_price(fill[t + 1], strike, max(tau - minute_years, 1e-9), iv[t], policy.rate, direction)
                units = lots * policy.lot_size
                buy_value, sell_value = units * p_entry, units * p_exit
                cost = round_trip_cost(buy_value, sell_value, units, costs)
                pnl = sell_value - buy_value - cost
                equity += pnl
                trades.append({"entry": t_in, "exit": t + 1, "dir": direction, "pnl": pnl,
                               "gross": sell_value - buy_value, "cost": cost, "label": lab, "reason": reason})
                pos = None
        elif desired[t] != 0 and bib <= policy.last_entry_bar:
            direction = int(desired[t])
            e_day = day + policy.min_days_to_expiry
            while e_day % policy.expiry_every_days != policy.expiry_every_days - 1:
                e_day += 1
            exp_bar = e_day * bars_per_day + bars_per_day - 1
            if exp_bar < n:
                tau = (exp_bar - t) * bar_years
                strike = select_strike(close[t], tau, iv[t], policy.rate, direction,
                                       policy.target_delta, policy.strike_step)
                p_dec = bs_price(close[t], strike, tau, iv[t], policy.rate, direction)
                lots = floor(policy.risk_fraction * equity / (policy.stop * p_dec * policy.lot_size)) if p_dec > 0 else 0
                if lots >= 1:
                    p_entry = bs_price(fill[t + 1], strike, tau - minute_years, iv[t], policy.rate, direction)
                    pos = (direction, strike, exp_bar, p_entry, lots, t + 1, label[t])
        if bib == bars_per_day - 1:
            daily.append(equity)
    daily.append(equity)    # the last bar closes the final day; no position is open by then
    return np.array(daily), trades


def performance(daily_equity: np.ndarray, trades: list, equity0: float, days_per_year: int = 252) -> dict:
    """Performance measures of Section VII-D (risk-free rate taken as zero)."""
    eq = np.concatenate([[equity0], daily_equity])
    ret = np.diff(eq) / eq[:-1]
    sd = ret.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2))
    peak = np.maximum.accumulate(eq)
    pnl = np.array([tr["pnl"] for tr in trades]) if trades else np.array([])
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    gross_gain = sum(tr["gross"] for tr in trades if tr["gross"] > 0)
    return {
        "annual_return": 100.0 * ret.mean() * days_per_year,
        "sharpe": np.sqrt(days_per_year) * ret.mean() / sd if sd > 0 else np.nan,
        "sortino": np.sqrt(days_per_year) * ret.mean() / downside if downside > 0 else np.nan,
        "max_drawdown": 100.0 * ((peak - eq) / peak).max(),
        "profit_factor": gains / losses if losses > 0 else np.nan,
        "win_rate": 100.0 * (pnl > 0).mean() if len(pnl) else np.nan,
        "trades": len(pnl),
        "cost_share": 100.0 * sum(tr["cost"] for tr in trades) / gross_gain if gross_gain > 0 else np.nan,
        "final_equity": float(eq[-1]),
    }
