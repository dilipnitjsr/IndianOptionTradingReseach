"""Checks of the synthetic market, option pricing and backtest loop."""
import numpy as np

from simulation.backtest import PolicyParams, performance, run_backtest
from simulation.options import CostParams, bs_delta, bs_price, round_trip_cost, select_strike
from simulation.synthetic import MarketParams, generate_path

MP = MarketParams()


def test_path_is_reproducible_and_well_formed():
    a, b = generate_path(5, 30, MP), generate_path(5, 30, MP)
    assert a.equals(b)
    assert len(a) == 30 * MP.bars_per_day
    assert (a["high"] >= a[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1) + 1e-9).all()
    assert set(a["regime"].unique()) <= {0, 1, 2, 3}
    assert (a["pcr"] > 0).all()


def test_transition_matrix_and_stationary_distribution():
    p = MP.transition_matrix()
    assert np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()
    pi = MP.stationary()
    assert np.allclose(pi @ p, pi) and np.isclose(pi.sum(), 1.0)


def test_black_scholes_put_call_parity_and_delta():
    s, k, tau, sig, r = 25000.0, 25100.0, 4 / 252, 0.14, 0.065
    call, put = bs_price(s, k, tau, sig, r, 1), bs_price(s, k, tau, sig, r, -1)
    assert np.isclose(call - put, s - k * np.exp(-r * tau))
    assert 0 < bs_delta(s, k, tau, sig, r, 1) < 1 and -1 < bs_delta(s, k, tau, sig, r, -1) < 0
    assert bs_price(s, k, 0.0, sig, r, 1) == 0.0 and bs_price(s, k, 0.0, sig, r, -1) == 100.0
    strike = select_strike(s, tau, sig, r, 1)
    assert strike % 50 == 0 and abs(strike - s) <= 100


def test_round_trip_cost_is_positive_and_itemised():
    c = CostParams()
    cost = round_trip_cost(10000.0, 12000.0, 65, c)
    assert cost > 2 * c.brokerage + 65 * 2 * c.half_spread


def test_backtest_runs_and_accounts_for_every_trade():
    path = generate_path(9, 40, MP)
    close, fill = path["close"].to_numpy(), path["fill"].to_numpy()
    iv = np.full(len(close), 0.13)
    regime = path["regime"].to_numpy()
    desired = np.where(regime == 0, 1, np.where(regime == 1, -1, 0)).astype(np.int8)
    label = np.array(["x"] * len(close), dtype=object)
    policy = PolicyParams()
    daily, trades = run_backtest(close, fill, iv, desired, label, MP.bars_per_day, MP.bars_per_year,
                                 5 * MP.bars_per_day, policy, CostParams())
    assert len(daily) == 35 and len(trades) > 0
    assert np.isclose(daily[-1], policy.equity0 + sum(t["pnl"] for t in trades))
    assert all(t["cost"] > 0 and t["exit"] > t["entry"] for t in trades)
    # positions are never held over the close of a day
    assert all(t["entry"] // MP.bars_per_day == t["exit"] // MP.bars_per_day for t in trades)
    perf = performance(daily, trades, policy.equity0)
    assert perf["trades"] == len(trades) and 0 <= perf["max_drawdown"] < 100

    flat, none = run_backtest(close, fill, iv, np.zeros(len(close), dtype=np.int8), label, MP.bars_per_day,
                              MP.bars_per_year, 5 * MP.bars_per_day, policy, CostParams())
    assert not none and (flat == policy.equity0).all()
