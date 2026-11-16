"""Numerical checks of the market state engine against the paper's propositions."""
import numpy as np
import pandas as pd
import pytest

from market_state import UNDETERMINED, EngineConfig, aggregate, candidate, persist, run_engine, switch_count
from market_state import indicators as ind
from market_state import layers as lay

STATES = lay.STATES
WEIGHTS = {"a": 0.3, "b": 0.2, "c": 0.25, "d": 0.25}


def random_votes(rng, rows, full=False):
    """Random vote vectors satisfying the constraint; ``full`` leaves no abstention."""
    raw = rng.dirichlet(np.ones(len(STATES) + (0 if full else 1)), size=rows)
    return pd.DataFrame(raw[:, : len(STATES)], columns=STATES)


def random_layers(seed, rows=2000, full=False):
    rng = np.random.default_rng(seed)
    return {name: random_votes(rng, rows, full) for name in WEIGHTS}


def sample_ohlc(rows=600, seed=7):
    rng = np.random.default_rng(seed)
    close = pd.Series(25000 * np.exp(np.cumsum(rng.normal(0, 0.001, rows))))
    spread = np.abs(rng.normal(0, 8, rows))
    return close + spread, close - spread, close


def test_ramp_limits():
    assert lay.ramp([-1, 0, 0.5, 1, 2], 0, 1).tolist() == [0, 0, 0.5, 1, 1]
    with pytest.raises(ValueError):
        lay.ramp(1.0, 2, 2)


def test_indicators_are_bounded():
    high, low, close = sample_ohlc()
    rsi = ind.rsi(close)
    assert rsi.between(0, 100).all()
    adx = ind.adx(high, low, close)
    assert adx["adx"].between(0, 100).all()
    assert (ind.atr(high, low, close).dropna() > 0).all()
    rank = ind.percentile_rank(close, 50).dropna()
    assert rank.between(0, 1).all()


def test_layers_satisfy_vote_constraint():
    high, low, close = sample_ohlc()
    adx = ind.adx(high, low, close)
    pcr = pd.Series(np.random.default_rng(1).uniform(0.5, 1.6, len(close)))
    votes = [
        lay.trend_layer(ind.ema(close, 9), ind.ema(close, 21), ind.atr(high, low, close)),
        lay.momentum_layer(ind.rsi(close)),
        lay.strength_layer(adx["adx"], adx["di_plus"], adx["di_minus"]),
        lay.derivatives_layer(pcr),
    ]
    nu = lay.volatility_evidence(ind.percentile_rank(ind.atr(high, low, close) / close, 100))
    assert nu.between(0, 1).all()
    shared = lay.with_volatility({str(i): v for i, v in enumerate(votes)}, nu)
    scores = aggregate(shared, {k: 1 / len(shared) for k in shared})
    assert np.allclose(scores["V"], nu)          # the score of V equals the shared evidence
    for v in votes + list(shared.values()):
        arr = v.to_numpy()
        assert (arr >= 0).all() and (arr <= 1).all()
        assert (arr.sum(axis=1) <= 1 + 1e-12).all()
        # no layer votes for U and D on the same bar
        assert ((v["U"] > 0) & (v["D"] > 0)).sum() == 0


def test_proposition_1_normalisation():
    scores = aggregate(random_layers(0), WEIGHTS)
    arr = scores.to_numpy()
    assert (arr >= 0).all() and (arr <= 1).all()
    assert (arr.sum(axis=1) <= 1 + 1e-12).all()
    full = aggregate(random_layers(0, full=True), WEIGHTS).to_numpy()
    assert np.allclose(full.sum(axis=1), 1.0)


@pytest.mark.parametrize("theta", [0.51, 0.6, 0.75])
def test_proposition_2_uniqueness(theta):
    rng = np.random.default_rng(3)
    scores = pd.DataFrame(rng.dirichlet(np.ones(len(STATES)) * 0.4, size=5000), columns=STATES)
    arr = scores.to_numpy()
    above = arr >= theta
    assert above.sum(axis=1).max() <= 1
    _, margin = candidate(scores, theta, 0.0)
    assert (margin[above.any(axis=1)] >= 2 * theta - 1 - 1e-12).all()


def test_proposition_3_robustness():
    theta, delta = 0.4, 0.05
    layers = random_layers(5, rows=20000)
    # concentrate evidence so that some bars meet the sufficient condition
    rng = np.random.default_rng(6)
    lead = rng.integers(0, len(STATES), size=20000)
    for name in ("a", "c", "d"):
        v = np.zeros((20000, len(STATES)))
        v[np.arange(20000), lead] = rng.uniform(0.6, 1.0, 20000)
        layers[name] = pd.DataFrame(v, columns=STATES)
    base_scores = aggregate(layers, WEIGHTS)
    base_cand, base_margin = candidate(base_scores, theta, delta)
    w_b = WEIGHTS["b"]
    corrupted = dict(layers)
    corrupted["b"] = random_votes(np.random.default_rng(99), 20000)
    new_scores = aggregate(corrupted, WEIGHTS)
    new_cand, _ = candidate(new_scores, theta, delta)
    assert (np.abs(new_scores.to_numpy() - base_scores.to_numpy()) <= w_b + 1e-12).all()
    safe = (base_scores.max(axis=1) >= theta + w_b) & (base_margin > delta + 2 * w_b)
    assert safe.sum() > 100
    assert (new_cand[safe] == base_cand[safe]).all()


@pytest.mark.parametrize("h", [1, 2, 3, 5, 8])
def test_proposition_5_switch_bound(h):
    rng = np.random.default_rng(11)
    cand = pd.Series(rng.choice(STATES + [UNDETERMINED], size=5000, p=[0.3, 0.3, 0.2, 0.1, 0.1]))
    declared = persist(cand, h)
    assert switch_count(declared) <= len(cand) // h
    # a declared state is only ever adopted after h equal candidates
    values, cands = declared.to_numpy(), cand.to_numpy()
    for t in range(1, len(values)):
        if values[t] != values[t - 1]:
            assert t + 1 >= h and (cands[t - h + 1 : t + 1] == values[t]).all()


def test_persistence_with_h_1_follows_candidate():
    cand = pd.Series(["U", "U", "s0", "D", "D"])
    assert persist(cand, 1).tolist() == cand.tolist()


def test_run_engine_output_and_config_checks():
    out = run_engine(random_layers(2), EngineConfig(weights=WEIGHTS, theta=0.55, delta=0.05, h=3))
    assert set(out["state"].unique()) <= set(STATES) | {UNDETERMINED}
    assert (out["abstention"] >= -1e-12).all()
    with pytest.raises(ValueError):
        EngineConfig(weights={"a": 0.5, "b": 0.6})
    bad = random_layers(2)
    bad["a"] = bad["a"] * 3
    with pytest.raises(ValueError):
        aggregate(bad, WEIGHTS)
