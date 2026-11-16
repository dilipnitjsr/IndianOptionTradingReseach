"""Simulation study: tune the engine on training paths, evaluate on independent test paths.

Run from the repository root:
    python -m simulation.experiment                # full study (30 training, 200 test paths)
    python -m simulation.experiment --train 2 --test 4 --out /tmp/quick   # smoke run

Writes tables to results/synthetic/ (CSV and summary.json) and the arrays the
figures need to results/synthetic/arrays.npz.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from market_state import EngineConfig, aggregate, candidate, persist, switch_count
from market_state import indicators as ind
from market_state import layers as lay

from .backtest import PolicyParams, performance, run_backtest
from .hmm import filter_low_vol, fit_hmm
from .options import CostParams
from .synthetic import REGIMES, MarketParams, generate_path

# ---- study design (fixed before any test path is generated) -----------------
MARKET = MarketParams()
POLICY = PolicyParams()
COSTS = CostParams()
WEIGHTS = {"trend": 0.30, "momentum": 0.20, "strength": 0.25, "derivatives": 0.25}
DELTA = 0.05
THETAS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
HS = [1, 2, 3, 4, 6, 8, 12]
WARM_DAYS = 20          # indicator warm-up, excluded from trading and from all statistics
EVAL_DAYS = 250
VOL_WINDOW = 500        # bars in the volatility percentile rank
IV_SPAN = 50            # bars in the EWMA realised volatility behind implied volatility (about two days)
IV_MARKUP = 1.10        # implied = realised x markup (volatility risk premium)
IV_FLOOR = 0.08
TREND_ON = 0.6          # |z| at which the single-layer trend rule (B0) takes a position
ADX_GATE = 25.0
TRAIN_SEEDS = range(1100, 1130)
TEST_SEEDS = range(3000, 3200)
PROP4_THETAS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
METHODS = ["Proposed", "B0", "B1", "B2", "B3", "Oracle"]
STATE_NAMES = REGIMES + ["s0"]


def prepare(seed: int) -> dict:
    """Generate a path and everything computed from it that does not depend on the engine parameters."""
    path = generate_path(seed, WARM_DAYS + EVAL_DAYS, MARKET)
    close, high, low = path["close"], path["high"], path["low"]
    atr = ind.atr(high, low, close)
    adx = ind.adx(high, low, close)
    z = (ind.ema(close, 9) - ind.ema(close, 21)) / atr
    base = {
        "trend": lay.trend_layer(ind.ema(close, 9), ind.ema(close, 21), atr),
        "momentum": lay.momentum_layer(ind.rsi(close)),
        "strength": lay.strength_layer(adx["adx"], adx["di_plus"], adx["di_minus"]),
        "derivatives": lay.derivatives_layer(path["pcr"]),
    }
    nu = lay.volatility_evidence(ind.percentile_rank(atr / close, VOL_WINDOW))
    ret = np.log(close).diff().fillna(0.0)
    realised = np.sqrt(ret.pow(2).ewm(span=IV_SPAN, adjust=False).mean() * MARKET.bars_per_year)
    iv = np.maximum(realised * IV_MARKUP, IV_FLOOR).to_numpy()
    return {"seed": seed, "path": path, "close": close.to_numpy(), "fill": path["fill"].to_numpy(),
            "regime": path["regime"].to_numpy(), "z": z.fillna(0.0).to_numpy(), "adx": adx["adx"].to_numpy(),
            "base": base, "nu": nu, "iv": iv, "ret": ret.to_numpy()}


def scores(prep: dict, weights: dict, use_vol: bool = True) -> pd.DataFrame:
    nu = prep["nu"] if use_vol else prep["nu"] * 0.0
    votes = lay.with_volatility({k: prep["base"][k] for k in weights}, nu)
    return aggregate(votes, weights)


def declared(sc: pd.DataFrame, theta: float, h: int, delta: float = DELTA) -> np.ndarray:
    cand, _ = candidate(sc, theta, delta)
    return persist(cand, h).to_numpy()


def to_desired(state: np.ndarray) -> np.ndarray:
    return np.where(state == "U", 1, np.where(state == "D", -1, 0)).astype(np.int8)


def backtest(prep: dict, desired: np.ndarray, label: np.ndarray):
    start = WARM_DAYS * MARKET.bars_per_day
    daily, trades = run_backtest(prep["close"], prep["fill"], prep["iv"], desired, label,
                                 MARKET.bars_per_day, MARKET.bars_per_year, start, POLICY, COSTS)
    return daily, trades, performance(daily, trades, POLICY.equity0)


def renormalised(drop: str) -> dict:
    kept = {k: v for k, v in WEIGHTS.items() if k != drop}
    total = sum(kept.values())
    return {k: v / total for k, v in kept.items()}


# ---- training: tune (theta, h) ------------------------------------------------
def train_one(seed: int) -> dict:
    prep = prepare(seed)
    sc = scores(prep, WEIGHTS)
    grid = {}
    for theta in THETAS:
        cand, _ = candidate(sc, theta, DELTA)
        for h in HS:
            state = persist(cand, h).to_numpy()
            _, _, perf = backtest(prep, to_desired(state), state)
            grid[(theta, h)] = 0.0 if np.isnan(perf["sharpe"]) else perf["sharpe"]
    start = WARM_DAYS * MARKET.bars_per_day
    return {"grid": grid, "ret": prep["ret"][start:]}


# ---- testing ------------------------------------------------------------------
def test_one(args) -> dict:
    seed, theta, h, hmm_model = args
    prep = prepare(seed)
    start = WARM_DAYS * MARKET.bars_per_day
    regime_names = np.array(REGIMES, dtype=object)[prep["regime"]]
    sc = scores(prep, WEIGHTS)
    state = declared(sc, theta, h)
    out = {"seed": seed, "perf": {}, "equity": {}}

    # strategies
    b0 = np.where(prep["z"] >= TREND_ON, 1, np.where(prep["z"] <= -TREND_ON, -1, 0)).astype(np.int8)
    low_vol = filter_low_vol(prep["ret"], hmm_model) > 0.5
    eq_w = {k: 1.0 / len(WEIGHTS) for k in WEIGHTS}
    sc_eq = scores(prep, eq_w).to_numpy()
    plain = np.array(REGIMES, dtype=object)[sc_eq.argmax(axis=1)]
    plain[sc_eq.max(axis=1) <= 0] = "s0"
    desired = {
        "Proposed": to_desired(state),
        "B0": b0,
        "B1": np.where(prep["adx"] >= ADX_GATE, b0, 0).astype(np.int8),
        "B2": np.where(low_vol, b0, 0).astype(np.int8),
        "B3": to_desired(plain),
        "Oracle": to_desired(regime_names),
    }
    b0_trades = None
    for name, des in desired.items():
        daily, trades, perf = backtest(prep, des, state)
        out["perf"][name] = perf
        out["equity"][name] = daily
        if name == "B0":
            b0_trades = trades
        if name == "Proposed":
            reasons = pd.Series([tr["reason"] for tr in trades]).value_counts().to_dict()
            out["exit_reasons"] = reasons

    # Proposition 6: the ungated trend rule, grouped by the declared state at entry
    by_state = {s: {"n": 0, "pnl": 0.0, "wins": 0} for s in STATE_NAMES}
    for tr in b0_trades:
        rec = by_state[tr["label"]]
        rec["n"] += 1
        rec["pnl"] += tr["pnl"]
        rec["wins"] += tr["pnl"] > 0
    out["b0_by_state"] = by_state

    # ablation
    out["ablation"] = {}
    for drop in WEIGHTS:
        st = declared(scores(prep, renormalised(drop)), theta, h)
        _, _, perf = backtest(prep, to_desired(st), st)
        out["ablation"]["Without " + drop] = {**perf, "switches": switch_count(pd.Series(st[start:]))}
    st = declared(scores(prep, WEIGHTS, use_vol=False), theta, h)
    _, _, perf = backtest(prep, to_desired(st), st)
    out["ablation"]["Without volatility"] = {**perf, "switches": switch_count(pd.Series(st[start:]))}
    out["ablation"]["All layers"] = {**out["perf"]["Proposed"], "switches": switch_count(pd.Series(state[start:]))}

    # state detection against the true regime
    true_eval, decl_eval = regime_names[start:], state[start:]
    conf = pd.crosstab(pd.Series(true_eval, name="true"), pd.Series(decl_eval, name="declared"))
    out["confusion"] = conf.reindex(index=REGIMES, columns=STATE_NAMES, fill_value=0).to_numpy()
    out["declared_share"] = pd.Series(decl_eval).value_counts().reindex(STATE_NAMES, fill_value=0).to_numpy()

    # candidate-level miss and false-declaration rates over theta (Proposition 4 comparison)
    rates = []
    for th in PROP4_THETAS:
        cand = candidate(sc, th, DELTA)[0].to_numpy()[start:]
        rates.append(((cand != true_eval).mean(), ((cand != true_eval) & (cand != "s0")).mean()))
    out["cand_rates"] = np.array(rates)

    # hard-vote accuracy of each layer
    votes = lay.with_volatility(prep["base"], prep["nu"])
    acc = {}
    for name, v in votes.items():
        arr = v.to_numpy()[start:]
        hard = np.array(REGIMES, dtype=object)[arr.argmax(axis=1)]
        hard[arr.max(axis=1) <= 0] = "s0"
        acc[name] = (hard == true_eval).mean()
    out["layer_accuracy"] = acc

    # Proposition 5: switches against the bound T / h
    cand = candidate(sc, theta, DELTA)[0]
    out["switches"] = {hh: switch_count(persist(cand, hh).iloc[start:]) for hh in [1, 2, 3, 4, 6, 8, 12]}
    out["eval_bars"] = len(decl_eval)

    # Proposition 3: corrupt the momentum layer with random votes
    rng = np.random.default_rng(seed + 7)
    bad = dict(lay.with_volatility(prep["base"], prep["nu"]))
    raw = rng.dirichlet(np.ones(5), size=len(prep["close"]))[:, :4]
    bad["momentum"] = pd.DataFrame(raw, columns=REGIMES, index=sc.index)
    sc_bad = aggregate(bad, WEIGHTS)
    c0, m0 = candidate(sc, theta, DELTA)
    c1, _ = candidate(sc_bad, theta, DELTA)
    w_b = WEIGHTS["momentum"]
    safe = ((sc.max(axis=1) >= theta + w_b) & (m0 > DELTA + 2 * w_b)).to_numpy()[start:]
    changed = (c0.to_numpy() != c1.to_numpy())[start:]
    out["prop3"] = {"safe": int(safe.sum()), "safe_changed": int((safe & changed).sum()),
                    "changed": int(changed.sum()),
                    "max_shift": float(np.abs(sc_bad.to_numpy() - sc.to_numpy()).max())}
    return out


def prop4_monte_carlo(p: dict, draws: int = 400_000, seed: int = 11) -> pd.DataFrame:
    """Independent hard-vote layers with accuracies p: empirical rates against the bounds of Proposition 4."""
    rng = np.random.default_rng(seed)
    names = list(WEIGHTS)
    w = np.array([WEIGHTS[k] for k in names])
    acc = np.array([p[k] for k in names])
    correct = rng.random((draws, len(names))) < acc
    wrong_state = rng.integers(1, 4, size=(draws, len(names)))         # true state is index 0
    vote = np.where(correct, 0, wrong_state)
    a = np.zeros((draws, 4))
    for j in range(len(names)):
        np.add.at(a, (np.arange(draws), vote[:, j]), w[j])
    top = a.argmax(axis=1)
    top_score = a.max(axis=1)
    pbar, leff = float(w @ acc), float(1.0 / (w**2).sum())
    rows = []
    for th in PROP4_THETAS:
        declared_any = top_score >= th
        miss = 1.0 - (declared_any & (top == 0)).mean()
        false = (declared_any & (top != 0)).mean()
        rows.append({"theta": th, "mc_miss": miss, "mc_false": false,
                     "bound_miss": np.exp(-2 * leff * (pbar - th) ** 2) if pbar > th else np.nan,
                     "bound_false": np.exp(-2 * leff * (pbar + th - 1) ** 2) if pbar > 1 - th else np.nan})
    df = pd.DataFrame(rows)
    df.attrs.update(pbar=pbar, leff=leff)
    return df


def mean_se(values) -> tuple:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def paired(a, b, rng) -> dict:
    d = np.nan_to_num(np.asarray(a, dtype=float)) - np.nan_to_num(np.asarray(b, dtype=float))
    boot = rng.choice(d, size=(10_000, len(d)), replace=True).mean(axis=1)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "t": float(d.mean() / se) if se > 0 else np.nan,
            "share_positive": float((d > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=len(TRAIN_SEEDS))
    ap.add_argument("--test", type=int, default=len(TEST_SEEDS))
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_seeds, test_seeds = list(TRAIN_SEEDS)[: args.train], list(TEST_SEEDS)[: args.test]

    with Pool(args.workers) as pool:
        trained = pool.map(train_one, train_seeds)
    grid = pd.DataFrame([{"theta": th, "h": h, "sharpe": np.mean([t["grid"][(th, h)] for t in trained])}
                         for th in THETAS for h in HS])
    best = grid.loc[grid["sharpe"].idxmax()]
    theta, h = float(best["theta"]), int(best["h"])
    grid.to_csv(out / "tuning_grid.csv", index=False)
    hmm_model = fit_hmm(np.concatenate([t["ret"] for t in trained[: min(8, len(trained))]]))
    print(f"tuned theta={theta} h={h} (training Sharpe {best['sharpe']:.3f})")

    with Pool(args.workers) as pool:
        res = pool.map(test_one, [(s, theta, h, hmm_model) for s in test_seeds], chunksize=4)

    rng = np.random.default_rng(123)
    metrics = ["annual_return", "sharpe", "sortino", "max_drawdown", "profit_factor", "win_rate", "trades", "cost_share"]
    perf_rows = []
    for m in METHODS:
        row = {"method": m}
        for k in metrics:
            row[k], row[k + "_se"] = mean_se([r["perf"][m][k] for r in res])
        perf_rows.append(row)
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out / "performance.csv", index=False)

    sig = pd.DataFrame([{"baseline": b, **paired([r["perf"]["Proposed"]["sharpe"] for r in res],
                                                 [r["perf"][b]["sharpe"] for r in res], rng)}
                        for b in ["B0", "B1", "B2", "B3"]])
    sig.to_csv(out / "significance.csv", index=False)

    abl_names = ["All layers"] + ["Without " + k for k in WEIGHTS] + ["Without volatility"]
    abl = pd.DataFrame([{"configuration": n,
                         "sharpe": mean_se([r["ablation"][n]["sharpe"] for r in res])[0],
                         "sharpe_se": mean_se([r["ablation"][n]["sharpe"] for r in res])[1],
                         "max_drawdown": mean_se([r["ablation"][n]["max_drawdown"] for r in res])[0],
                         "switches_per_day": np.mean([r["ablation"][n]["switches"] for r in res]) / EVAL_DAYS}
                        for n in abl_names])
    abl.to_csv(out / "ablation.csv", index=False)

    conf = sum(r["confusion"] for r in res)
    conf_df = pd.DataFrame(conf, index=REGIMES, columns=STATE_NAMES)
    conf_df.to_csv(out / "confusion_counts.csv")
    total = conf.sum()
    correct = sum(conf[i, i] for i in range(4))
    abstain = conf[:, 4].sum()
    detection = {"correct": correct / total, "abstain": abstain / total, "false": (total - correct - abstain) / total,
                 "true_share": (conf.sum(axis=1) / total).tolist(), "declared_share": (conf.sum(axis=0) / total).tolist()}

    by_state = pd.DataFrame([{"state": s,
                              "share_of_bars": 100 * conf[:, i].sum() / total,
                              "trades": sum(r["b0_by_state"][s]["n"] for r in res),
                              "mean_pnl": (sum(r["b0_by_state"][s]["pnl"] for r in res)
                                           / max(1, sum(r["b0_by_state"][s]["n"] for r in res))),
                              "win_rate": 100 * (sum(r["b0_by_state"][s]["wins"] for r in res)
                                                 / max(1, sum(r["b0_by_state"][s]["n"] for r in res)))}
                             for i, s in enumerate(STATE_NAMES)])
    by_state.to_csv(out / "b0_by_state.csv", index=False)

    layer_acc = {k: float(np.mean([r["layer_accuracy"][k] for r in res])) for k in WEIGHTS}
    p4 = prop4_monte_carlo(layer_acc)
    rates = np.mean([r["cand_rates"] for r in res], axis=0)
    p4["engine_miss"], p4["engine_false"] = rates[:, 0], rates[:, 1]
    p4.to_csv(out / "prop4.csv", index=False)

    bars = res[0]["eval_bars"]
    p5 = pd.DataFrame([{"h": hh, "mean_switches": np.mean([r["switches"][hh] for r in res]),
                        "max_switches": max(r["switches"][hh] for r in res), "bound": bars // hh}
                       for hh in res[0]["switches"]])
    p5.to_csv(out / "prop5.csv", index=False)

    prop3 = {k: int(sum(r["prop3"][k] for r in res)) for k in ("safe", "safe_changed", "changed")}
    prop3["max_shift"] = max(r["prop3"]["max_shift"] for r in res)
    prop3["bars"] = bars * len(res)
    reasons = pd.DataFrame([r["exit_reasons"] for r in res]).fillna(0).sum().to_dict()

    summary = {
        "design": {"train_paths": len(train_seeds), "test_paths": len(test_seeds), "eval_days": EVAL_DAYS,
                   "warm_days": WARM_DAYS, "weights": WEIGHTS, "delta": DELTA, "thetas": THETAS, "hs": HS,
                   "market": asdict(MARKET), "policy": asdict(POLICY), "costs": asdict(COSTS),
                   "stationary": dict(zip(REGIMES, MARKET.stationary().round(4).tolist()))},
        "tuned": {"theta": theta, "h": h, "train_sharpe": float(best["sharpe"])},
        "hmm": hmm_model, "detection": detection, "layer_accuracy": layer_acc,
        "prop4": {"pbar": p4.attrs["pbar"], "leff": p4.attrs["leff"]}, "prop3": prop3, "exit_reasons": reasons,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(out / "arrays.npz",
                        **{"equity_" + m: np.array([r["equity"][m] for r in res]) for m in METHODS})
    print(perf.round(3).to_string(index=False))
    print(sig.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
