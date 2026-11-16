# Simulation study

Study of the state-gated options strategy on a synthetic market with known regimes. It is the evidence used in the paper until NIFTY market data is available. Every market parameter is a stylised assumption, not an estimate from NIFTY data, so the study tests how the framework behaves and not whether it is profitable on the exchange.

## Run

```bash
pip install -e ".[dev,figures]"
python -m simulation.experiment      # about one minute on two cores
python -m simulation.figures
```

`experiment` writes tables to `results/synthetic/`; `figures` writes PDF and PNG files to `paper/figures/`. A quick check: `python -m simulation.experiment --train 2 --test 4 --out /tmp/quick`.

## Modules

| File | Purpose |
| --- | --- |
| `synthetic.py` | Four-regime Markov market on 15-minute bars, with a put-call ratio series tied to the regime |
| `options.py` | Black-Scholes prices, strike selection by delta, transaction-cost model |
| `backtest.py` | Bar-by-bar backtest of a long-option strategy and the performance measures |
| `hmm.py` | Two-state Gaussian hidden Markov model used by one baseline |
| `experiment.py` | Study design: tuning on training paths, evaluation on test paths, ablation, checks of the propositions |
| `figures.py` | The four figures of the paper |

## Design

- The threshold and persistence length of the engine are tuned on 30 training paths and applied unchanged to 200 independent test paths of 250 trading days.
- Methods compared: the proposed engine, a trend rule (B0), the trend rule with an ADX gate (B1) or a hidden-Markov volatility gate (B2), a plain majority vote of the layers (B3), and an oracle that knows the true regime.
- All design constants are at the top of `experiment.py`; market, policy and cost parameters are the dataclasses in `synthetic.py`, `backtest.py` and `options.py`.

## Output files in `results/synthetic/`

| File | Content |
| --- | --- |
| `summary.json` | Design, tuned parameters, state-detection rates, layer accuracies, proposition checks |
| `performance.csv` | Mean and standard error of each performance measure, per method |
| `significance.csv` | Paired difference in Sharpe ratio, proposed minus each baseline |
| `ablation.csv` | Performance with one layer, or the volatility evidence, removed |
| `b0_by_state.csv` | Trades of the trend rule grouped by the declared state at entry |
| `confusion_counts.csv` | True regime against declared state, in bars |
| `tuning_grid.csv` | Training Sharpe ratio over the parameter grid |
| `prop4.csv`, `prop5.csv` | Numerical checks of Propositions 4 and 5 |
| `arrays.npz` | Daily equity of every method on every test path |

## Known limits

The put-call ratio is generated with an assumed link to the regime, and the ablation shows that most of the gain depends on it. Premiums come from Black-Scholes with an implied volatility that lags realised volatility; there are no overnight gaps; stops are checked at bar closes; the cost rates and the lot size are assumptions to verify.

## Market data

When the NIFTY data has been pulled (see `scripts/pull_nifty_breeze.py`), the same engine and backtest will be run on it. The data lives on the `nifty-data` branch.
