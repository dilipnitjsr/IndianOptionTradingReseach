# Indian Option Trading Research

Research code for two lines of work on Indian markets:

1. **Market-state-aware options trading.** A market state engine that fuses technical-analysis layers by weighted evidence voting, and a backtest of state-gated NIFTY 50 option trades. This supports the paper *A Market-State-Aware Framework for Algorithmic Options Trading in Indian Derivatives Markets*.
2. **Stock-return forecasting.** A leakage-aware next-period return forecasting and backtesting framework (the original content of this repository).

## Repository layout

```text
market_state/     Market state engine: indicators, evidence layers, voting, declaration rule
simulation/       Backtest and simulation of the options strategy (see simulation/README.md)
scripts/          Data collection: pull_nifty_breeze.py (NIFTY spot, India VIX, weekly options)
paper/            Paper figures and notes (see paper/README.md)
results/          Tables and metrics written by the backtest
stock_forecast/   Stock-return forecasting framework
tests/            Tests for market_state and stock_forecast
notes/            Research notes on forecasting
notebooks/        Exploratory notebook
sample_data/      AAPL daily sample used by the forecasting benchmark
data/             Market data pulled from Breeze; not tracked on main (kept on the nifty-data branch)
```

## Options research

### Market state engine

`market_state` implements the engine of the paper: each layer casts a graded vote over the states (bullish trend, bearish trend, range-bound, high volatility), the votes are fused by a weighted sum, and a state is declared only when the leading score passes a threshold, a margin test and a persistence filter.

```python
from market_state import EngineConfig, run_engine
from market_state import indicators as ind, layers as lay

adx = ind.adx(high, low, close)
votes = {
    "trend": lay.trend_layer(ind.ema(close, 9), ind.ema(close, 21), ind.atr(high, low, close)),
    "momentum": lay.momentum_layer(ind.rsi(close)),
    "strength": lay.strength_layer(adx["adx"], adx["di_plus"], adx["di_minus"]),
    "derivatives": lay.derivatives_layer(pcr),
}
config = EngineConfig(
    weights={"trend": 0.3, "momentum": 0.2, "strength": 0.25, "derivatives": 0.25},
    theta=0.55, delta=0.05, h=3,
)
states = run_engine(votes, config)["state"]
```

The layer thresholds and weights are a reference configuration, not yet matched to the production engine. One known limit: only the volatility layer votes for the high-volatility state, so that state's score cannot exceed that layer's weight, and it is never declared when the acceptance threshold is above that weight.

`tests/test_market_state.py` checks the engine numerically against the paper's propositions (score normalisation, uniqueness of the declared state, robustness to a failed layer, bound on state switches).

### Market data

`scripts/pull_nifty_breeze.py` downloads NIFTY 50 spot, India VIX and weekly option contracts (with open interest) from ICICI Breeze. Run it locally, or through the "Pull NIFTY data (Breeze)" workflow in the Actions tab, which commits the CSVs to the `nifty-data` branch. Credentials are read from the environment (`BREEZE_API_KEY`, `BREEZE_API_SECRET`, `BREEZE_SESSION_TOKEN`); never commit them.

```bash
pip install breeze-connect pandas
python scripts/pull_nifty_breeze.py --only spot
python scripts/pull_nifty_breeze.py
```

# Stock-return forecasting framework

Research framework for leakage-aware next-period stock-return forecasting and simple backtesting.

The repository retains the original research notes and notebook, but the executable workflow now lives in the `stock_forecast` package.

## Research protocol

The framework predicts **next-period return**, rather than fitting directly to future prices:

```text
OHLCV history
    ↓
features available at time t
    ↓
target = Close[t+1] / Close[t] - 1
    ↓
chronological train/test split
    ↓
naive / Ridge / Random Forest / HistGradientBoosting
    ↓
forecast metrics + simple sign-strategy diagnostics
```

No random train/test split is used. Any preprocessing that learns parameters (for example StandardScaler in the Ridge pipeline) is fitted only on the training segment.

## Metrics

Forecasting:
- MAE
- RMSE
- R²
- directional accuracy

Research backtest diagnostics:
- cumulative strategy return
- annualized Sharpe ratio
- maximum drawdown
- turnover

The backtest includes a configurable transaction-cost assumption and is intentionally simple. It is for research comparison, **not investment advice or a production trading strategy**.

## Install

Core framework:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To download market data with yfinance:

```bash
pip install -e ".[data,dev]"
```

## Use the existing AAPL data

The repository contains a historical AAPL sample in `sample_data/stock_data.csv`.

```bash
python -m stock_forecast.benchmark \
  --data sample_data/stock_data.csv \
  --output-dir artifacts
```

Outputs:

```text
artifacts/
  leaderboard.csv
  predictions.csv
  config.json
  ridge.joblib
  random_forest.joblib
  hist_gradient_boosting.joblib
```

## Download another ticker

```bash
python -m stock_forecast.download \
  --ticker MSFT \
  --start 2015-01-01 \
  --end 2026-01-01 \
  --output data/msft.csv
```

Then:

```bash
python -m stock_forecast.benchmark --data data/msft.csv
```

## Leakage controls

The older LSTM experiment scaled the complete dataset before splitting it. That lets test-period extrema influence training transformations. The new framework avoids this by:

1. constructing only backward-looking features;
2. using next-period return as the target;
3. splitting chronologically;
4. fitting trainable preprocessing and models only on the training segment.

## Historical material

The Markdown files in `notes/` and `notebooks/stockdata.ipynb` are retained as research notes/history. The historical CSV dataset is in `sample_data/`.

## Tests

```bash
pytest
```

GitHub Actions validates the feature pipeline and runs the benchmark against the committed sample dataset.

## Research caution

Financial time series are non-stationary and noisy. Good performance on one historical split is not evidence of persistent out-of-sample profitability. For serious studies, add rolling/walk-forward evaluation, multiple assets/regimes, robust cost/slippage assumptions, and statistical comparison against simple benchmarks.
