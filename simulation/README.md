# Simulation and backtest

Backtest of the state-gated NIFTY 50 options strategy from the paper. This folder is a placeholder until the market data is pulled; the engine it will call is already in `market_state/`.

Planned modules:

| File | Purpose |
| --- | --- |
| `data.py` | Load the spot, VIX and option CSVs from `data/`; build bars and the put-call ratio |
| `policy.py` | State-conditioned policy: strike selection by delta, position sizing, exit rule |
| `costs.py` | Indian transaction-cost model (brokerage, STT, exchange and SEBI fees, stamp duty, GST, slippage) |
| `backtest.py` | Walk-forward backtest of the proposed method and the baselines |
| `metrics.py` | Sharpe, Sortino, maximum drawdown, profit factor, significance tests |
| `figures.py` | Equity curves, state sequence over price, parameter heat map |

Outputs go to `results/` (tables, metrics) and `paper/figures/` (plots).

Input data comes from the `nifty-data` branch:

```bash
git fetch origin nifty-data
git worktree add ../nifty-data nifty-data
ln -s ../nifty-data/data data
```
