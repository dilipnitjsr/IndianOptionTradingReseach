"""Figures of the simulation study.

Run after the experiment, from the repository root:
    python -m simulation.figures

Reads results/synthetic/ and writes PDF and PNG files to paper/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from . import experiment as ex
from .synthetic import REGIMES

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
# Three hues that stay distinct for every pair, a neutral for V, blank for undetermined.
STATE_COLOR = {"U": BLUE, "D": ORANGE, "R": AQUA, "V": INK2, "s0": SURFACE}
STATE_LABEL = {"U": "Bullish trend (U)", "D": "Bearish trend (D)", "R": "Range-bound (R)",
               "V": "High volatility (V)", "s0": "Undetermined"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
COL, WIDE = 3.5, 7.16      # IEEE column and page widths in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 400,
})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def strip(ax, y, states, x0):
    """Draw one row of coloured cells, merging runs of the same state."""
    change = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(states)]])
    for s, e in zip(starts, ends):
        ax.add_patch(Rectangle((x0 + s, y), e - s, 0.8, facecolor=STATE_COLOR[states[s]], edgecolor="none"))
    ax.add_patch(Rectangle((x0, y), len(states), 0.8, facecolor="none", edgecolor=INK2, linewidth=0.4))


def fig_states(summary: dict, out: Path, seed: int = 3000, first_day: int = 20, n_days: int = 30):
    theta, h = summary["tuned"]["theta"], summary["tuned"]["h"]
    prep = ex.prepare(seed)
    state = ex.declared(ex.scores(prep, ex.WEIGHTS), theta, h)
    bpd = ex.MARKET.bars_per_day
    a, b = first_day * bpd, (first_day + n_days) * bpd
    x = np.arange(b - a) / bpd
    true = np.array(REGIMES, dtype=object)[prep["regime"][a:b]]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(WIDE, 2.7), sharex=True,
                                  gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08})
    ax.plot(x, prep["close"][a:b], color=INK, linewidth=0.8)
    ax.set_ylabel("Simulated index level")
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    for row, (name, arr) in enumerate([("Declared", state[a:b]), ("True", true)]):
        change = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        for s, e in zip(np.concatenate([[0], change]), np.concatenate([change, [len(arr)]])):
            ax2.add_patch(Rectangle((s / bpd, row), (e - s) / bpd, 0.8,
                                    facecolor=STATE_COLOR[arr[s]], edgecolor="none"))
        ax2.add_patch(Rectangle((0, row), n_days, 0.8, facecolor="none", edgecolor=INK2, linewidth=0.4))
    ax2.set_ylim(-0.15, 1.95)
    ax2.set_yticks([0.4, 1.4], ["Declared state", "True regime"])
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlim(0, n_days)
    ax2.set_xlabel("Trading day")
    handles = [Patch(facecolor=STATE_COLOR[s], edgecolor=INK2, linewidth=0.4, label=STATE_LABEL[s])
               for s in ["U", "D", "R", "V", "s0"]]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.2, columnspacing=1.4)
    save(fig, out, "fig_states")


def fig_equity(arrays, out: Path):
    e0 = ex.POLICY.equity0
    days = np.arange(arrays["equity_Proposed"].shape[1] + 1)
    series = [("Proposed", BLUE, "-", 1.6), ("B1", ORANGE, (0, (5, 2)), 1.2), ("B0", AQUA, (0, (5, 1.5, 1, 1.5)), 1.2),
              ("B3", YELLOW, (0, (1, 1.2)), 1.4), ("B2", MAGENTA, (0, (8, 2, 2, 2)), 1.2)]
    names = {"Proposed": "Proposed", "B0": "B0 trend rule", "B1": "B1 ADX gate",
             "B2": "B2 HMM gate", "B3": "B3 majority vote"}
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    prop = np.column_stack([np.full(len(arrays["equity_Proposed"]), e0), arrays["equity_Proposed"]])
    lo, hi = np.percentile(100 * (prop / e0 - 1), [25, 75], axis=0)
    ax.fill_between(days, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
    ends = []
    for key, color, style, width in series:
        eq = np.column_stack([np.full(len(arrays["equity_" + key]), e0), arrays["equity_" + key]])
        mean = 100 * (eq.mean(axis=0) / e0 - 1)
        ax.plot(days, mean, color=color, linestyle=style, linewidth=width, label=names[key])
        ends.append((mean[-1], key))
    ax.axhline(0, color=INK2, linewidth=0.5)
    ax.set_xlim(0, days[-1] + 42)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Mean change in equity (%)")
    ax.grid(axis="y")
    # end labels, nudged apart so they do not collide
    ends.sort()
    placed = []
    for value, key in ends:
        y = value if not placed or value - placed[-1] >= 1.6 else placed[-1] + 1.6
        placed.append(y)
        ax.text(days[-1] + 3, y, f"{key} {value:+.1f}", va="center", fontsize=6.5, color=INK)
    ax.legend(loc="upper left", handlelength=2.6)
    save(fig, out, "fig_equity")


def fig_tuning(grid: pd.DataFrame, summary: dict, out: Path):
    piv = grid.pivot(index="theta", columns="h", values="sharpe")
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    vals = piv.to_numpy()
    ax.imshow(vals, cmap=SEQ, aspect="auto", origin="lower")
    mid = (vals.min() + vals.max()) / 2
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if vals[i, j] > mid else INK)
    ti = list(piv.index).index(summary["tuned"]["theta"])
    tj = list(piv.columns).index(summary["tuned"]["h"])
    ax.add_patch(Rectangle((tj - 0.5, ti - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4))
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{t:.2f}" for t in piv.index])
    ax.set_xlabel("Persistence length $h$ (bars)")
    ax.set_ylabel(r"Acceptance threshold $\theta$")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    save(fig, out, "fig_tuning")


def fig_propositions(p4: pd.DataFrame, p5: pd.DataFrame, out: Path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL, 4.3), gridspec_kw={"hspace": 0.5})
    ax.plot(p4["theta"], p4["bound_false"], color=INK, linestyle=(0, (5, 2)), linewidth=1.2, label="Bound of Proposition 4(b)")
    ax.plot(p4["theta"], p4["engine_false"], color=ORANGE, marker="s", markersize=3.5, linewidth=1.2,
            label="Engine on simulated market")
    ax.plot(p4["theta"], p4["mc_false"], color=BLUE, marker="o", markersize=3.5, linewidth=1.2,
            label="Independent hard votes")
    ax.set_yscale("log")
    ax.set_xlabel(r"Acceptance threshold $\theta$")
    ax.set_ylabel("Probability of a false declaration")
    ax.grid(axis="y")
    ax.legend(loc="lower left")
    ax.set_title("(a)", loc="left", fontsize=8)

    ax2.plot(p5["h"], p5["bound"], color=INK, linestyle=(0, (5, 2)), linewidth=1.2, label=r"Bound $\lfloor T/h \rfloor$")
    ax2.plot(p5["h"], p5["max_switches"], color=ORANGE, marker="s", markersize=3.5, linewidth=1.2,
             label="Largest count over test paths")
    ax2.plot(p5["h"], p5["mean_switches"], color=BLUE, marker="o", markersize=3.5, linewidth=1.2,
             label="Mean count over test paths")
    ax2.set_yscale("log")
    ax2.set_xticks(p5["h"])
    ax2.set_xlabel("Persistence length $h$ (bars)")
    ax2.set_ylabel("State switches in 250 days")
    ax2.grid(axis="y")
    ax2.legend(loc="upper right")
    ax2.set_title("(b)", loc="left", fontsize=8)
    save(fig, out, "fig_propositions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/synthetic")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    summary = json.loads((res / "summary.json").read_text())
    fig_states(summary, out)
    fig_equity(np.load(res / "arrays.npz"), out)
    fig_tuning(pd.read_csv(res / "tuning_grid.csv"), summary, out)
    fig_propositions(pd.read_csv(res / "prop4.csv"), pd.read_csv(res / "prop5.csv"), out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
