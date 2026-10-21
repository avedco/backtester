"""Draw all charts from the files saved by run.py. Fast: no backtests are re-run
(except the optional single-pair diagnostic, which only fits one window).

Usage:  python make_charts.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load
from engine import select_pairs, zscore_positions

RESULTS = "results"
FORM_DAYS, TRADE_DAYS = 504, 126


def save(fig, name):
    fig.savefig(f"{RESULTS}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)       # each chart gets its own figure, so they can't bleed together


def equity_chart(returns):
    fig, ax = plt.subplots(figsize=(10, 4))
    (1 + returns["gross"]).cumprod().plot(ax=ax, label="gross")
    (1 + returns["net"]).cumprod().plot(ax=ax, label="net")
    ax.set_title("Equity curve (growth of 1)")
    ax.set_ylabel("Equity")
    ax.legend()
    save(fig, "equity")


def drawdown_chart(returns):
    eq = (1 + returns["net"]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    (eq / eq.cummax() - 1).plot(ax=ax)
    ax.set_title("Net drawdown")
    ax.set_ylabel("Drawdown")
    save(fig, "drawdown")


def train_test_chart(summary):
    """Gross vs net Sharpe, train vs test, with 95% bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    w = 0.35
    for i, label in enumerate(["gross", "net"]):
        s = summary[label]
        vals = [s["train"]["sharpe"], s["test"]["sharpe"]]
        cis = [s["train_ci"], s["test_ci"]]
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        ax.bar(x + (i - 0.5) * w, vals, w, label=label.capitalize(),
               yerr=[lo, hi], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Train (to 2018)", "Test (2019+)"])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.5, color="r", ls="--", label="Pre-registered target (0.5)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Sharpe with 95% bootstrap CI")
    # legend below the plot so it can't hide the bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    save(fig, "train_test")


def cost_chart(sens, summary):
    sens = sens.rename(columns={"sharpe_train": "Train (to 2018)",
                                "sharpe_test": "Test (2019+)"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sens.plot(ax=ax, marker="o")
    ax.axhline(0, color="k", lw=0.8)
    be = summary["breakeven_bps"]
    ax.axvline(be["sharpe_train"], color="C0", ls=":", lw=1)
    ax.axvline(be["sharpe_test"], color="C1", ls=":", lw=1)
    ax.set_xlabel("Trading cost (bps per unit traded; 0.5% borrow always included)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Net Sharpe vs transaction cost")
    save(fig, "cost_sensitivity")


def one_pair_chart(logp, k=3, p_index=0):
    """Diagnostic: z-score and desired position for one selected pair in window k.
    Window 3 trades from mid-2015, so it is inside the TRAIN period."""
    start = k * TRADE_DAYS
    f_end = start + FORM_DAYS
    form = logp.iloc[start:f_end]
    idx = logp.index[f_end:f_end + TRADE_DAYS]

    pairs, _ = select_pairs(form)
    if p_index >= len(pairs):
        print(f"Window {k} has only {len(pairs)} pairs; skipping pair {p_index}")
        return
    p = pairs[p_index]
    print("One-pair chart:", p)

    spread = logp[p["a"]] - p["beta"] * logp[p["b"]]
    z = (spread - p["mu"]) / p["sd"]                # formation-window stats only
    pos = zscore_positions(z.loc[idx])

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    z.iloc[start:f_end + TRADE_DAYS].plot(ax=ax[0])
    for lvl in [-4, -2, -0.5, 0.5, 2, 4]:
        ax[0].axhline(lvl, color="grey", ls="--", lw=0.7)
    ax[0].axvline(logp.index[f_end], color="red")   # red line = formation | trading
    ax[0].set_title(f"z-score of {p['a']}/{p['b']} (left of red line = formation)")
    pos.plot(ax=ax[1], drawstyle="steps-post")
    ax[1].set_title("desired position (+1 long spread, -1 short, 0 flat)")
    fig.tight_layout()
    save(fig, f"one_pair{p_index}")


def main():
    returns = pd.read_csv(f"{RESULTS}/returns.csv", index_col=0, parse_dates=True)
    sens = pd.read_csv(f"{RESULTS}/sens.csv", index_col="cost_bps")
    with open(f"{RESULTS}/summary.json") as fh:
        summary = json.load(fh)

    equity_chart(returns)
    drawdown_chart(returns)
    train_test_chart(summary)
    cost_chart(sens, summary)

    # optional diagnostic charts (need the price data, but only fit one window)
    logp = np.log(load().dropna(axis=1))
    for p_index in [0, 1]:
        one_pair_chart(logp, k=3, p_index=p_index)

    print(f"Charts saved to {RESULTS}/")


if __name__ == "__main__":
    main()
