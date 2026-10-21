"""Run the full pairs-trading experiment and save everything to results/.

Usage:
    python run.py                 # full run (the cost sweep takes a few minutes)
    python run.py --skip-sweep    # reuse results/sens.csv if it already exists

Then run `python make_charts.py` to (re)draw the charts from the saved files.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from data import load
from engine import walk_forward, metrics, bootstrap_sharpe_ci, select_pairs

RESULTS = "results"
TRAIN_END = "2018-12-31"        # fixed in PROTOCOL.md before looking at results
TEST_START = "2019-01-01"
FORM_DAYS = 504                 # formation window (~2 years)
TRADE_DAYS = 126                # trading window (~6 months)
COSTS_BPS = [0, 1, 2, 5, 10, 20]


def split(r):
    """Split a return series into (train, test) using the protocol dates."""
    return r.loc[:TRAIN_END], r.loc[TEST_START:]


def f(x):
    """Convert numpy scalars to plain floats so they can be written to JSON."""
    return float(x)


def cointegration_stability(logp):
    """For every pair selected in a formation window, re-test cointegration in the
    following trading window. Returns one row per (window, pair)."""
    rows = []
    start = 0
    while start + FORM_DAYS + TRADE_DAYS <= len(logp):
        f_end = start + FORM_DAYS
        pairs, _ = select_pairs(logp.iloc[start:f_end])
        idx = logp.index[f_end:f_end + TRADE_DAYS]
        for p in pairs:
            p_after = coint(logp[p["a"]].loc[idx], logp[p["b"]].loc[idx])[1]
            rows.append({"trade_start": idx[0], "a": p["a"], "b": p["b"],
                         "p_before": p["p"], "p_after": p_after})
        start += TRADE_DAYS
    return pd.DataFrame(rows)


def cost_sweep(logp):
    """Re-run the whole walk-forward at several per-trade cost levels.
    Borrow cost (0.5%/yr) is always included, even at 0 bps."""
    rows = []
    for c in COSTS_BPS:
        print(f"  cost sweep: {c} bps ...")
        n, _, _ = walk_forward(logp, cost_bps=c)
        tr, te = split(n)
        rows.append({"cost_bps": c,
                     "sharpe_train": metrics(tr)["sharpe"],
                     "sharpe_test": metrics(te)["sharpe"]})
    return pd.DataFrame(rows).set_index("cost_bps")


def breakevens(sens):
    """Cost level (bps) where each Sharpe curve crosses zero, by linear interpolation.
    np.interp needs increasing x; Sharpe falls as cost rises, so reverse both arrays.
    (If a curve never crosses zero, np.interp clamps to an end point, so check the chart.)"""
    return {col: f(np.interp(0, sens[col].to_numpy()[::-1],
                             sens.index.to_numpy()[::-1]))
            for col in sens.columns}


def main(skip_sweep=False):
    os.makedirs(RESULTS, exist_ok=True)

    # ---- data ----
    prices = load().dropna(axis=1)           # drop tickers missing history
    logp = np.log(prices)

    # ---- main walk-forward run (base case: 5 bps + 0.5% borrow) ----
    net, gross, log = walk_forward(logp)
    net.name, gross.name = "net", "gross"
    pd.DataFrame({"net": net, "gross": gross}).to_csv(f"{RESULTS}/returns.csv")
    log.to_csv(f"{RESULTS}/windows.csv", index=False)

    print(log[["trade_start", "n_sig", "n_pairs"]])
    print("GROSS all:", metrics(gross))
    print("NET   all:", metrics(net))

    # ---- train / test, with bootstrap CIs ----
    summary = {"all": {"gross": metrics(gross), "net": metrics(net)}}
    for label, series in [("gross", gross), ("net", net)]:
        tr, te = split(series)
        summary[label] = {
            "train": metrics(tr), "test": metrics(te),
            "train_ci": list(bootstrap_sharpe_ci(tr)),
            "test_ci": list(bootstrap_sharpe_ci(te)),
        }
        print(f"{label.upper()} TRAIN:", metrics(tr))
        print(f"{label.upper()} TEST: ", metrics(te))
        print(f"{label.upper()} TEST Sharpe 95% CI:", summary[label]["test_ci"])

    # ---- cointegration stability ----
    stab = cointegration_stability(logp)
    stab.to_csv(f"{RESULTS}/stability.csv", index=False)
    summary["stability"] = {
        "n_pairs": int(len(stab)),
        "share_p05": f((stab["p_after"] < 0.05).mean()),
        "share_p10": f((stab["p_after"] < 0.10).mean()),
    }
    print("share still cointegrated (p<0.05):", summary["stability"]["share_p05"])
    print("share still cointegrated (p<0.10):", summary["stability"]["share_p10"])

    # ---- cost sensitivity ----
    sens_path = f"{RESULTS}/sens.csv"
    if skip_sweep and os.path.exists(sens_path):
        print("Reusing", sens_path)
        sens = pd.read_csv(sens_path, index_col="cost_bps")
    else:
        sens = cost_sweep(logp)
        sens.to_csv(sens_path)
    print(sens)
    summary["breakeven_bps"] = breakevens(sens)
    for col, be in summary["breakeven_bps"].items():
        print(col, "breakeven (bps):", round(be, 1))

    # ---- save summary (json can't hold numpy types, so convert) ----
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)):
            return [clean(v) for v in o]
        return f(o) if isinstance(o, (np.floating, float)) else o

    with open(f"{RESULTS}/summary.json", "w") as fh:
        json.dump(clean(summary), fh, indent=2)
    print(f"\nSaved results to {RESULTS}/. Now run: python make_charts.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sweep", action="store_true",
                        help="reuse results/sens.csv instead of re-running the cost sweep")
    args = parser.parse_args()
    main(skip_sweep=args.skip_sweep)
