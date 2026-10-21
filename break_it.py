"""Deliberately inject a lookahead leak into the engine and check that our
sanity test notices. Nothing in engine.py is edited: we temporarily swap out
engine.select_pairs while this script runs.

The leak: the z-score mean and std are taken from the TRADING window itself
(i.e. from the future) instead of from the formation window. This is the most
common real-world pairs-trading bug.

Usage:  python break_it.py
Expect: honest Sharpe on random walks ~ 0, leaky Sharpe clearly > 1.
"""
import numpy as np
import pandas as pd

import engine
from engine import walk_forward, metrics

TRADE_DAYS = 126
REAL_SELECT = engine.select_pairs          # keep a handle so we can restore it


def make_rw(seed, n=2000, k=10):
    """Independent random walks: there is NOTHING to predict in this data."""
    rng = np.random.default_rng(seed)
    data = np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)
    return pd.DataFrame(data,
                        index=pd.date_range("2015-01-01", periods=n, freq="B"),
                        columns=[f"S{i}" for i in range(k)])


def make_leaky_select(logp):
    """Same pair selection as the real one, but mu/sd come from the FUTURE window."""
    def leaky_select(form, p_thresh=0.05, top_n=5, hl_range=(2, 60)):
        pairs, n_sig = REAL_SELECT(form, p_thresh, top_n, hl_range)
        end = logp.index.get_loc(form.index[-1]) + 1
        future = logp.iloc[end:end + TRADE_DAYS]           # <-- the leak
        for p in pairs:
            s = future[p["a"]] - p["beta"] * future[p["b"]]
            p["mu"], p["sd"] = s.mean(), s.std()
        return pairs, n_sig
    return leaky_select


def avg_gross_sharpe(leaky, seeds=range(5)):
    out = []
    for sd in seeds:
        lp = make_rw(sd)
        engine.select_pairs = make_leaky_select(lp) if leaky else REAL_SELECT
        try:
            _, gross, _ = walk_forward(lp, cost_bps=0, borrow=0)
        finally:
            engine.select_pairs = REAL_SELECT              # always restore
        out.append(metrics(gross)["sharpe"])
    return float(np.nanmean(np.nan_to_num(out)))


if __name__ == "__main__":
    print("Random-walk data, zero costs, mean gross Sharpe over 5 seeds")
    print("  honest engine :", round(avg_gross_sharpe(leaky=False), 2))
    print("  leaky  engine :", round(avg_gross_sharpe(leaky=True), 2))
