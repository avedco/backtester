import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def zscore_positions(z, entry=2.0, exit_=0.5, stop=4.0):
    pos = np.zeros(len(z))
    cur = 0
    for i, v in enumerate(z.to_numpy()):
        if np.isnan(v):
            cur = 0
        elif cur == 0:                       # flat: look for an entry
            if entry < v < stop:
                cur = -1                     # spread too high -> short it
            elif -stop < v < -entry:
                cur = 1                      # spread too low -> long it
        elif abs(v) < exit_ or abs(v) > stop:
            cur = 0                          # reverted, or stop-loss hit
        pos[i] = cur
    return pd.Series(pos, index=z.index)


def half_life(spread):
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    b = sm.OLS(delta, sm.add_constant(lag)).fit().params.iloc[1]
    return -np.log(2) / b if b < 0 else np.inf


def select_pairs(form, p_thresh=0.05, top_n=5, hl_range=(2, 60)):
    n_sig = 0
    found = []
    for a, b in itertools.combinations(form.columns, 2):
        p = coint(form[a], form[b])[1]
        if p >= p_thresh:
            continue
        n_sig += 1
        fit = sm.OLS(form[a], sm.add_constant(form[b])).fit()
        beta = fit.params.iloc[1]
        if beta <= 0:
            continue
        spread = form[a] - beta * form[b]
        hl = half_life(spread)
        if not (hl_range[0] <= hl <= hl_range[1]):
            continue
        found.append(dict(a=a, b=b, p=p, beta=beta,
                          mu=spread.mean(), sd=spread.std(), hl=hl))
    found.sort(key=lambda d: d["p"])
    return found[:top_n], n_sig


def pair_pnl(pos, rA, rB, beta, cost_bps=5, borrow=0.005):
    held = pos.shift(1).fillna(0)            # THE timing rule: trade next day
    gross = held * (rA - beta * rB) / (1 + beta)   # return on gross capital
    turnover = held.diff().abs().fillna(0)
    turnover.iloc[-1] += abs(held.iloc[-1])  # force-close at end of window
    costs = turnover * cost_bps / 1e4 + held.abs() * 0.5 * borrow / 252
    return gross - costs, gross


def walk_forward(logp, form_days=504, trade_days=126, entry=2.0, exit_=0.5,
                 stop=4.0, cost_bps=5, borrow=0.005, top_n=5, p_thresh=0.05):
    rets = logp.diff()
    nets, grosses, log = [], [], []
    start = 0
    while start + form_days + trade_days <= len(logp):
        f_end = start + form_days
        form = logp.iloc[start:f_end]                 # past only
        idx = logp.index[f_end:f_end + trade_days]    # future of the formation
        pairs, n_sig = select_pairs(form, p_thresh, top_n)

        net = pd.Series(0.0, index=idx)
        gross = pd.Series(0.0, index=idx)
        for p in pairs:
            spread = logp[p["a"]] - p["beta"] * logp[p["b"]]
            z = ((spread - p["mu"]) / p["sd"]).loc[idx]   # FIXED formation stats
            pos = zscore_positions(z, entry, exit_, stop)
            n, g = pair_pnl(pos, rets[p["a"]].loc[idx], rets[p["b"]].loc[idx],
                            p["beta"], cost_bps, borrow)
            net += n / top_n                               # capital split evenly
            gross += g / top_n
        nets.append(net)
        grosses.append(gross)
        log.append(dict(trade_start=idx[0], n_sig=n_sig, n_pairs=len(pairs),
                        pairs=[(p["a"], p["b"]) for p in pairs]))
        start += trade_days
    return pd.concat(nets), pd.concat(grosses), pd.DataFrame(log)


def metrics(r):
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    return {"ann_ret": ann, "vol": vol,
            "sharpe": ann / vol if vol > 0 else np.nan,
            "max_dd": (eq / eq.cummax() - 1).min()}

def bootstrap_sharpe_ci(r, n=2000, block=10, seed=0):
    """95% CI for Sharpe using block bootstrap (blocks keep autocorrelation)."""
    rng = np.random.default_rng(seed)
    x = r.to_numpy()
    T = len(x)
    k = int(np.ceil(T / block))
    out = []
    for _ in range(n):
        starts = rng.integers(0, T - block + 1, k)
        s = np.concatenate([x[i:i + block] for i in starts])[:T]
        out.append(s.mean() / s.std() * np.sqrt(252))
    return np.percentile(out, [2.5, 97.5])