import numpy as np
import pandas as pd
from engine import zscore_positions, walk_forward, metrics

def make_rw(seed, n=1500, k=8):
    rng = np.random.default_rng(seed)
    data = np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)
    return pd.DataFrame(data,
                        index=pd.date_range("2015-01-01", periods=n, freq="B"),
                        columns=[f"S{i}" for i in range(k)])

def test_signal_truncation():
    rng = np.random.default_rng(1)
    z = pd.Series(rng.normal(size=500) * 2,
                  index=pd.date_range("2020-01-01", periods=500, freq="B"))
    full = zscore_positions(z)
    part = zscore_positions(z.iloc[:300])
    assert (full.iloc[:300] == part).all()

def test_walk_forward_no_lookahead():
    # Chop off the last 300 days. Results for dates both runs cover must match
    # exactly. If they differ, something used future data.
    lp = make_rw(0)
    full, _, _ = walk_forward(lp)
    part, _, _ = walk_forward(lp.iloc[:1200])
    common = part.index
    assert np.allclose(full.loc[common], part.loc[common])

def test_random_walks_have_no_edge():
    sharpes = []
    for s in range(5):
        _, gross, _ = walk_forward(make_rw(s, n=2000, k=10),
                                   cost_bps=0, borrow=0)
        sharpes.append(metrics(gross)["sharpe"])
    assert abs(np.nanmean(np.nan_to_num(sharpes))) < 0.5