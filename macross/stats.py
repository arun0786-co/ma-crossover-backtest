"""Bootstrap tests: is the strategy better than buy and hold, or is it luck and tuning?"""

import numpy as np

TRADING_DAYS = 252


def stationary_index(rng, n: int, block: float) -> np.ndarray:
    """Index for one stationary bootstrap resample (Politis and Romano 1994)."""
    starts = rng.integers(0, n, n)
    new = rng.random(n) < 1.0 / block
    new[0] = True
    # position inside the current block, then offset from the block's start
    block_id = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    offset = np.arange(n) - first[block_id]
    return (starts[first][block_id] + offset) % n


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return x.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0


def sharpe_diff_ci(strategy, benchmark, n_boot=2000, block=20.0, level=0.95, seed=0):
    """Bootstrap interval for Sharpe(strategy) - Sharpe(benchmark), resampling days jointly."""
    a, b = np.asarray(strategy), np.asarray(benchmark)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = stationary_index(rng, len(a), block)
        diffs[i] = _sharpe(a[idx]) - _sharpe(b[idx])
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    lo, hi = np.quantile(diffs, q)
    return _sharpe(a) - _sharpe(b), float(lo), float(hi), float(np.mean(diffs <= 0))


def reality_check(grid_returns, benchmark, n_boot=2000, block=20.0, seed=0):
    """White (2000) reality check for the best rule in a grid.

    Tests whether the best mean excess return over the benchmark could come from trying many
    rules on the same data. Returns (best mean daily excess, p-value).
    """
    f = np.asarray(grid_returns) - np.asarray(benchmark)[:, None]
    n = len(f)
    mean = f.mean(axis=0)
    stat = np.sqrt(n) * mean.max()
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = stationary_index(rng, n, block)
        boot[i] = np.sqrt(n) * (f[idx].mean(axis=0) - mean).max()
    return float(mean.max()), float(np.mean(boot >= stat))
