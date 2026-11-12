"""Moving average crossover: long when the fast average is above the slow one, else cash."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def signal(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """1 when SMA(fast) > SMA(slow) at that day's close, 0 otherwise and during warm-up."""
    if fast >= slow:
        raise ValueError("fast must be shorter than slow")
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    return (f > s).astype(float)


def strategy_returns(prices: pd.Series, fast: int, slow: int, cost: float = 0.001,
                     lag: int = 1, cash_rate: float = 0.0) -> pd.DataFrame:
    """Daily returns of the strategy.

    lag = 1 trades at the close that produced the signal, lag = 2 at the next close.
    cost is charged per unit of turnover (0.001 = 10 basis points each way).
    cash_rate is the yearly rate earned while out of the market.
    """
    if lag < 1:
        raise ValueError("lag below 1 would trade on information from the future")
    ret = prices.pct_change().fillna(0.0)
    pos = signal(prices, fast, slow).shift(lag).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    strat = pos * ret + (1 - pos) * daily_cash - cost * turnover
    return pd.DataFrame({"asset": ret, "position": pos, "strategy": strat})


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((1 - equity / equity.cummax()).max())


def metrics(returns: pd.Series) -> dict:
    n = len(returns)
    total = float((1 + returns).prod())
    sd = returns.std()
    return {
        "cagr": total ** (TRADING_DAYS / n) - 1,
        "vol": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": float(returns.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "max_dd": max_drawdown(returns),
        "total": total - 1,
    }


def trades(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per round trip: entry date, exit date, days held, return after costs."""
    pos = bt["position"].to_numpy()
    grp = np.cumsum(np.diff(pos, prepend=0.0) != 0)
    rows = []
    for _, block in bt[pos > 0].groupby(grp[pos > 0]):
        rows.append(
            {
                "entry": block.index[0],
                "exit": block.index[-1],
                "days": len(block),
                "return": float((1 + block["strategy"]).prod() - 1),
            }
        )
    return pd.DataFrame(rows, columns=["entry", "exit", "days", "return"])


def grid(prices: pd.Series, fasts, slows, start=None, **kw) -> pd.DataFrame:
    """Daily strategy returns for every valid (fast, slow) pair, as columns."""
    out = {}
    for f in fasts:
        for s in slows:
            if f < s:
                out[(f, s)] = strategy_returns(prices, f, s, **kw)["strategy"]
    df = pd.DataFrame(out)
    return df if start is None else df.loc[start:]


def sharpe_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Sharpe ratio of each grid column, laid out as fast (rows) by slow (columns)."""
    sh = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)
    return sh.unstack()


def walk_forward(grid_returns: pd.DataFrame, train_years: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """Each calendar year, trade the pair with the best Sharpe over the previous train_years."""
    years = sorted(set(grid_returns.index.year))
    pieces, picks = [], []
    for y in years[train_years:]:
        train = grid_returns[(grid_returns.index.year >= y - train_years) & (grid_returns.index.year < y)]
        sh = train.mean() / train.std()
        best = sh.idxmax()
        test = grid_returns.loc[grid_returns.index.year == y, best]
        pieces.append(test)
        picks.append({"year": y, "fast": best[0], "slow": best[1],
                      "train_sharpe": float(sh[best] * np.sqrt(TRADING_DAYS)),
                      "test_return": float((1 + test).prod() - 1)})
    return pd.concat(pieces), pd.DataFrame(picks)
