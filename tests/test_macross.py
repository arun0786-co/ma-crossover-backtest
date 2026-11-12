import numpy as np
import pandas as pd
import pytest

from macross.backtest import (grid, max_drawdown, metrics, sharpe_table, signal,
                              strategy_returns, trades, walk_forward)
from macross.stats import reality_check, sharpe_diff_ci, stationary_index


def series(values, start="2020-01-01"):
    return pd.Series(np.asarray(values, float), index=pd.bdate_range(start, periods=len(values)))


def test_signal_by_hand():
    p = series([1, 2, 3, 4, 3, 2, 1, 1, 5])
    # sma2: -, 1.5, 2.5, 3.5, 3.5, 2.5, 1.5, 1, 3 ; sma3: -, -, 2, 3, 3.33, 3, 2, 1.33, 2.33
    assert signal(p, 2, 3).tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 1]


def test_position_lags_the_signal_and_costs_are_charged_on_each_switch():
    p = series([1, 2, 3, 4, 3, 2, 1, 1, 5])
    bt = strategy_returns(p, 2, 3, cost=0.01, lag=1)
    assert bt["position"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    expected = [0, 0, 0, 4 / 3 - 1 - 0.01, 3 / 4 - 1, 2 / 3 - 1, -0.01, 0, 0]
    assert np.allclose(bt["strategy"], expected)
    assert strategy_returns(p, 2, 3, lag=2)["position"].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0]


def test_no_lookahead():
    rng = np.random.default_rng(0)
    p = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))))
    q = p.copy()
    q.iloc[400:] *= 3
    a = strategy_returns(p, 10, 50)
    b = strategy_returns(q, 10, 50)
    assert a.iloc[:400].equals(b.iloc[:400])
    with pytest.raises(ValueError):
        strategy_returns(p, 10, 50, lag=0)
    with pytest.raises(ValueError):
        signal(p, 50, 10)


def test_always_long_in_a_steady_uptrend_matches_buy_and_hold_after_warmup():
    p = series(100 * 1.001 ** np.arange(400))
    bt = strategy_returns(p, 5, 20, cost=0.0)
    assert bt["position"].iloc[20:].eq(1).all()
    assert np.allclose(bt["strategy"].iloc[20:], bt["asset"].iloc[20:])


def test_cash_rate_is_earned_only_when_out():
    p = series(100 * 0.999 ** np.arange(300))
    bt = strategy_returns(p, 5, 20, cash_rate=0.0652)
    assert bt["position"].eq(0).all()
    assert (1 + bt["strategy"]).prod() ** (252 / 300) - 1 == pytest.approx(0.0652)


def test_metrics_and_drawdown():
    r = series([0.1, -0.5, 1.0, 0.0])
    assert max_drawdown(r) == pytest.approx(0.5)
    m = metrics(series([0.01] * 126 + [0.0] * 126))
    assert m["cagr"] == pytest.approx(1.01**126 - 1)
    assert m["total"] == pytest.approx(1.01**126 - 1)


def test_trades_round_trips():
    idx = pd.bdate_range("2020-01-01", periods=7)
    bt = pd.DataFrame({"position": [0, 1, 1, 0, 0, 1, 1],
                       "strategy": [0, 0.1, 0.1, -0.01, 0, -0.2, 0.0]}, index=idx)
    t = trades(bt)
    assert t["days"].tolist() == [2, 2]
    assert np.allclose(t["return"], [0.21, -0.2])


def test_grid_and_walk_forward_use_only_past_years():
    rng = np.random.default_rng(1)
    p = series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 2600))), start="2010-01-01")
    g = grid(p, [5, 20], [20, 50, 100], cost=0.001)
    assert list(g.columns) == [(5, 20), (5, 50), (5, 100), (20, 50), (20, 100)]
    assert sharpe_table(g).shape == (2, 3)
    oos, picks = walk_forward(g, train_years=3)
    assert oos.index[0].year == 2013 and oos.index.is_unique
    # blowing up the last year must not change any earlier pick
    g2 = g.copy()
    g2.loc[g2.index.year == g2.index.year.max()] *= -7
    _, picks2 = walk_forward(g2, train_years=3)
    cols = ["year", "fast", "slow"]
    assert picks[cols].equals(picks2[cols])


def test_stationary_index_block_structure():
    rng = np.random.default_rng(2)
    idx = stationary_index(rng, 50_000, 20)
    assert idx.min() >= 0 and idx.max() < 50_000
    assert np.mean(np.diff(idx) % 50_000 == 1) == pytest.approx(0.95, abs=0.01)


def test_bootstrap_tests_separate_skill_from_noise():
    rng = np.random.default_rng(3)
    bench = rng.normal(0.0003, 0.01, 3000)
    noise = bench[:, None] + rng.normal(0, 0.004, (3000, 30))
    assert reality_check(noise, bench, n_boot=500)[1] > 0.1
    skilled = noise.copy()
    skilled[:, 7] += 0.0008
    assert reality_check(skilled, bench, n_boot=500)[1] < 0.02
    d, lo, hi, _ = sharpe_diff_ci(skilled[:, 7], bench, n_boot=500)
    assert lo > 0 and lo < d < hi
    d, lo, hi, _ = sharpe_diff_ci(noise[:, 0], bench, n_boot=500)
    assert lo < 0 < hi
