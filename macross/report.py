"""Run the crossover study for a config and write a markdown report with figures."""

import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import grid, metrics, sharpe_table, strategy_returns, trades, walk_forward
from .data import load_prices
from .stats import reality_check, sharpe_diff_ci

ACCENT, DARK, GREY, LIGHT = "#b3261e", "#222222", "#8a8a8a", "#d9d9d9"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "text.color": DARK, "axes.labelcolor": DARK, "axes.edgecolor": DARK,
                     "xtick.color": DARK, "ytick.color": DARK})


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def mrow(name, r):
    m = metrics(r)
    return [name, pct(m["cagr"]), pct(m["vol"]), f"{m['sharpe']:.2f}", pct(m["max_dd"]), pct(m["total"], 0)]


MHEAD = ["", "return / yr", "vol / yr", "Sharpe", "max drawdown", "total return"]


def plot_equity(bt, path, title):
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, height_ratios=[2, 1])
    eq_s, eq_a = (1 + bt["strategy"]).cumprod(), (1 + bt["asset"]).cumprod()
    a.fill_between(bt.index, 0, 1, where=bt["position"] > 0, color=LIGHT, alpha=0.5, lw=0,
                   transform=a.get_xaxis_transform(), label="in the market")
    a.plot(eq_a, color=GREY, lw=1.1, label="buy and hold")
    a.plot(eq_s, color=ACCENT, lw=1.3, label="crossover")
    a.set_yscale("log")
    a.set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    a.minorticks_off()
    a.set_ylabel("growth of 1 (log scale)")
    a.set_title(title, loc="left")
    a.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    b.plot(-(1 - eq_a / eq_a.cummax()) * 100, color=GREY, lw=0.9)
    b.plot(-(1 - eq_s / eq_s.cummax()) * 100, color=ACCENT, lw=1.1)
    b.set_ylabel("drawdown (%)")
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(sh, bench_sharpe, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = sh.to_numpy(dtype=float)
    span = np.nanmax(np.abs(data - bench_sharpe))
    im = ax.imshow(data, cmap="RdGy_r", vmin=bench_sharpe - span, vmax=bench_sharpe + span, aspect="auto")
    ax.set_xticks(range(len(sh.columns)), sh.columns)
    ax.set_yticks(range(len(sh.index)), sh.index)
    ax.set_xlabel("slow average (days)")
    ax.set_ylabel("fast average (days)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8, color=DARK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Sharpe ratio by window pair (buy and hold = {bench_sharpe:.2f}, red is better)",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(path)
    plt.close(fig)


def plot_walk(curves: dict, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [ACCENT, DARK, GREY]
    for (name, r), c in zip(curves.items(), colors):
        ax.plot((1 + r).cumprod(), color=c, lw=1.2, label=name)
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4], ["1", "1.5", "2", "3", "4"])
    ax.minorticks_off()
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title("out of sample: pair re-chosen every year from the previous years", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def run(config_path="config.toml", out_dir="results", refresh=False, log=print):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset, others = cfg["asset"], cfg.get("others", [])
    fast, slow, cost = cfg["fast"], cfg["slow"], cfg["cost"]
    fasts, slows, seed = cfg["fasts"], cfg["slows"], cfg.get("seed", 0)
    block, n_boot = cfg.get("block", 20), cfg.get("n_boot", 2000)

    prices = load_prices([asset] + others, cfg["start"], cfg["end"], refresh=refresh)
    p = prices[asset]
    warm = max(slows) + 2
    first = p.index[warm]  # every rule is live from here, so all are compared on the same days
    log(f"{len(p)} days, {p.index[0].date()} to {p.index[-1].date()}, evaluated from {first.date()}")

    bt = strategy_returns(p, fast, slow, cost=cost).loc[first:]
    tr = trades(bt)
    md = ["# Moving average crossover on " + asset, "",
          f"Long when the {fast}-day average is above the {slow}-day average, otherwise in cash at "
          f"{pct(cfg.get('cash_rate', 0.0))}. Signal at the close, trade at that close, "
          f"{cost * 1e4:.0f} basis points per trade. Daily closes {p.index[0].date()} to "
          f"{p.index[-1].date()}, results from {first.date()} ({len(bt)} days) so that every "
          f"window pair has finished warming up.", "",
          f"## {fast}/{slow} against buy and hold", "",
          table(MHEAD, [mrow(f"crossover {fast}/{slow}", bt["strategy"]), mrow("buy and hold", bt["asset"])]), ""]
    wins = tr["return"] > 0
    md += [f"{len(tr)} round trips, {int(wins.sum())} profitable. Mean holding period "
           f"{tr['days'].mean():.0f} days. Best trade {pct(tr['return'].max())}, worst "
           f"{pct(tr['return'].min())}. In the market {pct(bt['position'].mean())} of days.", ""]
    d, lo, hi, _ = sharpe_diff_ci(bt["strategy"], bt["asset"], n_boot=n_boot, block=block, seed=seed)
    md += [f"Sharpe difference (crossover minus buy and hold): {d:+.2f}, 95% bootstrap interval "
           f"{lo:+.2f} to {hi:+.2f}.", "", "![equity](equity.png)", ""]
    plot_equity(bt, out / "equity.png", f"{asset}: {fast}/{slow} crossover against buy and hold")

    md += ["## Costs, execution delay and cash interest", ""]
    variants = {
        "no costs": dict(cost=0.0),
        f"{cost * 1e4:.0f} bp (base case)": dict(cost=cost),
        "25 bp": dict(cost=0.0025),
        "trade one day later": dict(cost=cost, lag=2),
        "cash earns 6% a year": dict(cost=cost, cash_rate=0.06),
    }
    md += [table(MHEAD, [mrow(k, strategy_returns(p, fast, slow, **v).loc[first:, "strategy"])
                         for k, v in variants.items()]), ""]

    md += ["## By period", ""]
    rows = []
    edges = pd.date_range(first, p.index[-1], periods=5)
    for a, b in zip(edges[:-1], edges[1:]):
        seg = bt.loc[a:b]
        ms, ma = metrics(seg["strategy"]), metrics(seg["asset"])
        rows.append([f"{seg.index[0].date()} to {seg.index[-1].date()}", pct(ms["cagr"]), pct(ma["cagr"]),
                     pct(ms["max_dd"]), pct(ma["max_dd"])])
    md += [table(["period", "crossover / yr", "buy and hold / yr", "crossover max dd", "buy and hold max dd"], rows), ""]

    # the whole grid
    g = grid(p, fasts, slows, start=first, cost=cost)
    sh = sharpe_table(g)
    bench = metrics(bt["asset"])["sharpe"]
    plot_heatmap(sh, bench, out / "heatmap.png")
    best = (g.mean() / g.std()).idxmax()
    better = int((sh.to_numpy() > bench).sum())
    _, p_rc = reality_check(g.to_numpy(), bt["asset"].to_numpy(), n_boot=n_boot, block=block, seed=seed)
    bm = metrics(g[best])
    md += [f"## All {g.shape[1]} window pairs", "",
           f"{better} of {g.shape[1]} pairs have a higher Sharpe ratio than buy and hold ({bench:.2f}). "
           f"The best in hindsight is {best[0]}/{best[1]} with Sharpe {bm['sharpe']:.2f} and "
           f"{pct(bm['cagr'])} a year. White's reality check asks whether the best pair's excess "
           f"return over buy and hold survives the fact that {g.shape[1]} pairs were tried: "
           f"p = {p_rc:.3f}.", "", "![heatmap](heatmap.png)", ""]

    # walk forward
    ty = cfg.get("train_years", 5)
    oos, picks = walk_forward(g, ty)
    fixed = bt.loc[oos.index, "strategy"]
    hold = bt.loc[oos.index, "asset"]
    plot_walk({"walk forward": oos, f"fixed {fast}/{slow}": fixed, "buy and hold": hold}, out / "walk_forward.png")
    d2, lo2, hi2, _ = sharpe_diff_ci(oos, hold, n_boot=n_boot, block=block, seed=seed)
    md += ["## Walk forward", "",
           f"Each year the pair with the best Sharpe ratio over the previous {ty} years is traded "
           f"for the next year. Out of sample period {oos.index[0].date()} to {oos.index[-1].date()}.", "",
           table(MHEAD, [mrow("walk forward", oos), mrow(f"fixed {fast}/{slow}", fixed), mrow("buy and hold", hold)]), "",
           f"Sharpe difference, walk forward minus buy and hold: {d2:+.2f} (95% interval {lo2:+.2f} to {hi2:+.2f}).", "",
           table(["year", "pair chosen", "Sharpe in training", "return that year", "buy and hold that year"],
                 [[r.year, f"{r.fast}/{r.slow}", f"{r.train_sharpe:.2f}", pct(r.test_return),
                   pct((1 + hold[hold.index.year == r.year]).prod() - 1)] for r in picks.itertuples()]), "",
           "![walk forward](walk_forward.png)", ""]

    if others:
        md += [f"## The same {fast}/{slow} rule on other assets", ""]
        rows = []
        n_sharpe = n_dd = 0
        for t in [asset] + others:
            b = strategy_returns(prices[t], fast, slow, cost=cost).loc[first:]
            ms, ma = metrics(b["strategy"]), metrics(b["asset"])
            n_sharpe += ms["sharpe"] > ma["sharpe"]
            n_dd += ms["max_dd"] < ma["max_dd"]
            rows.append([t, pct(ms["cagr"]), pct(ma["cagr"]), f"{ms['sharpe']:.2f}", f"{ma['sharpe']:.2f}",
                         pct(ms["max_dd"]), pct(ma["max_dd"])])
        md += [table(["asset", "crossover / yr", "hold / yr", "crossover Sharpe", "hold Sharpe",
                      "crossover max dd", "hold max dd"], rows), "",
               f"Higher Sharpe than buy and hold on {n_sharpe} of {len(rows)} assets, smaller maximum "
               f"drawdown on {n_dd} of {len(rows)}.", ""]

    (out / "report.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="macross")
    ap.add_argument("config", nargs="?", default="config.toml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    run(a.config, a.out, a.refresh)
