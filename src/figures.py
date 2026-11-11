"""Figures for the sector-ETF backtest.

Figure types:
  1. cumulative wealth plots
  2. rolling Sharpe plots
  3. portfolio-weight heatmaps
  4. efficient-frontier comparison

Each ``plot_*`` function builds one matplotlib Figure (and can save it).
``save_all_figures`` regenerates the full set for every window length from a
cached/passed ``all_results`` dict.  Uses the non-interactive Agg backend so it
runs headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless; safe to import anywhere
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import metrics as MT
    from . import models as M
except ImportError:  # allows running the file directly (e.g. an IDE "Run" button)
    import metrics as MT
    import models as M

FIG_DIR = "figures"


def _ensure(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(exist_ok=True)
    return p


def stable_strategies(results: dict, max_final_wealth: float = 50.0) -> list[str]:
    """Strategies whose net cumulative wealth stays finite and < max_final_wealth.

    Used to drop the blown-up MVO / ellipsoidal-robust series (which explode to
    ~1e37 at short windows) so the comparison plots stay readable.  The blow-up
    itself is documented by ``plot_instability`` and discussed in README.md.
    """
    keep = []
    for s, d in results.items():
        W = MT.cumulative_wealth(d["records"]["net"])
        if np.isfinite(W.to_numpy()).all() and abs(W.iloc[-1]) < max_final_wealth:
            keep.append(s)
    return keep


def unstable_strategies(results: dict, max_final_wealth: float = 50.0) -> list[str]:
    """Complement of stable_strategies (the ones that blew up)."""
    stable = set(stable_strategies(results, max_final_wealth))
    return [s for s in results if s not in stable]


# --------------------------------------------------------------------------- #
# 1. Cumulative wealth
# --------------------------------------------------------------------------- #
def plot_cumulative_wealth(results: dict, L: int, strategies: list[str] | None = None,
                           exclude_unstable: bool = True, outdir: str | None = FIG_DIR):
    """Cumulative wealth W_t for each strategy in one window length L.

    By default drops strategies that blow up (MVO / ellipsoidal-robust at short
    windows) so the readable strategies aren't flattened; the title notes the
    dropped ones. Pass ``exclude_unstable=False`` to plot everything.
    """
    if strategies is None:
        strategies = stable_strategies(results) if exclude_unstable else list(results.keys())
    dropped = [s for s in results if s not in strategies] if exclude_unstable else []
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in strategies:
        net = results[s]["records"]["net"]
        W = MT.cumulative_wealth(net)
        ax.plot(W.index, W.values, label=s, lw=1.5)
    title = f"Cumulative wealth (L = {L} months, net of TC)"
    if dropped:
        title += f"\n(excluded as unstable: {', '.join(dropped)})"
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Wealth (W_0 = 1)")
    ax.set_xlabel("Date")
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if outdir:
        fig.savefig(_ensure(outdir) / f"cumwealth_L{L}.png", dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# 2. Rolling Sharpe
# --------------------------------------------------------------------------- #
def plot_rolling_sharpe(results: dict, L: int, window: int = 12,
                        strategies: list[str] | None = None,
                        exclude_unstable: bool = True,
                        outdir: str | None = FIG_DIR):
    """Trailing `window`-month rolling Sharpe per strategy."""
    if strategies is None:
        strategies = stable_strategies(results) if exclude_unstable else list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in strategies:
        net = results[s]["records"]["net"]
        rs = MT.rolling_sharpe(net, window)
        ax.plot(rs.index, rs.values, label=s, lw=1.3)
    ax.set_title(f"Rolling {window}-month Sharpe (L = {L})")
    ax.set_ylabel("Sharpe (monthly, rf = 0)")
    ax.set_xlabel("Date")
    ax.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if outdir:
        fig.savefig(_ensure(outdir) / f"rolling_sharpe_L{L}.png", dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# 3. Portfolio-weight heatmap
# --------------------------------------------------------------------------- #
def plot_weight_heatmap(results: dict, L: int, strategy: str,
                        outdir: str | None = FIG_DIR):
    """Heatmap of weights (assets x time) for a single strategy."""
    W = results[strategy]["weights"]            # date x assets
    data = W.T                                   # assets x date
    fig, ax = plt.subplots(figsize=(11, 5))
    vmax = np.nanmax(np.abs(data.values))
    im = ax.imshow(data.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    # sparse x ticks (dates)
    ncol = data.shape[1]
    step = max(1, ncol // 12)
    ax.set_xticks(range(0, ncol, step))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in data.columns[::step]],
                       rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Portfolio weights — {strategy} (L = {L})")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    if outdir:
        fig.savefig(_ensure(outdir) / f"weights_{strategy}_L{L}.png", dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# 4. Efficient-frontier comparison
# --------------------------------------------------------------------------- #
def plot_efficient_frontier(results: dict, returns: pd.DataFrame, L: int,
                            strategies: list[str] | None = None,
                            n_frontier: int = 40, outdir: str | None = FIG_DIR):
    """Risk-return picture: realized (std, mean) of each strategy + an ex-ante
    MVO frontier built from the *last* estimation window's sample moments.

    The frontier is the locus traced by sweeping the risk-aversion lambda in the
    plain MVO (no-short for a clean bounded frontier); each strategy is plotted
    as its realized out-of-sample (sigma, mean) point for comparison.
    """
    if strategies is None:
        strategies = stable_strategies(results)
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- realized OOS points per strategy ---
    for s in strategies:
        net = results[s]["records"]["net"]
        ax.scatter(net.std(ddof=1), net.mean(), s=45, zorder=3)
        ax.annotate(s, (net.std(ddof=1), net.mean()), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")

    # --- ex-ante MVO frontier from the last window's sample moments ---
    window = returns.iloc[-L:]
    mu = window.mean(axis=0).to_numpy()
    Q = window.cov(ddof=1).to_numpy()
    fr_risk, fr_ret = [], []
    for lam in np.logspace(-1, 2.5, n_frontier):
        try:
            x = M.mean_variance(mu, Q, lam, allow_short=False)
            fr_ret.append(float(mu @ x))
            fr_risk.append(float(np.sqrt(x @ Q @ x)))
        except Exception:
            continue
    order = np.argsort(fr_risk)
    ax.plot(np.array(fr_risk)[order], np.array(fr_ret)[order],
            "k-", lw=1.2, alpha=0.7, label="ex-ante MVO frontier (last window, no-short)")

    ax.set_title(f"Risk-return: realized OOS points vs ex-ante frontier (L = {L})")
    ax.set_xlabel("Std dev (monthly)")
    ax.set_ylabel("Mean return (monthly)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if outdir:
        fig.savefig(_ensure(outdir) / f"frontier_L{L}.png", dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# 5. MVO instability (documents the blow-up finding)
# --------------------------------------------------------------------------- #
def plot_instability(results: dict, L: int, outdir: str | None = FIG_DIR):
    """Log-scale average |weight| (gross leverage proxy) per strategy.

    Makes the MVO / ellipsoidal-robust blow-up at short windows legible without
    flattening the stable strategies. Bars on a log axis -> orders-of-magnitude
    difference is obvious.
    """
    names, lev = [], []
    for s, d in results.items():
        W = d["weights"]
        names.append(s)
        lev.append(float(W.abs().sum(axis=1).mean()))   # avg gross leverage ||x||_1
    order = np.argsort(lev)
    names = [names[i] for i in order]
    lev = [lev[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, lev, color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("avg gross leverage  ||x||_1  (log scale)")
    ax.axvline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="fully invested (=1)")
    ax.set_title(f"Average gross leverage by strategy (L = {L})\nMVO / ellipsoidal blow up at short windows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if outdir:
        fig.savefig(_ensure(outdir) / f"instability_L{L}.png", dpi=150)
    return fig


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def save_all_figures(all_results: dict, returns: pd.DataFrame,
                     outdir: str = FIG_DIR, heatmap_strategies: list[str] | None = None):
    """Regenerate every required figure for every window length."""
    _ensure(outdir)
    if heatmap_strategies is None:
        heatmap_strategies = ["mvo", "risk_parity", "black_litterman", "equal_weight"]
    paths = []
    for L, res in all_results.items():
        plot_cumulative_wealth(res, L, outdir=outdir)
        plot_rolling_sharpe(res, L, outdir=outdir)
        plot_efficient_frontier(res, returns, L, outdir=outdir)
        plot_instability(res, L, outdir=outdir)
        for s in heatmap_strategies:
            if s in res:
                plot_weight_heatmap(res, L, s, outdir=outdir)
        plt.close("all")
        paths.append(L)
    return outdir
