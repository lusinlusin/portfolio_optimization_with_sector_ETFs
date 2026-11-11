"""Out-of-sample performance metrics and summary tables.

All metrics operate on the per-strategy ``records`` DataFrames produced by
``backtest.run_backtest`` (columns: ``gross, turnover, tc, net`` indexed by
test-month date).  Definitions:

* mean OOS return          : ``R_bar = mean(R_net)``
* std dev                  : sample std with ``1/(T-1)``  (ddof=1)
* Sharpe (rf ignored)      : ``R_bar / sigma``            (monthly)
* Sortino                  : ``R_bar / downside_deviation``
* max drawdown             : on cumulative wealth ``W_t = W_{t-1}(1+R_net_t)``, W_0=1
* average turnover         : ``mean(Turnover_t)``

All figures are monthly.  Annualized Sharpe/Sortino
(``* sqrt(12)``) are added as extra columns for convenience and clearly labeled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Scalar metrics on a return series
# --------------------------------------------------------------------------- #
def cumulative_wealth(net: pd.Series) -> pd.Series:
    """W_t = W_{t-1} (1 + R_net_t), W_0 = 1 (so the series starts just above 1)."""
    return (1.0 + net).cumprod()


def max_drawdown(net: pd.Series) -> float:
    """Largest peak-to-trough drop of cumulative wealth, returned as a magnitude.

    e.g. 0.23 means a 23% max drawdown.
    """
    W = cumulative_wealth(net)
    running_max = W.cummax()
    drawdown = W / running_max - 1.0       # <= 0
    return float(-drawdown.min())          # report as positive magnitude


def downside_deviation(net: pd.Series, mar: float = 0.0) -> float:
    """sqrt(mean(min(r - MAR, 0)^2)) over all observations (MAR = 0, rf ignored)."""
    downside = np.minimum(net - mar, 0.0)
    return float(np.sqrt(np.mean(downside ** 2)))


def sharpe(net: pd.Series, annualize: bool = False) -> float:
    """R_bar / sigma (rf ignored). sigma uses ddof=1."""
    sd = net.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    s = net.mean() / sd
    return s * np.sqrt(PERIODS_PER_YEAR) if annualize else s


def sortino(net: pd.Series, mar: float = 0.0, annualize: bool = False) -> float:
    """R_bar / downside_deviation."""
    dd = downside_deviation(net, mar)
    if dd == 0 or np.isnan(dd):
        return np.nan
    s = (net.mean() - mar) / dd
    return s * np.sqrt(PERIODS_PER_YEAR) if annualize else s


def rolling_sharpe(net: pd.Series, window: int = 12) -> pd.Series:
    """Rolling monthly Sharpe over a trailing `window`-month window."""
    mean = net.rolling(window).mean()
    std = net.rolling(window).std(ddof=1)
    return mean / std


# --------------------------------------------------------------------------- #
# Tables across (L, strategy)
# --------------------------------------------------------------------------- #
def performance_table(all_results: dict) -> pd.DataFrame:
    """Full per-(L, strategy) performance summary on NET returns.

    Columns: mean_net, std_net, sharpe, sharpe_ann, sortino, sortino_ann,
             max_drawdown, final_wealth, avg_turnover, n_months.
    """
    rows = []
    for L, res in all_results.items():
        for s, d in res.items():
            net = d["records"]["net"]
            turn = d["records"]["turnover"]
            rows.append({
                "L": L, "strategy": s,
                "mean_net": net.mean(),
                "std_net": net.std(ddof=1),
                "sharpe": sharpe(net),
                "sharpe_ann": sharpe(net, annualize=True),
                "sortino": sortino(net),
                "sortino_ann": sortino(net, annualize=True),
                "max_drawdown": max_drawdown(net),
                "final_wealth": cumulative_wealth(net).iloc[-1],
                "avg_turnover": turn.mean(),
                "n_months": len(net),
            })
    return pd.DataFrame(rows).set_index(["L", "strategy"]).sort_index()


def turnover_table(all_results: dict) -> pd.DataFrame:
    """Turnover summary per (L, strategy): average, median, max, total."""
    rows = []
    for L, res in all_results.items():
        for s, d in res.items():
            turn = d["records"]["turnover"]
            rows.append({
                "L": L, "strategy": s,
                "avg_turnover": turn.mean(),
                "median_turnover": turn.median(),
                "max_turnover": turn.max(),
                "total_turnover": turn.sum(),
            })
    return pd.DataFrame(rows).set_index(["L", "strategy"]).sort_index()


def tc_impact_table(all_results: dict) -> pd.DataFrame:
    """Transaction-cost impact: gross vs net mean return and Sharpe, plus the drag.

    ``mean_drag`` = mean_gross - mean_net = avg monthly TC paid.
    ``sharpe_drag`` = sharpe_gross - sharpe_net.
    """
    rows = []
    for L, res in all_results.items():
        for s, d in res.items():
            r = d["records"]
            gross, net = r["gross"], r["net"]
            rows.append({
                "L": L, "strategy": s,
                "mean_gross": gross.mean(),
                "mean_net": net.mean(),
                "mean_drag": gross.mean() - net.mean(),
                "sharpe_gross": sharpe(gross),
                "sharpe_net": sharpe(net),
                "sharpe_drag": sharpe(gross) - sharpe(net),
                "avg_tc": r["tc"].mean(),
            })
    return pd.DataFrame(rows).set_index(["L", "strategy"]).sort_index()


def discussion_answers(all_results: dict, max_final_wealth: float = 50.0) -> dict:
    """Programmatic answers to the study's comparison questions (NET, per window).

    Strategies that blew up (non-finite or |final wealth| >= max_final_wealth —
    i.e. MVO / ellipsoidal-robust at short windows) are excluded from the
    "best" rankings so they aren't spuriously crowned; they are listed
    separately under ``unstable`` as the instability finding. The written
    discussion should be grounded in these numbers rather than eyeballing.
    """
    perf = performance_table(all_results)
    tc = tc_impact_table(all_results)
    out = {}
    for L in sorted({idx[0] for idx in perf.index}):
        p = perf.loc[L]
        t = tc.loc[L]
        fw = p["final_wealth"]
        stable = p.index[np.isfinite(fw) & (fw.abs() < max_final_wealth)]
        unstable = [s for s in p.index if s not in stable]
        ps, ts = p.loc[stable], t.loc[stable]
        out[L] = {
            "best_sharpe": ps["sharpe"].idxmax(),
            "best_mean_net": ps["mean_net"].idxmax(),
            "highest_turnover": ps["avg_turnover"].idxmax(),
            "most_tc_sensitive": ts["sharpe_drag"].idxmax(),
            "smallest_max_drawdown": ps["max_drawdown"].idxmin(),
            "unstable": unstable,
        }
    return out
