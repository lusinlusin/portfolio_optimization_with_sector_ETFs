"""Rolling-window, out-of-sample backtest driver.

For each estimation-window length ``L in {6, 12, 24}`` we run a *separate*
rolling backtest.  At each test month ``t``:

1. estimate ``mu_hat`` / ``Q_hat`` from the prior ``L`` months  (rows t-L .. t-1),
2. solve every strategy to get weights ``x_t``,
3. hold one month: realized **gross** return = ``x_t . r_t``  (r_t = row t),
4. charge transaction cost on turnover vs. last month's weights,
5. roll forward one month.

The single most important correctness invariant: month-``t``
returns are used **only to evaluate** the portfolio chosen at ``t`` — never to
estimate ``mu_hat`` / ``Q_hat``.  That invariant lives in exactly one place
here: the estimation slice is ``R.iloc[t-L:t]`` and the evaluation row is
``R.iloc[t]``.  Keep them disjoint and there is no look-ahead.

The loop is **model-agnostic**: it iterates a dict of ``strategy(mu, Q, L) ->
weights`` callables, so adding/removing a model never touches the loop.  The
three robust variants are expanded into one strategy per uncertainty level
(90 / 95 / 99), giving 10 strategies that map onto the study's 6 models.

Results are cached to ``results/backtest_L{L}.pkl`` keyed by strategy so the
figures/tables can be regenerated without re-running the solvers.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import models as M
except ImportError:  # allows running the file directly (e.g. an IDE "Run" button)
    import models as M

# --------------------------------------------------------------------------- #
# Configuration (all design choices live here; see README.md)
# --------------------------------------------------------------------------- #
WINDOWS = [6, 12, 24]              # estimation-window lengths L (months)
ROBUST_LEVELS = [0.90, 0.95, 0.99]  # uncertainty levels for robust MVO
RISK_AVERSION = 2.5                # lambda for MVO / robust / BL (design choice)
TC_RATE = 0.001                    # transaction cost c = 10 bps
DELTA_MKT = 2.5                    # BL market risk aversion (reverse optimization)
TAU = 0.05                         # BL prior scaling
X0_ZERO = True                     # x_0 = 0 (charge full cost to build the first
                                   # position). If False, x_0 = x_1 (no first-month TC).

DATA_PATH = "data/monthly_returns.csv"
RESULTS_DIR = "results"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_returns(path: str = DATA_PATH,
                 assets: list[str] | None = None) -> pd.DataFrame:
    """Load the monthly return panel, ordered by ASSETS, with complete rows only."""
    if assets is None:
        assets = M.ASSETS
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df[assets]                       # enforce canonical column order
    df = df.dropna(how="any").sort_index()  # need all assets present each month
    if df.empty:
        raise ValueError(f"No complete return rows in {path}; did the download succeed?")
    return df


# --------------------------------------------------------------------------- #
# Black-Litterman view (a single illustrative view)
# --------------------------------------------------------------------------- #
def build_bl_view(assets: list[str], long_asset: str = "XLK",
                  short_asset: str = "XLU", spread: float = 0.0025):
    """One absolute-vs-relative view: long_asset outperforms short_asset by `spread`.

    Returns (w_mkt, P, q). Market proxy w_mkt is equal-weight (a design choice;
    swap in real SPDR cap weights if available).
    """
    n = len(assets)
    w_mkt = np.full(n, 1.0 / n)
    P = np.zeros((1, n))
    P[0, assets.index(long_asset)] = 1.0
    P[0, assets.index(short_asset)] = -1.0
    q = np.array([spread])
    return w_mkt, P, q


def build_strategies(assets: list[str]) -> dict:
    """Map strategy name -> callable(mu, Q, L) -> weights.

    Robust models are expanded to one callable per uncertainty level.
    """
    n = len(assets)
    w_mkt, P, q = build_bl_view(assets)

    strat: dict = {
        "mvo": lambda mu, Q, L: M.mean_variance(mu, Q, RISK_AVERSION),
        "risk_parity": lambda mu, Q, L: M.risk_parity(Q),
        "equal_weight": lambda mu, Q, L: M.equal_weight(n),
        "black_litterman": lambda mu, Q, L: M.black_litterman(
            Q, w_mkt, P, q, RISK_AVERSION, tau=TAU, delta_mkt=DELTA_MKT),
    }
    for lvl in ROBUST_LEVELS:
        tag = int(round(lvl * 100))
        # default-arg binding to capture lvl by value, not by reference
        strat[f"robust_box_{tag}"] = (
            lambda mu, Q, L, lvl=lvl: M.robust_box(mu, Q, RISK_AVERSION, L, level=lvl))
        strat[f"robust_ellipsoidal_{tag}"] = (
            lambda mu, Q, L, lvl=lvl: M.robust_ellipsoidal(mu, Q, RISK_AVERSION, L, level=lvl))
    return strat


# --------------------------------------------------------------------------- #
# Estimation (the look-ahead invariant lives here)
# --------------------------------------------------------------------------- #
def estimate(window: pd.DataFrame):
    """Sample mean and covariance over an estimation window (rows t-L .. t-1)."""
    mu = window.mean(axis=0).to_numpy()
    Q = window.cov(ddof=1).to_numpy()    # 1/(L-1)
    return mu, Q


# --------------------------------------------------------------------------- #
# Core rolling loop
# --------------------------------------------------------------------------- #
def run_backtest(returns: pd.DataFrame, L: int,
                 strategies: dict | None = None) -> dict:
    """Run all strategies for a single window length L.

    Returns ``{strategy: {"records": DataFrame, "weights": DataFrame}}`` where
    ``records`` is indexed by test-month date with columns
    ``[gross, turnover, tc, net]`` and ``weights`` is (date x assets).
    """
    if strategies is None:
        strategies = build_strategies(list(returns.columns))

    assets = list(returns.columns)
    R = returns.to_numpy()
    dates = returns.index
    T, n = R.shape
    if T <= L:
        raise ValueError(f"Need more than L={L} months; only have {T}.")

    rec = {s: [] for s in strategies}              # per-strategy list of dicts
    wts = {s: [] for s in strategies}              # per-strategy list of weight rows
    prev_w = {s: (np.zeros(n) if X0_ZERO else None) for s in strategies}

    for t in range(L, T):
        window = returns.iloc[t - L:t]             # ESTIMATION: rows t-L .. t-1
        r_t = R[t]                                 # EVALUATION: row t (out of sample)
        mu, Q = estimate(window)
        date_t = dates[t]

        for s, fn in strategies.items():
            x = fn(mu, Q, L)
            x = np.asarray(x, dtype=float).ravel()

            pw = prev_w[s]
            if pw is None:                         # X0_ZERO=False: first month free
                pw = x
            turnover = float(np.abs(x - pw).sum())  # ||x_t - x_{t-1}||_1
            tc = TC_RATE * turnover
            gross = float(x @ r_t)
            net = gross - tc

            rec[s].append({"date": date_t, "gross": gross,
                           "turnover": turnover, "tc": tc, "net": net})
            wts[s].append(pd.Series(x, index=assets, name=date_t))
            prev_w[s] = x

    out = {}
    for s in strategies:
        records = pd.DataFrame(rec[s]).set_index("date")
        weights = pd.DataFrame(wts[s])
        out[s] = {"records": records, "weights": weights}
    return out


# --------------------------------------------------------------------------- #
# Driver over all windows + caching
# --------------------------------------------------------------------------- #
def run_all(returns: pd.DataFrame, windows: list[int] | None = None,
            outdir: str = RESULTS_DIR) -> dict:
    """Run every window length and cache each to results/backtest_L{L}.pkl."""
    if windows is None:
        windows = WINDOWS
    Path(outdir).mkdir(exist_ok=True)
    all_results = {}
    for L in windows:
        res = run_backtest(returns, L)
        all_results[L] = res
        with open(Path(outdir) / f"backtest_L{L}.pkl", "wb") as f:
            pickle.dump(res, f)
    return all_results


def load_results(L: int, outdir: str = RESULTS_DIR) -> dict:
    """Reload a cached window result."""
    with open(Path(outdir) / f"backtest_L{L}.pkl", "rb") as f:
        return pickle.load(f)


def summarize(all_results: dict) -> pd.DataFrame:
    """Quick mean-net-return / avg-turnover table per (strategy, L) for a sanity glance."""
    rows = []
    for L, res in all_results.items():
        for s, d in res.items():
            r = d["records"]
            rows.append({"L": L, "strategy": s,
                         "mean_net": r["net"].mean(),
                         "std_net": r["net"].std(ddof=1),
                         "avg_turnover": r["turnover"].mean(),
                         "n_months": len(r)})
    return pd.DataFrame(rows).set_index(["L", "strategy"]).sort_index()


if __name__ == "__main__":
    rets = load_returns()
    results = run_all(rets)
    print(summarize(results).round(4).to_string())
    print(f"\nCached to {RESULTS_DIR}/backtest_L{{{','.join(map(str, WINDOWS))}}}.pkl")
