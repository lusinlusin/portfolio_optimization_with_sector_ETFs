"""Shared fixtures: deterministic synthetic returns so tests need no network.

The panel mimics the real one (11 assets, monthly scale) but is generated from a
fixed seed, so every test is reproducible and the solvers behave deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import models as M


@pytest.fixture(scope="session")
def synthetic_returns() -> pd.DataFrame:
    """60 months x 11 assets of plausible monthly returns.

    The covariance is deliberately *well-conditioned* (a correlation matrix shrunk
    toward the identity, scaled by realistic monthly vols) so the convex solvers
    converge tightly and deterministically — keeping tests free of interior-point
    tolerance noise.
    """
    rng = np.random.default_rng(0)
    n = len(M.ASSETS)
    T = 60
    vols = rng.uniform(0.03, 0.06, size=n)          # monthly vols 3-6%
    B = rng.normal(size=(n, n))
    C = B @ B.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)                           # -> correlation matrix
    C = 0.6 * C + 0.4 * np.eye(n)                    # shrink toward I (well-conditioned)
    cov = np.outer(vols, vols) * C
    mean = rng.normal(0.008, 0.003, size=n)
    R = rng.multivariate_normal(mean, cov, size=T)
    idx = pd.date_range("2019-01-31", periods=T, freq="ME")
    return pd.DataFrame(R, index=idx, columns=M.ASSETS)


@pytest.fixture(scope="session")
def window_moments(synthetic_returns):
    """(mu, Q, L) estimated from the first 12 months — a single estimation window."""
    L = 12
    win = synthetic_returns.iloc[:L]
    mu = win.mean(axis=0).to_numpy()
    Q = win.cov(ddof=1).to_numpy()
    return mu, Q, L


@pytest.fixture(scope="session")
def bl_view(synthetic_returns):
    """(w_mkt, P, q): one view, XLK outperforms XLU by 0.25%/month (the illustrative view)."""
    from src import backtest as BT
    return BT.build_bl_view(list(synthetic_returns.columns))
