"""Performance-metric formulas, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as MT


def test_cumulative_wealth():
    net = pd.Series([0.10, -0.05, 0.20])
    W = MT.cumulative_wealth(net)
    assert np.isclose(W.iloc[-1], 1.10 * 0.95 * 1.20)


def test_max_drawdown():
    # W = 1.10, 0.88, 0.924 -> peak 1.10, trough 0.88 -> DD = 1 - 0.88/1.10 = 0.20
    net = pd.Series([0.10, -0.20, 0.05])
    assert np.isclose(MT.max_drawdown(net), 0.20)


def test_downside_deviation():
    net = pd.Series([0.01, -0.02, 0.03, -0.04])
    expected = np.sqrt(np.mean(np.minimum(net.to_numpy(), 0.0) ** 2))
    assert np.isclose(MT.downside_deviation(net), expected)


def test_sharpe_monthly_and_annualized():
    net = pd.Series([0.01, 0.02, -0.01, 0.03])
    expected = net.mean() / net.std(ddof=1)        # rf ignored, ddof=1
    assert np.isclose(MT.sharpe(net), expected)
    assert np.isclose(MT.sharpe(net, annualize=True), expected * np.sqrt(12))


def test_sortino_uses_downside_deviation():
    net = pd.Series([0.01, 0.02, -0.01, 0.03])
    dd = np.sqrt(np.mean(np.minimum(net.to_numpy(), 0.0) ** 2))
    assert np.isclose(MT.sortino(net), net.mean() / dd)


def test_sharpe_handles_zero_variance():
    assert np.isnan(MT.sharpe(pd.Series([0.01, 0.01, 0.01])))


def test_rolling_sharpe_shape():
    net = pd.Series(np.linspace(-0.02, 0.03, 24))
    rs = MT.rolling_sharpe(net, window=12)
    assert len(rs) == len(net)
    assert rs.iloc[:11].isna().all()               # first window-1 are NaN
    assert rs.iloc[11:].notna().any()
