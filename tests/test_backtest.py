"""Backtest-level invariants: no look-ahead, and the transaction-cost identity."""

from __future__ import annotations

import numpy as np

from src import models as M
from src import backtest as BT


def _stable_strategies(assets):
    """A small, deterministic, non-exploding strategy set (keeps tests fast)."""
    w_mkt, P, q = BT.build_bl_view(assets)
    return {
        "equal_weight": lambda mu, Q, L: M.equal_weight(len(mu)),
        "risk_parity": lambda mu, Q, L: M.risk_parity(Q),
        "black_litterman": lambda mu, Q, L: M.black_litterman(Q, w_mkt, P, q, 2.5),
    }


def test_no_look_ahead(synthetic_returns):
    """Perturbing a future month's return must not change any earlier weight.

    Weights at test month t are estimated from rows t-L .. t-1, so row k can only
    enter the *estimation* of months t in (k, k+L]; for every t <= k the weights
    are independent of row k. Row k is, however, the *evaluation* row for month k,
    so the realized gross return at k must change. That is exactly the look-ahead
    invariant the study relies on.
    """
    L = 12
    assets = list(synthetic_returns.columns)
    strategies = _stable_strategies(assets)

    base = BT.run_backtest(synthetic_returns, L, strategies)

    k = 30
    shocked = synthetic_returns.copy()
    shocked.iloc[k] = shocked.iloc[k] + 5.0          # huge shock to a single month
    pert = BT.run_backtest(shocked, L, strategies)

    date_k = synthetic_returns.index[k]
    # row k enters estimation only for test months in (k, k+L]; those weights MUST move.
    later = synthetic_returns.index[k + 1: k + 1 + L]
    for s in strategies:
        wb = base[s]["weights"]
        wp = pert[s]["weights"]

        # (a) no leakage: weights at test months t <= k are unchanged.
        # atol=1e-3 absorbs interior-point solver noise (~1e-5 on identical inputs)
        # while still catching a real look-ahead bug, which would move weights by
        # O(0.1-1) because the +5 shock blows up the estimated mu / Q (see (b)).
        early = wb.index[wb.index <= date_k]
        np.testing.assert_allclose(
            wb.loc[early].to_numpy(), wp.loc[early].to_numpy(), atol=1e-3, rtol=0.0,
            err_msg=f"{s}: weights at/before month k changed (look-ahead!)")

        # (b) the shock DOES propagate where it should: at least one later month's
        # weights move materially (guards against an off-by-one that drops row k).
        later_in = [d for d in later if d in wp.index]
        if later_in and s != "equal_weight":          # equal_weight ignores the data
            moved = np.abs(wb.loc[later_in].to_numpy()
                           - wp.loc[later_in].to_numpy()).max()
            assert moved > 1e-2, f"{s}: shock did not reach the windows that use row k"

        # (c) the realized return at month k *should* change (row k is the eval row).
        assert not np.isclose(base[s]["records"].loc[date_k, "gross"],
                              pert[s]["records"].loc[date_k, "gross"])


def test_transaction_cost_identity(synthetic_returns):
    """net == gross - tc, tc == c * turnover, turnover == ||x_t - x_{t-1}||_1 (x_0=0)."""
    L = 12
    strategies = {"risk_parity": lambda mu, Q, L: M.risk_parity(Q)}
    res = BT.run_backtest(synthetic_returns, L, strategies)
    rec = res["risk_parity"]["records"]
    W = res["risk_parity"]["weights"].to_numpy()

    np.testing.assert_allclose(rec["net"], rec["gross"] - rec["tc"], atol=1e-12)
    np.testing.assert_allclose(rec["tc"], BT.TC_RATE * rec["turnover"], atol=1e-15)

    assert BT.X0_ZERO, "this test assumes the x_0 = 0 convention"
    prev = np.zeros(W.shape[1])
    expected = []
    for row in W:
        expected.append(np.abs(row - prev).sum())
        prev = row
    np.testing.assert_allclose(rec["turnover"].to_numpy(), expected, atol=1e-10)


def test_estimation_excludes_evaluation_row(synthetic_returns):
    """The estimation window for month t never contains row t (disjointness)."""
    L = 6
    R = synthetic_returns
    for t in range(L, len(R)):
        window = R.iloc[t - L:t]
        assert R.index[t] not in window.index
        assert len(window) == L
