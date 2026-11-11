"""Model-level invariants (budget, long-only, ERC, BL tilt)."""

from __future__ import annotations

import numpy as np
import pytest

from src import models as M

LAM = 2.5


def _all_models(mu, Q, L, w_mkt, P, q):
    """Every model's weight vector, keyed by name."""
    return {
        "mvo": M.mean_variance(mu, Q, LAM),
        "robust_box": M.robust_box(mu, Q, LAM, L, level=0.95),
        "robust_ellipsoidal": M.robust_ellipsoidal(mu, Q, LAM, L, level=0.95),
        "risk_parity": M.risk_parity(Q),
        "equal_weight": M.equal_weight(len(mu)),
        "black_litterman": M.black_litterman(Q, w_mkt, P, q, LAM),
    }


def test_budget_constraint(window_moments, bl_view):
    """e^T x = 1 for every model."""
    mu, Q, L = window_moments
    w_mkt, P, q = bl_view
    for name, x in _all_models(mu, Q, L, w_mkt, P, q).items():
        assert x.shape == (len(mu),), name
        assert abs(x.sum() - 1.0) < 1e-6, f"{name}: sum={x.sum()}"


def test_long_only_models_nonnegative(window_moments):
    """Risk parity and equal weight are long-only by construction."""
    mu, Q, L = window_moments
    assert (M.risk_parity(Q) >= -1e-8).all()
    assert (M.equal_weight(len(mu)) >= 0).all()


def test_equal_weight_is_uniform():
    n = 11
    np.testing.assert_allclose(M.equal_weight(n), np.full(n, 1.0 / n))


def test_risk_parity_equalizes_risk_contributions(window_moments):
    """x_i (Q x)_i should be ~equal across assets (that is the definition)."""
    _, Q, _ = window_moments
    x = M.risk_parity(Q)
    rc = x * (Q @ x)                       # per-asset risk contribution
    # coefficient of variation across assets is tiny for a good RP solution
    assert rc.std(ddof=0) / rc.mean() < 0.02


def test_box_more_conservative_than_plain_mvo(window_moments):
    """The box penalty -delta^T|x| shrinks gross leverage vs plain MVO."""
    mu, Q, L = window_moments
    lev_mvo = np.abs(M.mean_variance(mu, Q, LAM)).sum()
    lev_box = np.abs(M.robust_box(mu, Q, LAM, L, level=0.95)).sum()
    assert lev_box <= lev_mvo + 1e-9


def test_box_delta_grows_with_confidence(window_moments):
    """Higher confidence -> wider box half-widths delta."""
    _, Q, L = window_moments
    d90 = M.box_delta(Q, L, 0.90)
    d99 = M.box_delta(Q, L, 0.99)
    assert (d99 >= d90 - 1e-12).all() and d99.sum() > d90.sum()


def test_bl_posterior_tilts_toward_view(window_moments, bl_view):
    """Posterior view-portfolio return moves from the prior toward the view value q."""
    _, Q, _ = window_moments
    w_mkt, P, q = bl_view
    pi = M.implied_equilibrium_returns(Q, w_mkt, delta_mkt=2.5)
    mu_bl, Sigma_bl = M.black_litterman_posterior(Q, w_mkt, P, q)
    prior_spread = float((P @ pi).item())
    post_spread = float((P @ mu_bl).item())
    qv = float(q[0])
    # the posterior is pulled strictly closer to the stated view than the prior was
    assert abs(post_spread - qv) < abs(prior_spread - qv)
    # posterior covariance of the mean is symmetric PSD
    assert np.allclose(Sigma_bl, Sigma_bl.T, atol=1e-12)
    assert (np.linalg.eigvalsh(Sigma_bl) > -1e-10).all()


def test_as_psd_returns_symmetric_psd():
    """_as_psd cleans a slightly-asymmetric / indefinite matrix into symmetric PSD."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    bad = A + A.T
    bad[0, 0] = -3.0                       # force an indefinite matrix
    P = M._as_psd(bad)
    assert np.allclose(P, P.T, atol=1e-12)
    assert (np.linalg.eigvalsh(P) >= -1e-9).all()


@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
def test_robust_models_solve_at_all_levels(window_moments, level):
    mu, Q, L = window_moments
    xb = M.robust_box(mu, Q, LAM, L, level=level)
    xe = M.robust_ellipsoidal(mu, Q, LAM, L, level=level)
    assert abs(xb.sum() - 1) < 1e-6
    assert abs(xe.sum() - 1) < 1e-6
