"""Portfolio-construction models for the sector-ETF backtest.

Every model is a *pure* function of the per-window estimates ``mu`` (expected
returns) and ``Q`` (covariance), plus a few hyper-parameters, and returns a
1-D numpy array of weights ``x`` with ``x.sum() == 1`` (the budget constraint
``e^T x = 1``).  Keeping the models data-agnostic lets the
rolling backtest driver stay model-agnostic: it only ever calls
``x = model(mu, Q, ...)``.

Conventions
-----------
* ``mu``  : shape (n,)   estimated mean monthly returns over the window.
* ``Q``   : shape (n, n) estimated covariance over the window (sample, 1/(L-1)).
* ``L``   : window length in months (needed by the robust models for the
            standard error of the estimated mean, se_i = sqrt(Q_ii / L)).
* Short selling is allowed for MVO and the two robust MVO variants; risk parity
  and equal weight are long-only by construction.

Solver: CLARABEL (conic) is the default — it handles the quadratic MVO terms,
the second-order-cone term in the ellipsoidal model, and the exponential-cone
log term in the risk-parity formulation.
"""

from __future__ import annotations

import warnings

import numpy as np
import cvxpy as cp
from scipy.stats import norm, chi2

# Canonical asset order — kept in sync with src/data.py ASSETS.
ASSETS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC"]

# Pick the best available conic solver instead of hard-coding one, so the code
# runs across cvxpy versions. CLARABEL (newer cvxpy) is preferred; ECOS/SCS are
# the older fallbacks. All of these handle the QP / SOCP / exp-cone programs here.
def _pick_solver():
    installed = set(cp.installed_solvers())
    for name in ("CLARABEL", "ECOS", "SCS", "OSQP"):
        if name in installed:
            return getattr(cp, name)
    raise RuntimeError(f"No suitable cvxpy solver found; installed: {sorted(installed)}")


_SOLVER = _pick_solver()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_psd(Q: np.ndarray) -> np.ndarray:
    """Symmetrize Q and clip tiny negative eigenvalues so cvxpy accepts it as PSD.

    Sample covariances are PSD in theory but can pick up asymmetry / tiny
    negative eigenvalues from floating point, which makes quad_form complain.
    """
    Q = np.asarray(Q, dtype=float)
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    w = np.clip(w, 1e-12, None)
    return (V * w) @ V.T


def _solve(prob: cp.Problem, x: cp.Variable, name: str) -> np.ndarray:
    """Solve a problem and return a clean, budget-normalized weight vector."""
    prob.solve(solver=_SOLVER)
    if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"{name}: solver failed with status '{prob.status}'")
    w = np.asarray(x.value, dtype=float).ravel()
    # Renormalize to kill tiny budget drift from the solver tolerance.
    return w / w.sum()


# --------------------------------------------------------------------------- #
# 1. Mean-variance optimization (short selling allowed)
# --------------------------------------------------------------------------- #
def mean_variance(mu: np.ndarray, Q: np.ndarray, lam: float,
                  allow_short: bool = True) -> np.ndarray:
    r"""max_x  mu^T x - lam * x^T Q x   s.t.  e^T x = 1.

    Parameters
    ----------
    lam : risk-aversion. Larger lam -> more weight on variance penalty.
          The backtest must state/justify how lam is chosen (fixed, swept, or
          market-implied); this function just takes it as given.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Q = _as_psd(Q)
    n = mu.size
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1]
    if not allow_short:
        cons.append(x >= 0)
    obj = cp.Maximize(mu @ x - lam * cp.quad_form(x, cp.psd_wrap(Q)))
    return _solve(cp.Problem(obj, cons), x, "mean_variance")


# --------------------------------------------------------------------------- #
# 2. Robust MVO — box uncertainty
# --------------------------------------------------------------------------- #
def box_delta(Q: np.ndarray, L: int, level: float) -> np.ndarray:
    """Half-widths delta_i of the box mu_i in [mu_hat_i - d_i, mu_hat_i + d_i].

    Each estimated mean has standard error se_i = sqrt(Q_ii / L).  We scale it
    by the normal quantile for the requested two-sided confidence ``level``
    (0.90 / 0.95 / 0.99), i.e. delta_i = z * se_i with z = Phi^{-1}((1+level)/2).
    """
    se = np.sqrt(np.diag(np.asarray(Q, dtype=float)) / L)
    z = norm.ppf((1.0 + level) / 2.0)
    return z * se


def robust_box(mu: np.ndarray, Q: np.ndarray, lam: float, L: int,
               level: float = 0.95, allow_short: bool = True) -> np.ndarray:
    r"""max_x  mu^T x - delta^T |x| - lam * x^T Q x   s.t.  e^T x = 1.

    The ``- delta^T |x|`` term is the worst case of ``mu^T x`` when each mu_i
    ranges over the box; it is convex (|x| is convex) so the program stays a QP.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Q = _as_psd(Q)
    n = mu.size
    delta = box_delta(Q, L, level)
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1]
    if not allow_short:
        cons.append(x >= 0)
    obj = cp.Maximize(mu @ x - delta @ cp.abs(x)
                      - lam * cp.quad_form(x, cp.psd_wrap(Q)))
    return _solve(cp.Problem(obj, cons), x, "robust_box")


# --------------------------------------------------------------------------- #
# 3. Robust MVO — ellipsoidal uncertainty
# --------------------------------------------------------------------------- #
def ellipsoid_rho(n: int, level: float) -> float:
    """Ellipsoid radius rho from the chi-square quantile with n d.o.f.

    U = {mu : (mu-mu_hat)^T S^{-1} (mu-mu_hat) <= rho^2}; rho^2 is the
    ``level`` quantile of chi^2_n so the true mean lies in U with prob ``level``.
    """
    return float(np.sqrt(chi2.ppf(level, df=n)))


def robust_ellipsoidal(mu: np.ndarray, Q: np.ndarray, lam: float, L: int,
                        level: float = 0.95, allow_short: bool = True,
                        S: np.ndarray | None = None) -> np.ndarray:
    r"""max_x  mu^T x - rho * sqrt(x^T S x) - lam * x^T Q x   s.t.  e^T x = 1.

    ``S`` is the covariance of the *estimated mean* (default Q / L).  The
    ``sqrt(x^T S x)`` term is a second-order-cone term, so this is an SOCP+QP.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Q = _as_psd(Q)
    n = mu.size
    if S is None:
        S = Q / L
    S = _as_psd(S)
    rho = ellipsoid_rho(n, level)
    # sqrt(x^T S x) = || S^{1/2} x ||_2
    w_eig, V = np.linalg.eigh(S)
    S_half = (V * np.sqrt(np.clip(w_eig, 0, None))) @ V.T
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1]
    if not allow_short:
        cons.append(x >= 0)
    obj = cp.Maximize(mu @ x - rho * cp.norm(S_half @ x, 2)
                      - lam * cp.quad_form(x, cp.psd_wrap(Q)))
    return _solve(cp.Problem(obj, cons), x, "robust_ellipsoidal")


# --------------------------------------------------------------------------- #
# 4. Risk parity (long-only, equal risk contributions)
# --------------------------------------------------------------------------- #
def risk_parity(Q: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    r"""Equal risk contribution portfolio, long-only.

    Exact RC equalization (x_i (Qx)_i all equal) is non-convex.  We use the
    standard convex log-barrier reformulation (Spinu / Maillard et al.):

        min_{y > 0}  (1/2) y^T Q y - sum_i b_i log(y_i)

    whose unique solution, renormalized to sum 1, has risk contributions
    proportional to the budget ``b`` (default 1/n each -> equal risk).
    """
    Q = _as_psd(Q)
    n = Q.shape[0]
    if b is None:
        b = np.full(n, 1.0 / n)
    b = np.asarray(b, dtype=float).ravel()
    y = cp.Variable(n, nonneg=True)
    obj = cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(Q)) - b @ cp.log(y))
    prob = cp.Problem(obj)
    prob.solve(solver=_SOLVER)
    if y.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"risk_parity: solver failed with status '{prob.status}'")
    w = np.asarray(y.value, dtype=float).ravel()
    return w / w.sum()


# --------------------------------------------------------------------------- #
# 5. Equal weight (trivial benchmark)
# --------------------------------------------------------------------------- #
def equal_weight(n: int) -> np.ndarray:
    """x_i = 1/n."""
    return np.full(n, 1.0 / n)


# --------------------------------------------------------------------------- #
# 6. Black-Litterman -> MVO
# --------------------------------------------------------------------------- #
def implied_equilibrium_returns(Q: np.ndarray, w_mkt: np.ndarray,
                                delta_mkt: float = 2.5) -> np.ndarray:
    """Reverse-optimization: Pi = delta_mkt * Q * w_mkt (CAPM equilibrium)."""
    Q = np.asarray(Q, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float).ravel()
    return delta_mkt * (Q @ w_mkt)


def black_litterman_posterior(Q: np.ndarray, w_mkt: np.ndarray,
                              P: np.ndarray, q: np.ndarray,
                              tau: float = 0.05, delta_mkt: float = 2.5,
                              omega: np.ndarray | None = None):
    """Black-Litterman posterior mean & covariance of expected returns.

    Returns
    -------
    mu_bl : posterior expected returns, shape (n,).
    Sigma_bl : posterior covariance of the mean estimate, shape (n, n).

    Notes
    -----
    * Pi      : implied equilibrium returns (prior mean).
    * P (k,n) : view-pick matrix; q (k,) : view values.
    * Omega   : view uncertainty; default diag(P (tau Q) P^T) (Idzorek-style
                proportional confidence), so a tighter view => smaller Omega.
    """
    Q = _as_psd(Q)
    n = Q.shape[0]
    P = np.atleast_2d(np.asarray(P, dtype=float))
    q = np.asarray(q, dtype=float).ravel()
    pi = implied_equilibrium_returns(Q, w_mkt, delta_mkt)
    tauQ = tau * Q
    if omega is None:
        omega = np.diag(np.diag(P @ tauQ @ P.T))
        # guard against a zero-variance view making Omega singular
        omega = omega + 1e-12 * np.eye(omega.shape[0])

    tauQ_inv = np.linalg.inv(tauQ)
    omega_inv = np.linalg.inv(omega)
    A = tauQ_inv + P.T @ omega_inv @ P
    b = tauQ_inv @ pi + P.T @ omega_inv @ q
    mu_bl = np.linalg.solve(A, b)
    Sigma_bl = np.linalg.inv(A)          # posterior covariance of the mean
    return mu_bl, Sigma_bl


def black_litterman(Q: np.ndarray, w_mkt: np.ndarray, P: np.ndarray,
                    q: np.ndarray, lam: float, tau: float = 0.05,
                    delta_mkt: float = 2.5, omega: np.ndarray | None = None,
                    allow_short: bool = True) -> np.ndarray:
    """Full BL pipeline: posterior expected returns -> MVO.

    The posterior mean ``mu_bl`` is fed into the plain MVO objective using the
    *return* covariance ``Q`` for the risk term (standard practice).  The prior
    ``w_mkt``, the view ``(P, q)`` and its confidence (tau/omega) are set by the caller.
    """
    mu_bl, _ = black_litterman_posterior(Q, w_mkt, P, q, tau, delta_mkt, omega)
    return mean_variance(mu_bl, Q, lam, allow_short=allow_short)


# Convenience registry so the backtest can iterate by name.
MODELS = {
    "mvo": mean_variance,
    "robust_box": robust_box,
    "robust_ellipsoidal": robust_ellipsoidal,
    "risk_parity": risk_parity,
    "equal_weight": equal_weight,
    "black_litterman": black_litterman,
}
