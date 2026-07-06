"""Risk-parity (equal risk contribution) portfolio construction.

The marginal risk contribution of asset :math:`i` is
:math:`\\partial \\sigma_p / \\partial w_i = (\\Sigma w)_i / \\sigma_p`,
so the *total* risk contribution is

.. math::

    RC_i = w_i \\frac{(\\Sigma w)_i}{\\sigma_p},
    \\qquad \\sum_i RC_i = \\sigma_p .

A risk-parity portfolio equalizes the fractional contributions
:math:`RC_i / \\sigma_p` (or matches an arbitrary risk budget
:math:`b_i`). We minimize the sum of squared deviations from the
budget with SLSQP and polish the solution with the fixed-point
iteration :math:`w_i \\propto b_i / (\\Sigma w)_i`.

References
----------
Maillard, S., Roncalli, T. & Teïletche, J. (2010). "The Properties of
Equally Weighted Risk Contribution Portfolios". *Journal of Portfolio
Management*, 36(4), 60-70.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio.mean_variance import _quiet_slsqp

__all__ = ["risk_contributions", "risk_parity_weights"]


def _validate_cov(cov: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cov, pd.DataFrame):
        raise TypeError(f"cov must be a pd.DataFrame, got {type(cov).__name__}")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if not cov.index.equals(cov.columns):
        raise ValueError("cov index and columns must be identical")
    vals = cov.to_numpy(dtype=float)
    if not np.allclose(vals, vals.T, atol=1e-10):
        raise ValueError("cov must be symmetric")
    if np.any(np.diag(vals) <= 0):
        raise ValueError("cov must have strictly positive variances on the diagonal")
    return cov


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Fractional risk contributions of each asset.

    .. math:: RC_i^{frac} = \\frac{w_i (\\Sigma w)_i}{w^\\top \\Sigma w}

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset (need not be long-only).
    cov : pd.DataFrame
        Covariance matrix with the same asset labels.

    Returns
    -------
    pd.Series
        Fractional risk contributions; sums to 1.

    References
    ----------
    Maillard, Roncalli & Teïletche (2010).
    """
    cov = _validate_cov(cov)
    if not isinstance(weights, pd.Series):
        raise TypeError(f"weights must be a pd.Series, got {type(weights).__name__}")
    if set(weights.index) != set(cov.index):
        raise ValueError("weights and cov are misaligned (different asset sets)")
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    sigma = cov.to_numpy(dtype=float)
    var = float(w @ sigma @ w)
    if var <= 0:
        raise ValueError("portfolio variance is non-positive; cannot compute risk contributions")
    rc = w * (sigma @ w) / var
    return pd.Series(rc, index=cov.index, name="risk_contribution")


def risk_parity_weights(cov: pd.DataFrame, budget: pd.Series | None = None) -> pd.Series:
    """Long-only risk-budgeted (default: equal risk contribution) weights.

    Minimizes :math:`\\sum_i (RC_i^{frac} - b_i)^2` with SLSQP subject
    to full investment and :math:`w_i \\ge 0`, then polishes with the
    fixed-point iteration :math:`w_i \\leftarrow b_i / (\\Sigma w)_i`
    (renormalized) until fractional contributions match the budget to
    machine-level precision.

    Parameters
    ----------
    cov : pd.DataFrame
        Annualized covariance matrix.
    budget : pd.Series, optional
        Target fractional risk budget per asset (positive, sums to 1).
        Defaults to equal budgets ``1/n``.

    Returns
    -------
    pd.Series
        Weights summing to 1 with ``risk_contributions(w, cov) ≈ budget``.

    References
    ----------
    Maillard, Roncalli & Teïletche (2010).
    """
    cov = _validate_cov(cov)
    n = cov.shape[0]
    sigma = cov.to_numpy(dtype=float)

    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        if not isinstance(budget, pd.Series):
            raise TypeError(f"budget must be a pd.Series, got {type(budget).__name__}")
        if set(budget.index) != set(cov.index):
            raise ValueError("budget and cov are misaligned (different asset sets)")
        b = budget.reindex(cov.index).to_numpy(dtype=float)
        if np.any(b <= 0):
            raise ValueError("budget entries must be strictly positive")
        if not np.isclose(b.sum(), 1.0, atol=1e-8):
            raise ValueError(f"budget must sum to 1, got {b.sum():.6f}")

    def objective(w: np.ndarray) -> float:
        var = w @ sigma @ w
        rc = w * (sigma @ w) / var
        return float(np.sum((rc - b) ** 2))

    # Inverse-volatility start (exact solution when correlations are equal).
    w0 = 1.0 / np.sqrt(np.diag(sigma))
    w0 /= w0.sum()
    with _quiet_slsqp():
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=[(1e-9, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-18},
        )
    w = np.asarray(res.x if res.success else w0, dtype=float)
    w = np.clip(w, 1e-12, None)

    # Newton polish on the strictly convex Roncalli formulation
    #     min_w  0.5 w' Sigma w - sum_i b_i log(w_i),   w > 0,
    # whose unique optimum satisfies w_i (Sigma w)_i = b_i exactly, i.e.
    # fractional risk contributions equal the budget. The scale-free
    # solution is then renormalized to sum to 1.
    w = w / np.sqrt(w @ sigma @ w)  # put iterate on the optimum's scale

    def _obj(x: np.ndarray) -> float:
        return 0.5 * x @ sigma @ x - b @ np.log(x)

    for _ in range(100):
        grad = sigma @ w - b / w
        if np.max(np.abs(grad)) < 1e-12:
            break
        hess = sigma + np.diag(b / w**2)
        step = np.linalg.solve(hess, grad)
        t, f0 = 1.0, _obj(w)
        while np.any(w - t * step <= 0) or _obj(w - t * step) > f0:
            t *= 0.5
            if t < 1e-12:
                break
        w = w - t * step

    w = w / w.sum()
    return pd.Series(w, index=cov.index, name="weight")
