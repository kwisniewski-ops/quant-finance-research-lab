"""Robust portfolio optimization.

Two guards against estimation error in expected returns:

* **Worst-case (box-uncertainty) max Sharpe** — expected returns are
  only known to lie in the box :math:`[\\mu_i - u_i, \\mu_i + u_i]`.
  The adversary's optimal choice against weights :math:`w` is
  :math:`\\mu_i - u_i \\, \\mathrm{sign}(w_i)`, so the worst-case
  portfolio return is :math:`w^\\top \\mu - u^\\top |w|` and we
  maximize the worst-case Sharpe ratio.

* **Michaud resampled frontier** — the frontier is re-estimated on
  many simulated return histories drawn from
  :math:`\\mathcal{N}(\\mu/252, \\Sigma/252)` and the rank-wise average
  of the resulting weight vectors is reported, smoothing out
  estimation noise.

References
----------
Michaud, R. (1998). *Efficient Asset Management*. Harvard Business
School Press.

Ben-Tal, A. & Nemirovski, A. (1998). "Robust Convex Optimization".
*Mathematics of Operations Research*, 23(4), 769-805.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio.mean_variance import (
    PortfolioResult,
    _align_mu_cov,
    _feasible_start,
    _make_result,
    _quiet_slsqp,
    _validate_bounds,
    _validate_cov,
    efficient_frontier,
)

__all__ = ["robust_max_sharpe", "resampled_frontier"]


def robust_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    mu_uncertainty: pd.Series | float,
    rf: float = 0.0,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> PortfolioResult:
    """Worst-case max-Sharpe portfolio under box uncertainty in ``mu``.

    Maximizes

    .. math::

        \\frac{w^\\top \\mu - u^\\top |w| - r_f}
             {\\sqrt{w^\\top \\Sigma w}}

    subject to full investment and box bounds. The reported
    ``expected_return``/``sharpe`` use the *nominal* ``mu`` so results
    are comparable with :func:`max_sharpe`.

    Parameters
    ----------
    mu : pd.Series
        Nominal annualized expected returns.
    cov : pd.DataFrame
        Annualized covariance matrix.
    mu_uncertainty : pd.Series or float
        Half-width of the uncertainty box per asset (scalar applies to
        all assets). Must be non-negative.
    rf : float, default 0.0
        Risk-free rate.
    bounds : tuple of float, default (0.0, 1.0)
        Per-asset weight bounds.

    Returns
    -------
    PortfolioResult

    References
    ----------
    Ben-Tal & Nemirovski (1998); Michaud (1998).
    """
    cov = _validate_cov(cov)
    mu, cov = _align_mu_cov(mu, cov)
    n = cov.shape[0]
    lo, hi = _validate_bounds(bounds, n)

    if isinstance(mu_uncertainty, pd.Series):
        if set(mu_uncertainty.index) != set(cov.index):
            raise ValueError("mu_uncertainty and cov are misaligned (different asset sets)")
        u = mu_uncertainty.reindex(cov.index).to_numpy(dtype=float)
    else:
        u = np.full(n, float(mu_uncertainty))
    if np.any(u < 0):
        raise ValueError("mu_uncertainty must be non-negative")

    sigma = cov.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)

    def neg_worst_case_sharpe(w: np.ndarray) -> float:
        worst_ret = w @ mu_vec - u @ np.abs(w)
        vol = np.sqrt(max(w @ sigma @ w, 1e-18))
        return -(worst_ret - rf) / vol

    with _quiet_slsqp():
        res = minimize(
            neg_worst_case_sharpe,
            _feasible_start(n, lo, hi),
            method="SLSQP",
            bounds=[(lo, hi)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-14},
        )
    if not res.success:
        raise RuntimeError(f"SLSQP failed to converge: {res.message}")
    w = np.clip(np.asarray(res.x, dtype=float), lo, hi)
    w /= w.sum()
    return _make_result(w, cov, mu, rf)


def resampled_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_samples: int = 200,
    n_points: int = 25,
    n_obs: int = 252,
    rng: np.random.Generator | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Michaud-style resampled efficient frontier.

    For each of ``n_samples`` draws, a synthetic return history of
    ``n_obs`` daily observations is simulated from
    :math:`\\mathcal{N}(\\mu/252, \\Sigma/252)`; sample moments are
    re-annualized and an ``n_points``-point frontier is computed. The
    resampled frontier averages weights rank-by-rank across samples and
    re-prices each averaged weight vector under the *original*
    ``(mu, cov)``.

    Parameters
    ----------
    mu : pd.Series
        Annualized expected returns.
    cov : pd.DataFrame
        Annualized covariance matrix.
    n_samples : int, default 200
        Number of resampled histories.
    n_points : int, default 25
        Frontier points per sample.
    n_obs : int, default 252
        Daily observations per simulated history.
    rng : np.random.Generator, optional
        Random generator; ``np.random.default_rng()`` if None.
    bounds : tuple of float, default (0.0, 1.0)
        Per-asset weight bounds.

    Returns
    -------
    pd.DataFrame
        Same layout as :func:`efficient_frontier`: columns
        ``expected_return``, ``volatility``, ``sharpe`` plus one weight
        column per asset.

    References
    ----------
    Michaud (1998).
    """
    cov = _validate_cov(cov)
    mu, cov = _align_mu_cov(mu, cov)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if n_obs < cov.shape[0] + 2:
        raise ValueError(f"n_obs={n_obs} too small to estimate a {cov.shape[0]}-asset covariance")
    if rng is None:
        rng = np.random.default_rng()

    assets = [str(a) for a in cov.index]
    mu_d = mu.to_numpy(dtype=float) / 252.0
    cov_d = cov.to_numpy(dtype=float) / 252.0

    weight_sum = np.zeros((n_points, len(assets)))
    n_ok = 0
    for _ in range(n_samples):
        sample = rng.multivariate_normal(mu_d, cov_d, size=n_obs, method="cholesky")
        mu_hat = pd.Series(sample.mean(axis=0) * 252.0, index=cov.index)
        cov_hat = pd.DataFrame(np.cov(sample, rowvar=False) * 252.0, index=cov.index, columns=cov.columns)
        try:
            fr = efficient_frontier(mu_hat, cov_hat, n_points=n_points, bounds=bounds)
        except RuntimeError:
            continue  # rare non-convergence on a pathological draw
        weight_sum += fr[assets].to_numpy(dtype=float)
        n_ok += 1
    if n_ok == 0:
        raise RuntimeError("all resampled frontier computations failed")

    avg_weights = weight_sum / n_ok
    avg_weights = avg_weights / avg_weights.sum(axis=1, keepdims=True)

    sigma = cov.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)
    rows = []
    for w in avg_weights:
        ret = float(w @ mu_vec)
        vol = float(np.sqrt(w @ sigma @ w))
        row = {"expected_return": ret, "volatility": vol, "sharpe": ret / vol if vol > 0 else np.nan}
        row.update({a: float(x) for a, x in zip(assets, w)})
        rows.append(row)
    return pd.DataFrame(rows)
