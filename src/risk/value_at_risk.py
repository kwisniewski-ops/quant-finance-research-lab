"""Value-at-Risk estimators.

VaR at confidence level :math:`\\alpha` is the loss threshold exceeded
with probability :math:`1 - \\alpha`:

.. math::

    \\mathrm{VaR}_\\alpha = -\\inf \\{ x : F_R(x) \\ge 1 - \\alpha \\}
    = -q_{1-\\alpha}(R),

reported here as a **positive loss number** in return units. Four
estimators are provided: empirical quantile (historical), Gaussian
(parametric), Cornish-Fisher (moment-adjusted quantile), and Monte
Carlo from a multivariate-normal asset model.

References
----------
Jorion, P. (2006). *Value at Risk*, 3rd ed. McGraw-Hill.

Cornish, E. A. & Fisher, R. A. (1938). "Moments and Cumulants in the
Specification of Distributions". *Revue de l'IIS*, 5(4), 307-320.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["historical_var", "parametric_var", "cornish_fisher_var", "monte_carlo_var"]


def _validate_returns(returns: pd.Series, min_obs: int = 2) -> np.ndarray:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns).__name__}")
    r = returns.dropna().to_numpy(dtype=float)
    if r.size < min_obs:
        raise ValueError(f"need at least {min_obs} non-NaN observations, got {r.size}")
    return r


def _validate_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(alpha)


def historical_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """Historical (empirical-quantile) VaR.

    Parameters
    ----------
    returns : pd.Series
        Period returns (decimal).
    alpha : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        VaR as a positive loss number:
        ``-quantile(returns, 1 - alpha)``.

    References
    ----------
    Jorion (2006).
    """
    r = _validate_returns(returns)
    alpha = _validate_alpha(alpha)
    return float(-np.quantile(r, 1.0 - alpha))


def parametric_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """Gaussian parametric VaR.

    .. math:: \\mathrm{VaR}_\\alpha = -(\\hat\\mu + \\hat\\sigma z_{1-\\alpha})

    where :math:`z_{1-\\alpha} = \\Phi^{-1}(1-\\alpha)`.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    alpha : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        VaR as a positive loss number.

    References
    ----------
    Jorion (2006).
    """
    r = _validate_returns(returns)
    alpha = _validate_alpha(alpha)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    z = stats.norm.ppf(1.0 - alpha)
    return float(-(mu + sd * z))


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """Cornish-Fisher (modified) VaR.

    Adjusts the Gaussian quantile for sample skewness :math:`S` and
    excess kurtosis :math:`K`:

    .. math::

        \\tilde{z} = z + \\frac{z^2 - 1}{6} S
            + \\frac{z^3 - 3z}{24} K
            - \\frac{2z^3 - 5z}{36} S^2 ,

    then :math:`\\mathrm{VaR} = -(\\hat\\mu + \\hat\\sigma \\tilde{z})`.
    Reduces exactly to :func:`parametric_var` when ``S = K = 0``.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    alpha : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        VaR as a positive loss number.

    References
    ----------
    Cornish & Fisher (1938); Favre & Galeano (2002).
    """
    r = _validate_returns(returns, min_obs=4)
    alpha = _validate_alpha(alpha)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    # Population (biased) moment estimators — standard in CF-VaR practice
    # and guarantees exact reduction to Gaussian VaR when the sample has
    # zero skew and zero excess kurtosis.
    s = float(stats.skew(r, bias=True))
    k = float(stats.kurtosis(r, fisher=True, bias=True))  # excess kurtosis
    z = stats.norm.ppf(1.0 - alpha)
    z_cf = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
    )
    return float(-(mu + sd * z_cf))


def monte_carlo_var(
    mu: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.95,
    n_sims: int = 100_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo VaR from a multivariate-normal asset model.

    Simulates ``n_sims`` one-period asset return vectors from
    :math:`\\mathcal{N}(\\mu, \\Sigma)` (daily parameters), forms
    portfolio returns :math:`w^\\top r`, and takes the empirical
    :math:`1-\\alpha` quantile.

    Parameters
    ----------
    mu : pd.Series
        Daily expected asset returns.
    cov : pd.DataFrame
        Daily covariance matrix, same assets as ``mu``.
    weights : pd.Series
        Portfolio weights, same assets, summing to 1.
    alpha : float, default 0.95
        Confidence level.
    n_sims : int, default 100_000
        Number of simulations.
    rng : np.random.Generator, optional
        Random generator; ``np.random.default_rng()`` if None.

    Returns
    -------
    float
        VaR as a positive loss number.

    References
    ----------
    Jorion (2006).
    """
    alpha = _validate_alpha(alpha)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if set(mu.index) != set(cov.index) or set(weights.index) != set(cov.index):
        raise ValueError("mu, cov, and weights must share the same asset labels")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"weights must sum to 1, got {weights.sum():.6f}")
    if n_sims < 1000:
        raise ValueError(f"n_sims must be >= 1000 for a stable quantile, got {n_sims}")
    if rng is None:
        rng = np.random.default_rng()

    mu_vec = mu.reindex(cov.index).to_numpy(dtype=float)
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    sims = rng.multivariate_normal(mu_vec, cov.to_numpy(dtype=float), size=n_sims, method="cholesky")
    port = sims @ w
    return float(-np.quantile(port, 1.0 - alpha))
