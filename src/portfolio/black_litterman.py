"""Black-Litterman expected-return model.

Combines CAPM-implied equilibrium returns with subjective investor
views in a Bayesian framework. The equilibrium (prior) returns are the
reverse-optimized

.. math::

    \\Pi = \\delta \\, \\Sigma \\, w_{mkt},

and, given a view matrix :math:`P`, view returns :math:`Q` and view
uncertainty :math:`\\Omega`, the posterior mean and covariance are

.. math::

    \\mu_{BL} = \\Pi + \\tau \\Sigma P^\\top
        (P \\tau \\Sigma P^\\top + \\Omega)^{-1} (Q - P \\Pi),

.. math::

    \\Sigma_{BL} = \\Sigma +
        \\left[ (\\tau \\Sigma)^{-1} + P^\\top \\Omega^{-1} P \\right]^{-1}.

References
----------
Black, F. & Litterman, R. (1992). "Global Portfolio Optimization".
*Financial Analysts Journal*, 48(5), 28-43.

Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman
Model". Working paper, Ibbotson Associates.

He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman
Model Portfolios". Goldman Sachs Investment Management Research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["implied_equilibrium_returns", "bl_posterior"]


def _validate_inputs(cov: pd.DataFrame, market_weights: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(cov, pd.DataFrame):
        raise TypeError(f"cov must be a pd.DataFrame, got {type(cov).__name__}")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if not cov.index.equals(cov.columns):
        raise ValueError("cov index and columns must be identical")
    if not isinstance(market_weights, pd.Series):
        raise TypeError(f"market_weights must be a pd.Series, got {type(market_weights).__name__}")
    if set(market_weights.index) != set(cov.index):
        raise ValueError("market_weights and cov are misaligned (different asset sets)")
    w = market_weights.reindex(cov.index)
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"market_weights must sum to 1, got {w.sum():.6f}")
    return cov, w


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    delta: float = 2.5,
) -> pd.Series:
    """CAPM-implied equilibrium excess returns via reverse optimization.

    .. math:: \\Pi = \\delta \\Sigma w_{mkt}

    Parameters
    ----------
    cov : pd.DataFrame
        Annualized covariance matrix of asset returns.
    market_weights : pd.Series
        Market-capitalization weights (sum to 1), same assets as ``cov``.
    delta : float, default 2.5
        Risk-aversion coefficient of the representative investor.

    Returns
    -------
    pd.Series
        Annualized equilibrium excess returns indexed by asset.

    References
    ----------
    Black & Litterman (1992); He & Litterman (1999).
    """
    cov, w = _validate_inputs(cov, market_weights)
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    pi = delta * cov.to_numpy(dtype=float) @ w.to_numpy(dtype=float)
    return pd.Series(pi, index=cov.index, name="equilibrium_return")


def bl_posterior(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float = 0.05,
    omega: np.ndarray | None = None,
    delta: float = 2.5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Black-Litterman posterior expected returns and covariance.

    Parameters
    ----------
    cov : pd.DataFrame
        Annualized covariance matrix (n x n).
    market_weights : pd.Series
        Market-cap weights, sum to 1.
    P : np.ndarray
        View pick matrix (k x n): each row selects the portfolio a view
        applies to (columns in ``cov``'s asset order).
    Q : np.ndarray
        View returns (k,): expected annualized return of each view
        portfolio.
    tau : float, default 0.05
        Scalar reflecting uncertainty in the prior; small values anchor
        the posterior to equilibrium.
    omega : np.ndarray, optional
        View uncertainty covariance (k x k). If None, uses the
        Idzorek-style proportional prior
        ``Omega = diag(tau * P Sigma P')``.
    delta : float, default 2.5
        Risk-aversion coefficient for the equilibrium prior.

    Returns
    -------
    tuple of (pd.Series, pd.DataFrame)
        Posterior expected returns ``mu_BL`` and posterior covariance
        ``Sigma_BL`` (both labeled by asset).

    Raises
    ------
    ValueError
        If P/Q/omega dimensions are inconsistent with ``cov``.

    References
    ----------
    Black & Litterman (1992); Idzorek (2005).
    """
    cov, w = _validate_inputs(cov, market_weights)
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    n = cov.shape[0]
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_1d(np.asarray(Q, dtype=float)).ravel()
    k = P.shape[0]
    if P.shape[1] != n:
        raise ValueError(f"P has {P.shape[1]} columns but cov has {n} assets")
    if Q.shape[0] != k:
        raise ValueError(f"Q has {Q.shape[0]} views but P has {k} rows")

    sigma = cov.to_numpy(dtype=float)
    pi = implied_equilibrium_returns(cov, w, delta=delta).to_numpy(dtype=float)
    tau_sigma = tau * sigma

    if omega is None:
        omega = np.diag(np.diag(P @ tau_sigma @ P.T))
    else:
        omega = np.asarray(omega, dtype=float)
        if omega.shape != (k, k):
            raise ValueError(f"omega must be ({k}, {k}), got {omega.shape}")

    # Posterior mean: solve rather than invert for numerical stability.
    A = P @ tau_sigma @ P.T + omega  # (k, k)
    rhs = Q - P @ pi
    adj = tau_sigma @ P.T @ np.linalg.solve(A, rhs)
    mu_bl = pi + adj

    # Posterior covariance of the mean estimate added to Sigma.
    m = tau_sigma - tau_sigma @ P.T @ np.linalg.solve(A, P @ tau_sigma)
    sigma_bl = sigma + m
    sigma_bl = 0.5 * (sigma_bl + sigma_bl.T)  # enforce symmetry

    return (
        pd.Series(mu_bl, index=cov.index, name="bl_posterior_return"),
        pd.DataFrame(sigma_bl, index=cov.index, columns=cov.columns),
    )
