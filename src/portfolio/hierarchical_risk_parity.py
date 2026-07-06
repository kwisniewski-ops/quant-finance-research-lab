"""Hierarchical Risk Parity (HRP).

Lopez de Prado's HRP allocates capital without inverting the covariance
matrix, in three stages:

1. **Tree clustering** — assets are clustered on the correlation
   distance :math:`d_{ij} = \\sqrt{(1 - \\rho_{ij}) / 2}` with single
   linkage.
2. **Quasi-diagonalization** — the covariance matrix is reordered so
   similar assets sit adjacent to each other.
3. **Recursive bisection** — capital is split top-down between the two
   halves of each cluster in inverse proportion to their
   inverse-variance cluster risks:

   .. math::

       \\alpha = 1 - \\frac{\\tilde{\\sigma}^2_{left}}
           {\\tilde{\\sigma}^2_{left} + \\tilde{\\sigma}^2_{right}},

   where :math:`\\tilde{\\sigma}^2` is the variance of the
   inverse-variance-weighted sub-portfolio.

Only :mod:`scipy.cluster.hierarchy` is required (no sklearn).

References
----------
Lopez de Prado, M. (2016). "Building Diversified Portfolios that
Outperform Out of Sample". *Journal of Portfolio Management*, 42(4),
59-69.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

__all__ = ["hrp_weights"]


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Distance matrix d_ij = sqrt((1 - rho_ij) / 2), zero diagonal."""
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def _cluster_var(cov: np.ndarray, idx: np.ndarray) -> float:
    """Variance of the inverse-variance-weighted sub-portfolio."""
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Top-down inverse-cluster-variance capital split."""
    n = cov.shape[0]
    weights = np.ones(n)
    clusters: list[list[int]] = [order]
    while clusters:
        clusters = [
            half
            for cluster in clusters
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
            if len(half) > 0 and len(cluster) > 1
        ]
        for i in range(0, len(clusters), 2):
            left = np.asarray(clusters[i])
            right = np.asarray(clusters[i + 1])
            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity weights from an asset-return panel.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns, columns = assets, rows = observations. At least
        two assets and three observations are required.

    Returns
    -------
    pd.Series
        Long-only weights indexed by asset, summing to 1.

    Raises
    ------
    ValueError
        If ``returns`` is too small, contains NaN, or has an asset with
        zero variance.

    References
    ----------
    Lopez de Prado (2016).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")
    if returns.shape[1] < 2:
        raise ValueError(f"need at least 2 assets, got {returns.shape[1]}")
    if returns.shape[0] < 3:
        raise ValueError(f"need at least 3 observations, got {returns.shape[0]}")
    if returns.isna().any().any():
        raise ValueError("returns contains NaN; clean the data before calling hrp_weights")

    cov = returns.cov().to_numpy(dtype=float)
    std = np.sqrt(np.diag(cov))
    if np.any(std <= 0):
        bad = list(returns.columns[std <= 0])
        raise ValueError(f"assets with zero variance: {bad}")
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)

    dist = _correlation_distance(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    order = leaves_list(link).tolist()

    weights = _recursive_bisection(cov, order)
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()
    return pd.Series(weights, index=returns.columns, name="weight")
