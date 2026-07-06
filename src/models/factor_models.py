"""Linear factor models: OLS estimation and rolling factor exposures.

Estimates the standard time-series factor regression

.. math::

    r_{i,t} - r_{f,t} = \\alpha_i + \\beta_i^\\top f_t + \\varepsilon_{i,t},

where :math:`f_t` are factor returns (e.g. Fama-French factors). Alpha and
betas are estimated by OLS; t-statistics use plain (non-robust) standard
errors :math:`\\widehat{SE} = \\sqrt{\\hat{\\sigma}^2 (X^\\top X)^{-1}_{jj}}`.

References
----------
Fama, E. F. and French, K. R. (1993). "Common Risk Factors in the Returns
on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3-56.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

__all__ = ["FactorModelResult", "fit_factor_model", "rolling_betas"]


@dataclass
class FactorModelResult:
    """Results of a time-series factor regression.

    Attributes
    ----------
    alpha : float
        Regression intercept (per-period, same frequency as the inputs).
    alpha_tstat : float
        t-statistic of the intercept.
    betas : pd.Series
        Factor loadings, indexed by factor name.
    tstats : pd.Series
        t-statistics of the loadings, indexed by factor name.
    r_squared : float
        Coefficient of determination of the regression.
    resid : pd.Series
        Residuals, indexed by the (aligned) observation dates.
    n_obs : int
        Number of observations used after alignment and NaN removal.
    """

    alpha: float
    alpha_tstat: float
    betas: pd.Series
    tstats: pd.Series
    r_squared: float
    resid: pd.Series
    n_obs: int


def _align(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    rf: Union[pd.Series, float],
) -> tuple[pd.Series, pd.DataFrame]:
    """Align series on a common index, subtract rf, drop NaNs."""
    df = factor_returns.copy()
    df["_asset"] = asset_returns
    if isinstance(rf, pd.Series):
        df["_rf"] = rf
    df = df.dropna()
    if isinstance(rf, pd.Series):
        excess = df["_asset"] - df["_rf"]
        factors = df.drop(columns=["_asset", "_rf"])
    else:
        excess = df["_asset"] - float(rf)
        factors = df.drop(columns=["_asset"])
    return excess, factors


def fit_factor_model(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    rf: Union[pd.Series, float] = 0.0,
) -> FactorModelResult:
    r"""Fit a linear factor model by OLS.

    Parameters
    ----------
    asset_returns : pd.Series
        Asset (simple) returns, indexed by date.
    factor_returns : pd.DataFrame
        Factor returns, indexed by date, one column per factor. Factor
        returns are assumed to already be excess/zero-investment returns.
    rf : pd.Series or float, optional
        Per-period risk-free rate subtracted from the asset returns.
        Default ``0.0``.

    Returns
    -------
    FactorModelResult
        Estimated alpha, betas, t-statistics, R-squared, residuals, n_obs.

    Raises
    ------
    ValueError
        If fewer aligned observations remain than parameters to estimate.

    Notes
    -----
    OLS via ``np.linalg.lstsq`` on the design matrix
    :math:`X = [\mathbf{1}, F]`. Plain standard errors:

    .. math::

        \hat{\sigma}^2 = \frac{e^\top e}{n - k}, \qquad
        \widehat{\mathrm{Var}}(\hat{b}) = \hat{\sigma}^2 (X^\top X)^{-1}.
    """
    excess, factors = _align(asset_returns, factor_returns, rf)
    n = len(excess)
    k = factors.shape[1] + 1
    if n <= k:
        raise ValueError(
            f"Need more than {k} aligned observations, got n={n}"
        )

    y = excess.to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), factors.to_numpy(dtype=float)])

    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefs
    resid = y - fitted

    dof = n - k
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    tstats = coefs / se

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

    factor_names = list(factors.columns)
    return FactorModelResult(
        alpha=float(coefs[0]),
        alpha_tstat=float(tstats[0]),
        betas=pd.Series(coefs[1:], index=factor_names, name="beta"),
        tstats=pd.Series(tstats[1:], index=factor_names, name="tstat"),
        r_squared=float(r_squared),
        resid=pd.Series(resid, index=excess.index, name="resid"),
        n_obs=n,
    )


def rolling_betas(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    r"""Rolling-window OLS factor betas.

    For each date :math:`t \ge \text{window}`, regresses the trailing
    ``window`` asset returns on the factor returns (with intercept) and
    records the slope coefficients.

    Parameters
    ----------
    asset_returns : pd.Series
        Asset returns indexed by date.
    factor_returns : pd.DataFrame
        Factor returns indexed by date, one column per factor.
    window : int, optional
        Rolling window length in observations. Default ``252``.

    Returns
    -------
    pd.DataFrame
        Rolling betas indexed by the window's end date, one column per
        factor. Dates before the first full window are omitted.

    Raises
    ------
    ValueError
        If ``window`` is not larger than the number of regressors + 1, or
        exceeds the number of aligned observations.
    """
    excess, factors = _align(asset_returns, factor_returns, rf=0.0)
    n = len(excess)
    k = factors.shape[1] + 1
    if window <= k:
        raise ValueError(f"window must exceed n_factors + 1 = {k}, got {window}")
    if n < window:
        raise ValueError(
            f"Need at least window={window} aligned observations, got n={n}"
        )

    y = excess.to_numpy(dtype=float)
    F = factors.to_numpy(dtype=float)
    dates = excess.index
    out = np.empty((n - window + 1, k - 1))

    for i in range(n - window + 1):
        Xw = np.column_stack([np.ones(window), F[i : i + window]])
        yw = y[i : i + window]
        coefs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        out[i] = coefs[1:]

    return pd.DataFrame(
        out, index=dates[window - 1 :], columns=list(factors.columns)
    )
