"""Performance and risk-adjusted return metrics.

All metrics take a series of *period* returns (daily by default) and
annualize with ``periods`` observations per year:

.. math::

    R_{ann} = \\left( \\prod_t (1 + r_t) \\right)^{P/T} - 1,
    \\qquad
    \\sigma_{ann} = \\hat\\sigma \\sqrt{P}.

References
----------
Sharpe, W. F. (1994). "The Sharpe Ratio". *Journal of Portfolio
Management*, 21(1), 49-58.

Sortino, F. & van der Meer, R. (1991). "Downside Risk". *Journal of
Portfolio Management*, 17(4), 27-31.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.risk.drawdown_analysis import max_drawdown
from src.risk.expected_shortfall import historical_es
from src.risk.value_at_risk import historical_var

__all__ = [
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "hit_rate",
    "summary",
]


def _clean(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns).__name__}")
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns has no non-NaN observations")
    return r


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """Geometric annualized return.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    periods : int, default 252
        Observations per year.

    Returns
    -------
    float
        Compound annual growth rate.
    """
    r = _clean(returns)
    total = float((1.0 + r).prod())
    if total <= 0:
        return -1.0
    return total ** (periods / len(r)) - 1.0


def annualized_vol(returns: pd.Series, periods: int = 252) -> float:
    """Annualized volatility (sample std x sqrt(periods))."""
    r = _clean(returns)
    if len(r) < 2:
        raise ValueError("need at least 2 observations for volatility")
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio.

    .. math:: SR = \\frac{\\bar{r} - r_f / P}{\\hat\\sigma} \\sqrt{P}

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    rf : float, default 0.0
        *Annualized* risk-free rate (de-annualized internally).
    periods : int, default 252
        Observations per year.

    References
    ----------
    Sharpe (1994).
    """
    r = _clean(returns)
    excess = r - rf / periods
    sd = float(excess.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation in the denominator).

    .. math::

        \\text{Sortino} = \\frac{\\bar{r} - r_f/P}
            {\\sqrt{\\frac{1}{T} \\sum_t \\min(r_t - r_f/P, 0)^2}}
            \\sqrt{P}

    References
    ----------
    Sortino & van der Meer (1991).
    """
    r = _clean(returns)
    excess = r - rf / periods
    downside = np.minimum(excess.to_numpy(), 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods))


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Calmar ratio: annualized return / max drawdown.

    Returns NaN if the series has no drawdown.
    """
    r = _clean(returns)
    mdd = max_drawdown(r)
    if mdd == 0:
        return float("nan")
    return annualized_return(r, periods=periods) / mdd


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    r = _clean(returns)
    return float((r > 0).mean())


def summary(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> pd.Series:
    """One-stop performance summary.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    rf : float, default 0.0
        Annualized risk-free rate.
    periods : int, default 252
        Observations per year.

    Returns
    -------
    pd.Series
        Keys: ``annualized_return``, ``annualized_vol``,
        ``sharpe_ratio``, ``sortino_ratio``, ``calmar_ratio``,
        ``hit_rate``, ``max_drawdown``, ``skew``, ``kurtosis``
        (excess), ``VaR95``, ``ES95``.
    """
    r = _clean(returns)
    return pd.Series(
        {
            "annualized_return": annualized_return(r, periods=periods),
            "annualized_vol": annualized_vol(r, periods=periods),
            "sharpe_ratio": sharpe_ratio(r, rf=rf, periods=periods),
            "sortino_ratio": sortino_ratio(r, rf=rf, periods=periods),
            "calmar_ratio": calmar_ratio(r, periods=periods),
            "hit_rate": hit_rate(r),
            "max_drawdown": max_drawdown(r),
            "skew": float(stats.skew(r.to_numpy(), bias=False)),
            "kurtosis": float(stats.kurtosis(r.to_numpy(), fisher=True, bias=False)),
            "VaR95": historical_var(r, alpha=0.95),
            "ES95": historical_es(r, alpha=0.95),
        },
        name="summary",
    )
