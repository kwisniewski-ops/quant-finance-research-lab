"""Expected shortfall (conditional VaR).

Expected shortfall at confidence :math:`\\alpha` is the expected loss
*given* that the loss exceeds VaR:

.. math::

    \\mathrm{ES}_\\alpha = -\\mathbb{E}\\left[ R \\mid R \\le
        q_{1-\\alpha}(R) \\right] \\ge \\mathrm{VaR}_\\alpha .

Unlike VaR, ES is a coherent risk measure (subadditive) and is the
object minimized in the Rockafellar-Uryasev CVaR optimization
framework. Reported as a **positive loss number**.

For Gaussian returns the closed form is

.. math::

    \\mathrm{ES}_\\alpha = -\\mu +
        \\sigma \\frac{\\varphi(z_{1-\\alpha})}{1 - \\alpha}.

References
----------
Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of Conditional
Value-at-Risk". *Journal of Risk*, 2(3), 21-41.

Acerbi, C. & Tasche, D. (2002). "On the Coherence of Expected
Shortfall". *Journal of Banking & Finance*, 26(7), 1487-1503.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.risk.value_at_risk import _validate_alpha, _validate_returns

__all__ = ["historical_es", "parametric_es"]


def historical_es(returns: pd.Series, alpha: float = 0.95) -> float:
    """Historical expected shortfall.

    Average of returns at or below the empirical :math:`1-\\alpha`
    quantile, sign-flipped to a positive loss.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    alpha : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        ES as a positive loss number; always >= historical VaR.

    References
    ----------
    Rockafellar & Uryasev (2000); Acerbi & Tasche (2002).
    """
    r = _validate_returns(returns)
    alpha = _validate_alpha(alpha)
    q = np.quantile(r, 1.0 - alpha)
    tail = r[r <= q]
    return float(-tail.mean())


def parametric_es(returns: pd.Series, alpha: float = 0.95) -> float:
    """Gaussian parametric expected shortfall.

    .. math::

        \\mathrm{ES}_\\alpha = -\\hat\\mu + \\hat\\sigma
            \\frac{\\varphi(z_{1-\\alpha})}{1-\\alpha}

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    alpha : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        ES as a positive loss number; always >= parametric VaR.

    References
    ----------
    Rockafellar & Uryasev (2000).
    """
    r = _validate_returns(returns)
    alpha = _validate_alpha(alpha)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    z = stats.norm.ppf(1.0 - alpha)
    return float(-mu + sd * stats.norm.pdf(z) / (1.0 - alpha))
