"""SABR asymptotic implied volatility and implied-vol surface construction.

The SABR model (Hagan et al. 2002) describes a forward :math:`F_t` with CEV
backbone and lognormal stochastic volatility:

.. math::

    dF_t = \\alpha_t F_t^{\\beta}\\, dW_t, \\qquad
    d\\alpha_t = \\nu\\, \\alpha_t\\, dZ_t, \\qquad
    d\\langle W, Z \\rangle_t = \\rho\\, dt.

Hagan's singular-perturbation expansion gives an accurate closed-form
approximation of the Black (lognormal) implied volatility, ubiquitous on
rates and FX vol desks for smile interpolation.

References
----------
Hagan, P. S., Kumar, D., Lesniewski, A. S., and Woodward, D. E. (2002).
"Managing Smile Risk." *Wilmott Magazine*, September, 84-108.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.models.black_scholes import implied_vol

__all__ = ["sabr_implied_vol", "implied_vol_surface"]


def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    r"""Hagan et al. (2002) lognormal SABR implied volatility.

    Parameters
    ----------
    F : float
        Forward price (must be > 0).
    K : float
        Strike (must be > 0).
    T : float
        Time to expiry in years (must be > 0).
    alpha : float
        Initial volatility level :math:`\alpha_0` (must be > 0).
    beta : float
        CEV exponent in [0, 1].
    rho : float
        Spot-vol correlation in (-1, 1).
    nu : float
        Volatility of volatility (must be >= 0).

    Returns
    -------
    float
        Black (lognormal) implied volatility.

    Raises
    ------
    ValueError
        On invalid inputs.

    Notes
    -----
    With :math:`z = \frac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ln(F/K)` and

    .. math::

        x(z) = \ln\!\left[
            \frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}
        \right],

    the implied volatility is

    .. math::

        \sigma_B(K, F) =
        \frac{\alpha}
             {(FK)^{(1-\beta)/2}
              \left[1 + \frac{(1-\beta)^2}{24}\ln^2\frac{F}{K}
                     + \frac{(1-\beta)^4}{1920}\ln^4\frac{F}{K}\right]}
        \cdot \frac{z}{x(z)} \cdot
        \left[1 + \left(\frac{(1-\beta)^2}{24}
                        \frac{\alpha^2}{(FK)^{1-\beta}}
                      + \frac{\rho\beta\nu\alpha}{4 (FK)^{(1-\beta)/2}}
                      + \frac{2 - 3\rho^2}{24}\nu^2\right) T\right].

    The at-the-money limit :math:`K \to F` is handled with the exact ATM
    branch; for small :math:`|z|` the ratio :math:`z/x(z)` is replaced by
    its Taylor expansion :math:`1 + \rho z / 2 + O(z^2)`, so the smile is
    continuous through the strike :math:`K = F`.
    """
    for name, x in (("F", F), ("K", K), ("T", T), ("alpha", alpha)):
        if x <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {name}={x}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got beta={beta}")
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must lie in (-1, 1), got rho={rho}")
    if nu < 0.0:
        raise ValueError(f"nu must be non-negative, got nu={nu}")

    one_m_beta = 1.0 - beta
    log_FK = np.log(F / K)
    FK_pow = (F * K) ** (0.5 * one_m_beta)

    # Common maturity correction term (evaluated at the geometric midpoint).
    correction = 1.0 + (
        one_m_beta**2 / 24.0 * alpha**2 / FK_pow**2
        + 0.25 * rho * beta * nu * alpha / FK_pow
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    ) * T

    if abs(log_FK) < 1e-12:
        # Exact ATM branch: sigma = alpha / F^{1-beta} * correction.
        return float(alpha / F**one_m_beta * correction)

    denom = FK_pow * (
        1.0
        + one_m_beta**2 / 24.0 * log_FK**2
        + one_m_beta**4 / 1920.0 * log_FK**4
    )

    if nu == 0.0:
        z_over_x = 1.0
    else:
        z = (nu / alpha) * FK_pow * log_FK
        if abs(z) < 1e-7:
            # Taylor expansion of z / x(z) near z = 0 for continuity.
            z_over_x = 1.0 + 0.5 * rho * z
        else:
            x_z = np.log(
                (np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho)
            )
            z_over_x = z / x_z

    return float(alpha / denom * z_over_x * correction)


def implied_vol_surface(
    S: float,
    r: float,
    strikes: Sequence[float],
    maturities: Sequence[float],
    price_fn: Callable[[float, float], float],
) -> pd.DataFrame:
    r"""Build a Black-Scholes implied-volatility surface from a pricer.

    For each maturity :math:`T` and strike :math:`K`, evaluates the supplied
    call pricer and inverts the Black-Scholes formula via
    :func:`src.models.black_scholes.implied_vol`.

    Parameters
    ----------
    S : float
        Spot price (must be > 0).
    r : float
        Continuously compounded risk-free rate.
    strikes : sequence of float
        Strikes (surface columns).
    maturities : sequence of float
        Times to expiry in years (surface rows).
    price_fn : callable
        European **call** pricer with signature ``price_fn(K, T) -> float``
        (e.g. ``lambda K, T: heston_price(S, K, T, r, params)``).

    Returns
    -------
    pd.DataFrame
        Implied volatilities with ``index=maturities`` (named ``"maturity"``)
        and ``columns=strikes`` (named ``"strike"``). Entries where the
        inversion fails (price outside no-arbitrage bounds) are ``np.nan``.

    Raises
    ------
    ValueError
        If ``S`` is non-positive or strikes/maturities are empty or contain
        non-positive values.
    """
    if S <= 0.0:
        raise ValueError(f"S must be strictly positive, got S={S}")
    strikes = list(strikes)
    maturities = list(maturities)
    if len(strikes) == 0 or len(maturities) == 0:
        raise ValueError("strikes and maturities must be non-empty")
    if min(strikes) <= 0.0 or min(maturities) <= 0.0:
        raise ValueError("strikes and maturities must be strictly positive")

    surface = np.full((len(maturities), len(strikes)), np.nan)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = float(price_fn(K, T))
            surface[i, j] = implied_vol(price, S, K, T, r, option_type="call")

    df = pd.DataFrame(surface, index=maturities, columns=strikes)
    df.index.name = "maturity"
    df.columns.name = "strike"
    return df
