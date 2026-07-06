"""Black-Scholes-Merton analytic pricing, Greeks, and implied volatility.

The Black-Scholes-Merton (1973) model assumes the underlying follows a
geometric Brownian motion under the risk-neutral measure :math:`\\mathbb{Q}`:

.. math::

    dS_t = (r - q)\\, S_t\\, dt + \\sigma S_t\\, dW_t^{\\mathbb{Q}},

which yields the closed-form European option prices

.. math::

    C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \\qquad
    P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),

with

.. math::

    d_1 = \\frac{\\ln(S/K) + (r - q + \\tfrac{1}{2}\\sigma^2) T}
               {\\sigma \\sqrt{T}}, \\qquad
    d_2 = d_1 - \\sigma \\sqrt{T}.

References
----------
Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
Liabilities." *Journal of Political Economy*, 81(3), 637-654.

Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of
Economics and Management Science*, 4(1), 141-183.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = ["bs_price", "bs_greeks", "implied_vol"]

ArrayLike = Union[float, np.ndarray]

_VALID_OPTION_TYPES = ("call", "put")


def _validate_option_type(option_type: str) -> None:
    if option_type not in _VALID_OPTION_TYPES:
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )


def _validate_positive(name: str, x: np.ndarray) -> None:
    if np.any(np.asarray(x) <= 0.0):
        raise ValueError(f"{name} must be strictly positive, got {name}={x}")


def _d1_d2(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (d1, d2) with broadcasting."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: str = "call",
    q: ArrayLike = 0.0,
) -> float | np.ndarray:
    r"""Black-Scholes-Merton price of a European call or put.

    Fully vectorized: any argument may be a scalar or a NumPy array; standard
    broadcasting rules apply. Scalar inputs return a Python ``float``.

    Parameters
    ----------
    S : float or np.ndarray
        Spot price of the underlying (must be > 0).
    K : float or np.ndarray
        Strike price (must be > 0).
    T : float or np.ndarray
        Time to expiry in years (must be > 0).
    r : float or np.ndarray
        Continuously compounded risk-free rate (annualized decimal).
    sigma : float or np.ndarray
        Volatility of the underlying (annualized decimal, must be > 0).
    option_type : {"call", "put"}, optional
        Payoff type. Default ``"call"``.
    q : float or np.ndarray, optional
        Continuous dividend yield (annualized decimal). Default ``0.0``.

    Returns
    -------
    float or np.ndarray
        Option value(s). Scalar in, scalar out; array in, array out.

    Raises
    ------
    ValueError
        If ``S``, ``K``, ``T`` or ``sigma`` contain non-positive entries, or
        ``option_type`` is invalid.

    Notes
    -----
    .. math::

        C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
        P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).

    Examples
    --------
    >>> round(bs_price(100.0, 100.0, 1.0, 0.05, 0.2), 4)
    10.4506
    """
    _validate_option_type(option_type)
    S_, K_, T_, r_, sigma_, q_ = (
        np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)
    )
    _validate_positive("S", S_)
    _validate_positive("K", K_)
    _validate_positive("T", T_)
    _validate_positive("sigma", sigma_)

    d1, d2 = _d1_d2(S_, K_, T_, r_, sigma_, q_)
    df_r = np.exp(-r_ * T_)
    df_q = np.exp(-q_ * T_)
    if option_type == "call":
        price = S_ * df_q * norm.cdf(d1) - K_ * df_r * norm.cdf(d2)
    else:
        price = K_ * df_r * norm.cdf(-d2) - S_ * df_q * norm.cdf(-d1)

    price = np.asarray(price)
    if price.ndim == 0:
        return float(price)
    return price


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> dict[str, float]:
    r"""Analytic Black-Scholes-Merton Greeks.

    Parameters
    ----------
    S, K, T, r, sigma, option_type, q
        As in :func:`bs_price`.

    Returns
    -------
    dict of str to float
        Keys ``"delta"``, ``"gamma"``, ``"vega"``, ``"theta"``, ``"rho"``.

        Conventions:

        - ``delta`` : :math:`\partial V/\partial S`.
        - ``gamma`` : :math:`\partial^2 V/\partial S^2`.
        - ``vega``  : :math:`\partial V/\partial \sigma` per **unit** of
          volatility (divide by 100 for "per vol point").
        - ``theta`` : :math:`\partial V/\partial t` per **year** of calendar
          time (typically negative; divide by 365 for per-day).
        - ``rho``   : :math:`\partial V/\partial r` per unit of rate.

    Raises
    ------
    ValueError
        If inputs are non-positive where positivity is required, or
        ``option_type`` is invalid.

    Notes
    -----
    For a call (put analogues via symmetry):

    .. math::

        \Delta = e^{-qT} N(d_1), \quad
        \Gamma = \frac{e^{-qT} \varphi(d_1)}{S \sigma \sqrt{T}}, \quad
        \mathcal{V} = S e^{-qT} \varphi(d_1) \sqrt{T},

    .. math::

        \Theta = -\frac{S e^{-qT} \varphi(d_1) \sigma}{2\sqrt{T}}
                 + q S e^{-qT} N(d_1) - r K e^{-rT} N(d_2), \quad
        \rho = K T e^{-rT} N(d_2).
    """
    _validate_option_type(option_type)
    for name, x in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        _validate_positive(name, np.asarray(x, dtype=float))

    d1, d2 = _d1_d2(*(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)))
    d1 = float(d1)
    d2 = float(d2)
    sqrt_T = np.sqrt(T)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * df_q * pdf_d1 * sqrt_T
    common_theta = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_T)

    if option_type == "call":
        delta = df_q * norm.cdf(d1)
        theta = (
            common_theta
            + q * S * df_q * norm.cdf(d1)
            - r * K * df_r * norm.cdf(d2)
        )
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = (
            common_theta
            - q * S * df_q * norm.cdf(-d1)
            + r * K * df_r * norm.cdf(-d2)
        )
        rho = -K * T * df_r * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
) -> float:
    r"""Black-Scholes implied volatility via Brent's root-finding method.

    Solves :math:`\sigma^\ast` such that
    :math:`\text{BS}(S, K, T, r, \sigma^\ast, q) = \text{price}`.

    Parameters
    ----------
    price : float
        Observed option price.
    S, K, T, r, option_type, q
        As in :func:`bs_price`.
    tol : float, optional
        Absolute tolerance passed to the Brent solver. Default ``1e-8``.

    Returns
    -------
    float
        Implied volatility (annualized decimal), or ``np.nan`` if the price
        lies outside the no-arbitrage bounds or no root is bracketed in
        :math:`\sigma \in [10^{-9},\, 5]`.

    Raises
    ------
    ValueError
        If ``S``, ``K`` or ``T`` are non-positive or ``option_type`` invalid.

    Notes
    -----
    No-arbitrage bounds used for early rejection:

    .. math::

        \max(S e^{-qT} - K e^{-rT}, 0) \le C < S e^{-qT}, \qquad
        \max(K e^{-rT} - S e^{-qT}, 0) \le P < K e^{-rT}.

    Deep ITM/OTM quotes near the intrinsic bound have vanishing vega; Brent's
    method remains robust there since it does not rely on derivatives.
    """
    _validate_option_type(option_type)
    for name, x in (("S", S), ("K", K), ("T", T)):
        _validate_positive(name, np.asarray(x, dtype=float))

    fwd_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        lower, upper = max(fwd_S - disc_K, 0.0), fwd_S
    else:
        lower, upper = max(disc_K - fwd_S, 0.0), disc_K

    # Reject prices violating (or numerically at) the no-arbitrage bounds.
    if not (lower - tol <= price < upper):
        return float("nan")

    sigma_lo, sigma_hi = 1e-9, 5.0

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type=option_type, q=q) - price

    f_lo, f_hi = objective(sigma_lo), objective(sigma_hi)
    if f_lo * f_hi > 0.0:
        return float("nan")
    try:
        root = brentq(objective, sigma_lo, sigma_hi, xtol=tol, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")
    return float(root)
