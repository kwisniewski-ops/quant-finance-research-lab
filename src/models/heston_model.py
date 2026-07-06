"""Heston (1993) stochastic-volatility model: semi-analytic pricing and MC.

Under the risk-neutral measure the Heston model reads

.. math::

    dS_t = r S_t\\, dt + \\sqrt{v_t}\\, S_t\\, dW_t^S, \\qquad
    dv_t = \\kappa (\\theta - v_t)\\, dt + \\xi \\sqrt{v_t}\\, dW_t^v,

with :math:`d\\langle W^S, W^v \\rangle_t = \\rho\\, dt`. European calls admit
the semi-closed form

.. math::

    C = S\\, P_1 - K e^{-rT} P_2, \\qquad
    P_j = \\tfrac{1}{2} + \\frac{1}{\\pi} \\int_0^\\infty
        \\mathrm{Re}\\!\\left[ \\frac{e^{-iu \\ln K} f_j(u)}{iu} \\right] du,

where :math:`f_j` are the characteristic functions of the log-price under
the two pricing measures. This module uses the numerically stable
"little Heston trap" formulation of Albrecher et al. (2007), which keeps the
complex logarithm on its principal branch for all maturities, integrated
with 128-node Gauss-Legendre quadrature on a transformed semi-infinite
domain.

References
----------
Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic
Volatility with Applications to Bond and Currency Options." *Review of
Financial Studies*, 6(2), 327-343.

Albrecher, H., Mayer, P., Schoutens, W., and Tistaert, J. (2007). "The
Little Heston Trap." *Wilmott Magazine*, January, 83-92.

Lord, R., Koekkoek, R., and van Dijk, D. (2010). "A Comparison of Biased
Simulation Schemes for Stochastic Volatility Models." *Quantitative
Finance*, 10(2), 177-194.  (Full-truncation Euler.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HestonParams", "heston_price", "simulate_heston"]

# 128-node Gauss-Legendre rule on [0, 1], computed once at import.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)
_GL_T = 0.5 * (_GL_NODES + 1.0)          # nodes mapped to (0, 1)
_GL_W = 0.5 * _GL_WEIGHTS                # weights for [0, 1]

# Transform u = c * t / (1 - t) maps (0, 1) -> (0, inf); Jacobian c/(1-t)^2.
# The scale c is chosen per call, adapted to the maturity: the integrand's
# effective support grows like 1/sqrt(T), so nodes are spread accordingly.
_U_SCALE_BASE = 8.0


def _u_grid(T: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature abscissae and weights (incl. Jacobian) for maturity T."""
    c = _U_SCALE_BASE / np.sqrt(min(T, 1.0))
    u = c * _GL_T / (1.0 - _GL_T)
    w = _GL_W * c / (1.0 - _GL_T) ** 2
    return u, w


@dataclass(frozen=True)
class HestonParams:
    """Parameters of the Heston stochastic-volatility model.

    Attributes
    ----------
    v0 : float
        Initial instantaneous variance (must be >= 0).
    kappa : float
        Mean-reversion speed of the variance process (must be > 0).
    theta : float
        Long-run variance level (must be > 0).
    xi : float
        Volatility of variance ("vol of vol", must be > 0).
    rho : float
        Correlation between the spot and variance Brownian motions,
        in [-1, 1].
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 < 0.0:
            raise ValueError(f"v0 must be non-negative, got v0={self.v0}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got kappa={self.kappa}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be positive, got theta={self.theta}")
        if self.xi <= 0.0:
            raise ValueError(f"xi must be positive, got xi={self.xi}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1], got rho={self.rho}")

    @property
    def feller_satisfied(self) -> bool:
        r"""Whether the Feller condition :math:`2\kappa\theta \ge \xi^2` holds.

        When satisfied, the CIR variance process almost surely stays
        strictly positive; when violated, the origin is attainable and
        simulation schemes must handle negative-variance excursions.
        """
        return 2.0 * self.kappa * self.theta >= self.xi**2


def _heston_char_fn(
    u: np.ndarray,
    j: int,
    x: float,
    T: float,
    r: float,
    p: HestonParams,
) -> np.ndarray:
    r"""Heston characteristic function :math:`f_j(u)`, little-trap form.

    Uses :math:`g_j = (b_j - \rho\xi i u - d_j)/(b_j - \rho\xi i u + d_j)`
    (the "minus" root convention), for which
    :math:`|g_j e^{-d_j T}| < 1` and the complex logarithm never crosses the
    negative real axis (Albrecher et al. 2007).
    """
    iu = 1j * u
    a = p.kappa * p.theta
    if j == 1:
        b = p.kappa - p.rho * p.xi
        u_j = 0.5
    else:
        b = p.kappa
        u_j = -0.5

    beta = b - p.rho * p.xi * iu
    d = np.sqrt(beta**2 - p.xi**2 * (2.0 * u_j * iu - u**2))
    g = (beta - d) / (beta + d)

    exp_dT = np.exp(-d * T)
    C = r * iu * T + (a / p.xi**2) * (
        (beta - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = ((beta - d) / p.xi**2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    return np.exp(C + D * p.v0 + iu * x)


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: str = "call",
) -> float:
    r"""Semi-analytic Heston price of a European option.

    Parameters
    ----------
    S : float
        Spot price (must be > 0).
    K : float
        Strike (must be > 0).
    T : float
        Time to expiry in years (must be > 0).
    r : float
        Continuously compounded risk-free rate.
    params : HestonParams
        Model parameters.
    option_type : {"call", "put"}, optional
        Payoff type; puts obtained via put-call parity. Default ``"call"``.

    Returns
    -------
    float
        Option value.

    Raises
    ------
    ValueError
        If ``S``, ``K`` or ``T`` non-positive, or ``option_type`` invalid.

    Notes
    -----
    The in-the-money probabilities are

    .. math::

        P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty
              \mathrm{Re}\!\left[
              \frac{e^{-iu\ln K} f_j(u)}{iu} \right] du, \quad j = 1, 2,

    evaluated with a 128-node Gauss-Legendre rule under the substitution
    :math:`u = c\,t/(1-t)` on :math:`t \in (0, 1)`, which covers the full
    semi-infinite domain and exploits the exponential decay of the
    integrand.
    """
    if option_type not in ("call", "put"):
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )
    for name, x in (("S", S), ("K", K), ("T", T)):
        if x <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {name}={x}")

    x_log = float(np.log(S))
    log_K = float(np.log(K))
    u, w = _u_grid(T)

    probs = []
    for j in (1, 2):
        f = _heston_char_fn(u, j, x_log, T, r, params)
        integrand = np.real(np.exp(-1j * u * log_K) * f / (1j * u))
        integral = float(np.sum(w * integrand))
        probs.append(0.5 + integral / np.pi)

    P1, P2 = probs
    call = S * P1 - K * np.exp(-r * T) * P2
    # Guard against tiny negative values from quadrature noise deep OTM.
    call = max(call, 0.0)
    if option_type == "call":
        return float(call)
    return float(call - S + K * np.exp(-r * T))


def simulate_heston(
    S0: float,
    params: HestonParams,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Simulate Heston paths with the full-truncation Euler scheme.

    The variance process is discretized as (Lord et al. 2010)

    .. math::

        v_{t+\Delta t} = v_t + \kappa(\theta - v_t^+)\Delta t
                         + \xi \sqrt{v_t^+}\sqrt{\Delta t}\, Z_v,
        \qquad v^+ = \max(v, 0),

    and the log-spot exactly integrates the drift given :math:`v_t^+`:

    .. math::

        \ln S_{t+\Delta t} = \ln S_t + (r - \tfrac{1}{2} v_t^+)\Delta t
            + \sqrt{v_t^+ \Delta t}\,
              \big(\rho Z_v + \sqrt{1-\rho^2}\, Z_\perp\big).

    Full truncation has the smallest positive bias among Euler variants and
    never produces negative variance inputs to the square root.

    Parameters
    ----------
    S0 : float
        Initial spot (must be > 0).
    params : HestonParams
        Model parameters.
    r : float
        Risk-free rate (drift of the spot under :math:`\mathbb{Q}`).
    T : float
        Horizon in years (must be > 0).
    n_steps : int
        Number of time steps (must be >= 1).
    n_paths : int
        Number of paths (must be >= 1).
    rng : np.random.Generator, optional
        Source of randomness; ``np.random.default_rng()`` if omitted.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(S_paths, v_paths)``, each of shape ``(n_paths, n_steps + 1)``.
        ``v_paths`` stores the truncated instantaneous variance
        ``max(v, 0)``; the internal Euler state may go negative (per the
        full-truncation scheme) but is never exposed.
    """
    if S0 <= 0.0:
        raise ValueError(f"S0 must be strictly positive, got S0={S0}")
    if T <= 0.0:
        raise ValueError(f"T must be strictly positive, got T={T}")
    if n_steps < 1 or n_paths < 1:
        raise ValueError(
            f"n_steps and n_paths must be >= 1, got {n_steps}, {n_paths}"
        )
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho = params.rho
    rho_perp = np.sqrt(1.0 - rho**2)

    S_paths = np.empty((n_paths, n_steps + 1))
    v_paths = np.empty((n_paths, n_steps + 1))
    log_S = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, params.v0)
    S_paths[:, 0] = S0
    v_paths[:, 0] = params.v0

    for k in range(1, n_steps + 1):
        z_v = rng.standard_normal(n_paths)
        z_perp = rng.standard_normal(n_paths)
        z_s = rho * z_v + rho_perp * z_perp

        v_plus = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_plus)
        log_S = log_S + (r - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * z_s
        v = v + params.kappa * (params.theta - v_plus) * dt \
            + params.xi * sqrt_v * sqrt_dt * z_v

        S_paths[:, k] = np.exp(log_S)
        v_paths[:, k] = np.maximum(v, 0.0)

    return S_paths, v_paths
