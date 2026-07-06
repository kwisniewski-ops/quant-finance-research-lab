"""Merton (1976) jump-diffusion model: series pricing and simulation.

Under the risk-neutral measure the Merton model augments geometric Brownian
motion with lognormally distributed jumps arriving at Poisson rate
:math:`\\lambda`:

.. math::

    \\frac{dS_t}{S_{t^-}} = (r - \\lambda \\bar{k})\\, dt
        + \\sigma\\, dW_t + (e^{Y} - 1)\\, dN_t,

where :math:`N_t \\sim \\mathrm{Poisson}(\\lambda t)`,
:math:`Y \\sim \\mathcal{N}(\\mu_J, \\sigma_J^2)` and
:math:`\\bar{k} = \\mathbb{E}[e^Y] - 1 = e^{\\mu_J + \\sigma_J^2/2} - 1`
is the compensator that keeps the discounted spot a martingale.

Conditioning on the number of jumps :math:`n` yields Merton's series
solution: a Poisson-weighted mixture of Black-Scholes prices with adjusted
rate and volatility.

References
----------
Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns Are
Discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.

Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
Springer. (Section 3.5, jump-diffusion simulation.)
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from src.models.black_scholes import bs_price

__all__ = ["merton_price", "simulate_merton"]


def merton_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    option_type: str = "call",
    n_terms: int = 60,
) -> float:
    r"""Merton jump-diffusion price of a European option (series solution).

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
    sigma : float
        Diffusive volatility (annualized decimal, must be > 0).
    lam : float
        Jump intensity :math:`\lambda \ge 0` (expected jumps per year).
    mu_j : float
        Mean of the lognormal jump size :math:`Y \sim N(\mu_J, \sigma_J^2)`.
    sigma_j : float
        Standard deviation of the jump size (must be >= 0).
    option_type : {"call", "put"}, optional
        Payoff type. Default ``"call"``.
    n_terms : int, optional
        Number of terms in the Poisson series. Default ``60``.

    Returns
    -------
    float
        Option value.

    Raises
    ------
    ValueError
        On invalid inputs.

    Notes
    -----
    With :math:`\bar{k} = e^{\mu_J + \sigma_J^2/2} - 1` and
    :math:`\lambda' = \lambda (1 + \bar{k})`:

    .. math::

        V = \sum_{n=0}^{\infty}
            \frac{e^{-\lambda' T} (\lambda' T)^n}{n!}\,
            \mathrm{BS}\!\left(S, K, T, r_n, \sigma_n\right),

    .. math::

        r_n = r - \lambda \bar{k} + \frac{n \ln(1 + \bar{k})}{T}, \qquad
        \sigma_n^2 = \sigma^2 + \frac{n \sigma_J^2}{T}.

    Poisson weights are computed in log-space via ``gammaln`` for numerical
    stability at large :math:`n`. With ``lam=0`` the series collapses to the
    Black-Scholes price exactly.
    """
    if option_type not in ("call", "put"):
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )
    for name, x in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        if x <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {name}={x}")
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got lam={lam}")
    if sigma_j < 0.0:
        raise ValueError(f"sigma_j must be non-negative, got sigma_j={sigma_j}")
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")

    if lam == 0.0:
        return float(bs_price(S, K, T, r, sigma, option_type=option_type))

    k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lam_prime = lam * (1.0 + k_bar)

    n = np.arange(n_terms)
    log_weights = -lam_prime * T + n * np.log(lam_prime * T) - gammaln(n + 1.0)
    weights = np.exp(log_weights)

    r_n = r - lam * k_bar + n * np.log1p(k_bar) / T
    sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)

    bs_values = bs_price(S, K, T, r_n, sigma_n, option_type=option_type)
    return float(np.sum(weights * np.asarray(bs_values)))


def simulate_merton(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Simulate Merton jump-diffusion paths (compensated drift convention).

    The log-price is advanced exactly between grid points, conditioning on
    the number of jumps per step:

    .. math::

        \ln S_{t+\Delta t} = \ln S_t
            + \left(\mu - \lambda \bar{k} - \tfrac{1}{2}\sigma^2\right)
              \Delta t
            + \sigma \sqrt{\Delta t}\, Z
            + \sum_{i=1}^{N} Y_i,

    with :math:`N \sim \mathrm{Poisson}(\lambda \Delta t)` and
    :math:`\sum_i Y_i \mid N \sim \mathcal{N}(N \mu_J, N \sigma_J^2)`.
    The compensator :math:`-\lambda \bar{k}` makes ``mu`` the total expected
    growth rate: :math:`\mathbb{E}[S_T] = S_0 e^{\mu T}`. Passing ``mu=r``
    therefore simulates risk-neutral dynamics consistent with
    :func:`merton_price`.

    Parameters
    ----------
    S0 : float
        Initial spot (must be > 0).
    mu : float
        Total expected growth rate of the spot (see above).
    sigma : float
        Diffusive volatility (must be > 0).
    lam : float
        Jump intensity (must be >= 0).
    mu_j : float
        Mean jump size in log-space.
    sigma_j : float
        Jump-size standard deviation (must be >= 0).
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
    np.ndarray
        Simulated price paths, shape ``(n_paths, n_steps + 1)``, with
        column 0 equal to ``S0``.
    """
    for name, x in (("S0", S0), ("sigma", sigma), ("T", T)):
        if x <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {name}={x}")
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got lam={lam}")
    if sigma_j < 0.0:
        raise ValueError(f"sigma_j must be non-negative, got sigma_j={sigma_j}")
    if n_steps < 1 or n_paths < 1:
        raise ValueError(
            f"n_steps and n_paths must be >= 1, got {n_steps}, {n_paths}"
        )
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    drift = (mu - lam * k_bar - 0.5 * sigma**2) * dt

    # Diffusive increments for all steps at once.
    z = rng.standard_normal((n_paths, n_steps))
    increments = drift + sigma * sqrt_dt * z

    if lam > 0.0:
        n_jumps = rng.poisson(lam * dt, size=(n_paths, n_steps))
        jump_z = rng.standard_normal((n_paths, n_steps))
        jump_sum = n_jumps * mu_j + np.sqrt(n_jumps) * sigma_j * jump_z
        increments += jump_sum

    log_paths = np.cumsum(increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths
