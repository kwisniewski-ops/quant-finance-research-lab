"""Monte Carlo path simulation and estimator utilities.

Simulators for the workhorse SDEs of the library — geometric Brownian
motion (exact scheme), Ornstein-Uhlenbeck (exact transition density), CIR
(full-truncation Euler), and a Markov regime-switching diffusion — plus a
discounted-payoff estimator with standard errors.

All simulators return arrays of shape ``(n_paths, n_steps + 1)`` with the
initial condition in column 0, and accept an optional
``np.random.Generator`` for reproducibility.

References
----------
Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
Springer.

Lord, R., Koekkoek, R., and van Dijk, D. (2010). "A Comparison of Biased
Simulation Schemes for Stochastic Volatility Models." *Quantitative
Finance*, 10(2), 177-194.

Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2),
357-384.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

__all__ = [
    "simulate_gbm",
    "simulate_ou",
    "simulate_cir",
    "simulate_regime_switching",
    "mc_price",
]


def _default_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _validate_grid(T: float, n_steps: int, n_paths: int) -> None:
    if T <= 0.0:
        raise ValueError(f"T must be strictly positive, got T={T}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got n_steps={n_steps}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got n_paths={n_paths}")


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    r"""Simulate geometric Brownian motion with the exact log-Euler scheme.

    .. math::

        S_{t+\Delta t} = S_t \exp\!\left[
            \left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t
            + \sigma \sqrt{\Delta t}\, Z \right],
        \qquad Z \sim \mathcal{N}(0, 1).

    The scheme is exact in distribution at the grid points (strong order
    :math:`\infty` on the grid), so ``n_steps`` only controls path
    granularity, not bias.

    Parameters
    ----------
    S0 : float
        Initial spot (must be > 0).
    mu : float
        Drift (annualized decimal).
    sigma : float
        Volatility (annualized decimal, must be > 0).
    T : float
        Horizon in years (must be > 0).
    n_steps : int
        Number of time steps.
    n_paths : int
        Total number of paths returned.
    rng : np.random.Generator, optional
        Source of randomness; ``np.random.default_rng()`` if omitted.
    antithetic : bool, optional
        If ``True``, generate ``n_paths // 2`` normal draws and mirror them
        (:math:`Z \mapsto -Z`), returning ``n_paths`` paths in total as
        antithetic pairs (variance-reduction; Glasserman 2004, §4.2).
        Requires ``n_paths`` to be even.

    Returns
    -------
    np.ndarray
        Paths of shape ``(n_paths, n_steps + 1)``; column 0 equals ``S0``.
        With ``antithetic=True``, row ``i`` and row ``i + n_paths//2`` form
        an antithetic pair.

    Raises
    ------
    ValueError
        On invalid inputs, or if ``antithetic=True`` and ``n_paths`` is odd.
    """
    if S0 <= 0.0:
        raise ValueError(f"S0 must be strictly positive, got S0={S0}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got sigma={sigma}")
    _validate_grid(T, n_steps, n_paths)
    if antithetic and n_paths % 2 != 0:
        raise ValueError(
            f"antithetic=True requires an even n_paths, got n_paths={n_paths}"
        )
    rng = _default_rng(rng)

    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        z = np.vstack([half, -half])
    else:
        z = rng.standard_normal((n_paths, n_steps))

    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def simulate_ou(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Simulate an Ornstein-Uhlenbeck process with its exact transition law.

    The OU SDE :math:`dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t` has a
    Gaussian transition density, sampled exactly:

    .. math::

        X_{t+\Delta t} = \theta + (X_t - \theta) e^{-\kappa \Delta t}
            + \sigma \sqrt{\frac{1 - e^{-2\kappa \Delta t}}{2\kappa}}\; Z.

    There is no discretization bias at the grid points; the terminal
    distribution is exactly
    :math:`\mathcal{N}\!\big(\theta + (x_0-\theta)e^{-\kappa T},\;
    \sigma^2 (1 - e^{-2\kappa T}) / (2\kappa)\big)`.

    Parameters
    ----------
    x0 : float
        Initial value.
    kappa : float
        Mean-reversion speed (must be > 0).
    theta : float
        Long-run mean.
    sigma : float
        Diffusion coefficient (must be > 0).
    T : float
        Horizon in years (must be > 0).
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    rng : np.random.Generator, optional
        Source of randomness; ``np.random.default_rng()`` if omitted.

    Returns
    -------
    np.ndarray
        Paths of shape ``(n_paths, n_steps + 1)``; column 0 equals ``x0``.
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be strictly positive, got kappa={kappa}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got sigma={sigma}")
    _validate_grid(T, n_steps, n_paths)
    rng = _default_rng(rng)

    dt = T / n_steps
    decay = np.exp(-kappa * dt)
    cond_std = sigma * np.sqrt((1.0 - decay**2) / (2.0 * kappa))

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    x = np.full(n_paths, float(x0))
    for k in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        x = theta + (x - theta) * decay + cond_std * z
        paths[:, k] = x
    return paths


def simulate_cir(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Simulate a CIR square-root process with full-truncation Euler.

    The CIR SDE :math:`dX_t = \kappa(\theta - X_t)\,dt
    + \sigma\sqrt{X_t}\,dW_t` is discretized as (Lord et al. 2010)

    .. math::

        \tilde{X}_{t+\Delta t} = \tilde{X}_t
            + \kappa(\theta - \tilde{X}_t^+)\Delta t
            + \sigma \sqrt{\tilde{X}_t^+}\sqrt{\Delta t}\, Z,
        \qquad X_t = \tilde{X}_t^+,

    where :math:`x^+ = \max(x, 0)`. Full truncation keeps the square-root
    argument non-negative regardless of the Feller condition
    :math:`2\kappa\theta \ge \sigma^2` and has the smallest bias among the
    standard Euler fixes. The **returned** paths are the truncated
    (non-negative) process :math:`X_t`.

    Parameters
    ----------
    x0 : float
        Initial value (must be >= 0).
    kappa : float
        Mean-reversion speed (must be > 0).
    theta : float
        Long-run mean (must be > 0).
    sigma : float
        Vol-of-vol coefficient (must be > 0).
    T : float
        Horizon in years (must be > 0).
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    rng : np.random.Generator, optional
        Source of randomness; ``np.random.default_rng()`` if omitted.

    Returns
    -------
    np.ndarray
        Non-negative paths of shape ``(n_paths, n_steps + 1)``.
    """
    if x0 < 0.0:
        raise ValueError(f"x0 must be non-negative, got x0={x0}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be strictly positive, got kappa={kappa}")
    if theta <= 0.0:
        raise ValueError(f"theta must be strictly positive, got theta={theta}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got sigma={sigma}")
    _validate_grid(T, n_steps, n_paths)
    rng = _default_rng(rng)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    x_tilde = np.full(n_paths, float(x0))
    for k in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        x_plus = np.maximum(x_tilde, 0.0)
        x_tilde = (
            x_tilde
            + kappa * (theta - x_plus) * dt
            + sigma * np.sqrt(x_plus) * sqrt_dt * z
        )
        paths[:, k] = np.maximum(x_tilde, 0.0)
    return paths


def simulate_regime_switching(
    S0: float,
    mus: Sequence[float],
    sigmas: Sequence[float],
    transition_matrix: Union[Sequence[Sequence[float]], np.ndarray],
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Simulate a Markov regime-switching geometric diffusion.

    A discrete-time Markov chain :math:`s_t \in \{0, \dots, m-1\}` with
    one-step transition matrix :math:`P` (applied per simulation step)
    selects the drift/volatility regime; conditional on the regime, the
    log-price advances as exact GBM:

    .. math::

        \ln S_{t+\Delta t} = \ln S_t
            + \left(\mu_{s_t} - \tfrac{1}{2}\sigma_{s_t}^2\right)\Delta t
            + \sigma_{s_t}\sqrt{\Delta t}\, Z.

    Regime models of this form (Hamilton 1989) reproduce volatility
    clustering and fat tails absent from single-regime GBM.

    Parameters
    ----------
    S0 : float
        Initial spot (must be > 0).
    mus : sequence of float
        Per-regime drifts (annualized), length ``m``.
    sigmas : sequence of float
        Per-regime volatilities (annualized, all > 0), length ``m``.
    transition_matrix : array_like
        Row-stochastic ``(m, m)`` matrix; entry ``[i, j]`` is the one-step
        probability of moving from regime ``i`` to regime ``j``.
    T : float
        Horizon in years (must be > 0).
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    rng : np.random.Generator, optional
        Source of randomness; ``np.random.default_rng()`` if omitted.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(paths, regime_paths)``: price paths of shape
        ``(n_paths, n_steps + 1)`` and integer regime labels of the same
        shape. All paths start in regime 0.

    Raises
    ------
    ValueError
        If shapes are inconsistent, the matrix is not row-stochastic, or
        any volatility is non-positive.
    """
    if S0 <= 0.0:
        raise ValueError(f"S0 must be strictly positive, got S0={S0}")
    _validate_grid(T, n_steps, n_paths)
    mus_arr = np.asarray(mus, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    P = np.asarray(transition_matrix, dtype=float)
    m = len(mus_arr)
    if len(sigmas_arr) != m:
        raise ValueError(
            f"mus and sigmas must have equal length, got {m} and {len(sigmas_arr)}"
        )
    if np.any(sigmas_arr <= 0.0):
        raise ValueError(f"All sigmas must be strictly positive, got {sigmas}")
    if P.shape != (m, m):
        raise ValueError(
            f"transition_matrix must have shape ({m}, {m}), got {P.shape}"
        )
    if np.any(P < 0.0) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("transition_matrix rows must be non-negative and sum to 1")
    rng = _default_rng(rng)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    cum_P = np.cumsum(P, axis=1)

    regimes = np.zeros((n_paths, n_steps + 1), dtype=np.int64)
    log_paths = np.empty((n_paths, n_steps + 1))
    log_paths[:, 0] = np.log(S0)
    state = np.zeros(n_paths, dtype=np.int64)

    for k in range(1, n_steps + 1):
        # Evolve the chain: inverse-CDF sampling against each row's cumsum.
        u = rng.random(n_paths)
        state = (u[:, None] > cum_P[state]).sum(axis=1)
        regimes[:, k] = state

        z = rng.standard_normal(n_paths)
        mu_k = mus_arr[state]
        sig_k = sigmas_arr[state]
        log_paths[:, k] = (
            log_paths[:, k - 1]
            + (mu_k - 0.5 * sig_k**2) * dt
            + sig_k * sqrt_dt * z
        )

    paths = np.exp(log_paths)
    paths[:, 0] = S0  # exact (avoids exp(log(S0)) round-off)
    return paths, regimes


def mc_price(payoff: np.ndarray, r: float, T: float) -> tuple[float, float]:
    r"""Discounted Monte Carlo estimator with standard error.

    .. math::

        \hat{V} = e^{-rT} \frac{1}{n} \sum_{i=1}^{n} X_i, \qquad
        \widehat{SE} = e^{-rT} \frac{s_X}{\sqrt{n}},

    where :math:`s_X` is the sample standard deviation (ddof=1) of the
    undiscounted payoffs.

    Parameters
    ----------
    payoff : np.ndarray
        Undiscounted terminal payoffs, one entry per path.
    r : float
        Continuously compounded discount rate.
    T : float
        Time to payment in years (must be >= 0).

    Returns
    -------
    tuple of (float, float)
        ``(price, standard_error)``.

    Raises
    ------
    ValueError
        If ``payoff`` has fewer than 2 entries or ``T`` is negative.
    """
    payoff = np.asarray(payoff, dtype=float).ravel()
    if payoff.size < 2:
        raise ValueError(
            f"payoff must contain at least 2 samples, got {payoff.size}"
        )
    if T < 0.0:
        raise ValueError(f"T must be non-negative, got T={T}")

    disc = np.exp(-r * T)
    price = disc * float(payoff.mean())
    se = disc * float(payoff.std(ddof=1)) / np.sqrt(payoff.size)
    return price, se
