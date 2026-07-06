"""Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

European and American vanilla options are priced by solving the
Black-Scholes partial differential equation

.. math::

    \\frac{\\partial V}{\\partial t}
    + \\frac{1}{2}\\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2}
    + r S \\frac{\\partial V}{\\partial S} - r V = 0,

backwards from the terminal payoff. Crank-Nicolson averages the explicit
and implicit Euler stencils, is unconditionally stable, and is second-order
accurate in both :math:`\\Delta S` and :math:`\\Delta t`. The American
early-exercise constraint turns each time step into a linear
complementarity problem, solved here with projected successive
over-relaxation (PSOR).

References
----------
Crank, J. and Nicolson, P. (1947). "A Practical Method for Numerical
Evaluation of Solutions of Partial Differential Equations of the
Heat-Conduction Type." *Proc. Cambridge Phil. Soc.*, 43(1), 50-67.

Wilmott, P., Howison, S., and Dewynne, J. (1995). *The Mathematics of
Financial Derivatives*. Cambridge University Press. (Chapter 9, LCP/PSOR.)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

__all__ = ["cn_bs_price", "cn_bs_grid"]


def _validate(
    S0: float, K: float, T: float, sigma: float,
    option_type: str, n_s: int, n_t: int, s_max_mult: float,
) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )
    for name, x in (("S0", S0), ("K", K), ("T", T), ("sigma", sigma)):
        if x <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {name}={x}")
    if n_s < 3 or n_t < 1:
        raise ValueError(f"Need n_s >= 3 and n_t >= 1, got n_s={n_s}, n_t={n_t}")
    if s_max_mult <= 1.0:
        raise ValueError(f"s_max_mult must exceed 1, got {s_max_mult}")


def _psor(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    psi: np.ndarray,
    x0: np.ndarray,
    omega: float = 1.2,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> np.ndarray:
    """Projected SOR for the LCP ``M x >= rhs, x >= psi`` (tridiagonal M)."""
    x = np.maximum(x0.copy(), psi)
    n = len(rhs)
    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(n):
            resid = rhs[i]
            if i > 0:
                resid -= lower[i] * x[i - 1]
            if i < n - 1:
                resid -= upper[i] * x[i + 1]
            x_gs = resid / diag[i]
            x_new = max(psi[i], x[i] + omega * (x_gs - x[i]))
            diff = abs(x_new - x[i])
            if diff > max_diff:
                max_diff = diff
            x[i] = x_new
        if max_diff < tol:
            break
    return x


def cn_bs_grid(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    american: bool = False,
    n_s: int = 200,
    n_t: int = 200,
    s_max_mult: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Solve the Black-Scholes PDE on a Crank-Nicolson grid.

    Parameters
    ----------
    S0 : float
        Spot price; sets the grid extent (must be > 0). The grid itself does
        not otherwise depend on ``S0``.
    K : float
        Strike (must be > 0).
    T : float
        Time to expiry in years (must be > 0).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (must be > 0).
    option_type : {"call", "put"}, optional
        Payoff type. Default ``"call"``.
    american : bool, optional
        Enforce the early-exercise constraint via PSOR. Default ``False``.
    n_s : int, optional
        Number of spatial intervals (grid has ``n_s + 1`` nodes on
        :math:`[0, S_{\max}]`). Default ``200``.
    n_t : int, optional
        Number of time steps. Default ``200``.
    s_max_mult : float, optional
        Upper grid boundary :math:`S_{\max} = \text{s\_max\_mult} \cdot
        \max(S_0, K)`. Default ``4.0``.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        ``(s_grid, t_grid, value_surface)`` where ``s_grid`` has shape
        ``(n_s + 1,)``, ``t_grid`` has shape ``(n_t + 1,)`` in **calendar
        time** from 0 to T, and ``value_surface`` has shape
        ``(n_t + 1, n_s + 1)``: row ``j`` is the option value at time
        ``t_grid[j]``, so row 0 is today's value function and the last row
        is the terminal payoff.

    Notes
    -----
    With :math:`L` the discretized spatial operator
    :math:`\tfrac{1}{2}\sigma^2 S^2 \partial_{SS} + r S \partial_S - r`
    (central differences), each step in backward time :math:`\tau = T - t`
    solves

    .. math::

        \left(I - \tfrac{\Delta\tau}{2} L\right) V^{j+1}
        = \left(I + \tfrac{\Delta\tau}{2} L\right) V^{j},

    a tridiagonal system handled by banded LU. Dirichlet boundaries:
    :math:`V(0) = K e^{-r\tau}` (put) or 0 (call), and
    :math:`V(S_{\max}) = S_{\max} - K e^{-r\tau}` (call) or 0 (put); the
    American constraint replaces discounted by intrinsic values at the
    boundary and projects the interior solution onto the payoff via PSOR.
    """
    _validate(S0, K, T, sigma, option_type, n_s, n_t, s_max_mult)

    s_max = s_max_mult * max(S0, K)
    s_grid = np.linspace(0.0, s_max, n_s + 1)
    dt = T / n_t
    t_grid = np.linspace(0.0, T, n_t + 1)

    sign = 1.0 if option_type == "call" else -1.0
    payoff = np.maximum(sign * (s_grid - K), 0.0)

    # Terminal condition: replace nodal payoff values near the strike by
    # their cell averages. This smooths the kink at K (which otherwise
    # dominates the spatial error constant) without biasing the solution;
    # the American exercise constraint below still uses the exact payoff.
    ds_half = 0.5 * (s_grid[1] - s_grid[0])
    ds_cell = s_grid[1] - s_grid[0]
    terminal = payoff.copy()
    kink = np.abs(s_grid - K) < ds_half
    if np.any(kink):
        lo_edge = s_grid[kink] - ds_half
        hi_edge = s_grid[kink] + ds_half
        if option_type == "call":
            terminal[kink] = 0.5 * (hi_edge - K) ** 2 / ds_cell
        else:
            terminal[kink] = 0.5 * (K - lo_edge) ** 2 / ds_cell

    # Spatial operator coefficients at interior nodes i = 1..n_s-1.
    i = np.arange(1, n_s)
    ds = s_grid[1] - s_grid[0]
    s_i = s_grid[i]
    alpha = 0.5 * sigma**2 * s_i**2 / ds**2
    beta = r * s_i / (2.0 * ds)
    a = alpha - beta          # coefficient on V_{i-1}
    b = -2.0 * alpha - r      # coefficient on V_i
    c = alpha + beta          # coefficient on V_{i+1}

    # Theta-scheme matrices M = I - theta dt L (implicit side) and
    # E = I + (1 - theta) dt L (explicit side). Crank-Nicolson uses
    # theta = 1/2; the first two steps use theta = 1 (implicit Euler,
    # Rannacher start-up) to damp the spurious oscillations seeded by the
    # non-smooth payoff and restore second-order convergence.
    n_rannacher = 2

    def _theta_matrices(theta: float):
        m_lower = -theta * dt * a
        m_diag = 1.0 - theta * dt * b
        m_upper = -theta * dt * c
        e_lower = (1.0 - theta) * dt * a
        e_diag = 1.0 + (1.0 - theta) * dt * b
        e_upper = (1.0 - theta) * dt * c
        ab = np.zeros((3, n_s - 1))  # banded storage for solve_banded
        ab[0, 1:] = m_upper[:-1]
        ab[1, :] = m_diag
        ab[2, :-1] = m_lower[1:]
        return m_lower, m_diag, m_upper, e_lower, e_diag, e_upper, ab

    mats_cn = _theta_matrices(0.5)
    mats_imp = _theta_matrices(1.0)

    surface_tau = np.empty((n_t + 1, n_s + 1))
    surface_tau[0] = payoff
    v = terminal.copy()

    for j in range(1, n_t + 1):
        m_lower, m_diag, m_upper, e_lower, e_diag, e_upper, ab = (
            mats_imp if j <= n_rannacher else mats_cn
        )
        tau_new = j * dt
        tau_old = (j - 1) * dt

        if option_type == "call":
            lo_old, lo_new = 0.0, 0.0
            hi_old = s_max - K * np.exp(-r * tau_old)
            hi_new = s_max - K * np.exp(-r * tau_new)
            if american:
                hi_old = max(hi_old, s_max - K)
                hi_new = max(hi_new, s_max - K)
        else:
            hi_old, hi_new = 0.0, 0.0
            lo_old = K * np.exp(-r * tau_old)
            lo_new = K * np.exp(-r * tau_new)
            if american:
                lo_old, lo_new = float(K), float(K)

        rhs = e_diag * v[1:-1]
        rhs[1:] += e_lower[1:] * v[1:-2]
        rhs[:-1] += e_upper[:-1] * v[2:-1]
        # Boundary contributions from both time levels.
        rhs[0] += e_lower[0] * lo_old + (-m_lower[0]) * lo_new
        rhs[-1] += e_upper[-1] * hi_old + (-m_upper[-1]) * hi_new

        if american:
            interior = _psor(
                m_lower, m_diag, m_upper, rhs, psi=payoff[1:-1], x0=v[1:-1]
            )
        else:
            interior = solve_banded((1, 1), ab, rhs)

        v = np.empty(n_s + 1)
        v[0], v[-1] = lo_new, hi_new
        v[1:-1] = interior
        if american:
            v = np.maximum(v, payoff)
        surface_tau[j] = v

    # Convert from backward time tau to calendar time t = T - tau.
    value_surface = surface_tau[::-1].copy()
    return s_grid, t_grid, value_surface


def cn_bs_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    american: bool = False,
    n_s: int = 200,
    n_t: int = 200,
    s_max_mult: float = 4.0,
) -> float:
    r"""Crank-Nicolson finite-difference price of a vanilla option.

    Solves the Black-Scholes PDE via :func:`cn_bs_grid` and interpolates
    today's value function at ``S0`` with a cubic spline (linear
    interpolation would reintroduce an :math:`O(\Gamma \, \Delta S^2)`
    read-out error when ``S0`` falls between grid nodes).

    Parameters
    ----------
    S0, K, T, r, sigma, option_type, american, n_s, n_t, s_max_mult
        As in :func:`cn_bs_grid`.

    Returns
    -------
    float
        Option value at ``(t=0, S=S0)``.

    Raises
    ------
    ValueError
        On invalid inputs (see :func:`cn_bs_grid`).
    """
    s_grid, _, surface = cn_bs_grid(
        S0, K, T, r, sigma,
        option_type=option_type, american=american,
        n_s=n_s, n_t=n_t, s_max_mult=s_max_mult,
    )
    return float(CubicSpline(s_grid, surface[0])(S0))
