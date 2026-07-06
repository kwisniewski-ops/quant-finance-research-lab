"""Cox-Ross-Rubinstein binomial tree for European and American options.

The CRR (1979) lattice discretizes geometric Brownian motion with up/down
multipliers

.. math::

    u = e^{\\sigma \\sqrt{\\Delta t}}, \\qquad d = 1/u, \\qquad
    p = \\frac{e^{(r - q)\\Delta t} - d}{u - d},

and prices by discounted risk-neutral backward induction. For American
options the continuation value is compared against immediate exercise at
every node. Convergence to Black-Scholes is :math:`O(1/n)` for European
payoffs.

References
----------
Cox, J. C., Ross, S. A., and Rubinstein, M. (1979). "Option Pricing: A
Simplified Approach." *Journal of Financial Economics*, 7(3), 229-263.
"""

from __future__ import annotations

import numpy as np

__all__ = ["binomial_price"]


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 500,
    option_type: str = "call",
    american: bool = False,
    q: float = 0.0,
) -> float:
    r"""Price a European or American option on a CRR binomial lattice.

    Parameters
    ----------
    S : float
        Spot price of the underlying (must be > 0).
    K : float
        Strike price (must be > 0).
    T : float
        Time to expiry in years (must be > 0).
    r : float
        Continuously compounded risk-free rate (annualized decimal).
    sigma : float
        Volatility (annualized decimal, must be > 0).
    n_steps : int, optional
        Number of time steps in the lattice. Default ``500``.
    option_type : {"call", "put"}, optional
        Payoff type. Default ``"call"``.
    american : bool, optional
        If ``True``, allow early exercise at every node. Default ``False``.
    q : float, optional
        Continuous dividend yield. Default ``0.0``.

    Returns
    -------
    float
        Option value at the root node.

    Raises
    ------
    ValueError
        If inputs are non-positive where positivity is required, if
        ``option_type`` is invalid, or if the induced risk-neutral
        probability lies outside :math:`(0, 1)` (increase ``n_steps``).

    Notes
    -----
    Backward induction is vectorized across each time slice:

    .. math::

        V_i^{(m)} = \max\Big( e^{-r \Delta t}
            \big[ p\, V_{i+1}^{(m+1)} + (1-p)\, V_i^{(m+1)} \big],\;
            \text{intrinsic}_i^{(m)} \cdot \mathbf{1}_{\text{american}} \Big).
    """
    if option_type not in ("call", "put"):
        raise ValueError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )
    for name, x in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        if x <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {name}={x}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"Risk-neutral probability p={p:.6f} outside (0, 1); "
            "increase n_steps or check (r - q) vs sigma."
        )
    disc = np.exp(-r * dt)

    # Terminal asset prices S * u^j * d^(n-j), j = 0..n.
    j = np.arange(n_steps + 1)
    s_terminal = S * u**j * d ** (n_steps - j)
    if option_type == "call":
        values = np.maximum(s_terminal - K, 0.0)
    else:
        values = np.maximum(K - s_terminal, 0.0)

    sign = 1.0 if option_type == "call" else -1.0
    for m in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(m + 1)
            s_slice = S * u**j * d ** (m - j)
            values = np.maximum(values, sign * (s_slice - K))

    return float(values[0])
