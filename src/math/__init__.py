"""Numerical methods: Monte Carlo, PDE solvers, linear algebra.

Public API re-exports for ``src.math``.
"""

from src.math.monte_carlo_methods import (
    mc_price,
    simulate_cir,
    simulate_gbm,
    simulate_ou,
    simulate_regime_switching,
)
from src.math.numerical_linear_algebra import (
    is_psd,
    ledoit_wolf_shrinkage,
    nearest_psd,
    safe_cholesky,
)
from src.math.pde_solvers import cn_bs_grid, cn_bs_price

__all__ = [
    "simulate_gbm",
    "simulate_ou",
    "simulate_cir",
    "simulate_regime_switching",
    "mc_price",
    "cn_bs_price",
    "cn_bs_grid",
    "nearest_psd",
    "is_psd",
    "ledoit_wolf_shrinkage",
    "safe_cholesky",
]
