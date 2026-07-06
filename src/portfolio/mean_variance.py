"""Mean-variance portfolio optimization.

Classical Markowitz portfolio construction: minimum-variance and
maximum-Sharpe portfolios plus the efficient frontier, solved
numerically with SLSQP under box bounds and a full-investment
constraint.

The optimization problem for the frontier point targeting return
:math:`\\bar{r}` is

.. math::

    \\min_{w} \\; w^\\top \\Sigma w
    \\quad \\text{s.t.} \\quad
    \\mathbf{1}^\\top w = 1, \\;
    \\mu^\\top w \\ge \\bar{r}, \\;
    l \\le w_i \\le u .

References
----------
Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*,
7(1), 77-91.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@contextmanager
def _quiet_slsqp() -> Iterator[None]:
    """Silence scipy's benign 'x outside bounds, clipping' RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Values in x were outside bounds", category=RuntimeWarning
        )
        yield

__all__ = ["PortfolioResult", "min_variance", "max_sharpe", "efficient_frontier"]


@dataclass
class PortfolioResult:
    """Container for a single optimized portfolio.

    Attributes
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset; sums to 1.
    expected_return : float
        Annualized expected portfolio return ``w' mu`` (NaN if ``mu``
        was not supplied).
    volatility : float
        Annualized portfolio volatility ``sqrt(w' Sigma w)``.
    sharpe : float
        Sharpe ratio ``(expected_return - rf) / volatility`` (NaN if
        ``mu`` was not supplied).
    """

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def _validate_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Validate a covariance matrix: square, symmetric, aligned labels."""
    if not isinstance(cov, pd.DataFrame):
        raise TypeError(f"cov must be a pd.DataFrame, got {type(cov).__name__}")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if not cov.index.equals(cov.columns):
        raise ValueError("cov index and columns must be identical (same assets, same order)")
    vals = cov.to_numpy(dtype=float)
    if np.isnan(vals).any():
        raise ValueError("cov contains NaN values")
    if not np.allclose(vals, vals.T, atol=1e-10):
        raise ValueError("cov must be symmetric")
    return cov


def _align_mu_cov(mu: pd.Series, cov: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Check mu/cov share the same assets and align mu to cov's ordering."""
    if not isinstance(mu, pd.Series):
        raise TypeError(f"mu must be a pd.Series, got {type(mu).__name__}")
    if set(mu.index) != set(cov.index):
        raise ValueError(
            "mu and cov are misaligned: "
            f"mu assets {sorted(map(str, mu.index))} != cov assets {sorted(map(str, cov.index))}"
        )
    return mu.reindex(cov.index), cov


def _validate_bounds(bounds: tuple[float, float], n: int) -> tuple[float, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    if lo > hi:
        raise ValueError(f"bounds lower {lo} exceeds upper {hi}")
    if n * hi < 1.0 - 1e-12 or n * lo > 1.0 + 1e-12:
        raise ValueError(f"bounds {bounds} make the full-investment constraint infeasible for {n} assets")
    return lo, hi


def _feasible_start(n: int, lo: float, hi: float) -> np.ndarray:
    w0 = np.full(n, 1.0 / n)
    return np.clip(w0, lo, hi)


def _solve(
    objective,
    n: int,
    lo: float,
    hi: float,
    extra_constraints: list | None = None,
) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        constraints += extra_constraints
    # SLSQP with a very tight ftol can stall on large, ill-conditioned
    # universes.  Retry with progressively relaxed tolerances and a
    # perturbed feasible start before giving up.
    attempts = (
        (_feasible_start(n, lo, hi), 1e-14, 1000),
        (_feasible_start(n, lo, hi), 1e-12, 2000),
        (
            np.clip(
                _feasible_start(n, lo, hi)
                + np.random.default_rng(0).uniform(-0.5 / n, 0.5 / n, n),
                lo,
                hi,
            ),
            1e-10,
            3000,
        ),
    )
    res = None
    for w0, ftol, maxiter in attempts:
        w0 = w0 / w0.sum() if w0.sum() > 0 else _feasible_start(n, lo, hi)
        with _quiet_slsqp():
            res = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=[(lo, hi)] * n,
                constraints=constraints,
                options={"maxiter": maxiter, "ftol": ftol},
            )
        if res.success:
            break
        # Accept a technically "failed" solve whose constraint violation is
        # negligible — SLSQP often reports failure on the last micro-step.
        w_try = np.asarray(res.x, dtype=float)
        feasible = (
            abs(w_try.sum() - 1.0) < 1e-8
            and np.all(w_try >= lo - 1e-8)
            and np.all(w_try <= hi + 1e-8)
            and all(
                abs(c["fun"](w_try)) < 1e-8 if c["type"] == "eq" else c["fun"](w_try) > -1e-8
                for c in constraints
            )
        )
        if feasible and np.isfinite(res.fun):
            break
    else:
        raise RuntimeError(f"SLSQP failed to converge: {res.message}")
    w = np.asarray(res.x, dtype=float)
    # Clean numerical dust and renormalize within bounds.
    w = np.clip(w, lo, hi)
    w = w / w.sum()
    return w


def _make_result(
    w: np.ndarray,
    cov: pd.DataFrame,
    mu: pd.Series | None,
    rf: float,
) -> PortfolioResult:
    sigma = cov.to_numpy(dtype=float)
    vol = float(np.sqrt(w @ sigma @ w))
    if mu is not None:
        ret = float(mu.to_numpy(dtype=float) @ w)
        sharpe = (ret - rf) / vol if vol > 0 else np.nan
    else:
        ret, sharpe = float("nan"), float("nan")
    return PortfolioResult(
        weights=pd.Series(w, index=cov.index, name="weight"),
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
    )


def min_variance(
    cov: pd.DataFrame,
    bounds: tuple[float, float] = (0.0, 1.0),
    mu: pd.Series | None = None,
    rf: float = 0.0,
) -> PortfolioResult:
    """Minimum-variance portfolio.

    Solves :math:`\\min_w w^\\top \\Sigma w` subject to full investment
    and box bounds.

    Parameters
    ----------
    cov : pd.DataFrame
        Annualized covariance matrix (square, symmetric, labeled).
    bounds : tuple of float, default (0.0, 1.0)
        (lower, upper) bound applied to every weight.
    mu : pd.Series, optional
        Annualized expected returns. If omitted, ``expected_return``
        and ``sharpe`` in the result are NaN.
    rf : float, default 0.0
        Annualized risk-free rate used for the Sharpe ratio.

    Returns
    -------
    PortfolioResult

    References
    ----------
    Markowitz (1952).
    """
    cov = _validate_cov(cov)
    if mu is not None:
        mu, cov = _align_mu_cov(mu, cov)
    n = cov.shape[0]
    lo, hi = _validate_bounds(bounds, n)
    sigma = cov.to_numpy(dtype=float)
    w = _solve(lambda w: w @ sigma @ w, n, lo, hi)
    return _make_result(w, cov, mu, rf)


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> PortfolioResult:
    """Maximum-Sharpe (tangency) portfolio.

    Maximizes :math:`(w^\\top \\mu - r_f) / \\sqrt{w^\\top \\Sigma w}`
    subject to full investment and box bounds.

    Parameters
    ----------
    mu : pd.Series
        Annualized expected returns indexed by asset.
    cov : pd.DataFrame
        Annualized covariance matrix; assets must match ``mu``.
    rf : float, default 0.0
        Annualized risk-free rate.
    bounds : tuple of float, default (0.0, 1.0)
        (lower, upper) bound applied to every weight.

    Returns
    -------
    PortfolioResult

    References
    ----------
    Markowitz (1952); Sharpe, W. F. (1966). "Mutual Fund Performance".
    """
    cov = _validate_cov(cov)
    mu, cov = _align_mu_cov(mu, cov)
    n = cov.shape[0]
    lo, hi = _validate_bounds(bounds, n)
    sigma = cov.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ sigma @ w, 1e-18))
        return -(w @ mu_vec - rf) / vol

    w = _solve(neg_sharpe, n, lo, hi)
    return _make_result(w, cov, mu, rf)


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    rf: float = 0.0,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Trace the constrained efficient frontier.

    For ``n_points`` target returns between the minimum-variance
    portfolio's return and the maximum attainable return, solves the
    variance-minimization problem with an added return constraint
    ``w' mu >= target``.

    Parameters
    ----------
    mu : pd.Series
        Annualized expected returns.
    cov : pd.DataFrame
        Annualized covariance matrix.
    n_points : int, default 50
        Number of frontier points.
    rf : float, default 0.0
        Risk-free rate for Sharpe ratios.
    bounds : tuple of float, default (0.0, 1.0)
        Per-asset weight bounds.

    Returns
    -------
    pd.DataFrame
        One row per frontier point with columns ``expected_return``,
        ``volatility``, ``sharpe`` plus one weight column per asset.

    References
    ----------
    Markowitz (1952).
    """
    cov = _validate_cov(cov)
    mu, cov = _align_mu_cov(mu, cov)
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    n = cov.shape[0]
    lo, hi = _validate_bounds(bounds, n)
    sigma = cov.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)

    ret_min = min_variance(cov, bounds=bounds, mu=mu, rf=rf).expected_return
    # Max attainable return under the constraints.
    w_max = _solve(lambda w: -(w @ mu_vec), n, lo, hi)
    ret_max = float(w_max @ mu_vec)

    targets = np.linspace(ret_min, ret_max, n_points)
    rows: list[dict[str, float]] = []
    for target in targets:
        w = _solve(
            lambda w: w @ sigma @ w,
            n,
            lo,
            hi,
            extra_constraints=[{"type": "ineq", "fun": lambda w, t=target: w @ mu_vec - t}],
        )
        res = _make_result(w, cov, mu, rf)
        row: dict[str, float] = {
            "expected_return": res.expected_return,
            "volatility": res.volatility,
            "sharpe": res.sharpe,
        }
        row.update({str(a): float(x) for a, x in res.weights.items()})
        rows.append(row)
    return pd.DataFrame(rows)
