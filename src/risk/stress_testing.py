"""Scenario analysis and stress testing.

Two complementary tools:

* **Discrete scenarios** — apply a vector of asset return shocks to a
  weight vector: :math:`\\Delta P = \\sum_i w_i \\, s_i`. A library of
  approximate historical episodes (GFC 2008, Covid March 2020, the 2022
  rate shock, the dot-com bust) ships as ``HISTORICAL_SCENARIOS``.

* **Correlation stress** — blend the correlation matrix toward 1,
  :math:`\\rho^{stress} = (1 - \\lambda) \\rho + \\lambda \\mathbf{1}
  \\mathbf{1}^\\top`, rebuild the covariance with unchanged volatilities
  and project back to the PSD cone. Captures the empirical tendency of
  correlations to spike in crises.

References
----------
Kupiec, P. (1998). "Stress Testing in a Value at Risk Framework".
*Journal of Derivatives*, 6(1), 7-24. Historical shock magnitudes are
approximate published asset-class moves for each episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "Scenario",
    "apply_scenario",
    "run_scenarios",
    "correlation_stress",
    "HISTORICAL_SCENARIOS",
]


@dataclass
class Scenario:
    """A named stress scenario.

    Attributes
    ----------
    name : str
        Scenario label.
    shocks : dict[str, float]
        Asset (or asset-class ticker) mapped to a decimal return shock,
        e.g. ``{"SPY": -0.50}`` for a 50% equity decline. Assets absent
        from a portfolio are ignored; portfolio assets absent from the
        scenario are shocked by 0.
    """

    name: str
    shocks: dict[str, float] = field(default_factory=dict)


def apply_scenario(weights: pd.Series, scenario: Scenario) -> float:
    """Portfolio P&L under a scenario.

    .. math:: \\Delta P = \\sum_i w_i \\, s_i

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset.
    scenario : Scenario
        Shock definition; assets not listed receive a zero shock.

    Returns
    -------
    float
        Decimal portfolio return under the scenario (negative = loss).
    """
    if not isinstance(weights, pd.Series):
        raise TypeError(f"weights must be a pd.Series, got {type(weights).__name__}")
    if not isinstance(scenario, Scenario):
        raise TypeError(f"scenario must be a Scenario, got {type(scenario).__name__}")
    shocks = pd.Series(scenario.shocks, dtype=float).reindex(weights.index).fillna(0.0)
    return float(weights.astype(float) @ shocks)


def run_scenarios(weights: pd.Series, scenarios: list[Scenario]) -> pd.DataFrame:
    """Evaluate a portfolio across a list of scenarios.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.
    scenarios : list of Scenario
        Scenarios to apply.

    Returns
    -------
    pd.DataFrame
        Indexed by scenario name with columns ``pnl`` (decimal
        portfolio return) and ``n_assets_shocked`` (portfolio assets
        with a non-zero shock).
    """
    if len(scenarios) == 0:
        raise ValueError("scenarios list is empty")
    rows = {}
    for sc in scenarios:
        n_hit = sum(1 for a in weights.index if float(sc.shocks.get(str(a), 0.0)) != 0.0)
        rows[sc.name] = {"pnl": apply_scenario(weights, sc), "n_assets_shocked": n_hit}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "scenario"
    return out


def _nearest_psd_fallback(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Eigenvalue clipping to the PSD cone (local fallback)."""
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T


def correlation_stress(cov: pd.DataFrame, stress_factor: float) -> pd.DataFrame:
    """Stress a covariance matrix by pushing correlations toward 1.

    Volatilities are preserved; the correlation matrix is blended,

    .. math::

        \\rho^{stress}_{ij} = (1 - \\lambda) \\rho_{ij} + \\lambda,
        \\quad i \\ne j,

    and the resulting covariance is projected onto the PSD cone
    (Higham-style eigenvalue clipping via
    ``src.math.numerical_linear_algebra.nearest_psd`` when available,
    with a local fallback).

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix.
    stress_factor : float
        Blend weight :math:`\\lambda \\in [0, 1]`; 0 returns ``cov``
        unchanged, 1 forces perfect correlation.

    Returns
    -------
    pd.DataFrame
        Stressed, PSD covariance matrix with the same labels.
    """
    if not isinstance(cov, pd.DataFrame):
        raise TypeError(f"cov must be a pd.DataFrame, got {type(cov).__name__}")
    if cov.shape[0] != cov.shape[1] or not cov.index.equals(cov.columns):
        raise ValueError("cov must be square with identical index/columns")
    if not 0.0 <= stress_factor <= 1.0:
        raise ValueError(f"stress_factor must be in [0, 1], got {stress_factor}")

    sigma = cov.to_numpy(dtype=float)
    vol = np.sqrt(np.diag(sigma))
    if np.any(vol <= 0):
        raise ValueError("cov must have strictly positive variances")
    corr = sigma / np.outer(vol, vol)
    ones = np.ones_like(corr)
    stressed_corr = (1.0 - stress_factor) * corr + stress_factor * ones
    np.fill_diagonal(stressed_corr, 1.0)
    stressed = stressed_corr * np.outer(vol, vol)

    try:  # prefer the shared implementation; lazy so this module stands alone
        from src.math.numerical_linear_algebra import nearest_psd  # noqa: PLC0415

        stressed = nearest_psd(stressed)
    except ImportError:
        stressed = _nearest_psd_fallback(stressed)

    return pd.DataFrame(stressed, index=cov.index, columns=cov.columns)


#: Approximate peak-to-trough asset-class moves in well-known crisis
#: episodes (decimal returns; keyed by the ETF proxies used across this
#: repo). Magnitudes are rounded from published index histories and are
#: intended for illustrative stress testing, not exact replication.
HISTORICAL_SCENARIOS: list[Scenario] = [
    Scenario(
        name="GFC-2008",
        shocks={
            "SPY": -0.55, "QQQ": -0.50, "IWM": -0.58, "EFA": -0.60, "EEM": -0.62,
            "AGG": 0.05, "TLT": 0.25, "LQD": -0.10, "GLD": 0.20, "DBC": -0.55,
            "VNQ": -0.68, "USMV": -0.45, "MTUM": -0.50, "VLUE": -0.58, "QUAL": -0.50,
        },
    ),
    Scenario(
        name="Covid-2020",
        shocks={
            "SPY": -0.34, "QQQ": -0.28, "IWM": -0.41, "EFA": -0.34, "EEM": -0.34,
            "AGG": -0.01, "TLT": 0.15, "LQD": -0.15, "GLD": -0.04, "DBC": -0.30,
            "VNQ": -0.42, "USMV": -0.28, "MTUM": -0.30, "VLUE": -0.40, "QUAL": -0.32,
        },
    ),
    Scenario(
        name="RateShock-2022",
        shocks={
            "SPY": -0.25, "QQQ": -0.35, "IWM": -0.27, "EFA": -0.28, "EEM": -0.31,
            "AGG": -0.17, "TLT": -0.33, "LQD": -0.21, "GLD": -0.09, "DBC": 0.15,
            "VNQ": -0.29, "USMV": -0.16, "MTUM": -0.28, "VLUE": -0.15, "QUAL": -0.28,
        },
    ),
    Scenario(
        name="DotCom-2000",
        shocks={
            "SPY": -0.49, "QQQ": -0.78, "IWM": -0.37, "EFA": -0.47, "EEM": -0.45,
            "AGG": 0.10, "TLT": 0.20, "LQD": 0.08, "GLD": 0.05, "DBC": -0.10,
            "VNQ": 0.10, "USMV": -0.30, "MTUM": -0.60, "VLUE": -0.20, "QUAL": -0.45,
        },
    ),
]
