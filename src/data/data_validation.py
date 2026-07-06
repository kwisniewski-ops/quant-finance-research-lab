"""Price data quality validation.

Screens a price panel for the classic data pathologies that silently
corrupt backtests:

* missing values (per column, against a tolerance),
* stale runs (a price repeated verbatim for many consecutive days —
  typically a dead feed or delisted series being forward-filled),
* return outliers (:math:`|z|` above a threshold on daily returns —
  fat fingers, bad splits, unadjusted dividends),
* a non-monotonic index (unsorted or duplicate dates).

The result is a :class:`ValidationReport` whose ``passed`` property is
True only when no issue was flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ValidationReport", "validate_prices"]


@dataclass
class ValidationReport:
    """Structured result of :func:`validate_prices`.

    Attributes
    ----------
    n_rows : int
        Number of rows validated.
    n_missing : pd.Series
        Missing-value count per column.
    stale_runs : pd.Series
        Length of the longest run of consecutive identical prices per
        column.
    outliers : pd.Series
        Count of daily returns with ``|z| > outlier_z`` per column.
    index_monotonic : bool
        True if the index is strictly increasing (sorted, no
        duplicates).
    issues : list of str
        Human-readable description of every flagged problem.
    """

    n_rows: int
    n_missing: pd.Series
    stale_runs: pd.Series
    outliers: pd.Series
    index_monotonic: bool
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no issues were flagged."""
        return len(self.issues) == 0


def _longest_run_of_equal_values(x: np.ndarray) -> int:
    """Longest run of consecutive equal, non-NaN values."""
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return int(valid.sum())
    same = (x[1:] == x[:-1]) & valid[1:] & valid[:-1]
    longest, current = 1, 1
    for s in same:
        current = current + 1 if s else 1
        longest = max(longest, current)
    return longest


def validate_prices(
    prices: pd.DataFrame,
    max_missing_frac: float = 0.05,
    outlier_z: float = 8.0,
    stale_len: int = 10,
) -> ValidationReport:
    """Validate a price panel and return a structured report.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel, DatetimeIndex, columns = tickers.
    max_missing_frac : float, default 0.05
        Maximum tolerated fraction of missing values per column.
    outlier_z : float, default 8.0
        Daily returns with ``|return - mean| / std > outlier_z`` are
        flagged as outliers.
    stale_len : int, default 10
        A run of >= ``stale_len`` consecutive identical prices is
        flagged as stale.

    Returns
    -------
    ValidationReport

    Examples
    --------
    >>> report = validate_prices(prices)
    >>> if not report.passed:
    ...     for issue in report.issues:
    ...         print(issue)
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a pd.DataFrame, got {type(prices).__name__}")
    if prices.empty:
        raise ValueError("prices is empty")
    if not 0.0 <= max_missing_frac <= 1.0:
        raise ValueError(f"max_missing_frac must be in [0, 1], got {max_missing_frac}")
    if outlier_z <= 0:
        raise ValueError(f"outlier_z must be positive, got {outlier_z}")
    if stale_len < 2:
        raise ValueError(f"stale_len must be >= 2, got {stale_len}")

    n_rows = len(prices)
    issues: list[str] = []

    # --- missing values -------------------------------------------------
    n_missing = prices.isna().sum()
    n_missing.name = "n_missing"
    for col, n in n_missing.items():
        frac = n / n_rows
        if frac > max_missing_frac:
            issues.append(
                f"{col}: {n}/{n_rows} missing values ({frac:.1%} > {max_missing_frac:.1%} allowed)"
            )

    # --- stale runs -----------------------------------------------------
    stale = pd.Series(
        {col: _longest_run_of_equal_values(prices[col].to_numpy(dtype=float)) for col in prices.columns},
        name="stale_run_length",
    )
    for col, run in stale.items():
        if run >= stale_len:
            issues.append(f"{col}: stale run of {run} consecutive identical prices (>= {stale_len})")

    # --- return outliers ------------------------------------------------
    rets = prices.pct_change(fill_method=None)
    out_counts = {}
    for col in prices.columns:
        r = rets[col].dropna().to_numpy(dtype=float)
        if r.size < 3:
            out_counts[col] = 0
            continue
        sd = r.std(ddof=1)
        if sd == 0:
            out_counts[col] = 0
            continue
        z = np.abs(r - r.mean()) / sd
        out_counts[col] = int((z > outlier_z).sum())
    outliers = pd.Series(out_counts, name="n_outliers")
    for col, n in outliers.items():
        if n > 0:
            issues.append(f"{col}: {n} daily return(s) beyond {outlier_z:.1f} sigma")

    # --- index ----------------------------------------------------------
    index_monotonic = bool(prices.index.is_monotonic_increasing and not prices.index.has_duplicates)
    if not index_monotonic:
        issues.append("index is not strictly increasing (unsorted or duplicate dates)")

    return ValidationReport(
        n_rows=n_rows,
        n_missing=n_missing,
        stale_runs=stale,
        outliers=outliers,
        index_monotonic=index_monotonic,
        issues=issues,
    )
