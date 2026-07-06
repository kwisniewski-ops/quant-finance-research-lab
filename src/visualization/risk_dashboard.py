"""Four-panel portfolio risk dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.risk.drawdown_analysis import drawdown_series
from src.risk.expected_shortfall import historical_es
from src.risk.value_at_risk import historical_var

__all__ = ["plot_risk_dashboard"]


def plot_risk_dashboard(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
) -> plt.Figure:
    """2x2 risk dashboard: equity curve, drawdown, rolling vol, histogram.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (decimal), DatetimeIndex.
    benchmark : pd.Series, optional
        Daily benchmark returns overlaid on the equity-curve and
        rolling-volatility panels (aligned on common dates).

    Returns
    -------
    matplotlib.figure.Figure
        The 2x2 dashboard figure (never shown or saved here).
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns).__name__}")
    r = returns.dropna().astype(float)
    if len(r) < 30:
        raise ValueError(f"need at least 30 observations for a meaningful dashboard, got {len(r)}")
    if benchmark is not None:
        benchmark = benchmark.dropna().astype(float).reindex(r.index).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_eq, ax_dd, ax_vol, ax_hist = axes.ravel()

    # --- equity curve -----------------------------------------------------
    equity = (1.0 + r).cumprod()
    ax_eq.plot(equity.index, equity, lw=1.4, label="Portfolio", color="tab:blue")
    if benchmark is not None and len(benchmark) > 1:
        ax_eq.plot(benchmark.index, (1.0 + benchmark).cumprod(), lw=1.1,
                   label="Benchmark", color="0.5", alpha=0.9)
        ax_eq.legend(frameon=False, fontsize=8)
    ax_eq.set_title("Equity curve (growth of 1)")
    ax_eq.set_ylabel("Cumulative growth")
    ax_eq.grid(True, alpha=0.3)

    # --- drawdown ---------------------------------------------------------
    dd = drawdown_series(r)
    ax_dd.fill_between(dd.index, dd.to_numpy() * 100.0, 0.0, color="tab:red", alpha=0.45)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.3)

    # --- rolling volatility -------------------------------------------------
    window = min(63, max(20, len(r) // 10))
    roll_vol = r.rolling(window).std() * np.sqrt(252) * 100.0
    ax_vol.plot(roll_vol.index, roll_vol, lw=1.2, color="tab:blue", label="Portfolio")
    if benchmark is not None and len(benchmark) > window:
        bvol = benchmark.rolling(window).std() * np.sqrt(252) * 100.0
        ax_vol.plot(bvol.index, bvol, lw=1.0, color="0.5", alpha=0.9, label="Benchmark")
        ax_vol.legend(frameon=False, fontsize=8)
    ax_vol.set_title(f"Rolling {window}-day annualized volatility")
    ax_vol.set_ylabel("Volatility (%)")
    ax_vol.grid(True, alpha=0.3)

    # --- histogram with VaR / ES -------------------------------------------
    var95 = historical_var(r, alpha=0.95)
    es95 = historical_es(r, alpha=0.95)
    ax_hist.hist(r.to_numpy() * 100.0, bins=60, color="tab:blue", alpha=0.7, density=True)
    ax_hist.axvline(-var95 * 100.0, color="tab:orange", lw=1.6, ls="--",
                    label=f"VaR 95%: {var95:.2%}")
    ax_hist.axvline(-es95 * 100.0, color="tab:red", lw=1.6, ls="-.",
                    label=f"ES 95%: {es95:.2%}")
    ax_hist.set_title("Daily return distribution")
    ax_hist.set_xlabel("Daily return (%)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(frameon=False, fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    for ax in (ax_eq, ax_dd, ax_vol):
        ax.tick_params(axis="x", labelrotation=25)

    fig.suptitle("Risk dashboard", y=0.995)
    fig.tight_layout()
    return fig
