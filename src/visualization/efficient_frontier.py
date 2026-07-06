"""Efficient frontier plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for typing
    from src.portfolio.mean_variance import PortfolioResult

__all__ = ["plot_efficient_frontier"]


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    assets_mu: pd.Series | None = None,
    assets_vol: pd.Series | None = None,
    highlight: "dict[str, PortfolioResult] | None" = None,
) -> plt.Figure:
    """Plot an efficient frontier colored by Sharpe ratio.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Output of :func:`src.portfolio.mean_variance.efficient_frontier`
        — must contain ``expected_return``, ``volatility`` and
        ``sharpe`` columns.
    assets_mu, assets_vol : pd.Series, optional
        Individual-asset expected returns and volatilities (same
        index); when both are given the single assets are scattered and
        labeled.
    highlight : dict of str -> PortfolioResult, optional
        Named portfolios (e.g. ``{"Max Sharpe": msr}``) marked with
        stars.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (never shown or saved here).
    """
    required = {"expected_return", "volatility", "sharpe"}
    missing = required - set(frontier_df.columns)
    if missing:
        raise ValueError(f"frontier_df is missing columns: {sorted(missing)}")
    if (assets_mu is None) != (assets_vol is None):
        raise ValueError("provide both assets_mu and assets_vol, or neither")

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        frontier_df["volatility"], frontier_df["expected_return"],
        c=frontier_df["sharpe"], cmap="viridis", s=28, zorder=3,
    )
    ax.plot(
        frontier_df["volatility"], frontier_df["expected_return"],
        color="0.55", lw=1.0, alpha=0.7, zorder=2,
    )
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")

    if assets_mu is not None and assets_vol is not None:
        if not assets_mu.index.equals(assets_vol.index):
            assets_vol = assets_vol.reindex(assets_mu.index)
        ax.scatter(assets_vol, assets_mu, marker="o", s=45, color="tab:red",
                   edgecolor="white", zorder=4, label="Assets")
        for name in assets_mu.index:
            ax.annotate(str(name), (assets_vol[name], assets_mu[name]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

    if highlight:
        markers = ["*", "P", "X", "D", "s"]
        for i, (name, res) in enumerate(highlight.items()):
            ax.scatter(res.volatility, res.expected_return,
                       marker=markers[i % len(markers)], s=260, zorder=5,
                       edgecolor="black", linewidth=0.8, label=name)

    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title("Efficient frontier")
    ax.grid(True, alpha=0.3)
    if highlight or assets_mu is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig
