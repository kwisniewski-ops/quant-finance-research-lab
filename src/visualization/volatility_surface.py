"""Implied-volatility surface plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_volatility_surface"]


def plot_volatility_surface(surface_df: pd.DataFrame) -> plt.Figure:
    """3-D implied-volatility surface.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Implied vols with rows = maturities (years) and columns =
        strikes, as produced by
        ``src.models.stochastic_volatility.implied_vol_surface``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing a single 3-D axes (never shown or saved
        here).
    """
    if not isinstance(surface_df, pd.DataFrame):
        raise TypeError(f"surface_df must be a pd.DataFrame, got {type(surface_df).__name__}")
    if surface_df.empty:
        raise ValueError("surface_df is empty")

    maturities = surface_df.index.to_numpy(dtype=float)
    strikes = surface_df.columns.to_numpy(dtype=float)
    K, T = np.meshgrid(strikes, maturities)
    vols = surface_df.to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K, T, vols, cmap="viridis", edgecolor="none",
                           antialiased=True, alpha=0.95)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Implied volatility")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title("Implied volatility surface")
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    return fig
