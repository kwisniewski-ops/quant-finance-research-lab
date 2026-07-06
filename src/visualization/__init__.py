"""Matplotlib visualizations (figures returned, never shown/saved)."""

from src.visualization.efficient_frontier import plot_efficient_frontier
from src.visualization.risk_dashboard import plot_risk_dashboard
from src.visualization.volatility_surface import plot_volatility_surface

__all__ = [
    "plot_efficient_frontier",
    "plot_volatility_surface",
    "plot_risk_dashboard",
]
