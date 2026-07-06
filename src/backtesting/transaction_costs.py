"""Transaction cost models.

Costs are expressed as a fraction of NAV charged against the
portfolio's return on the rebalance day, as a function of one-sided
turnover :math:`\\tau = \\tfrac12 \\sum_i |\\Delta w_i|`.
"""

from __future__ import annotations

__all__ = ["ProportionalCost"]


class ProportionalCost:
    """Linear (proportional) transaction costs.

    .. math:: c(\\tau) = \\tau \\cdot \\text{bps} \\times 10^{-4}

    Parameters
    ----------
    bps : float, default 10.0
        Round-trip cost per unit of turnover, in basis points. 10 bps
        means trading 100% of NAV costs 0.10% of NAV.

    Examples
    --------
    >>> ProportionalCost(bps=10).cost(0.5)
    0.0005
    """

    def __init__(self, bps: float = 10.0) -> None:
        if bps < 0:
            raise ValueError(f"bps must be non-negative, got {bps}")
        self.bps = float(bps)

    def cost(self, turnover: float) -> float:
        """Cost as a fraction of NAV for a given turnover.

        Parameters
        ----------
        turnover : float
            One-sided turnover (fraction of NAV traded), >= 0.

        Returns
        -------
        float
            Cost as a decimal fraction of NAV.
        """
        if turnover < 0:
            raise ValueError(f"turnover must be non-negative, got {turnover}")
        return turnover * self.bps * 1e-4

    def __repr__(self) -> str:  # pragma: no cover
        return f"ProportionalCost(bps={self.bps})"
