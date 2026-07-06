"""Slippage models.

Slippage is the adverse difference between the decision price and the
achieved execution price. Modeled here, like explicit transaction
costs, as a fraction of NAV per unit of one-sided turnover.
"""

from __future__ import annotations

__all__ = ["FixedSlippage"]


class FixedSlippage:
    """Constant per-unit-turnover slippage.

    .. math:: c(\\tau) = \\tau \\cdot \\text{bps} \\times 10^{-4}

    Parameters
    ----------
    bps : float, default 5.0
        Slippage per unit of turnover, in basis points.

    Examples
    --------
    >>> FixedSlippage(bps=5).cost(1.0)
    0.0005
    """

    def __init__(self, bps: float = 5.0) -> None:
        if bps < 0:
            raise ValueError(f"bps must be non-negative, got {bps}")
        self.bps = float(bps)

    def cost(self, turnover: float) -> float:
        """Slippage cost as a fraction of NAV for a given turnover.

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
        return f"FixedSlippage(bps={self.bps})"
