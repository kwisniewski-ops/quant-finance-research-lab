"""Event-free vectorized backtesting engine.

Design invariants (enforced, and asserted in the test suite):

* **No look-ahead.** At each rebalance date :math:`t`, ``weight_fn``
  receives a trailing window of prices whose last row is *strictly
  before* :math:`t` (the most recent ``lookback`` observations). The
  first trade occurs on the first rebalance date with a full lookback
  window available.
* **Weight drift.** Between rebalances, weights evolve with realized
  returns,

  .. math::

      w_{i,d+1} = \\frac{w_{i,d} (1 + r_{i,d})}
          {\\sum_j w_{j,d} (1 + r_{j,d})},

  so turnover at the next rebalance is measured against the *drifted*
  holdings, not the previous targets.
* **Cost accounting.** On each rebalance date, one-sided turnover is
  :math:`\\tau_t = \\tfrac12 \\sum_i |w^{new}_i - w^{drift}_i|` and
  ``cost_model.cost(tau) + slippage_model.cost(tau)`` is deducted from
  that day's portfolio return.

References
----------
Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio".
*Journal of Portfolio Management*, 40(5), 94-107 — on why leak-free
backtests matter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

__all__ = ["BacktestResult", "Backtester"]

logger = logging.getLogger(__name__)

_FREQ_TO_PERIOD = {"D": "D", "W": "W", "M": "M", "Q": "Q", "A": "A", "Y": "A"}


class _CostModel(Protocol):
    def cost(self, turnover: float) -> float: ...


@dataclass
class BacktestResult:
    """Output of a backtest run.

    Attributes
    ----------
    returns : pd.Series
        Daily net portfolio returns (costs deducted on rebalance days),
        starting on the first rebalance date.
    equity_curve : pd.Series
        Cumulative growth of 1 unit: ``(1 + returns).cumprod()``.
    weights : pd.DataFrame
        Start-of-day weights actually held each day (targets on
        rebalance days, drifted otherwise).
    turnover : pd.Series
        One-sided turnover on each rebalance date
        (``0.5 * sum |w_new - w_drifted|``).
    costs : pd.Series
        Total transaction + slippage cost (fraction of NAV) charged on
        each rebalance date.
    """

    returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


class Backtester:
    """Rebalanced, cost-aware, look-ahead-free portfolio backtester.

    Parameters
    ----------
    cost_model : object, optional
        Object with ``cost(turnover) -> float`` (fraction of NAV), e.g.
        :class:`src.backtesting.transaction_costs.ProportionalCost`.
        ``None`` means free trading.
    slippage_model : object, optional
        Same interface, e.g.
        :class:`src.backtesting.slippage.FixedSlippage`.
    rebalance : str, default "M"
        Rebalance frequency: ``"D"``, ``"W"``, ``"M"``, ``"Q"`` or
        ``"A"``. Trades execute on the first trading day of each
        period.

    Examples
    --------
    >>> bt = Backtester(cost_model=ProportionalCost(10), rebalance="M")
    >>> result = bt.run(prices, lambda window: hrp_weights(
    ...     window.pct_change().dropna()), lookback=252)
    """

    def __init__(
        self,
        cost_model: _CostModel | None = None,
        slippage_model: _CostModel | None = None,
        rebalance: str = "M",
    ) -> None:
        if rebalance not in _FREQ_TO_PERIOD:
            raise ValueError(
                f"rebalance must be one of {sorted(_FREQ_TO_PERIOD)}, got {rebalance!r}"
            )
        for name, model in (("cost_model", cost_model), ("slippage_model", slippage_model)):
            if model is not None and not callable(getattr(model, "cost", None)):
                raise TypeError(f"{name} must expose a callable .cost(turnover)")
        self.cost_model = cost_model
        self.slippage_model = slippage_model
        self.rebalance = rebalance

    # ------------------------------------------------------------------
    def _rebalance_dates(self, dates: pd.DatetimeIndex, first_valid: pd.Timestamp) -> list[pd.Timestamp]:
        """First trading day of each period, at/after ``first_valid``."""
        eligible = dates[dates >= first_valid]
        periods = eligible.to_period(_FREQ_TO_PERIOD[self.rebalance])
        first_of_period = pd.Series(eligible, index=periods).groupby(level=0).first()
        return list(first_of_period)

    def _trade_cost(self, turnover: float) -> float:
        total = 0.0
        for model in (self.cost_model, self.slippage_model):
            if model is not None:
                c = float(model.cost(turnover))
                if c < 0:
                    raise ValueError(f"{type(model).__name__}.cost returned negative cost {c}")
                total += c
        return total

    # ------------------------------------------------------------------
    def run(
        self,
        prices: pd.DataFrame,
        weight_fn: Callable[[pd.DataFrame], pd.Series],
        lookback: int = 252,
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily prices, DatetimeIndex ascending, columns = assets.
        weight_fn : callable
            ``weight_fn(window) -> pd.Series`` mapping a trailing price
            window (the ``lookback`` most recent rows *strictly before*
            the rebalance date) to target weights summing to 1. Assets
            omitted from the returned Series get weight 0.
        lookback : int, default 252
            Number of past observations passed to ``weight_fn``.

        Returns
        -------
        BacktestResult

        Raises
        ------
        ValueError
            If the index is not a sorted DatetimeIndex, the history is
            shorter than ``lookback + 2`` rows, or ``weight_fn``
            returns weights that do not sum to 1.
        """
        if not isinstance(prices, pd.DataFrame):
            raise TypeError(f"prices must be a pd.DataFrame, got {type(prices).__name__}")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must be indexed by a DatetimeIndex")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be sorted ascending")
        if prices.index.has_duplicates:
            raise ValueError("prices index contains duplicate dates")
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if len(prices) < lookback + 2:
            raise ValueError(
                f"need at least lookback + 2 = {lookback + 2} rows of prices, got {len(prices)}"
            )

        returns = prices.pct_change(fill_method=None).iloc[1:]
        dates = returns.index
        assets = prices.columns

        # First date with `lookback` price rows strictly before it.
        first_valid = prices.index[lookback]
        rebalance_dates = set(self._rebalance_dates(dates, first_valid))
        if not rebalance_dates:
            raise ValueError("no rebalance dates fall inside the price history")
        start = min(rebalance_dates)
        active = dates[dates >= start]

        w = np.zeros(len(assets))
        port_returns = np.zeros(len(active))
        weight_rows = np.zeros((len(active), len(assets)))
        turnover_out: dict[pd.Timestamp, float] = {}
        costs_out: dict[pd.Timestamp, float] = {}

        for k, date in enumerate(active):
            cost_today = 0.0
            if date in rebalance_dates:
                loc = prices.index.get_loc(date)
                window = prices.iloc[max(0, loc - lookback) : loc]  # strictly before `date`
                assert window.index.max() < date, "look-ahead guard violated"
                target = weight_fn(window)
                if not isinstance(target, pd.Series):
                    raise TypeError("weight_fn must return a pd.Series of weights")
                if not np.isclose(float(target.sum()), 1.0, atol=1e-6):
                    raise ValueError(
                        f"weight_fn weights must sum to 1 at {date.date()}, got {target.sum():.6f}"
                    )
                unknown = set(target.index) - set(assets)
                if unknown:
                    raise ValueError(f"weight_fn returned unknown assets: {sorted(map(str, unknown))}")
                w_new = target.reindex(assets).fillna(0.0).to_numpy(dtype=float)
                turnover = 0.5 * float(np.abs(w_new - w).sum())
                cost_today = self._trade_cost(turnover)
                turnover_out[date] = turnover
                costs_out[date] = cost_today
                w = w_new

            r = returns.loc[date].to_numpy(dtype=float)
            r = np.where(np.isnan(r), 0.0, r)
            weight_rows[k] = w
            port_returns[k] = float(w @ r) - cost_today
            # Drift weights with realized returns.
            grown = w * (1.0 + r)
            total = grown.sum()
            if total > 0:
                w = grown / total

        ret = pd.Series(port_returns, index=active, name="return")
        result = BacktestResult(
            returns=ret,
            equity_curve=(1.0 + ret).cumprod().rename("equity"),
            weights=pd.DataFrame(weight_rows, index=active, columns=assets),
            turnover=pd.Series(turnover_out, name="turnover", dtype=float),
            costs=pd.Series(costs_out, name="cost", dtype=float),
        )
        logger.info(
            "backtest: %d days, %d rebalances, total cost %.4f",
            len(active), len(turnover_out), result.costs.sum(),
        )
        return result
