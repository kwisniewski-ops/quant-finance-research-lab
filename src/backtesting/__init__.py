"""Backtesting: engine, cost/slippage models, performance metrics."""

from src.backtesting.engine import Backtester, BacktestResult
from src.backtesting.performance_metrics import (
    annualized_return,
    annualized_vol,
    calmar_ratio,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
    summary,
)
from src.backtesting.slippage import FixedSlippage
from src.backtesting.transaction_costs import ProportionalCost

__all__ = [
    "Backtester",
    "BacktestResult",
    "ProportionalCost",
    "FixedSlippage",
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "hit_rate",
    "summary",
]
