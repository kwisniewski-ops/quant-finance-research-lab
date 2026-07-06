"""Tests for src.backtesting: engine (incl. look-ahead freedom), costs,
slippage, and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import Backtester
from src.backtesting.performance_metrics import (
    annualized_return,
    annualized_vol,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
    summary,
)
from src.backtesting.slippage import FixedSlippage
from src.backtesting.transaction_costs import ProportionalCost


def make_prices(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """Seeded GBM-ish price panel for three assets."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n_days)
    rets = rng.normal(0.0004, 0.011, size=(n_days, 3))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=idx, columns=["AAA", "BBB", "CCC"])


# ---------------------------------------------------------------------------
# Cost / slippage models
# ---------------------------------------------------------------------------
class TestCostModels:
    def test_proportional_cost_values(self):
        assert np.isclose(ProportionalCost(bps=10).cost(1.0), 0.001)
        assert np.isclose(ProportionalCost(bps=10).cost(0.5), 0.0005)
        assert ProportionalCost(bps=0).cost(1.0) == 0.0

    def test_fixed_slippage_values(self):
        assert np.isclose(FixedSlippage(bps=5).cost(1.0), 0.0005)

    def test_negative_inputs_raise(self):
        with pytest.raises(ValueError):
            ProportionalCost(bps=-1)
        with pytest.raises(ValueError):
            FixedSlippage(bps=5).cost(-0.1)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TestBacktester:
    def test_buy_and_hold_reproduces_asset_return_exactly(self):
        prices = make_prices()[["AAA"]]
        bt = Backtester(rebalance="M")  # zero costs
        result = bt.run(prices, lambda window: pd.Series({"AAA": 1.0}), lookback=60)

        start = result.returns.index[0]
        asset_rets = prices["AAA"].pct_change().loc[start:]
        assert np.allclose(result.returns.values, asset_rets.values, atol=1e-12)

        # Equity curve equals price relative from the day before entry.
        entry_pos = prices.index.get_loc(start) - 1
        price_relative = prices["AAA"].iloc[-1] / prices["AAA"].iloc[entry_pos]
        assert np.isclose(result.equity_curve.iloc[-1], price_relative, atol=1e-10)

        # Only the initial trade generates turnover; drift keeps w = 1.
        assert np.isclose(result.turnover.iloc[0], 0.5)
        assert np.allclose(result.turnover.iloc[1:], 0.0, atol=1e-12)
        assert np.allclose(result.weights["AAA"].values, 1.0, atol=1e-12)

    def test_costs_strictly_reduce_performance(self):
        prices = make_prices()
        rng = np.random.default_rng(7)

        def churny_weight_fn(window: pd.DataFrame) -> pd.Series:
            w = rng.dirichlet(np.ones(3))
            return pd.Series(w, index=window.columns)

        # Same weight sequence for both runs (reset the generator).
        rng = np.random.default_rng(7)
        free = Backtester(rebalance="M").run(prices, churny_weight_fn, lookback=120)
        rng = np.random.default_rng(7)
        costly = Backtester(
            cost_model=ProportionalCost(bps=25),
            slippage_model=FixedSlippage(bps=10),
            rebalance="M",
        ).run(prices, churny_weight_fn, lookback=120)

        assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
        assert (costly.costs > 0).all()
        assert costly.costs.sum() > 0
        # Gross-of-cost daily returns are identical.
        gross = costly.returns.add(
            costly.costs.reindex(costly.returns.index).fillna(0.0)
        )
        assert np.allclose(gross.values, free.returns.values, atol=1e-12)

    def test_no_look_ahead_window_ends_before_rebalance(self):
        """The window passed to weight_fn must end strictly before t."""
        prices = make_prices()
        seen: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []

        def spy_weight_fn(window: pd.DataFrame) -> pd.Series:
            seen.append((window.index.min(), window.index.max(), len(window)))
            return pd.Series(1 / 3, index=window.columns)

        lookback = 120
        result = Backtester(rebalance="M").run(prices, spy_weight_fn, lookback=lookback)
        rebalance_dates = list(result.turnover.index)
        assert len(seen) == len(rebalance_dates)
        for (w_start, w_end, w_len), t in zip(seen, rebalance_dates):
            assert w_end < t, f"window leaks data at/after rebalance {t}"
            assert w_len == lookback
            # Window is the *trailing* lookback: it ends on the trading
            # day immediately before t.
            pos = prices.index.get_loc(t)
            assert w_end == prices.index[pos - 1]
            assert w_start == prices.index[pos - lookback]

    def test_future_peeking_is_impossible(self):
        """A cheater strategy putting 100% in the next-day winner must fail:
        the engine never hands weight_fn data at or beyond the rebalance
        date, so 'peeking' at the window's future is just the past."""
        prices = make_prices()
        max_future_seen: list[pd.Timestamp] = []

        def cheater(window: pd.DataFrame) -> pd.Series:
            max_future_seen.append(window.index.max())
            best = window.pct_change().iloc[-1].idxmax()  # best it can do
            return pd.Series({best: 1.0})

        result = Backtester(rebalance="M").run(prices, cheater, lookback=60)
        # Every date the cheater ever saw precedes its own trade date.
        for w_end, t in zip(max_future_seen, result.turnover.index):
            assert w_end < t

    def test_weights_drift_between_rebalances(self):
        prices = make_prices()
        target = pd.Series({"AAA": 0.5, "BBB": 0.5, "CCC": 0.0})
        result = Backtester(rebalance="M").run(prices, lambda w: target, lookback=60)

        rets = prices.pct_change()
        rebalance_dates = set(result.turnover.index)
        dates = list(result.returns.index)
        for today, tomorrow in zip(dates[:-1], dates[1:]):
            if tomorrow in rebalance_dates:
                continue  # reset day, not a drift day
            w_today = result.weights.loc[today].to_numpy()
            grown = w_today * (1.0 + rets.loc[today].to_numpy())
            expected = grown / grown.sum()
            assert np.allclose(result.weights.loc[tomorrow].to_numpy(), expected, atol=1e-12)

        # Turnover after the first rebalance is measured vs *drifted*
        # weights, hence strictly positive for a fixed 50/50 target.
        assert (result.turnover.iloc[1:] > 0).all()

    def test_input_validation(self):
        prices = make_prices(100)
        bt = Backtester(rebalance="M")
        with pytest.raises(ValueError, match="lookback"):
            bt.run(prices, lambda w: pd.Series(), lookback=200)
        with pytest.raises(ValueError, match="sum to 1"):
            bt.run(prices, lambda w: pd.Series({"AAA": 0.7}), lookback=30)
        with pytest.raises(ValueError, match="rebalance"):
            Backtester(rebalance="hourly")
        unsorted = prices.iloc[::-1]
        with pytest.raises(ValueError, match="sorted"):
            bt.run(unsorted, lambda w: pd.Series({"AAA": 1.0}), lookback=30)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rets() -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2019-01-01", periods=1260)
    return pd.Series(rng.normal(0.0005, 0.01, 1260), index=idx)


class TestPerformanceMetrics:
    def test_annualized_return_constant_series(self):
        r = pd.Series([0.001] * 252)
        assert np.isclose(annualized_return(r), 1.001**252 - 1, atol=1e-12)

    def test_annualized_vol_scaling(self, rets):
        assert np.isclose(annualized_vol(rets), rets.std(ddof=1) * np.sqrt(252))

    def test_sharpe_and_sortino_signs(self, rets):
        assert sharpe_ratio(rets) > 0
        assert sortino_ratio(rets) > sharpe_ratio(rets)  # normal data: downside dev < full std
        assert sharpe_ratio(rets, rf=0.5) < sharpe_ratio(rets, rf=0.0)

    def test_hit_rate_bounds(self, rets):
        hr = hit_rate(rets)
        assert 0.4 < hr < 0.7

    def test_summary_keys_and_consistency(self, rets):
        s = summary(rets)
        expected_keys = {
            "annualized_return", "annualized_vol", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "hit_rate", "max_drawdown", "skew", "kurtosis",
            "VaR95", "ES95",
        }
        assert set(s.index) == expected_keys
        assert s["ES95"] >= s["VaR95"]
        assert s["max_drawdown"] > 0
        assert np.isclose(s["sharpe_ratio"], sharpe_ratio(rets))
