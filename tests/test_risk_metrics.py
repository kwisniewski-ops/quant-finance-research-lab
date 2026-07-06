"""Tests for src.risk: VaR, expected shortfall, drawdowns, stress tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.risk.drawdown_analysis import drawdown_series, drawdown_stats, max_drawdown
from src.risk.expected_shortfall import historical_es, parametric_es
from src.risk.stress_testing import (
    HISTORICAL_SCENARIOS,
    Scenario,
    apply_scenario,
    correlation_stress,
    run_scenarios,
)
from src.risk.value_at_risk import (
    cornish_fisher_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
)

MU_D, SD_D = 0.0004, 0.012  # daily normal parameters used throughout


@pytest.fixture(scope="module")
def normal_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(MU_D, SD_D, 100_000))


@pytest.fixture(scope="module")
def skewed_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    r = -np.abs(rng.standard_t(df=4, size=20_000)) * 0.01 + 0.005
    return pd.Series(r)


# ---------------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------------
class TestVaR:
    def test_parametric_matches_closed_form_on_normal(self, normal_returns):
        # True VaR of N(mu, sd): -(mu + sd * z_{0.05})
        z = stats.norm.ppf(0.05)
        true_var = -(MU_D + SD_D * z)
        assert np.isclose(parametric_var(normal_returns, 0.95), true_var, atol=2e-4)
        # Historical converges to the same number on a big normal sample.
        assert np.isclose(historical_var(normal_returns, 0.95), true_var, atol=4e-4)

    def test_var_positive_for_loss_making_quantile(self, normal_returns):
        assert historical_var(normal_returns, 0.99) > historical_var(normal_returns, 0.95) > 0
        assert parametric_var(normal_returns, 0.99) > parametric_var(normal_returns, 0.95) > 0

    def test_cornish_fisher_reduces_to_normal_when_moments_vanish(self):
        # Symmetric 3-point sample with population skew = 0 and
        # population excess kurtosis = 0 exactly (P(|x|=c) = 1/3).
        c = 0.01
        block = [-c, 0.0, 0.0, 0.0, 0.0, c]
        r = pd.Series(block * 500)
        s = float(stats.skew(r, bias=True))
        k = float(stats.kurtosis(r, fisher=True, bias=True))
        assert abs(s) < 1e-12 and abs(k) < 1e-12
        assert np.isclose(cornish_fisher_var(r, 0.95), parametric_var(r, 0.95), atol=1e-12)

    def test_cornish_fisher_penalizes_negative_skew(self, skewed_returns):
        # Left-skewed, fat-tailed returns -> CF VaR above Gaussian VaR.
        assert cornish_fisher_var(skewed_returns, 0.99) > parametric_var(skewed_returns, 0.99)

    def test_monte_carlo_var_matches_closed_form(self):
        assets = ["A", "B"]
        mu = pd.Series([0.0004, 0.0002], index=assets)
        cov = pd.DataFrame(
            [[0.012**2, 0.3 * 0.012 * 0.008], [0.3 * 0.012 * 0.008, 0.008**2]],
            index=assets, columns=assets,
        )
        w = pd.Series([0.6, 0.4], index=assets)
        port_mu = float(w @ mu)
        port_sd = float(np.sqrt(w @ cov.values @ w))
        true_var = -(port_mu + port_sd * stats.norm.ppf(0.05))
        mc = monte_carlo_var(mu, cov, w, alpha=0.95, n_sims=200_000,
                             rng=np.random.default_rng(42))
        assert np.isclose(mc, true_var, rtol=0.02)

    def test_input_validation(self, normal_returns):
        with pytest.raises(ValueError, match="alpha"):
            historical_var(normal_returns, alpha=1.5)
        with pytest.raises(ValueError, match="observations"):
            parametric_var(pd.Series([0.01]))


# ---------------------------------------------------------------------------
# Expected shortfall
# ---------------------------------------------------------------------------
class TestES:
    def test_es_geq_var_historical_and_parametric(self, normal_returns, skewed_returns):
        for r in (normal_returns, skewed_returns):
            for alpha in (0.90, 0.95, 0.99):
                assert historical_es(r, alpha) >= historical_var(r, alpha)
                assert parametric_es(r, alpha) >= parametric_var(r, alpha)

    def test_parametric_es_closed_form(self, normal_returns):
        z = stats.norm.ppf(0.05)
        true_es = -MU_D + SD_D * stats.norm.pdf(z) / 0.05
        assert np.isclose(parametric_es(normal_returns, 0.95), true_es, atol=2e-4)
        assert np.isclose(historical_es(normal_returns, 0.95), true_es, atol=5e-4)


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------
class TestDrawdown:
    def test_hand_computed_sequence(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        r = pd.Series([0.10, -0.20, 0.05, 0.10], index=idx)
        dd = drawdown_series(r)
        # equity: 1.1, 0.88, 0.924, 1.0164 ; running peak: 1.1 throughout
        expected = np.array([0.0, 0.88 / 1.1 - 1, 0.924 / 1.1 - 1, 1.0164 / 1.1 - 1])
        assert np.allclose(dd.values, expected, atol=1e-12)
        assert np.isclose(max_drawdown(r), 0.20, atol=1e-12)
        assert max_drawdown(r) > 0

    def test_drawdown_stats_episodes(self):
        idx = pd.bdate_range("2020-01-01", periods=8)
        # Episode 1: -10% then full recovery; episode 2: -20%, unrecovered.
        r = pd.Series([0.05, -0.10, 0.12, 0.02, -0.15, -0.06, 0.01, 0.02], index=idx)
        stats_df = drawdown_stats(r)
        assert list(stats_df.columns) == ["depth", "start", "trough", "recovery", "duration_days"]
        assert len(stats_df) == 2
        assert stats_df.loc[0, "depth"] > stats_df.loc[1, "depth"]  # sorted by depth
        assert np.isclose(stats_df.loc[1, "depth"], 0.10, atol=1e-12)
        assert pd.isna(stats_df.loc[0, "recovery"])  # deepest one never recovers
        assert stats_df.loc[1, "recovery"] == idx[2]

    def test_monotone_gain_has_zero_drawdown(self):
        r = pd.Series([0.01] * 50, index=pd.bdate_range("2020-01-01", periods=50))
        assert max_drawdown(r) == 0.0
        assert (drawdown_series(r) == 0).all()


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------
class TestStress:
    def test_apply_scenario_hand_computed(self):
        w = pd.Series({"SPY": 0.6, "TLT": 0.4})
        sc = Scenario("crash", {"SPY": -0.50, "TLT": 0.20, "GLD": 0.10})
        assert np.isclose(apply_scenario(w, sc), 0.6 * -0.5 + 0.4 * 0.2)

    def test_run_scenarios_frame(self):
        w = pd.Series({"SPY": 0.6, "AGG": 0.4})
        out = run_scenarios(w, HISTORICAL_SCENARIOS)
        assert set(out.columns) == {"pnl", "n_assets_shocked"}
        assert len(out) == len(HISTORICAL_SCENARIOS)
        assert out.loc["GFC-2008", "pnl"] < 0  # 60/40 loses in 2008

    def test_correlation_stress_properties(self):
        vols = np.array([0.15, 0.20, 0.10])
        corr = np.array([[1.0, 0.2, -0.3], [0.2, 1.0, 0.1], [-0.3, 0.1, 1.0]])
        cov = pd.DataFrame(corr * np.outer(vols, vols), index=list("xyz"), columns=list("xyz"))
        stressed = correlation_stress(cov, 0.5)
        # Variances preserved, correlations pushed toward 1, PSD.
        assert np.allclose(np.diag(stressed.values), vols**2, rtol=1e-6)
        s_vol = np.sqrt(np.diag(stressed.values))
        s_corr = stressed.values / np.outer(s_vol, s_vol)
        assert s_corr[0, 2] > corr[0, 2]
        assert (np.linalg.eigvalsh(stressed.values) > -1e-12).all()
        # lambda = 0 is a no-op.
        assert np.allclose(correlation_stress(cov, 0.0).values, cov.values, atol=1e-12)

    def test_stress_factor_out_of_range(self):
        cov = pd.DataFrame(np.eye(2), index=list("ab"), columns=list("ab"))
        with pytest.raises(ValueError, match="stress_factor"):
            correlation_stress(cov, 1.5)
