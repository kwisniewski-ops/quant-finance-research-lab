"""Tests for src.portfolio: mean-variance, Black-Litterman, risk parity,
HRP, and robust optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.black_litterman import bl_posterior, implied_equilibrium_returns
from src.portfolio.hierarchical_risk_parity import hrp_weights
from src.portfolio.mean_variance import efficient_frontier, max_sharpe, min_variance
from src.portfolio.risk_parity import risk_contributions, risk_parity_weights
from src.portfolio.robust_optimization import resampled_frontier, robust_max_sharpe

RNG = np.random.default_rng(42)
ASSETS = ["A", "B", "C", "D"]


@pytest.fixture(scope="module")
def mu_cov() -> tuple[pd.Series, pd.DataFrame]:
    mu = pd.Series([0.08, 0.10, 0.12, 0.05], index=ASSETS)
    vols = np.array([0.15, 0.20, 0.25, 0.08])
    corr = np.array(
        [
            [1.00, 0.50, 0.40, 0.10],
            [0.50, 1.00, 0.60, 0.05],
            [0.40, 0.60, 1.00, 0.00],
            [0.10, 0.05, 0.00, 1.00],
        ]
    )
    cov = pd.DataFrame(corr * np.outer(vols, vols), index=ASSETS, columns=ASSETS)
    return mu, cov


# ---------------------------------------------------------------------------
# mean_variance
# ---------------------------------------------------------------------------
class TestMeanVariance:
    def test_min_variance_weights_sum_and_bounds(self, mu_cov):
        _, cov = mu_cov
        res = min_variance(cov)
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-8)
        assert (res.weights >= -1e-9).all() and (res.weights <= 1.0 + 1e-9).all()
        assert res.volatility > 0
        assert np.isnan(res.expected_return)  # mu not supplied

    def test_min_variance_two_asset_analytic(self):
        # w1* = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12)
        s1, s2, rho = 0.20, 0.10, 0.30
        s12 = rho * s1 * s2
        cov = pd.DataFrame(
            [[s1**2, s12], [s12, s2**2]], index=["X", "Y"], columns=["X", "Y"]
        )
        w1_analytic = (s2**2 - s12) / (s1**2 + s2**2 - 2 * s12)
        res = min_variance(cov, bounds=(0.0, 1.0))
        assert np.isclose(res.weights["X"], w1_analytic, atol=1e-6)
        assert np.isclose(res.weights["Y"], 1 - w1_analytic, atol=1e-6)

    def test_max_sharpe_weights_and_optimality(self, mu_cov):
        mu, cov = mu_cov
        res = max_sharpe(mu, cov, rf=0.02)
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-8)
        assert (res.weights >= -1e-9).all() and (res.weights <= 1.0 + 1e-9).all()
        # No random long-only portfolio should beat the optimizer.
        for _ in range(200):
            w = RNG.dirichlet(np.ones(len(mu)))
            sharpe = (w @ mu.values - 0.02) / np.sqrt(w @ cov.values @ w)
            assert sharpe <= res.sharpe + 1e-6

    def test_max_sharpe_respects_tight_bounds(self, mu_cov):
        mu, cov = mu_cov
        res = max_sharpe(mu, cov, bounds=(0.05, 0.40))
        assert (res.weights >= 0.05 - 1e-8).all()
        assert (res.weights <= 0.40 + 1e-8).all()
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-8)

    def test_frontier_monotone_and_weights(self, mu_cov):
        mu, cov = mu_cov
        fr = efficient_frontier(mu, cov, n_points=15)
        assert len(fr) == 15
        # Volatility non-decreasing along increasing expected return.
        assert (np.diff(fr["volatility"].to_numpy()) >= -1e-8).all()
        assert (np.diff(fr["expected_return"].to_numpy()) >= -1e-10).all()
        w = fr[ASSETS].to_numpy()
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
        assert (w >= -1e-8).all() and (w <= 1 + 1e-8).all()

    def test_misaligned_and_nonsquare_inputs_raise(self, mu_cov):
        mu, cov = mu_cov
        with pytest.raises(ValueError, match="misaligned"):
            max_sharpe(mu.rename({"A": "Z"}), cov)
        with pytest.raises(ValueError, match="square"):
            min_variance(cov.iloc[:3, :])
        with pytest.raises(ValueError, match="symmetric"):
            bad = cov.copy()
            bad.iloc[0, 1] += 0.01
            min_variance(bad)


# ---------------------------------------------------------------------------
# black_litterman
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def market(mu_cov):
    _, cov = mu_cov
    w_mkt = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
    return cov, w_mkt


class TestBlackLitterman:
    def test_equilibrium_returns_formula(self, market):
        cov, w_mkt = market
        pi = implied_equilibrium_returns(cov, w_mkt, delta=2.5)
        expected = 2.5 * cov.values @ w_mkt.values
        assert np.allclose(pi.values, expected)

    def test_zero_confidence_recovers_equilibrium(self, market):
        cov, w_mkt = market
        pi = implied_equilibrium_returns(cov, w_mkt)
        P = np.zeros((1, 4))
        P[0, 0] = 1.0
        Q = np.array([0.30])  # extreme view...
        omega = np.array([[1e6]])  # ...held with zero confidence
        mu_bl, cov_bl = bl_posterior(cov, w_mkt, P, Q, omega=omega)
        assert np.allclose(mu_bl.values, pi.values, atol=1e-6)
        assert cov_bl.shape == cov.shape

    def test_posterior_tilts_toward_view(self, market):
        cov, w_mkt = market
        pi = implied_equilibrium_returns(cov, w_mkt)
        P = np.zeros((1, 4))
        P[0, 0] = 1.0
        Q = np.array([pi["A"] + 0.05])  # bullish on A
        mu_bl, _ = bl_posterior(cov, w_mkt, P, Q)
        assert mu_bl["A"] > pi["A"]
        assert mu_bl["A"] < Q[0]  # shrunk between prior and view

    def test_posterior_cov_psd_and_bigger(self, market):
        cov, w_mkt = market
        P = np.array([[1.0, -1.0, 0.0, 0.0]])
        Q = np.array([0.02])
        _, cov_bl = bl_posterior(cov, w_mkt, P, Q)
        eigs = np.linalg.eigvalsh(cov_bl.values)
        assert (eigs > 0).all()
        # Posterior covariance adds estimation uncertainty to Sigma.
        assert np.trace(cov_bl.values) >= np.trace(cov.values)

    def test_dimension_mismatch_raises(self, market):
        cov, w_mkt = market
        with pytest.raises(ValueError, match="columns"):
            bl_posterior(cov, w_mkt, np.ones((1, 3)), np.array([0.1]))
        with pytest.raises(ValueError, match="sum to 1"):
            implied_equilibrium_returns(cov, w_mkt * 2)


# ---------------------------------------------------------------------------
# risk_parity
# ---------------------------------------------------------------------------
class TestRiskParity:
    def test_equal_risk_contributions(self, mu_cov):
        _, cov = mu_cov
        w = risk_parity_weights(cov)
        assert np.isclose(w.sum(), 1.0, atol=1e-10)
        assert (w > 0).all()
        rc = risk_contributions(w, cov)
        assert np.isclose(rc.sum(), 1.0, atol=1e-10)
        assert np.max(np.abs(rc.values - 0.25)) < 1e-6

    def test_custom_budget(self, mu_cov):
        _, cov = mu_cov
        budget = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
        w = risk_parity_weights(cov, budget=budget)
        rc = risk_contributions(w, cov)
        assert np.max(np.abs(rc.values - budget.values)) < 1e-6

    def test_uncorrelated_case_matches_inverse_variance(self):
        # With zero correlation, ERC weights are proportional to 1/vol.
        vols = np.array([0.10, 0.20, 0.40])
        cov = pd.DataFrame(np.diag(vols**2), index=list("abc"), columns=list("abc"))
        w = risk_parity_weights(cov)
        expected = (1 / vols) / (1 / vols).sum()
        assert np.allclose(w.values, expected, atol=1e-8)

    def test_erc_on_dense_correlated_covariance(self):
        # Regression: naive fixed-point iteration oscillates on strongly
        # correlated dense covariances; the Newton polish must not.
        rng = np.random.default_rng(42)
        n = 12
        loadings = rng.normal(0.8, 0.3, size=(n, 3))
        vols = rng.uniform(0.05, 0.35, n)
        c = loadings @ loadings.T + np.diag(rng.uniform(0.1, 0.5, n))
        d = np.sqrt(np.diag(c))
        corr = c / np.outer(d, d)
        names = [f"A{i}" for i in range(n)]
        cov = pd.DataFrame(corr * np.outer(vols, vols), index=names, columns=names)
        w = risk_parity_weights(cov)
        rc = risk_contributions(w, cov)
        assert np.isclose(w.sum(), 1.0, atol=1e-10)
        assert (w > 0).all()
        assert np.max(np.abs(rc.values - 1.0 / n)) < 1e-6

    def test_bad_budget_raises(self, mu_cov):
        _, cov = mu_cov
        with pytest.raises(ValueError, match="sum to 1"):
            risk_parity_weights(cov, budget=pd.Series([0.5] * 4, index=ASSETS))


# ---------------------------------------------------------------------------
# hierarchical_risk_parity
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def block_returns() -> pd.DataFrame:
    """Two blocks: 3 correlated equities (high vol), 3 correlated bonds (low vol)."""
    rng = np.random.default_rng(42)
    n = 1500
    eq_factor = rng.normal(0, 0.012, n)
    bd_factor = rng.normal(0, 0.003, n)
    data = {}
    for i in range(3):
        data[f"EQ{i}"] = eq_factor + rng.normal(0, 0.004, n)
    for i in range(3):
        data[f"BD{i}"] = bd_factor + rng.normal(0, 0.001, n)
    return pd.DataFrame(data)


class TestHRP:
    def test_weights_valid(self, block_returns):
        w = hrp_weights(block_returns)
        assert np.isclose(w.sum(), 1.0, atol=1e-10)
        assert ((w >= 0) & (w <= 1)).all()
        assert set(w.index) == set(block_returns.columns)

    def test_low_vol_block_gets_more_weight(self, block_returns):
        w = hrp_weights(block_returns)
        eq = w[[c for c in w.index if c.startswith("EQ")]].sum()
        bd = w[[c for c in w.index if c.startswith("BD")]].sum()
        assert bd > eq  # inverse-variance splits favor the low-vol block
        # Within a block, weights should be of similar magnitude.
        eq_w = w[[c for c in w.index if c.startswith("EQ")]]
        assert eq_w.max() / eq_w.min() < 3.0

    def test_input_validation(self, block_returns):
        with pytest.raises(ValueError, match="NaN"):
            bad = block_returns.copy()
            bad.iloc[5, 0] = np.nan
            hrp_weights(bad)
        with pytest.raises(ValueError, match="2 assets"):
            hrp_weights(block_returns.iloc[:, :1])


# ---------------------------------------------------------------------------
# robust_optimization
# ---------------------------------------------------------------------------
class TestRobust:
    def test_zero_uncertainty_matches_max_sharpe(self, mu_cov):
        mu, cov = mu_cov
        nominal = max_sharpe(mu, cov)
        robust = robust_max_sharpe(mu, cov, mu_uncertainty=0.0)
        assert np.allclose(robust.weights.values, nominal.weights.values, atol=1e-4)

    def test_uncertainty_diversifies(self, mu_cov):
        mu, cov = mu_cov
        nominal = max_sharpe(mu, cov)
        robust = robust_max_sharpe(mu, cov, mu_uncertainty=0.04)
        assert np.isclose(robust.weights.sum(), 1.0, atol=1e-8)
        # Worst-case optimization cannot beat the nominal optimum on nominal mu.
        assert robust.sharpe <= nominal.sharpe + 1e-8
        # And should spread bets at least as much (lower concentration).
        assert (robust.weights**2).sum() <= (nominal.weights**2).sum() + 1e-6

    def test_negative_uncertainty_raises(self, mu_cov):
        mu, cov = mu_cov
        with pytest.raises(ValueError, match="non-negative"):
            robust_max_sharpe(mu, cov, mu_uncertainty=-0.01)

    def test_resampled_frontier_shape_and_weights(self, mu_cov):
        mu, cov = mu_cov
        rng = np.random.default_rng(42)
        fr = resampled_frontier(mu, cov, n_samples=25, n_points=8, n_obs=252, rng=rng)
        assert len(fr) == 8
        w = fr[ASSETS].to_numpy()
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-8)
        assert (w >= -1e-8).all()
        # Resampling smooths: highest-return point is not 100% one asset.
        assert w[-1].max() < 0.999
