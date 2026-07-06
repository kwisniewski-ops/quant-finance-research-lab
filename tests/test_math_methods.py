"""Tests for src.math: Monte Carlo simulators and numerical linear algebra."""

import numpy as np
import pandas as pd
import pytest

from src.math.monte_carlo_methods import (
    mc_price,
    simulate_cir,
    simulate_gbm,
    simulate_ou,
    simulate_regime_switching,
)
from src.math.numerical_linear_algebra import (
    is_psd,
    ledoit_wolf_shrinkage,
    nearest_psd,
    safe_cholesky,
)
from src.models.black_scholes import bs_price


class TestSimulateGbm:
    def test_shape_and_initial_value(self):
        paths = simulate_gbm(100.0, 0.05, 0.2, 1.0, n_steps=52, n_paths=10,
                             rng=np.random.default_rng(42))
        assert paths.shape == (10, 53)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0.0)

    def test_terminal_mean_matches_theory(self):
        S0, mu, T = 100.0, 0.07, 2.0
        rng = np.random.default_rng(42)
        paths = simulate_gbm(S0, mu, 0.2, T, n_steps=1, n_paths=200_000,
                             rng=rng)
        terminal = paths[:, -1]
        se = terminal.std(ddof=1) / np.sqrt(len(terminal))
        assert abs(terminal.mean() - S0 * np.exp(mu * T)) < 3.5 * se

    def test_terminal_log_variance_matches_theory(self):
        sigma, T = 0.3, 1.5
        rng = np.random.default_rng(42)
        paths = simulate_gbm(100.0, 0.05, sigma, T, n_steps=1,
                             n_paths=200_000, rng=rng)
        log_ret = np.log(paths[:, -1] / 100.0)
        assert log_ret.var(ddof=1) == pytest.approx(sigma**2 * T, rel=0.02)

    def test_antithetic_returns_n_paths_in_mirrored_pairs(self):
        n_paths = 1000
        paths = simulate_gbm(100.0, 0.05, 0.2, 1.0, n_steps=10,
                             n_paths=n_paths, rng=np.random.default_rng(42),
                             antithetic=True)
        assert paths.shape == (n_paths, 11)
        # Rows i and i + n/2 use mirrored normals: log-returns sum to
        # twice the deterministic drift.
        half = n_paths // 2
        lr = np.log(paths[:, -1] / 100.0)
        drift = (0.05 - 0.5 * 0.2**2) * 1.0
        np.testing.assert_allclose(lr[:half] + lr[half:], 2.0 * drift,
                                   atol=1e-12)

    def test_antithetic_reduces_variance_of_mean(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        n = 50_000
        plain = simulate_gbm(100.0, 0.05, 0.2, 1.0, 1, n, rng=rng1)
        anti = simulate_gbm(100.0, 0.05, 0.2, 1.0, 1, n, rng=rng2,
                            antithetic=True)
        half = n // 2
        pair_means = 0.5 * (anti[:half, -1] + anti[half:, -1])
        var_plain = plain[:, -1].var(ddof=1) / n
        var_anti = pair_means.var(ddof=1) / half
        assert var_anti < var_plain

    def test_antithetic_odd_paths_raises(self):
        with pytest.raises(ValueError, match="even"):
            simulate_gbm(100.0, 0.05, 0.2, 1.0, 10, 11, antithetic=True)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            simulate_gbm(-100.0, 0.05, 0.2, 1.0, 10, 10)
        with pytest.raises(ValueError):
            simulate_gbm(100.0, 0.05, 0.2, 0.0, 10, 10)

    def test_gbm_prices_european_call(self):
        # End-to-end: GBM paths + mc_price reproduce Black-Scholes.
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        rng = np.random.default_rng(42)
        paths = simulate_gbm(S, r, sigma, T, n_steps=1, n_paths=200_000,
                             rng=rng, antithetic=True)
        payoff = np.maximum(paths[:, -1] - K, 0.0)
        price, se = mc_price(payoff, r, T)
        assert abs(price - bs_price(S, K, T, r, sigma)) < 3.5 * se


class TestSimulateOu:
    def test_exact_terminal_distribution(self):
        # Terminal mean/variance must match the closed-form OU law within
        # Monte Carlo error (the discretization itself is exact).
        x0, kappa, theta, sigma, T = 0.5, 1.8, 0.1, 0.4, 2.0
        rng = np.random.default_rng(42)
        n = 200_000
        paths = simulate_ou(x0, kappa, theta, sigma, T, n_steps=25,
                            n_paths=n, rng=rng)
        terminal = paths[:, -1]

        mean_theory = theta + (x0 - theta) * np.exp(-kappa * T)
        var_theory = sigma**2 * (1.0 - np.exp(-2.0 * kappa * T)) / (2.0 * kappa)

        se_mean = np.sqrt(var_theory / n)
        assert abs(terminal.mean() - mean_theory) < 3.5 * se_mean
        # Var of the sample variance ~ 2 var^2 / n for Gaussians.
        se_var = var_theory * np.sqrt(2.0 / n)
        assert abs(terminal.var(ddof=1) - var_theory) < 3.5 * se_var

    def test_step_count_invariance_of_law(self):
        # Exact discretization: terminal moments don't drift with n_steps.
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        coarse = simulate_ou(0.0, 2.0, 0.05, 0.3, 1.0, 2, 100_000, rng=rng1)
        fine = simulate_ou(0.0, 2.0, 0.05, 0.3, 1.0, 200, 100_000, rng=rng2)
        assert coarse[:, -1].mean() == pytest.approx(fine[:, -1].mean(),
                                                     abs=3e-3)
        assert coarse[:, -1].var() == pytest.approx(fine[:, -1].var(),
                                                    rel=0.02)

    def test_shape_and_validation(self):
        paths = simulate_ou(1.0, 1.0, 0.0, 0.2, 1.0, 10, 5,
                            rng=np.random.default_rng(0))
        assert paths.shape == (5, 11)
        assert np.all(paths[:, 0] == 1.0)
        with pytest.raises(ValueError):
            simulate_ou(1.0, -1.0, 0.0, 0.2, 1.0, 10, 5)


class TestSimulateCir:
    def test_paths_non_negative_even_when_feller_violated(self):
        # 2*kappa*theta = 0.04 < sigma^2 = 0.36: origin attainable.
        rng = np.random.default_rng(42)
        paths = simulate_cir(0.02, 1.0, 0.02, 0.6, 2.0, n_steps=200,
                             n_paths=2000, rng=rng)
        assert np.all(paths >= 0.0)

    def test_long_run_mean(self):
        # After many mean-reversion times, E[X_t] ~ theta.
        kappa, theta = 3.0, 0.05
        rng = np.random.default_rng(42)
        paths = simulate_cir(0.2, kappa, theta, 0.25, 5.0, n_steps=500,
                             n_paths=20_000, rng=rng)
        assert paths[:, -1].mean() == pytest.approx(theta, rel=0.05)

    def test_conditional_mean_matches_theory(self):
        # E[X_T | X_0] = theta + (x0 - theta) e^{-kappa T} for CIR.
        x0, kappa, theta, T = 0.09, 2.0, 0.04, 1.0
        rng = np.random.default_rng(42)
        paths = simulate_cir(x0, kappa, theta, 0.3, T, n_steps=400,
                             n_paths=50_000, rng=rng)
        mean_theory = theta + (x0 - theta) * np.exp(-kappa * T)
        assert paths[:, -1].mean() == pytest.approx(mean_theory, rel=0.02)

    def test_validation(self):
        with pytest.raises(ValueError):
            simulate_cir(-0.01, 1.0, 0.04, 0.3, 1.0, 10, 10)
        with pytest.raises(ValueError):
            simulate_cir(0.04, 1.0, -0.04, 0.3, 1.0, 10, 10)


class TestRegimeSwitching:
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    def test_shapes_and_labels(self):
        paths, regimes = simulate_regime_switching(
            100.0, [0.08, -0.05], [0.12, 0.35], self.P, 1.0,
            n_steps=252, n_paths=500, rng=np.random.default_rng(42),
        )
        assert paths.shape == (500, 253)
        assert regimes.shape == (500, 253)
        assert regimes.dtype.kind == "i"
        assert set(np.unique(regimes)) <= {0, 1}
        assert np.all(regimes[:, 0] == 0)
        assert np.all(paths[:, 0] == 100.0)

    def test_stationary_regime_frequencies(self):
        # Stationary distribution of P is (2/3, 1/3).
        rng = np.random.default_rng(42)
        _, regimes = simulate_regime_switching(
            100.0, [0.05, 0.0], [0.1, 0.3], self.P, 4.0,
            n_steps=1000, n_paths=400, rng=rng,
        )
        freq1 = regimes[:, 500:].mean()
        assert freq1 == pytest.approx(1.0 / 3.0, abs=0.03)

    def test_high_vol_regime_raises_realized_vol(self):
        rng = np.random.default_rng(42)
        paths, regimes = simulate_regime_switching(
            100.0, [0.0, 0.0], [0.1, 0.5], self.P, 1.0,
            n_steps=252, n_paths=300, rng=rng,
        )
        log_ret = np.diff(np.log(paths), axis=1)
        in_regime1 = regimes[:, 1:] == 1
        vol0 = log_ret[~in_regime1].std() * np.sqrt(252)
        vol1 = log_ret[in_regime1].std() * np.sqrt(252)
        assert vol1 > 3.0 * vol0

    def test_validation(self):
        bad_P = np.array([[0.9, 0.2], [0.1, 0.9]])  # rows don't sum to 1
        with pytest.raises(ValueError):
            simulate_regime_switching(100.0, [0.05, 0.0], [0.1, 0.3],
                                      bad_P, 1.0, 10, 10)
        with pytest.raises(ValueError):
            simulate_regime_switching(100.0, [0.05], [0.1, 0.3], self.P,
                                      1.0, 10, 10)


class TestMcPrice:
    def test_deterministic_payoff(self):
        payoff = np.full(1000, 7.0)
        price, se = mc_price(payoff, 0.05, 2.0)
        assert price == pytest.approx(7.0 * np.exp(-0.1))
        assert se == 0.0

    def test_standard_error_scaling(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(400_00) + 10.0
        _, se_full = mc_price(x, 0.0, 1.0)
        _, se_quarter = mc_price(x[: len(x) // 4], 0.0, 1.0)
        assert se_quarter == pytest.approx(2.0 * se_full, rel=0.05)

    def test_validation(self):
        with pytest.raises(ValueError):
            mc_price(np.array([1.0]), 0.05, 1.0)
        with pytest.raises(ValueError):
            mc_price(np.array([1.0, 2.0]), 0.05, -1.0)


class TestNearestPsd:
    def test_output_is_psd_and_close(self):
        # Indefinite "correlation-like" matrix.
        A = np.array([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.3],
            [0.7, 0.3, 1.0],
        ])
        A[0, 2] = A[2, 0] = -0.9  # force indefiniteness
        assert not is_psd(A)
        B = nearest_psd(A)
        assert is_psd(B)
        assert np.linalg.norm(B - A, "fro") < 0.5
        np.testing.assert_allclose(B, B.T, atol=1e-14)

    def test_psd_input_unchanged(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        A = X.T @ X
        np.testing.assert_allclose(nearest_psd(A), A, atol=1e-10)

    def test_corr_mode_preserves_unit_diagonal(self):
        A = np.array([
            [1.0, 0.95, -0.9],
            [0.95, 1.0, 0.6],
            [-0.9, 0.6, 1.0],
        ])
        assert not is_psd(A)
        B = nearest_psd(A, corr=True)
        assert is_psd(B, tol=1e-8)
        np.testing.assert_allclose(np.diag(B), 1.0, atol=1e-12)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            nearest_psd(np.ones((2, 3)))


class TestIsPsd:
    def test_identity(self):
        assert is_psd(np.eye(4))

    def test_negative_definite(self):
        assert not is_psd(-np.eye(3))

    def test_boundary_zero_eigenvalue(self):
        A = np.ones((3, 3))  # rank-1, eigenvalues {3, 0, 0}
        assert is_psd(A)


class TestLedoitWolf:
    @staticmethod
    def _returns(n_obs=300, n_assets=8, seed=42):
        # Heterogeneous correlations and vols, so the constant-correlation
        # target is misspecified and the optimal intensity is interior.
        rng = np.random.default_rng(seed)
        B = rng.uniform(-1.0, 1.0, size=(n_assets, 2))
        cov = B @ B.T + np.diag(rng.uniform(0.5, 2.0, n_assets))
        L = np.linalg.cholesky(cov)
        X = rng.standard_normal((n_obs, n_assets)) @ L.T * 0.01
        return pd.DataFrame(X, columns=[f"A{i}" for i in range(n_assets)])

    def test_intensity_in_unit_interval(self):
        _, delta = ledoit_wolf_shrinkage(self._returns())
        assert 0.0 <= delta <= 1.0

    def test_output_shape_psd_and_between_targets(self):
        rets = self._returns()
        shrunk, delta = ledoit_wolf_shrinkage(rets)
        n = rets.shape[1]
        assert shrunk.shape == (n, n)
        assert is_psd(shrunk, tol=1e-12)
        # Diagonal preserved (both S and F share the sample variances).
        X = rets.to_numpy()
        sample_var = ((X - X.mean(0)) ** 2).mean(0)
        np.testing.assert_allclose(np.diag(shrunk), sample_var, rtol=1e-10)

    def test_more_data_less_shrinkage(self):
        _, d_small = ledoit_wolf_shrinkage(self._returns(n_obs=60))
        _, d_large = ledoit_wolf_shrinkage(self._returns(n_obs=2000))
        assert d_large < d_small

    def test_too_few_observations_raise(self):
        with pytest.raises(ValueError):
            ledoit_wolf_shrinkage(pd.DataFrame([[0.01, 0.02]]))


class TestSafeCholesky:
    def test_exact_for_positive_definite(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        A = X.T @ X / 100 + 0.1 * np.eye(5)
        L = np.linalg.cholesky(A)
        np.testing.assert_allclose(safe_cholesky(A), L, atol=1e-12)

    def test_fallback_for_indefinite(self):
        A = np.array([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.3],
            [0.7, 0.3, 1.0],
        ])
        A[0, 2] = A[2, 0] = -0.9
        assert not is_psd(A)
        L = safe_cholesky(A)
        assert np.all(np.isfinite(L))
        recon = L @ L.T
        assert is_psd(recon)
        # Reconstruction close to the PSD projection of A.
        assert np.linalg.norm(recon - nearest_psd(A), "fro") < 1e-4

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            safe_cholesky(np.ones((3, 2)))
