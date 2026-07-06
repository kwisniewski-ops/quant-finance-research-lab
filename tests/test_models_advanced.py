"""Cross-model tests: Heston, Merton, SABR, binomial trees, CN PDE.

Anchors every advanced model against an independent implementation:
binomial and CN-PDE against Black-Scholes; Merton and Heston semi-analytic
prices against seeded Monte Carlo; SABR against its ATM branch.
"""

import numpy as np
import pandas as pd
import pytest

from src.math.monte_carlo_methods import mc_price
from src.math.pde_solvers import cn_bs_grid, cn_bs_price
from src.models.binomial_tree import binomial_price
from src.models.black_scholes import bs_price
from src.models.heston_model import HestonParams, heston_price, simulate_heston
from src.models.jump_diffusion import merton_price, simulate_merton
from src.models.factor_models import fit_factor_model, rolling_betas
from src.models.stochastic_volatility import implied_vol_surface, sabr_implied_vol


class TestBinomialTree:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
    def test_converges_to_black_scholes(self, option_type, K):
        S, T, r, sigma = 100.0, 1.0, 0.05, 0.2
        tree = binomial_price(S, K, T, r, sigma, n_steps=500,
                              option_type=option_type)
        bs = bs_price(S, K, T, r, sigma, option_type)
        assert tree == pytest.approx(bs, abs=1e-2)

    def test_american_put_geq_european_put(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        amer = binomial_price(S, K, T, r, sigma, option_type="put",
                              american=True)
        euro = binomial_price(S, K, T, r, sigma, option_type="put")
        assert amer >= euro
        # ATM put with positive rates has strictly positive premium.
        assert amer - euro > 1e-3

    def test_american_call_no_dividend_equals_european(self):
        # Without dividends, early exercise of a call is never optimal.
        S, K, T, r, sigma = 100.0, 95.0, 1.0, 0.05, 0.25
        amer = binomial_price(S, K, T, r, sigma, american=True)
        euro = binomial_price(S, K, T, r, sigma)
        assert amer == pytest.approx(euro, abs=1e-10)

    def test_american_put_never_below_intrinsic(self):
        price = binomial_price(80.0, 100.0, 1.0, 0.05, 0.2,
                               option_type="put", american=True)
        assert price >= 20.0 - 1e-12

    def test_put_call_parity_european(self):
        S, K, T, r, sigma = 102.0, 97.0, 0.6, 0.04, 0.3
        call = binomial_price(S, K, T, r, sigma, n_steps=800)
        put = binomial_price(S, K, T, r, sigma, n_steps=800,
                             option_type="put")
        assert call - put == pytest.approx(
            S - K * np.exp(-r * T), abs=1e-10
        )

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            binomial_price(100.0, 100.0, -1.0, 0.05, 0.2)
        with pytest.raises(ValueError):
            binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, n_steps=0)


class TestCrankNicolson:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    def test_european_matches_black_scholes(self, option_type, K):
        S, T, r, sigma = 100.0, 1.0, 0.05, 0.2
        pde = cn_bs_price(S, K, T, r, sigma, option_type=option_type)
        bs = bs_price(S, K, T, r, sigma, option_type)
        assert pde == pytest.approx(bs, abs=1e-2)

    def test_american_put_matches_fine_binomial(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        pde = cn_bs_price(S, K, T, r, sigma, option_type="put",
                          american=True)
        tree = binomial_price(S, K, T, r, sigma, n_steps=2000,
                              option_type="put", american=True)
        assert pde == pytest.approx(tree, abs=2e-2)

    def test_american_geq_european_on_grid(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        amer = cn_bs_price(S, K, T, r, sigma, option_type="put",
                           american=True)
        euro = cn_bs_price(S, K, T, r, sigma, option_type="put")
        assert amer >= euro - 1e-10

    def test_grid_shapes_and_terminal_row(self):
        s_grid, t_grid, surface = cn_bs_grid(
            100.0, 100.0, 1.0, 0.05, 0.2, n_s=120, n_t=80
        )
        assert s_grid.shape == (121,)
        assert t_grid.shape == (81,)
        assert surface.shape == (81, 121)
        assert t_grid[0] == 0.0 and t_grid[-1] == pytest.approx(1.0)
        # Last row is the terminal payoff.
        np.testing.assert_allclose(
            surface[-1], np.maximum(s_grid - 100.0, 0.0), atol=1e-12
        )

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            cn_bs_price(100.0, 100.0, 1.0, 0.05, -0.2)
        with pytest.raises(ValueError):
            cn_bs_price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="digital")


class TestMerton:
    def test_lam_zero_equals_black_scholes(self):
        S, K, T, r, sigma = 100.0, 105.0, 0.9, 0.04, 0.22
        for option_type in ("call", "put"):
            merton = merton_price(S, K, T, r, sigma, lam=0.0, mu_j=-0.1,
                                  sigma_j=0.2, option_type=option_type)
            bs = bs_price(S, K, T, r, sigma, option_type)
            assert merton == pytest.approx(bs, abs=1e-8)

    def test_put_call_parity(self):
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        kw = dict(sigma=0.2, lam=0.75, mu_j=-0.08, sigma_j=0.18)
        call = merton_price(S, K, T, r, option_type="call", **kw)
        put = merton_price(S, K, T, r, option_type="put", **kw)
        assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)

    def test_jump_risk_increases_option_value(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        with_jumps = merton_price(S, K, T, r, sigma, lam=1.0, mu_j=-0.1,
                                  sigma_j=0.2)
        without = bs_price(S, K, T, r, sigma)
        assert with_jumps > without

    def test_mc_matches_series_price(self):
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        sigma, lam, mu_j, sigma_j = 0.2, 0.5, -0.1, 0.2
        analytic = merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j)

        rng = np.random.default_rng(42)
        paths = simulate_merton(S, r, sigma, lam, mu_j, sigma_j, T,
                                n_steps=50, n_paths=100_000, rng=rng)
        payoff = np.maximum(paths[:, -1] - K, 0.0)
        price, se = mc_price(payoff, r, T)
        assert abs(price - analytic) < 3.0 * se
        assert se < 0.15  # sanity: enough paths for a meaningful test

    def test_mc_martingale_property(self):
        # E[S_T] = S0 * exp(mu * T) under the compensated convention.
        rng = np.random.default_rng(42)
        paths = simulate_merton(100.0, 0.07, 0.2, 1.0, -0.05, 0.15, 2.0,
                                n_steps=40, n_paths=100_000, rng=rng)
        expected = 100.0 * np.exp(0.07 * 2.0)
        se = paths[:, -1].std(ddof=1) / np.sqrt(paths.shape[0])
        assert abs(paths[:, -1].mean() - expected) < 3.5 * se

    def test_paths_shape_and_start(self):
        paths = simulate_merton(50.0, 0.05, 0.2, 0.3, 0.0, 0.1, 1.0,
                                n_steps=12, n_paths=7,
                                rng=np.random.default_rng(0))
        assert paths.shape == (7, 13)
        assert np.all(paths[:, 0] == 50.0)
        assert np.all(paths > 0.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            merton_price(100.0, 100.0, 1.0, 0.05, 0.2, lam=-0.1,
                         mu_j=0.0, sigma_j=0.1)
        with pytest.raises(ValueError):
            simulate_merton(100.0, 0.05, -0.2, 0.5, 0.0, 0.1, 1.0, 10, 10)


class TestHeston:
    PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)

    def test_feller_property(self):
        assert HestonParams(0.04, 2.0, 0.04, 0.3, -0.5).feller_satisfied
        assert not HestonParams(0.04, 1.0, 0.04, 0.5, -0.5).feller_satisfied

    def test_params_validation(self):
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=-1.0, theta=0.04, xi=0.5, rho=-0.7)
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-1.5)

    def test_xi_to_zero_recovers_black_scholes(self):
        # With v0 = theta and vanishing vol-of-vol, variance is frozen at
        # theta and Heston degenerates to BS with sigma = sqrt(theta).
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-6,
                              rho=-0.7)
        for K in (85.0, 100.0, 115.0):
            heston = heston_price(100.0, K, 1.0, 0.03, params)
            bs = bs_price(100.0, K, 1.0, 0.03, np.sqrt(0.04))
            assert heston == pytest.approx(bs, abs=1e-2)

    def test_put_call_parity(self):
        S, K, T, r = 100.0, 110.0, 1.5, 0.03
        call = heston_price(S, K, T, r, self.PARAMS, "call")
        put = heston_price(S, K, T, r, self.PARAMS, "put")
        assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)

    def test_semi_analytic_vs_monte_carlo(self):
        S, K, T, r = 100.0, 100.0, 1.0, 0.03
        analytic = heston_price(S, K, T, r, self.PARAMS)

        rng = np.random.default_rng(42)
        s_paths, _ = simulate_heston(S, self.PARAMS, r, T, n_steps=200,
                                     n_paths=60_000, rng=rng)
        payoff = np.maximum(s_paths[:, -1] - K, 0.0)
        price, se = mc_price(payoff, r, T)
        # Allow 3.5 SE plus a small allowance for Euler discretization bias.
        assert abs(price - analytic) < 3.5 * se + 0.05

    def test_negative_rho_produces_skew(self):
        # Downside puts should be richer (in implied vol) than upside calls.
        from src.models.black_scholes import implied_vol

        S, T, r = 100.0, 1.0, 0.03
        iv_down = implied_vol(
            heston_price(S, 80.0, T, r, self.PARAMS), S, 80.0, T, r
        )
        iv_up = implied_vol(
            heston_price(S, 120.0, T, r, self.PARAMS), S, 120.0, T, r
        )
        assert iv_down > iv_up

    def test_simulation_shapes_and_positivity(self):
        rng = np.random.default_rng(0)
        s_paths, v_paths = simulate_heston(100.0, self.PARAMS, 0.03, 1.0,
                                           n_steps=50, n_paths=11, rng=rng)
        assert s_paths.shape == (11, 51)
        assert v_paths.shape == (11, 51)
        assert np.all(s_paths > 0.0)
        assert np.all(s_paths[:, 0] == 100.0)
        assert np.all(v_paths[:, 0] == self.PARAMS.v0)

    def test_deep_otm_short_maturity_stable(self):
        # Numerically challenging corner: quadrature must not produce
        # spurious mass far out of the money at short maturities.
        price = heston_price(100.0, 160.0, 0.05, 0.03, self.PARAMS)
        assert 0.0 <= price < 1e-4

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            heston_price(100.0, 100.0, -1.0, 0.03, self.PARAMS)
        with pytest.raises(ValueError):
            simulate_heston(-5.0, self.PARAMS, 0.03, 1.0, 10, 10)


class TestSabr:
    def test_atm_branch_continuous(self):
        # K -> F must approach the exact ATM value smoothly.
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.3, 0.5, -0.3, 0.4
        atm = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
        for bump in (1e-3, 1e-5, 1e-7):
            up = sabr_implied_vol(F, F * (1 + bump), T, alpha, beta, rho, nu)
            dn = sabr_implied_vol(F, F * (1 - bump), T, alpha, beta, rho, nu)
            assert up == pytest.approx(atm, rel=10 * bump + 1e-9)
            assert dn == pytest.approx(atm, rel=10 * bump + 1e-9)

    def test_atm_lognormal_case(self):
        # beta = 1, nu -> 0: implied vol collapses to alpha at the money.
        vol = sabr_implied_vol(100.0, 100.0, 1.0, alpha=0.25, beta=1.0,
                               rho=0.0, nu=1e-12)
        assert vol == pytest.approx(0.25, abs=1e-8)

    def test_negative_rho_skew(self):
        F, T = 100.0, 1.0
        kw = dict(alpha=0.3, beta=0.7, rho=-0.5, nu=0.5)
        low = sabr_implied_vol(F, 80.0, T, **kw)
        high = sabr_implied_vol(F, 120.0, T, **kw)
        assert low > high

    def test_smile_convexity_with_nu(self):
        # Higher vol-of-vol lifts the wings relative to ATM.
        F, T = 100.0, 1.0
        base = dict(alpha=0.3, beta=1.0, rho=0.0)
        wing_lo = sabr_implied_vol(F, 60.0, T, nu=0.8, **base)
        atm_lo = sabr_implied_vol(F, F, T, nu=0.8, **base)
        assert wing_lo > atm_lo

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            sabr_implied_vol(100.0, 100.0, 1.0, alpha=-0.1, beta=0.5,
                             rho=0.0, nu=0.3)
        with pytest.raises(ValueError):
            sabr_implied_vol(100.0, 100.0, 1.0, alpha=0.3, beta=1.5,
                             rho=0.0, nu=0.3)


class TestImpliedVolSurface:
    def test_flat_bs_surface_recovered(self):
        S, r, sigma = 100.0, 0.03, 0.25
        strikes = [80.0, 100.0, 120.0]
        maturities = [0.25, 1.0]
        surf = implied_vol_surface(
            S, r, strikes, maturities,
            price_fn=lambda K, T: bs_price(S, K, T, r, sigma),
        )
        assert isinstance(surf, pd.DataFrame)
        assert list(surf.index) == maturities
        assert list(surf.columns) == strikes
        np.testing.assert_allclose(surf.to_numpy(), sigma, atol=1e-6)

    def test_heston_surface_has_skew(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5,
                              rho=-0.7)
        S, r = 100.0, 0.03
        surf = implied_vol_surface(
            S, r, [85.0, 100.0, 115.0], [0.5, 1.0],
            price_fn=lambda K, T: heston_price(S, K, T, r, params),
        )
        vols = surf.to_numpy()
        assert np.all(np.isfinite(vols))
        # Downside strikes carry higher implied vol at every maturity.
        assert np.all(vols[:, 0] > vols[:, -1])


class TestFactorModels:
    @staticmethod
    def _synthetic(n=1000, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2018-01-01", periods=n)
        factors = pd.DataFrame(
            rng.standard_normal((n, 2)) * 0.01,
            index=dates, columns=["MKT", "SMB"],
        )
        true_alpha, true_betas = 0.0002, np.array([1.2, -0.4])
        noise = rng.standard_normal(n) * 0.005
        asset = pd.Series(
            true_alpha + factors.to_numpy() @ true_betas + noise,
            index=dates, name="asset",
        )
        return asset, factors, true_alpha, true_betas

    def test_recovers_true_coefficients(self):
        asset, factors, true_alpha, true_betas = self._synthetic()
        res = fit_factor_model(asset, factors)
        assert res.n_obs == len(asset)
        assert res.alpha == pytest.approx(true_alpha, abs=6e-4)
        np.testing.assert_allclose(res.betas.to_numpy(), true_betas,
                                   atol=0.06)
        assert list(res.betas.index) == ["MKT", "SMB"]
        assert 0.7 < res.r_squared <= 1.0
        # Strong loadings should be highly significant.
        assert abs(res.tstats["MKT"]) > 10.0

    def test_residuals_orthogonal_to_factors(self):
        asset, factors, *_ = self._synthetic()
        res = fit_factor_model(asset, factors)
        for col in factors.columns:
            corr = np.corrcoef(res.resid, factors[col])[0, 1]
            assert abs(corr) < 1e-10

    def test_rf_subtraction(self):
        asset, factors, *_ = self._synthetic()
        rf = 0.0001
        res0 = fit_factor_model(asset, factors, rf=0.0)
        res1 = fit_factor_model(asset, factors, rf=rf)
        # Constant rf only shifts alpha, leaves betas untouched.
        assert res1.alpha == pytest.approx(res0.alpha - rf, abs=1e-12)
        np.testing.assert_allclose(res1.betas, res0.betas, atol=1e-12)

    def test_rolling_betas_shape_and_accuracy(self):
        asset, factors, _, true_betas = self._synthetic()
        rb = rolling_betas(asset, factors, window=252)
        assert rb.shape == (len(asset) - 252 + 1, 2)
        assert list(rb.columns) == ["MKT", "SMB"]
        # Every rolling estimate stays near the (constant) true loadings.
        assert (rb["MKT"] - true_betas[0]).abs().max() < 0.25

    def test_insufficient_data_raises(self):
        asset, factors, *_ = self._synthetic(n=3)
        with pytest.raises(ValueError):
            fit_factor_model(asset, factors)
