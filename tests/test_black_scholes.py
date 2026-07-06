"""Tests for src.models.black_scholes: prices, Greeks, implied vol."""

import numpy as np
import pytest

from src.models.black_scholes import bs_greeks, bs_price, implied_vol


class TestBsPrice:
    def test_known_value_atm_call(self):
        # Standard textbook anchor: S=K=100, T=1, r=5%, sigma=20%.
        assert bs_price(100.0, 100.0, 1.0, 0.05, 0.2) == pytest.approx(
            10.450583572185565, abs=1e-4
        )

    def test_known_value_atm_put(self):
        assert bs_price(100.0, 100.0, 1.0, 0.05, 0.2, "put") == pytest.approx(
            5.573526022256971, abs=1e-4
        )

    def test_put_call_parity(self):
        S, K, T, r, sigma, q = 105.0, 98.0, 0.75, 0.03, 0.27, 0.01
        call = bs_price(S, K, T, r, sigma, "call", q=q)
        put = bs_price(S, K, T, r, sigma, "put", q=q)
        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_scalar_in_scalar_out(self):
        out = bs_price(100.0, 100.0, 1.0, 0.05, 0.2)
        assert isinstance(out, float)

    def test_vectorized_over_strikes(self):
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = bs_price(100.0, strikes, 1.0, 0.05, 0.2)
        assert isinstance(prices, np.ndarray)
        assert prices.shape == strikes.shape
        # Matches scalar evaluation elementwise.
        for K, p in zip(strikes, prices):
            assert p == pytest.approx(bs_price(100.0, float(K), 1.0, 0.05, 0.2))
        # Call price decreasing in strike.
        assert np.all(np.diff(prices) < 0)

    def test_broadcasting_2d(self):
        strikes = np.array([90.0, 100.0, 110.0])[None, :]
        mats = np.array([0.25, 1.0])[:, None]
        prices = bs_price(100.0, strikes, mats, 0.05, 0.2)
        assert prices.shape == (2, 3)
        # Longer maturity is worth more for calls (r > 0, q = 0).
        assert np.all(prices[1] > prices[0])

    def test_monotone_in_vol(self):
        sigmas = np.linspace(0.05, 1.0, 25)
        prices = bs_price(100.0, 100.0, 1.0, 0.05, sigmas)
        assert np.all(np.diff(prices) > 0)

    def test_deep_itm_call_approaches_forward_intrinsic(self):
        S, K, T, r = 100.0, 1.0, 1.0, 0.05
        price = bs_price(S, K, T, r, 0.2)
        assert price == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)

    def test_deep_otm_call_near_zero(self):
        assert bs_price(100.0, 10_000.0, 0.1, 0.05, 0.2) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_short_maturity_limit_is_intrinsic(self):
        # T -> 0: value converges to intrinsic. Away from the money the
        # convergence is exponentially fast; ATM the residual time value is
        # ~ 0.4 * S * sigma * sqrt(T).
        for K in (80.0, 120.0):
            price = bs_price(100.0, K, 1e-10, 0.05, 0.2)
            assert price == pytest.approx(max(100.0 - K, 0.0), abs=1e-8)
        atm = bs_price(100.0, 100.0, 1e-10, 0.05, 0.2)
        assert atm == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"S": -1.0}, {"S": 0.0}, {"K": 0.0}, {"T": 0.0},
            {"T": -0.5}, {"sigma": 0.0}, {"sigma": -0.2},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs):
        base = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        base.update(kwargs)
        with pytest.raises(ValueError):
            bs_price(**base)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            bs_price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="straddle")


class TestBsGreeks:
    def test_keys_present(self):
        greeks = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2)
        assert set(greeks) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_call_put_delta_relation(self):
        # delta_call - delta_put = exp(-qT).
        q = 0.02
        gc = bs_greeks(100.0, 95.0, 0.5, 0.03, 0.25, "call", q=q)
        gp = bs_greeks(100.0, 95.0, 0.5, 0.03, 0.25, "put", q=q)
        assert gc["delta"] - gp["delta"] == pytest.approx(
            np.exp(-q * 0.5), abs=1e-12
        )
        # Gamma and vega identical for calls and puts.
        assert gc["gamma"] == pytest.approx(gp["gamma"], abs=1e-12)
        assert gc["vega"] == pytest.approx(gp["vega"], abs=1e-12)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_greeks_match_finite_differences(self, option_type):
        S, K, T, r, sigma, q = 100.0, 105.0, 0.8, 0.04, 0.3, 0.01
        g = bs_greeks(S, K, T, r, sigma, option_type, q=q)
        h = 1e-4

        def price(**over):
            args = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
            args.update(over)
            return bs_price(option_type=option_type, **args)

        delta_fd = (price(S=S + h) - price(S=S - h)) / (2 * h)
        gamma_fd = (price(S=S + h) - 2 * price() + price(S=S - h)) / h**2
        vega_fd = (price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h)
        theta_fd = -(price(T=T + h) - price(T=T - h)) / (2 * h)
        rho_fd = (price(r=r + h) - price(r=r - h)) / (2 * h)

        assert g["delta"] == pytest.approx(delta_fd, abs=1e-6)
        assert g["gamma"] == pytest.approx(gamma_fd, abs=1e-4)
        assert g["vega"] == pytest.approx(vega_fd, abs=1e-4)
        assert g["theta"] == pytest.approx(theta_fd, abs=1e-4)
        assert g["rho"] == pytest.approx(rho_fd, abs=1e-4)

    def test_call_theta_negative_typical(self):
        assert bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2)["theta"] < 0.0


class TestImpliedVol:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("sigma_true", [0.08, 0.2, 0.55, 1.2])
    @pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
    def test_round_trip(self, option_type, sigma_true, K):
        S, T, r, q = 100.0, 0.75, 0.04, 0.015
        price = bs_price(S, K, T, r, sigma_true, option_type, q=q)
        iv = implied_vol(price, S, K, T, r, option_type, q=q)
        assert iv == pytest.approx(sigma_true, abs=1e-6)

    def test_round_trip_deep_itm_and_otm(self):
        # Low-vega corners: deep ITM and deep OTM at short maturity.
        S, T, r, sigma_true = 100.0, 0.1, 0.02, 0.3
        for K in (55.0, 160.0):
            price = bs_price(S, K, T, r, sigma_true)
            iv = implied_vol(price, S, K, T, r)
            assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_price_below_intrinsic_returns_nan(self):
        # Call below the no-arbitrage lower bound.
        assert np.isnan(implied_vol(1.0, 100.0, 50.0, 1.0, 0.05))

    def test_price_above_upper_bound_returns_nan(self):
        assert np.isnan(implied_vol(150.0, 100.0, 100.0, 1.0, 0.05))

    def test_negative_time_raises(self):
        with pytest.raises(ValueError):
            implied_vol(10.0, 100.0, 100.0, -1.0, 0.05)
