"""Tests for src.data: validation report, cache-first loaders, returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.data_validation import ValidationReport, validate_prices
from src.data.market_data_loader import load_prices, to_returns


def make_clean_prices(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(0.0003, 0.01, size=(n, 3))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=idx, columns=["SPY", "AGG", "GLD"])


# ---------------------------------------------------------------------------
# validate_prices
# ---------------------------------------------------------------------------
class TestValidation:
    def test_clean_data_passes(self):
        report = validate_prices(make_clean_prices())
        assert isinstance(report, ValidationReport)
        assert report.passed
        assert report.issues == []
        assert report.index_monotonic
        assert report.n_rows == 500
        assert (report.n_missing == 0).all()
        assert (report.outliers == 0).all()

    def test_catches_injected_nan_block(self):
        prices = make_clean_prices()
        prices.loc[prices.index[100:160], "AGG"] = np.nan  # 12% missing
        report = validate_prices(prices, max_missing_frac=0.05)
        assert not report.passed
        assert report.n_missing["AGG"] == 60
        assert any("AGG" in msg and "missing" in msg for msg in report.issues)

    def test_catches_stale_run(self):
        prices = make_clean_prices()
        prices.loc[prices.index[200:215], "GLD"] = prices.iloc[200]["GLD"]  # 15 frozen days
        report = validate_prices(prices, stale_len=10)
        assert not report.passed
        assert report.stale_runs["GLD"] >= 15
        assert any("GLD" in msg and "stale" in msg for msg in report.issues)

    def test_catches_10_sigma_outlier(self):
        prices = make_clean_prices()
        # Inject a one-day ~12-sigma crash (daily sigma ~1%).
        prices.loc[prices.index[300]:, "SPY"] *= 0.88
        report = validate_prices(prices, outlier_z=8.0)
        assert not report.passed
        assert report.outliers["SPY"] >= 1
        assert any("SPY" in msg and "sigma" in msg for msg in report.issues)

    def test_catches_unsorted_index(self):
        prices = make_clean_prices().iloc[::-1]
        report = validate_prices(prices)
        assert not report.passed
        assert not report.index_monotonic

    def test_parameter_validation(self):
        prices = make_clean_prices(50)
        with pytest.raises(ValueError, match="max_missing_frac"):
            validate_prices(prices, max_missing_frac=2.0)
        with pytest.raises(ValueError, match="stale_len"):
            validate_prices(prices, stale_len=1)
        with pytest.raises(ValueError, match="empty"):
            validate_prices(prices.iloc[0:0])


# ---------------------------------------------------------------------------
# market_data_loader (offline, cache-first behavior)
# ---------------------------------------------------------------------------
class TestLoader:
    def test_reads_cache_without_network(self, tmp_path):
        cache = make_clean_prices()
        (tmp_path / "prices.csv").write_text(cache.to_csv())
        out = load_prices(["SPY", "GLD"], start="2020-06-01", cache_dir=str(tmp_path))
        assert list(out.columns) == ["SPY", "GLD"]
        assert out.index.min() >= pd.Timestamp("2020-06-01")
        assert isinstance(out.index, pd.DatetimeIndex)
        assert (out.dtypes == float).all()

    def test_end_bound_applied(self, tmp_path):
        cache = make_clean_prices()
        (tmp_path / "prices.csv").write_text(cache.to_csv())
        out = load_prices(["SPY"], start="2020-02-01", end="2020-03-31", cache_dir=str(tmp_path))
        assert out.index.max() <= pd.Timestamp("2020-03-31")

    def test_missing_cache_raises_with_instructions(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--refresh"):
            load_prices(["SPY"], cache_dir=str(tmp_path / "nowhere"))

    def test_missing_ticker_raises_keyerror(self, tmp_path):
        (tmp_path / "prices.csv").write_text(make_clean_prices().to_csv())
        with pytest.raises(KeyError, match="ZZZT"):
            load_prices(["SPY", "ZZZT"], cache_dir=str(tmp_path))

    def test_synthetic_fallback_warns(self, tmp_path, caplog):
        (tmp_path / "prices_synthetic_demo.csv").write_text(make_clean_prices().to_csv())
        with caplog.at_level("WARNING", logger="src.data.market_data_loader"):
            out = load_prices(["SPY"], cache_dir=str(tmp_path))
        assert not out.empty
        assert any("SYNTHETIC" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# to_returns
# ---------------------------------------------------------------------------
class TestToReturns:
    def test_simple_and_log_returns(self):
        prices = make_clean_prices(50)
        simple = to_returns(prices)
        logr = to_returns(prices, log=True)
        assert len(simple) == len(prices) - 1
        assert np.allclose(np.log1p(simple.values), logr.values, atol=1e-12)
        # Hand check on the first observation.
        assert np.isclose(
            simple.iloc[0, 0], prices.iloc[1, 0] / prices.iloc[0, 0] - 1, atol=1e-12
        )

    def test_nonpositive_prices_raise(self):
        prices = make_clean_prices(20)
        prices.iloc[3, 1] = -5.0
        with pytest.raises(ValueError, match="positive"):
            to_returns(prices)
