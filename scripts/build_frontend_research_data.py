"""Build deterministic, browser-ready empirical research snapshots.

This script never fetches the network. It reads the repository's frozen price
and factor CSVs, runs the estimators used by the public Factor and Risk pages,
and writes compact JSON artifacts beside the frontend.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from src.backtesting.engine import Backtester
from src.backtesting.transaction_costs import ProportionalCost


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "snapshots"
OUT = ROOT / "app" / "frontend" / "data"
SOURCE_COMMIT = "ad24c49995832f84ebeceb2f8243011463591a4a"
BUILD_DATE = "2026-08-03"
FACTOR_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
REPO = "https://github.com/kwisniewski-ops/quant-finance-research-lab"
FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
ETF_NAMES = ["QQQ", "USMV", "MTUM", "VLUE", "QUAL", "IWM"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_number(value: float, digits: int = 8) -> float | None:
    value = float(value)
    return None if not math.isfinite(value) else round(value, digits)


def write_json(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def factor_study() -> None:
    factor_path = DATA / "ff_factors.csv"
    price_path = DATA / "prices.csv"
    factors = pd.read_csv(factor_path, index_col=0, parse_dates=True).sort_index()
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    returns = prices[ETF_NAMES].pct_change(fill_method=None)
    aligned = returns.join(factors, how="inner").dropna()

    y_by_etf: dict[str, pd.Series] = {}
    regressions: list[dict] = []
    alpha_pvalues: list[float] = []
    fitted: dict[str, object] = {}
    X = sm.add_constant(aligned[FACTOR_NAMES], has_constant="add")

    for ticker in ETF_NAMES:
        y = aligned[ticker] - aligned["RF"]
        y_by_etf[ticker] = y
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        fitted[ticker] = model
        alpha_pvalues.append(float(model.pvalues["const"]))
        exposures = []
        ci = model.conf_int(alpha=0.05)
        for factor in FACTOR_NAMES:
            exposures.append(
                {
                    "factor": factor,
                    "beta": clean_number(model.params[factor]),
                    "ci_low": clean_number(ci.loc[factor, 0]),
                    "ci_high": clean_number(ci.loc[factor, 1]),
                    "t_hac": clean_number(model.tvalues[factor], 4),
                }
            )
        regressions.append(
            {
                "ticker": ticker,
                "alpha_annualized": clean_number(model.params["const"] * 252),
                "alpha_p_raw": clean_number(model.pvalues["const"]),
                "r_squared": clean_number(model.rsquared),
                "n": int(model.nobs),
                "exposures": exposures,
            }
        )

    holm = multipletests(alpha_pvalues, method="holm")[1]
    bh = multipletests(alpha_pvalues, method="fdr_bh")[1]
    for row, holm_p, bh_q in zip(regressions, holm, bh, strict=True):
        row["alpha_p_holm"] = clean_number(holm_p)
        row["alpha_q_bh"] = clean_number(bh_q)
        row["alpha_survives_5pct_bh"] = bool(bh_q <= 0.05)

    # Rolling one-year QQQ exposures, sampled at month-end for a compact payload.
    rolling_rows = []
    qqq = y_by_etf["QQQ"]
    window = 252
    for end in range(window, len(aligned) + 1):
        if end < len(aligned) and aligned.index[end - 1].month == aligned.index[end].month:
            continue
        sl = slice(end - window, end)
        roll_model = sm.OLS(qqq.iloc[sl], X.iloc[sl]).fit()
        rolling_rows.append(
            {
                "date": aligned.index[end - 1].date().isoformat(),
                **{f: clean_number(roll_model.params[f]) for f in ["Mkt-RF", "HML", "Mom"]},
            }
        )

    qqq_full = fitted["QQQ"]
    split = len(aligned) // 2
    oos_rows = []
    for ticker in ETF_NAMES:
        y = y_by_etf[ticker]
        train_y, test_y = y.iloc[:split], y.iloc[split:]
        train_X, test_X = X.iloc[:split], X.iloc[split:]
        train_model = sm.OLS(train_y, train_X).fit()
        train_pred = train_model.predict(train_X)
        test_pred = train_model.predict(test_X)
        train_r2 = 1 - float(((train_y - train_pred) ** 2).sum()) / float(((train_y - train_y.mean()) ** 2).sum())
        test_r2 = 1 - float(((test_y - test_pred) ** 2).sum()) / float(((test_y - test_y.mean()) ** 2).sum())
        oos_rows.append(
            {
                "ticker": ticker,
                "train_start": train_y.index[0].date().isoformat(),
                "train_end": train_y.index[-1].date().isoformat(),
                "test_start": test_y.index[0].date().isoformat(),
                "test_end": test_y.index[-1].date().isoformat(),
                "in_sample_r2": clean_number(train_r2),
                "out_of_sample_r2": clean_number(test_r2),
                "decay": clean_number(test_r2 - train_r2),
                "test_rmse_daily": clean_number(np.sqrt(np.mean((test_y - test_pred) ** 2))),
            }
        )

    payload = {
        "schema_version": 1,
        "built_on": BUILD_DATE,
        "method": {
            "regression": "Daily ETF excess returns on FF5 plus momentum; OLS coefficients with Newey-West HAC standard errors (5 lags).",
            "rolling": "Trailing 252-observation QQQ regression, sampled at each month end.",
            "out_of_sample": "Chronological 50/50 split; training coefficients frozen before evaluating predictive R-squared on the second half.",
            "multiple_testing": "Six ETF alpha p-values adjusted by Holm family-wise control and Benjamini-Hochberg false-discovery control.",
        },
        "provenance": {
            "factor_source": FACTOR_URL,
            "factor_file": "data/snapshots/ff_factors.csv",
            "factor_sha256": sha256(factor_path),
            "factor_snapshot_dates": [factors.index[0].date().isoformat(), factors.index[-1].date().isoformat()],
            "price_file": "data/snapshots/prices.csv",
            "price_sha256": sha256(price_path),
            "price_snapshot_dates": [prices.index[0].date().isoformat(), prices.index[-1].date().isoformat()],
            "aligned_sample_dates": [aligned.index[0].date().isoformat(), aligned.index[-1].date().isoformat()],
            "source_commit": SOURCE_COMMIT,
            "notebook_url": f"{REPO}/blob/{SOURCE_COMMIT}/notebooks/05_factor_investing.ipynb",
        },
        "regressions": regressions,
        "rolling_qqq": {
            "full_sample": {f: clean_number(qqq_full.params[f]) for f in ["Mkt-RF", "HML", "Mom"]},
            "rows": rolling_rows,
        },
        "out_of_sample": oos_rows,
    }
    write_json("factor-study.json", payload)


def _safe_log_likelihood(counts: list[int], probs: list[float]) -> float:
    total = 0.0
    for count, prob in zip(counts, probs, strict=True):
        if count:
            total += count * math.log(min(max(prob, 1e-12), 1 - 1e-12))
    return total


def coverage_tests(breaches: np.ndarray, expected_probability: float) -> dict:
    b = breaches.astype(int)
    n, x = len(b), int(b.sum())
    observed = x / n
    ll_null = _safe_log_likelihood([x, n - x], [expected_probability, 1 - expected_probability])
    ll_alt = _safe_log_likelihood([x, n - x], [observed, 1 - observed])
    lr_uc = max(0.0, 2 * (ll_alt - ll_null))

    prev, nxt = b[:-1], b[1:]
    n00 = int(((prev == 0) & (nxt == 0)).sum())
    n01 = int(((prev == 0) & (nxt == 1)).sum())
    n10 = int(((prev == 1) & (nxt == 0)).sum())
    n11 = int(((prev == 1) & (nxt == 1)).sum())
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    ll_ind = _safe_log_likelihood([n01 + n11, n00 + n10], [p, 1 - p])
    ll_markov = _safe_log_likelihood([n01, n00, n11, n10], [p01, 1 - p01, p11, 1 - p11])
    lr_ind = max(0.0, 2 * (ll_markov - ll_ind))
    lr_cc = lr_uc + lr_ind
    return {
        "forecast_days": n,
        "breaches": x,
        "expected_breaches": clean_number(n * expected_probability, 2),
        "breach_rate": clean_number(observed),
        "kupiec_lr": clean_number(lr_uc, 5),
        "kupiec_p": clean_number(chi2.sf(lr_uc, 1)),
        "christoffersen_independence_lr": clean_number(lr_ind, 5),
        "christoffersen_independence_p": clean_number(chi2.sf(lr_ind, 1)),
        "christoffersen_conditional_coverage_lr": clean_number(lr_cc, 5),
        "christoffersen_conditional_coverage_p": clean_number(chi2.sf(lr_cc, 2)),
        "transition_counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    }


def risk_study() -> None:
    price_path = DATA / "prices.csv"
    tickers = ["SPY", "QQQ", "IWM", "EFA", "AGG", "TLT", "GLD", "VNQ"]
    target = pd.Series(
        {"SPY": 0.30, "QQQ": 0.10, "IWM": 0.05, "EFA": 0.10,
         "AGG": 0.20, "TLT": 0.10, "GLD": 0.075, "VNQ": 0.075}
    )
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()[tickers]
    engine = Backtester(cost_model=ProportionalCost(bps=10.0), rebalance="M")
    result = engine.run(prices, lambda _window: target, lookback=252)
    returns = result.returns
    equity = result.equity_curve
    drawdown = equity / equity.cummax() - 1
    roll_vol = returns.rolling(63).std(ddof=1) * math.sqrt(252)

    sorted_returns = np.sort(returns.to_numpy())
    threshold = float(np.quantile(sorted_returns, 0.05))
    var95 = -threshold
    es95 = -float(sorted_returns[sorted_returns <= threshold].mean())

    rng = np.random.default_rng(20260803)
    block, reps, n = 21, 2000, len(returns)
    n_blocks = math.ceil(n / block)
    starts = np.arange(0, n - block + 1)
    boot_var = np.empty(reps)
    boot_es = np.empty(reps)
    source = returns.to_numpy()
    for i in range(reps):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([source[s : s + block] for s in chosen])[:n]
        q = float(np.quantile(sample, 0.05))
        boot_var[i] = -q
        boot_es[i] = -float(sample[sample <= q].mean())

    window = 500
    forecasts = []
    breach_values = []
    for i in range(window, len(returns)):
        history = returns.iloc[i - window : i].to_numpy()
        q = float(np.quantile(history, 0.05))
        breached = bool(returns.iloc[i] < q)
        breach_values.append(breached)
        forecasts.append(
            {
                "date": returns.index[i].date().isoformat(),
                "var95": clean_number(-q),
                "return": clean_number(returns.iloc[i]),
                "breach": breached,
            }
        )
    tests = coverage_tests(np.array(breach_values), 0.05)

    stress_defs = [
        ("Volatility shock of Q4 2018", "2018-09-20", "2018-12-24"),
        ("COVID-19 selloff", "2020-02-19", "2020-03-23"),
        ("2022 inflation and rate shock", "2022-01-03", "2022-10-14"),
    ]
    stresses = []
    forecast_by_date = {row["date"]: row for row in forecasts}
    for label, start, end in stress_defs:
        r = returns.loc[start:end]
        e = (1 + r).cumprod()
        dd = e / e.cummax() - 1
        stress_breaches = sum(
            bool(forecast_by_date.get(d.date().isoformat(), {}).get("breach")) for d in r.index
        )
        stresses.append(
            {
                "label": label,
                "start": r.index[0].date().isoformat(),
                "end": r.index[-1].date().isoformat(),
                "cumulative_return": clean_number(e.iloc[-1] - 1),
                "max_drawdown": clean_number(dd.min()),
                "worst_day": clean_number(r.min()),
                "var_breaches": int(stress_breaches),
            }
        )

    mean_d = float(returns.mean())
    sd_d = float(returns.std(ddof=1))
    annual_return = float(equity.iloc[-1] ** (252 / len(equity)) - 1)
    annual_vol = sd_d * math.sqrt(252)
    histogram_counts, histogram_edges = np.histogram(returns.to_numpy() * 100, bins=56)
    histogram_centers = ((histogram_edges[:-1] + histogram_edges[1:]) / 2).tolist()
    normal_counts = [
        len(returns) * (histogram_edges[1] - histogram_edges[0]) / 100
        * math.exp(-0.5 * (((x / 100) - mean_d) / sd_d) ** 2)
        / (sd_d * math.sqrt(2 * math.pi))
        for x in histogram_centers
    ]

    sample_idx = list(range(0, len(returns), 5))
    if sample_idx[-1] != len(returns) - 1:
        sample_idx.append(len(returns) - 1)
    chart_dates = [returns.index[i].date().isoformat() for i in sample_idx]
    payload = {
        "schema_version": 1,
        "built_on": BUILD_DATE,
        "method": {
            "portfolio": "Monthly rebalance to fixed multi-asset target weights; 252-day warm-up; 10bp proportional cost on one-sided turnover.",
            "tail": "One-day historical 95% VaR and Expected Shortfall from net daily portfolio returns.",
            "uncertainty": "Moving-block bootstrap, 21-day blocks, 2,000 replications, seed 20260803; percentile 95% intervals.",
            "backtest": "Rolling 500-observation historical VaR forecast; Kupiec unconditional coverage and Christoffersen independence/conditional-coverage likelihood-ratio tests.",
        },
        "provenance": {
            "price_file": "data/snapshots/prices.csv",
            "price_sha256": sha256(price_path),
            "price_snapshot_dates": [prices.index[0].date().isoformat(), prices.index[-1].date().isoformat()],
            "backtest_dates": [returns.index[0].date().isoformat(), returns.index[-1].date().isoformat()],
            "source_commit": SOURCE_COMMIT,
            "notebook_url": f"{REPO}/blob/{SOURCE_COMMIT}/notebooks/06_market_risk_dashboard.ipynb",
            "engine_url": f"{REPO}/blob/{SOURCE_COMMIT}/src/backtesting/engine.py",
        },
        "weights": {key: clean_number(value) for key, value in target.items()},
        "metrics": {
            "n": int(len(returns)),
            "annual_return": clean_number(annual_return),
            "annual_volatility": clean_number(annual_vol),
            "sharpe_rf_3pct": clean_number((mean_d * 252 - 0.03) / annual_vol),
            "max_drawdown": clean_number(drawdown.min()),
            "var95": clean_number(var95),
            "es95": clean_number(es95),
            "var95_ci": [clean_number(x) for x in np.quantile(boot_var, [0.025, 0.975])],
            "es95_ci": [clean_number(x) for x in np.quantile(boot_es, [0.025, 0.975])],
            "skewness": clean_number(returns.skew()),
            "excess_kurtosis": clean_number(returns.kurt()),
            "total_cost": clean_number(result.costs.sum()),
        },
        "coverage_tests": tests,
        "stress_windows": stresses,
        "charts": {
            "dates": chart_dates,
            "equity": [clean_number(equity.iloc[i]) for i in sample_idx],
            "drawdown": [clean_number(drawdown.iloc[i]) for i in sample_idx],
            "rolling_volatility": [clean_number(roll_vol.iloc[i]) for i in sample_idx],
            "histogram_centers_pct": [clean_number(x, 5) for x in histogram_centers],
            "histogram_counts": histogram_counts.astype(int).tolist(),
            "normal_counts": [clean_number(x, 5) for x in normal_counts],
        },
    }
    write_json("risk-study.json", payload)


if __name__ == "__main__":
    factor_study()
    risk_study()
    print("Wrote app/frontend/data/factor-study.json and risk-study.json")
