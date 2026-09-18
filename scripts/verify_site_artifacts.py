"""Assert that the site artifacts reproduce every number the published pages state.

Each check names the artifact, the ``key_numbers`` entry, the value the page
prints, and the page that prints it. Numeric checks compare the artifact's
``count`` rounded to the page's printed precision; string checks compare the
display ``value``. Entries the exporter marked ``unverified`` are reported,
never silently passed.

Run from the repository root after ``python scripts/export_site_artifacts.py``::

    python scripts/verify_site_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "app" / "frontend" / "site-artifacts"

# (artifact, key id, expected, decimals or None for a string match, page)
CHECKS: list[tuple[str, str, object, int | None, str]] = [
    # finding-portfolio-estimation.html · README · research-log.html
    ("portfolio-estimation", "mvo-sharpe-fit", 1.12, 2, "finding-portfolio-estimation.html"),
    ("portfolio-estimation", "mvo-sharpe-eval", 0.44, 2, "finding-portfolio-estimation.html"),
    ("portfolio-estimation", "sharpe-gap", 0.68, 2, "finding-portfolio-estimation.html (figure)"),
    ("portfolio-estimation", "sharpe-sampling-error", 0.5, 1, "finding-portfolio-estimation.html"),
    ("portfolio-estimation", "robust-sharpe-eval", 0.6, 1, "finding-portfolio-estimation.html"),
    ("portfolio-estimation", "observations", 2891, 0, "finding-portfolio-estimation.html"),
    ("portfolio-estimation", "assets", 15, 0, "finding-portfolio-estimation.html"),
    ("portfolio-estimation", "sample-start", "2015-01-05", None, "research-log.html"),
    ("portfolio-estimation", "sample-end", "2026-07-06", None, "research-log.html"),
    # finding-var-backtest.html · risk.html · research-log.html
    ("var-backtest", "breaches", 135, 0, "finding-var-backtest.html"),
    ("var-backtest", "expected-breaches", 107, 0, "finding-var-backtest.html"),
    ("var-backtest", "breach-rate", 6.31, 2, "finding-var-backtest.html"),
    ("var-backtest", "forecast-days", 2140, 0, "finding-var-backtest.html"),
    ("var-backtest", "net-returns", 2640, 0, "finding-var-backtest.html"),
    ("var-backtest", "kupiec-p", 0.0075, 4, "finding-var-backtest.html"),
    ("var-backtest", "independence-p", 0.00025, 5, "finding-var-backtest.html"),
    ("var-backtest", "conditional-coverage-p", 0.000035, 6, "finding-var-backtest.html"),
    ("var-backtest", "var95", 1.079, 3, "finding-var-backtest.html"),
    ("var-backtest", "var95-ci-low", 0.930, 3, "finding-var-backtest.html"),
    ("var-backtest", "var95-ci-high", 1.219, 3, "finding-var-backtest.html"),
    ("var-backtest", "es95", 1.718, 3, "finding-var-backtest.html"),
    ("var-backtest", "es95-ci-low", 1.421, 3, "finding-var-backtest.html"),
    ("var-backtest", "es95-ci-high", 2.126, 3, "finding-var-backtest.html"),
    ("var-backtest", "n11", 20, 0, "finding-var-backtest.html (figure)"),
    ("var-backtest", "n11-expected", 8.5, 1, "finding-var-backtest.html (figure)"),
    ("var-backtest", "runs-single", 99, 0, "finding-var-backtest.html (figure caption)"),
    ("var-backtest", "runs-two", 12, 0, "finding-var-backtest.html (figure caption)"),
    ("var-backtest", "runs-three", 4, 0, "finding-var-backtest.html (figure caption)"),
    ("var-backtest", "annual-return", 10.14, 2, "risk.html"),
    ("var-backtest", "annual-vol", 11.50, 2, "risk.html"),
    ("var-backtest", "sharpe-rf3", 0.64, 2, "risk.html"),
    ("var-backtest", "max-drawdown", -24.3, 1, "risk.html"),
    ("var-backtest", "skewness", -0.54, 2, "risk.html"),
    ("var-backtest", "excess-kurtosis", 12.98, 2, "risk.html"),
    ("var-backtest", "ending-value", 2.750, 3, "risk.html"),
    ("var-backtest", "total-cost", 0.183, 3, "risk.html"),
    ("var-backtest", "backtest-start", "2016-01-04", None, "risk.html"),
    ("var-backtest", "backtest-end", "2026-07-06", None, "risk.html"),
    # finding-factor-etfs.html · factors.html · README
    ("factor-etfs", "mtum-momentum-beta", 0.34, 2, "finding-factor-etfs.html"),
    ("factor-etfs", "mtum-momentum-t", 49, 0, "finding-factor-etfs.html"),
    ("factor-etfs", "vlue-value-beta", 0.31, 2, "finding-factor-etfs.html"),
    ("factor-etfs", "vlue-value-t", 29, 0, "finding-factor-etfs.html"),
    ("factor-etfs", "qqq-alpha", 3.29, 2, "finding-factor-etfs.html"),
    ("factor-etfs", "qqq-alpha-p", 0.024, 3, "finding-factor-etfs.html"),
    ("factor-etfs", "qqq-alpha-q", 0.147, 3, "finding-factor-etfs.html"),
    ("factor-etfs", "alphas-surviving", 0, 0, "finding-factor-etfs.html"),
    ("factor-etfs", "observations", 2867, 0, "finding-factor-etfs.html"),
    ("factor-etfs", "sample-start", "2015-01-05", None, "finding-factor-etfs.html"),
    ("factor-etfs", "sample-end", "2026-05-29", None, "finding-factor-etfs.html"),
    ("factor-etfs", "holdout-decay-min", 0.4, 1, "finding-factor-etfs.html"),
    ("factor-etfs", "holdout-decay-max", 29.6, 1, "finding-factor-etfs.html"),
    ("factor-etfs", "holdout-decay-max-etf", "USMV", None, "finding-factor-etfs.html"),
    ("factor-etfs", "iwm-size-beta", 0.85, 2, "notebook 05"),
    ("factor-etfs", "usmv-market-beta", 0.74, 2, "notebook 05"),
    ("factor-etfs", "qqq-market-beta", 1.11, 2, "notebook 05"),
    ("factor-etfs", "qqq-hml-beta", -0.31, 2, "notebook 05"),
    # finding-volatility-smile.html · README · case-study.html
    ("volatility", "realised-vol-min", 3.4, 1, "README (finding page: ≈3%)"),
    ("volatility", "realised-vol-max", 93, 0, "finding-volatility-smile.html"),
    ("volatility", "excess-kurtosis", 14, 0, "finding-volatility-smile.html"),
    ("volatility", "worst-day-sigma", 10, 0, "finding-volatility-smile.html"),
    ("volatility", "abs-acf-lag1", 0.35, 2, "finding-volatility-smile.html"),
    ("volatility", "skew-low-strike", 25, 0, "finding-volatility-smile.html"),
    ("volatility", "skew-high-strike", 16, 0, "finding-volatility-smile.html"),
    ("volatility", "feller-lhs", 0.180, 3, "notebook 03"),
    ("volatility", "feller-rhs", 0.122, 3, "notebook 03"),
    ("volatility", "case-study-atm-iv", 19.2, 1, "case-study.html"),
    ("volatility", "worst-day", -10.9, 1, "notebook 03"),
    # finding-monte-carlo-error.html · research-log.html · notebook 04
    ("monte-carlo", "antithetic-se-ratio", 1.30, 2, "notebook 04 (page: ≈30%)"),
    ("monte-carlo", "antithetic-se-reduction", 30, 0, "finding-monte-carlo-error.html"),
    ("monte-carlo", "primary-seed", 42, 0, "finding-monte-carlo-error.html"),
    ("monte-carlo", "law-1e3", 1.00, 2, "finding-monte-carlo-error.html (figure)"),
    ("monte-carlo", "law-1e4", 0.32, 2, "finding-monte-carlo-error.html (figure)"),
    ("monte-carlo", "law-1e5", 0.10, 2, "finding-monte-carlo-error.html (figure)"),
    ("monte-carlo", "ou-kappa-recovered", 2.011, 3, "notebook 04"),
    ("monte-carlo", "regime-crisis-share", 14.0, 1, "notebook 04"),
    # research-log.html (pricing validation) · findings.html
    ("pricing-validation", "mc-paths", 200000, 0, "research-log.html"),
    ("pricing-validation", "seed", 42, 0, "research-log.html"),
    ("pricing-validation", "all-budgets-met", "pass", None, "findings.html"),
    ("pricing-validation", "iv-roundtrip-error", 1e-8, 8, "research-log.html (about 10⁻⁸)"),
    # finding-diversification-regimes.html · dossier · README
    ("stock-bond-regimes", "corr-2015-2019", -0.35, 2, "finding-diversification-regimes.html"),
    ("stock-bond-regimes", "corr-2022-2023-peak", 0.37, 2, "finding-diversification-regimes.html"),
    ("stock-bond-regimes", "corr-sample-end", 0.31, 2, "finding-diversification-regimes.html"),
    ("stock-bond-regimes", "vol-observed", 11.6, 1, "finding-diversification-regimes.html"),
    ("stock-bond-regimes", "vol-stressed", 15.8, 1, "finding-diversification-regimes.html"),
    ("stock-bond-regimes", "undiversified-vol", 15.8, 1, "finding-diversification-regimes.html"),
    ("stock-bond-regimes", "diversification-points", 4, 0, "finding-diversification-regimes.html"),
    ("stock-bond-regimes", "vol-half-stress", 13.86, 2, "notebook 06"),
    ("stock-bond-regimes", "corr-series-end", "2026-07-06", None, "stock-bond-regime-dossier.html"),
]

# Numbers that live in figures/tables rather than key_numbers.
SPECIAL_CHECKS = [
    ("pricing-validation", "tree/PDE error ≈ 10⁻³ (research-log.html)",
     lambda a: 1e-4 <= max(row[3] for row in _table(a, "cross-method")["rows"] if row[1] in ("Binomial(1000)", "Crank-Nicolson")) < 1e-2),
    ("pricing-validation", "Monte Carlo consistent within 3 SE (research-log.html)",
     lambda a: max(row[5] for row in _table(a, "cross-method")["rows"] if row[1] == "Monte Carlo") < 3.0),
    ("volatility", "case-study.js Heston calls reproduced to 4 dp",
     lambda a: all(abs(row[1] - typed) <= 5.5e-5 for row, typed in zip(_figure(a, "bs-vs-heston-calls")["table"]["rows"],
                   [30.9860, 26.2467, 21.6299, 17.1919, 13.0108, 9.1946, 5.8920, 3.2897, 1.5472, 0.6244, 0.2338, 0.0869, 0.0330]))),
    ("volatility", "case-study.js Heston implied vols reproduced to 4 dp",
     lambda a: all(abs(row[2] - typed) <= 5.5e-5 for row, typed in zip(_figure(a, "bs-vs-heston-calls")["table"]["rows"],
                   [0.2980, 0.2808, 0.2635, 0.2459, 0.2281, 0.2099, 0.1918, 0.1747, 0.1605, 0.1514, 0.1476, 0.1474, 0.1490]))),
    ("var-backtest", "stress windows (risk.html): -10.2%/-21.3%/-24.2% cumulative, 16/9/26 breaches",
     lambda a: [(round(100 * r[3], 1), r[6]) for r in _table(a, "stress-windows")["rows"]] == [(-10.2, 16), (-21.3, 9), (-24.2, 26)]),
    ("var-backtest", "coverage statistics (risk.html): 7.148 / 13.392 / 20.540",
     lambda a: [round(r[1], 3) for r in _table(a, "coverage-tests")["rows"]] == [7.148, 13.392, 20.540]),
    ("factor-etfs", "holdout R² (factors.html): USMV 89.6% → 60.0%",
     lambda a: [round(100 * x, 1) for x in next(r for r in _table(a, "holdout")["rows"] if r[0] == "USMV")[1:3]] == [89.6, 60.0]),
    ("stock-bond-regimes", "full daily correlation series (≥ 2,700 points)",
     lambda a: len(_figure(a, "spy-tlt-correlation")["series"][0]["values"]) >= 2700),
    ("stock-bond-regimes", "six atlas episodes", lambda a: len(_table(a, "regime-episodes")["rows"]) == 6),
    ("monte-carlo", "fitted log–log RMSE slope within noise of −0.5 (finding-monte-carlo-error.html)",
     lambda a: (lambda k: not k.get("unverified") and abs(k["count"] + 0.5) <= 2 * k["noise_se"])(next(k for k in a["key_numbers"] if k["id"] == "fitted-slope"))),
]


def _table(artifact: dict, id_: str) -> dict:
    return next(t for t in artifact["tables"] if t["id"] == id_)


def _figure(artifact: dict, id_: str) -> dict:
    return next(f for f in artifact["figures"] if f["id"] == id_)


def _round(value: float, decimals: int) -> float:
    """Round the way the pages and notebooks print (Python ``format``), so 0.1225 -> 0.122."""
    return float(f"{value:.{decimals}f}")


def main() -> int:
    artifacts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in ARTIFACTS.glob("*.json") if p.stem not in {"ledger", "manifest"}}
    failures, unverified, passed = [], [], 0
    for slug, key, expected, decimals, page in CHECKS:
        entry = next((k for k in artifacts[slug]["key_numbers"] if k["id"] == key), None)
        if entry is None:
            failures.append(f"{slug}:{key} missing")
            continue
        if entry.get("unverified"):
            unverified.append(f"{slug}:{key} page states {expected!r} ({page}); artifact carries {entry['value']} — {entry.get('note', '')}")
            continue
        if decimals is None:
            ok = entry["value"] == expected
            got = entry["value"]
        else:
            got = _round(entry["count"], decimals)
            ok = got == _round(float(expected), decimals)
        if ok:
            passed += 1
        else:
            failures.append(f"{slug}:{key} expected {expected!r} ({page}) got {got!r} (unrounded {entry.get('count')!r})")
    for slug, label, check in SPECIAL_CHECKS:
        try:
            ok = bool(check(artifacts[slug]))
        except Exception as exc:  # noqa: BLE001 — report, do not hide
            ok, label = False, f"{label} [{exc!r}]"
        if ok:
            passed += 1
        else:
            failures.append(f"{slug}: {label}")
    print(f"verified {passed} page-stated numbers; {len(unverified)} unverified; {len(failures)} failures")
    for line in unverified:
        print("  UNVERIFIED", line)
    for line in failures:
        print("  FAIL", line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
