"""Render default empirical research tables into the static HTML pages.

The browser scripts still enhance these regions with charts and selectors, but the
default evidence remains readable, accessible, and indexable without JavaScript.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "app" / "frontend"
FACTOR_LABELS = {
    "Mkt-RF": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "Mom": "Momentum",
}


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f'<th scope="col">{cell}</th>' for cell in headers)
    body = []
    for row in rows:
        cells = [f'<th scope="row">{row[0]}</th>']
        cells.extend(f"<td>{cell}</td>" for cell in row[1:])
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><caption>{caption}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def definition_grid(rows: list[tuple[str, str]], item_class: str = "") -> str:
    class_attr = f' class="{item_class}"' if item_class else ""
    return "".join(
        f"<div{class_attr}><dt>{label}</dt><dd>{value}</dd></div>"
        for label, value in rows
    )


def replace_region(page: Path, key: str, markup: str) -> None:
    start = f"<!-- research-data:{key}:start -->"
    end = f"<!-- research-data:{key}:end -->"
    source = page.read_text(encoding="utf-8")
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f"Expected one prerender region {key!r} in {page}")
    prefix, rest = source.split(start, 1)
    _, suffix = rest.split(end, 1)
    page.write_text(f"{prefix}{start}{markup}{end}{suffix}", encoding="utf-8")


def render_risk() -> None:
    data = json.loads((FRONTEND / "data" / "risk-study.json").read_text(encoding="utf-8"))
    page = FRONTEND / "risk.html"
    provenance = data["provenance"]
    metrics = data["metrics"]
    tests = data["coverage_tests"]
    charts = data["charts"]

    replace_region(
        page,
        "risk-provenance",
        definition_grid(
            [
                ("Backtest sample", f'{provenance["backtest_dates"][0]} – {provenance["backtest_dates"][1]}'),
                ("Price vintage", provenance["price_snapshot_dates"][1]),
                ("Snapshot hash", f'<code>sha256:{provenance["price_sha256"][:16]}…</code>'),
                (
                    "Pinned source",
                    f'<a href="{provenance["notebook_url"]}"><code>{provenance["source_commit"][:12]}</code></a>',
                ),
                ("Daily observations", f'{metrics["n"]:,}'),
                ("Total modeled costs", f'{pct(metrics["total_cost"], 3)} of NAV'),
            ]
        ),
    )

    risk_stats = [
        ("Ann. return", pct(metrics["annual_return"], 2)),
        ("Ann. volatility", pct(metrics["annual_volatility"], 2)),
        ("Sharpe (rf 3%)", num(metrics["sharpe_rf_3pct"], 2)),
        ("Max drawdown", pct(metrics["max_drawdown"], 1)),
        ("VaR 95 (1d)", pct(metrics["var95"], 2)),
        ("ES 95 (1d)", pct(metrics["es95"], 2)),
        ("Skewness", num(metrics["skewness"], 2)),
        ("Excess kurtosis", num(metrics["excess_kurtosis"], 2)),
    ]
    replace_region(
        page,
        "risk-stats",
        "".join(
            '<div class="stat"><dt class="k">{}</dt><dd class="v{}">{}</dd></div>'.format(
                label, " accent" if index in (3, 4, 5) else "", value
            )
            for index, (label, value) in enumerate(risk_stats)
        ),
    )
    replace_region(
        page,
        "risk-data-table",
        table(
            ["Measure", "Value"],
            [
                ["Ending value of $1", f'${num(charts["equity"][-1], 3)}'],
                ["Maximum drawdown", pct(metrics["max_drawdown"], 2)],
                ["VaR 95, one day", pct(metrics["var95"], 2)],
                ["Expected Shortfall 95, one day", pct(metrics["es95"], 2)],
                ["Backtest observations", f'{metrics["n"]:,}'],
            ],
            "Key values represented in the empirical risk charts",
        ),
    )
    replace_region(
        page,
        "risk-uncertainty-table",
        table(
            ["Measure", "Point estimate", "Bootstrap 95% interval"],
            [
                [
                    "Historical VaR 95",
                    pct(metrics["var95"], 3),
                    f'{pct(metrics["var95_ci"][0], 3)} – {pct(metrics["var95_ci"][1], 3)}',
                ],
                [
                    "Historical ES 95",
                    pct(metrics["es95"], 3),
                    f'{pct(metrics["es95_ci"][0], 3)} – {pct(metrics["es95_ci"][1], 3)}',
                ],
            ],
            "Moving-block bootstrap uncertainty; 2,000 replications, 21-day blocks",
        ),
    )

    def decision(p_value: float) -> str:
        return '<span class="result-reject">Reject at 5%</span>' if p_value < 0.05 else "Do not reject at 5%"

    replace_region(
        page,
        "risk-coverage-table",
        table(
            ["Test", "Statistic", "p-value", "Decision"],
            [
                [
                    "Observed breaches",
                    f'{tests["breaches"]} / {tests["forecast_days"]}',
                    f'{pct(tests["breach_rate"], 2)} rate',
                    f'Expected {num(tests["expected_breaches"], 0)}',
                ],
                ["Kupiec coverage", num(tests["kupiec_lr"], 3), num(tests["kupiec_p"], 5), decision(tests["kupiec_p"])],
                [
                    "Christoffersen independence",
                    num(tests["christoffersen_independence_lr"], 3),
                    num(tests["christoffersen_independence_p"], 5),
                    decision(tests["christoffersen_independence_p"]),
                ],
                [
                    "Conditional coverage",
                    num(tests["christoffersen_conditional_coverage_lr"], 3),
                    num(tests["christoffersen_conditional_coverage_p"], 5),
                    decision(tests["christoffersen_conditional_coverage_p"]),
                ],
            ],
            "Rolling 500-observation historical VaR forecast tests",
        ),
    )
    replace_region(
        page,
        "risk-stress-table",
        table(
            ["Window", "Dates", "Cumulative return", "Max drawdown", "Worst day", "VaR breaches"],
            [
                [
                    row["label"],
                    f'{row["start"]} – {row["end"]}',
                    pct(row["cumulative_return"], 1),
                    pct(row["max_drawdown"], 1),
                    pct(row["worst_day"], 2),
                    str(row["var_breaches"]),
                ]
                for row in data["stress_windows"]
            ],
            "Fixed historical stress windows within the frozen sample",
        ),
    )


def render_factors() -> None:
    data = json.loads((FRONTEND / "data" / "factor-study.json").read_text(encoding="utf-8"))
    page = FRONTEND / "factors.html"
    provenance = data["provenance"]
    default = next(row for row in data["regressions"] if row["ticker"] == "QQQ")

    replace_region(
        page,
        "factor-provenance",
        definition_grid(
            [
                ("Aligned sample", f'{provenance["aligned_sample_dates"][0]} – {provenance["aligned_sample_dates"][1]}'),
                ("Factor vintage", f'{provenance["factor_snapshot_dates"][1]} · sha256:{provenance["factor_sha256"][:12]}…'),
                ("Price vintage", f'{provenance["price_snapshot_dates"][1]} · sha256:{provenance["price_sha256"][:12]}…'),
                (
                    "Pinned source",
                    f'<a href="{provenance["notebook_url"]}"><code>{provenance["source_commit"][:12]}</code></a>',
                ),
                ("Observations", f'{default["n"]:,} daily rows'),
                ("Estimator", "OLS · Newey–West HAC(5) intervals"),
            ]
        ),
    )
    replace_region(
        page,
        "factor-selector",
        "".join(
            f'<option value="{row["ticker"]}"{" selected" if row["ticker"] == "QQQ" else ""}>{row["ticker"]}</option>'
            for row in data["regressions"]
        ),
    )
    replace_region(
        page,
        "factor-regression-table",
        table(
            ["Factor", "Beta", "95% CI", "HAC t"],
            [
                [
                    FACTOR_LABELS[row["factor"]],
                    num(row["beta"], 3),
                    f'[{num(row["ci_low"], 3)}, {num(row["ci_high"], 3)}]',
                    num(row["t_hac"], 2),
                ]
                for row in default["exposures"]
            ],
            f'QQQ factor estimates; alpha {pct(default["alpha_annualized"], 2)} annualized; R² {pct(default["r_squared"], 1)}',
        ),
    )
    replace_region(
        page,
        "factor-oos-table",
        table(
            ["ETF", "Training R²", "Holdout R²", "Change", "Holdout dates"],
            [
                [
                    row["ticker"],
                    pct(row["in_sample_r2"], 1),
                    pct(row["out_of_sample_r2"], 1),
                    pct(row["decay"], 1),
                    f'{row["test_start"]} – {row["test_end"]}',
                ]
                for row in data["out_of_sample"]
            ],
            "Chronological holdout results; coefficients fitted only on the first half",
        ),
    )
    replace_region(
        page,
        "factor-multiple-table",
        table(
            ["ETF", "Annualized alpha", "Raw p", "Holm adjusted p", "BH q", "BH 5% decision"],
            [
                [
                    row["ticker"],
                    pct(row["alpha_annualized"], 2),
                    num(row["alpha_p_raw"], 4),
                    num(row["alpha_p_holm"], 4),
                    num(row["alpha_q_bh"], 4),
                    "Survives" if row["alpha_survives_5pct_bh"] else "Does not survive",
                ]
                for row in data["regressions"]
            ],
            "ETF alpha tests with family-wise and false-discovery corrections",
        ),
    )


def prerender_frontend() -> None:
    render_risk()
    render_factors()


if __name__ == "__main__":
    prerender_frontend()
    print("Prerendered default evidence in risk.html and factors.html")
