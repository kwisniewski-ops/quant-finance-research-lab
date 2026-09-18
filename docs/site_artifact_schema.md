# Lab site-artifact schema (`site-artifact/1`)

The contract between the Quantitative Markets & Institutions Lab repository
(`github.com/kwisniewski-ops/quant-finance-research-lab`, which computes) and
this site (which renders). One JSON file per investigation, written by the lab's
`scripts/export_site_artifacts.py` to `app/frontend/site-artifacts/`, vendored
by `scripts/sync-lab.mjs` into `src/content/lab/`. The site never restates a
number: everything a page shows comes from an artifact.

## Envelope

```json
{
  "schema_version": "site-artifact/1",
  "investigation": "var-backtest",
  "title": "Historical VaR under coverage and independence tests",
  "kicker": "Market risk · Forecast calibration",
  "result": "Breaches arrived too often and in clusters; the forecast failed its calibration tests.",
  "evidence_class": "Empirical finding",
  "provenance": {
    "source_commit": "4806df9…",
    "generated_at": "2026-09-17T23:00:00Z",
    "data": [
      { "file": "data/snapshots/prices.csv", "sha256": "…", "rows": 2892, "start": "2015-01-02", "end": "2026-08-29" }
    ],
    "universe": ["SPY", "QQQ", "…"],
    "notebook": "notebooks/06_market_risk_dashboard.ipynb",
    "tests": { "suite": "tests/test_risk.py", "count": 41, "passed": 41 }
  },
  "key_numbers": [ … ],
  "figures": [ … ],
  "tables": [ … ],
  "prose": { … },
  "math_notes": [ … ]
}
```

Rules: dates are ISO `YYYY-MM-DD`; numbers are numbers, not strings, except the
display `value` of a key number; series colours are the site's fixed
categorical slots `c1…c8`; prose is lifted verbatim from the lab's published
pages, never composed by the exporter; `null` is allowed for a missing value.

## key_numbers

```json
{ "id": "breaches", "value": "135", "label": "breaches at 95% VaR, 107 expected",
  "count": 135, "decimals": 0, "prefix": "", "suffix": "" }
```

`count` is the numeric target for the count-up animation; when it is absent the
site prints `value` as static text.

## figures

Every figure has `id`, `type`, `title`, `sub` (one line: unit · window · source),
`alt` (a sentence a screen reader can use), optional `unit` (`"%"`, `"$"`,
`"σ"`, …), optional `yMax`/`yMin`, optional `rules` and `annotations`, and an
optional `table` (`{ "columns": [], "rows": [[]] }`) that becomes the Table
toggle; when `table` is absent the site derives one from the series.

| `type` | Shape | Used for |
|---|---|---|
| `line` | `x: { label, values: [dates or numbers] }`, `series: [{ key, color, values: [numbers], band?: { lo: [], hi: [] }, dash?: true }]` | rolling betas with HAC bands, rolling vol, equity, drawdown, rolling correlation, SE by N, smile by strike |
| `columns` | `x: { label, values: [categories] }`, `series: [{ key, color, values }]`, `stacked?: bool` | in-sample vs holdout R², per-method Sharpe |
| `bars` | `items: [{ label, value, color, note?, strong? }]`, `max?` | ranked single values |
| `histogram` | `x: { label, values: [bin centres] }`, `series: [{ key, color, values: [counts] }]`, `overlay?: { key, color, values: [expected counts] }` | daily return distribution with matched Gaussian |
| `dots` | `x: { label, values: [categories] }`, `series: [{ key, color, values, band?: { lo, hi } }]`, `zero?: true` | alphas and betas with intervals; Sharpe in/out of sample |
| `scatter` | `points: [{ x, y, label?, color?, strong? }]`, `x: { label }`, `y: { label }` | the efficient frontier with named portfolios; price vs strike |
| `heatmap` | `grid: { rows: [labels], cols: [labels], values: [[numbers]] }`, `scale?: { min, max }` | the regime plane |

`rules`: `[{ "axis": "x" | "y", "at": <number or date>, "label": "VaR 95%", "color": "c5" }]`.
`annotations`: `[{ "x": <value>, "y": <value>, "text": "…" }]`.

## tables

```json
{ "id": "coverage-tests", "title": "Coverage and independence tests",
  "columns": ["Test", "Statistic", "p-value", "Verdict"],
  "rows": [["Kupiec", 6.91, 0.009, "Reject"], ["Christoffersen", 12.4, 0.0004, "Reject"]],
  "note": "…" }
```

## prose

```json
{ "decision_summary_html": "<p>…</p>",
  "sections": [ { "id": "method", "heading": "Method", "html": "<p>…</p>" } ],
  "limitations_html": "<p>…</p>",
  "sources": [ { "n": 1, "title": "…", "url": "…", "publisher": "…" } ] }
```

HTML is the lab page's own markup, sanitised to `p, h3, ul, ol, li, em, strong,
a, code, table, thead, tbody, tr, th, td, sup, sub, span, br`. Links to other
lab pages are clean `/lab/<slug>` paths.

## math_notes

`[{ "id": "heston", "heading": "…", "markdown": "…$…$…" }]` — Markdown with
KaTeX source. The site renders KaTeX at build time.

## ledger.json

```json
{ "schema_version": "site-artifact/1", "entries": [
  { "investigation": "var-backtest", "question": "…", "claim": "…", "assumptions": ["…"],
    "method": "…", "falsifier": "…", "result": "…", "failures": ["…"], "revisions": [{ "date": "…", "note": "…" }],
    "commit": "…", "notebook_url": "…", "updated": "YYYY-MM-DD" } ] }
```

## manifest.json

`{ "schema_version": "site-artifact/1", "generated_at": "…", "source_commit": "…", "artifacts": ["portfolio-estimation.json", …] }`
