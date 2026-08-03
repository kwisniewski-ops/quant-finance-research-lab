/* Empirical portfolio risk dashboard, rendered from a frozen JSON artifact. */
(function () {
  "use strict";
  function table(headers, rows, caption) {
    var html = "<table><caption>" + caption + "</caption><thead><tr>";
    headers.forEach(function (h) { html += '<th scope="col">' + h + "</th>"; });
    html += "</tr></thead><tbody>";
    rows.forEach(function (row) { html += "<tr>" + row.map(function (cell, i) { return (i ? "<td>" : '<th scope="row">') + cell + (i ? "</td>" : "</th>"); }).join("") + "</tr>"; });
    return html + "</tbody></table>";
  }
  function decision(p) { return p < 0.05 ? '<span class="result-reject">Reject at 5%</span>' : "Do not reject at 5%"; }
  function render(data) {
    var p = data.provenance, m = data.metrics, tests = data.coverage_tests, charts = data.charts;
    document.getElementById("risk-provenance").innerHTML = [
      ["Backtest sample", p.backtest_dates[0] + " – " + p.backtest_dates[1]],
      ["Price vintage", p.price_snapshot_dates[1]],
      ["Snapshot hash", "<code>sha256:" + p.price_sha256.slice(0, 16) + "…</code>"],
      ["Pinned source", '<a href="' + p.notebook_url + '"><code>' + p.source_commit.slice(0, 12) + "</code></a>"],
      ["Daily observations", m.n.toLocaleString("en-US")],
      ["Total modeled costs", QL.pct(m.total_cost, 3) + " of NAV"]
    ].map(function (row) { return "<div><dt>" + row[0] + "</dt><dd>" + row[1] + "</dd></div>"; }).join("");
    document.getElementById("risk-notebook-link").href = p.notebook_url;
    document.getElementById("risk-engine-link").href = p.engine_url;

    [["Ann. return", QL.pct(m.annual_return, 2)], ["Ann. volatility", QL.pct(m.annual_volatility, 2)],
      ["Sharpe (rf 3%)", QL.fmt(m.sharpe_rf_3pct, 2)], ["Max drawdown", QL.pct(m.max_drawdown, 1)],
      ["VaR 95 (1d)", QL.pct(m.var95, 2)], ["ES 95 (1d)", QL.pct(m.es95, 2)],
      ["Skewness", QL.fmt(m.skewness, 2)], ["Excess kurtosis", QL.fmt(m.excess_kurtosis, 2)]
    ].forEach(function (d, i) {
      var el = document.createElement("div"); el.className = "stat";
      el.innerHTML = '<dt class="k">' + d[0] + '</dt><dd class="v' + (i === 3 || i === 4 || i === 5 ? " accent" : "") + '">' + d[1] + "</dd>";
      document.getElementById("risk-stats").appendChild(el);
    });

    Plotly.newPlot("chart-equity", [{ x: charts.dates, y: charts.equity, mode: "lines", line: { width: 1.3, color: QL.colors.ink } }], QL.layout({ title: { text: "GROWTH OF $1 · EMPIRICAL MULTI-ASSET PORTFOLIO", font: { size: 11 }, x: 0 }, xaxis: { title: { text: "date" } }, yaxis: { title: { text: "portfolio value" } }, showlegend: false }), QL.plotConfig);
    Plotly.newPlot("chart-dd", [{ x: charts.dates, y: charts.drawdown.map(function (x) { return 100 * x; }), mode: "lines", fill: "tozeroy", line: { width: 1, color: QL.colors.accent }, fillcolor: "rgba(138,48,51,0.12)" }], QL.layout({ title: { text: "DRAWDOWN FROM PEAK (%)", font: { size: 11 }, x: 0 }, xaxis: { title: { text: "date" } }, yaxis: { title: { text: "drawdown (%)" } }, showlegend: false }), QL.plotConfig);
    Plotly.newPlot("chart-vol", [{ x: charts.dates, y: charts.rolling_volatility.map(function (x) { return x == null ? null : 100 * x; }), mode: "lines", line: { width: 1.2, color: QL.colors.slate } }], QL.layout({ title: { text: "ROLLING 63-DAY VOL, ANNUALIZED (%)", font: { size: 11 }, x: 0 }, xaxis: { title: { text: "date" } }, yaxis: { title: { text: "volatility (%)" } }, showlegend: false }), QL.plotConfig);
    Plotly.newPlot("chart-hist", [
      { x: charts.histogram_centers_pct, y: charts.histogram_counts, type: "bar", name: "empirical count", marker: { color: "rgba(62,92,118,0.55)", pattern: { shape: "/" } } },
      { x: charts.histogram_centers_pct, y: charts.normal_counts, mode: "lines", name: "matched Gaussian", line: { width: 1.4, color: QL.colors.ink, dash: "dot" } }
    ], QL.layout({ title: { text: "DAILY RETURN DISTRIBUTION (%)", font: { size: 11 }, x: 0 }, xaxis: { title: { text: "daily return (%)" } }, yaxis: { title: { text: "count" } }, shapes: [
      { type: "line", x0: -100 * m.var95, x1: -100 * m.var95, y0: 0, y1: 1, yref: "paper", line: { color: QL.colors.accent, width: 1.4, dash: "dash" } },
      { type: "line", x0: -100 * m.es95, x1: -100 * m.es95, y0: 0, y1: 1, yref: "paper", line: { color: QL.colors.accent, width: 1.4 } }
    ] }), QL.plotConfig);

    document.getElementById("risk-data-table").innerHTML = table(["Measure", "Value"], [
      ["Ending value of $1", "$" + QL.fmt(charts.equity[charts.equity.length - 1], 3)], ["Maximum drawdown", QL.pct(m.max_drawdown, 2)],
      ["VaR 95, one day", QL.pct(m.var95, 2)], ["Expected Shortfall 95, one day", QL.pct(m.es95, 2)],
      ["Backtest observations", m.n.toLocaleString("en-US")]
    ], "Key values represented in the empirical risk charts");
    document.getElementById("risk-uncertainty-table").innerHTML = table(["Measure", "Point estimate", "Bootstrap 95% interval"], [
      ["Historical VaR 95", QL.pct(m.var95, 3), QL.pct(m.var95_ci[0], 3) + " – " + QL.pct(m.var95_ci[1], 3)],
      ["Historical ES 95", QL.pct(m.es95, 3), QL.pct(m.es95_ci[0], 3) + " – " + QL.pct(m.es95_ci[1], 3)]
    ], "Moving-block bootstrap uncertainty; 2,000 replications, 21-day blocks");
    document.getElementById("risk-coverage-table").innerHTML = table(["Test", "Statistic", "p-value", "Decision"], [
      ["Observed breaches", tests.breaches + " / " + tests.forecast_days, QL.pct(tests.breach_rate, 2) + " rate", "Expected " + tests.expected_breaches],
      ["Kupiec unconditional coverage", QL.fmt(tests.kupiec_lr, 3), QL.fmt(tests.kupiec_p, 5), decision(tests.kupiec_p)],
      ["Christoffersen independence", QL.fmt(tests.christoffersen_independence_lr, 3), QL.fmt(tests.christoffersen_independence_p, 5), decision(tests.christoffersen_independence_p)],
      ["Christoffersen conditional coverage", QL.fmt(tests.christoffersen_conditional_coverage_lr, 3), QL.fmt(tests.christoffersen_conditional_coverage_p, 5), decision(tests.christoffersen_conditional_coverage_p)]
    ], "Rolling 500-observation historical VaR forecast tests");
    document.getElementById("risk-stress-table").innerHTML = table(["Window", "Dates", "Cumulative return", "Max drawdown", "Worst day", "VaR breaches"], data.stress_windows.map(function (x) {
      return [x.label, x.start + " – " + x.end, QL.pct(x.cumulative_return, 1), QL.pct(x.max_drawdown, 1), QL.pct(x.worst_day, 2), x.var_breaches];
    }), "Historical stress windows fixed by event dates");
    document.getElementById("risk-status").textContent = "Empirical risk artifact loaded. Coverage model rejected at five percent.";
  }
  fetch("data/risk-study.json").then(function (response) { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); }).then(render).catch(function (error) {
    document.getElementById("risk-status").textContent = "Risk artifact failed to load.";
    document.getElementById("risk-provenance").innerHTML = "<div><dt>Status</dt><dd>Research artifact unavailable: " + error.message + "</dd></div>";
  });
})();
