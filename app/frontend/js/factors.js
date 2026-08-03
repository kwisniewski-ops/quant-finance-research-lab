/* Empirical ETF factor study, rendered from a deterministic JSON artifact. */
(function () {
  "use strict";

  var FACTOR_LABELS = { "Mkt-RF": "Market", SMB: "Size", HML: "Value", RMW: "Profitability", CMA: "Investment", Mom: "Momentum" };
  var PATTERNS = ["/", "x"];

  function pct(value, digits) { return QL.pct(value, digits == null ? 2 : digits); }
  function hashLabel(hash) { return "sha256:" + hash.slice(0, 12) + "…"; }
  function table(headers, rows, caption) {
    var html = "<table><caption>" + caption + "</caption><thead><tr>";
    headers.forEach(function (h) { html += '<th scope="col">' + h + "</th>"; });
    html += "</tr></thead><tbody>";
    rows.forEach(function (row) {
      html += "<tr>";
      row.forEach(function (cell, i) { html += (i === 0 ? '<th scope="row">' : "<td>") + cell + (i === 0 ? "</th>" : "</td>"); });
      html += "</tr>";
    });
    return html + "</tbody></table>";
  }

  function render(data) {
    var p = data.provenance;
    document.getElementById("factor-provenance").innerHTML = [
      ["Aligned sample", p.aligned_sample_dates[0] + " – " + p.aligned_sample_dates[1]],
      ["Factor vintage", p.factor_snapshot_dates[1] + " · " + hashLabel(p.factor_sha256)],
      ["Price vintage", p.price_snapshot_dates[1] + " · " + hashLabel(p.price_sha256)],
      ["Pinned source", '<a href="' + p.notebook_url + '"><code>' + p.source_commit.slice(0, 12) + "</code></a>"],
      ["Observations", data.regressions[0].n.toLocaleString("en-US") + " daily rows"],
      ["Estimator", "OLS · Newey–West HAC(5) intervals"]
    ].map(function (row) { return "<div><dt>" + row[0] + "</dt><dd>" + row[1] + "</dd></div>"; }).join("");
    document.getElementById("factor-notebook-link").href = p.notebook_url;

    var selector = document.getElementById("factor-etf");
    data.regressions.forEach(function (result) {
      var option = document.createElement("option");
      option.value = result.ticker;
      option.textContent = result.ticker;
      selector.appendChild(option);
    });

    function renderExposure() {
      var result = data.regressions.find(function (x) { return x.ticker === selector.value; });
      var labels = result.exposures.map(function (x) { return FACTOR_LABELS[x.factor]; });
      var beta = result.exposures.map(function (x) { return x.beta; });
      Plotly.react("chart-factor-exposures", [{
        x: labels, y: beta, mode: "markers", name: result.ticker,
        marker: {
          size: 10,
          color: beta.map(function (x) { return x >= 0 ? QL.colors.slate : QL.colors.accent; }),
          symbol: beta.map(function (x) { return x >= 0 ? "circle" : "x"; })
        },
        error_y: {
          type: "data", symmetric: false,
          array: result.exposures.map(function (x) { return x.ci_high - x.beta; }),
          arrayminus: result.exposures.map(function (x) { return x.beta - x.ci_low; }),
          color: QL.colors.ink, thickness: 1.2, width: 4
        }
      }], QL.layout({
        title: { text: result.ticker + " · DAILY FF5 + MOMENTUM EXPOSURES", font: { size: 11 }, x: 0 },
        yaxis: { title: { text: "beta (95% HAC interval)" }, zeroline: true, zerolinewidth: 1.4 },
        showlegend: false
      }), QL.plotConfig);
      document.getElementById("factor-regression-table").innerHTML = table(
        ["Factor", "Beta", "95% CI", "HAC t"],
        result.exposures.map(function (x) { return [FACTOR_LABELS[x.factor], QL.fmt(x.beta, 3), "[" + QL.fmt(x.ci_low, 3) + ", " + QL.fmt(x.ci_high, 3) + "]", QL.fmt(x.t_hac, 2)]; }),
        result.ticker + " factor estimates; alpha " + pct(result.alpha_annualized, 2) + " annualized; R² " + pct(result.r_squared, 1)
      );
      document.getElementById("factor-status").textContent = result.ticker + " factor estimates displayed.";
    }
    selector.addEventListener("change", renderExposure);
    renderExposure();

    var rolling = data.rolling_qqq;
    var rollFactors = ["Mkt-RF", "HML", "Mom"];
    Plotly.newPlot("chart-factor-rolling", rollFactors.map(function (factor, i) {
      return {
        x: rolling.rows.map(function (x) { return x.date; }),
        y: rolling.rows.map(function (x) { return x[factor]; }),
        mode: "lines", name: FACTOR_LABELS[factor] + " rolling",
        line: { width: 1.4, color: [QL.colors.ink, QL.colors.accent, QL.colors.slate][i], dash: ["solid", "dash", "dot"][i] }
      };
    }), QL.layout({
      title: { text: "QQQ · TRAILING 252-DAY BETAS", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "window end" } }, yaxis: { title: { text: "beta" } },
      shapes: rollFactors.map(function (factor, i) { return { type: "line", xref: "paper", x0: 0, x1: 1, y0: rolling.full_sample[factor], y1: rolling.full_sample[factor], line: { color: [QL.colors.ink, QL.colors.accent, QL.colors.slate][i], width: 1, dash: "dashdot" } }; })
    }), QL.plotConfig);

    Plotly.newPlot("chart-factor-oos", [
      { x: data.out_of_sample.map(function (x) { return x.ticker; }), y: data.out_of_sample.map(function (x) { return 100 * x.in_sample_r2; }), type: "bar", name: "training", marker: { color: QL.colors.slate, pattern: { shape: PATTERNS[0] } } },
      { x: data.out_of_sample.map(function (x) { return x.ticker; }), y: data.out_of_sample.map(function (x) { return 100 * x.out_of_sample_r2; }), type: "bar", name: "holdout", marker: { color: QL.colors.accent, pattern: { shape: PATTERNS[1] } } }
    ], QL.layout({ title: { text: "FROZEN-COEFFICIENT HOLDOUT TEST", font: { size: 11 }, x: 0 }, barmode: "group", yaxis: { title: { text: "R² (%)" } } }), QL.plotConfig);
    document.getElementById("factor-oos-table").innerHTML = table(
      ["ETF", "Training R²", "Holdout R²", "Change", "Holdout dates"],
      data.out_of_sample.map(function (x) { return [x.ticker, pct(x.in_sample_r2, 1), pct(x.out_of_sample_r2, 1), pct(x.decay, 1), x.test_start + " – " + x.test_end]; }),
      "Chronological holdout results; coefficients fitted only on the first half"
    );
    document.getElementById("factor-multiple-table").innerHTML = table(
      ["ETF", "Annualized alpha", "Raw p", "Holm adjusted p", "BH q", "BH 5% decision"],
      data.regressions.map(function (x) { return [x.ticker, pct(x.alpha_annualized, 2), QL.fmt(x.alpha_p_raw, 4), QL.fmt(x.alpha_p_holm, 4), QL.fmt(x.alpha_q_bh, 4), x.alpha_survives_5pct_bh ? "Survives" : "Does not survive"]; }),
      "ETF alpha tests with family-wise and false-discovery corrections"
    );
  }

  fetch("data/factor-study.json").then(function (response) {
    if (!response.ok) throw new Error("HTTP " + response.status);
    return response.json();
  }).then(render).catch(function (error) {
    document.getElementById("factor-status").textContent = "Factor artifact failed to load.";
    document.getElementById("factor-provenance").innerHTML = "<div><dt>Status</dt><dd>Research artifact unavailable: " + error.message + "</dd></div>";
  });
})();
