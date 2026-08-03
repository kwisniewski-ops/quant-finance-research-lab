/* ============================================================
   Efficient frontier explorer — random long-only portfolios
   over a 6-asset universe. All inputs illustrative.
   ============================================================ */
(function () {
  "use strict";

  /* ---------- illustrative capital market assumptions (annualized) ---------- */
  var ASSETS = [
    { key: "us",   name: "US Equity",        mu: 0.070, vol: 0.16 },
    { key: "intl", name: "Intl Dev Equity",  mu: 0.065, vol: 0.17 },
    { key: "em",   name: "EM Equity",        mu: 0.080, vol: 0.22 },
    { key: "bond", name: "US Agg Bonds",     mu: 0.040, vol: 0.055 },
    { key: "gold", name: "Gold",             mu: 0.045, vol: 0.15 },
    { key: "reit", name: "REITs",            mu: 0.068, vol: 0.19 }
  ];

  var CORR = [
    [1.00, 0.85, 0.75, 0.10, 0.05, 0.75],
    [0.85, 1.00, 0.80, 0.12, 0.10, 0.65],
    [0.75, 0.80, 1.00, 0.08, 0.15, 0.60],
    [0.10, 0.12, 0.08, 1.00, 0.20, 0.25],
    [0.05, 0.10, 0.15, 0.20, 1.00, 0.10],
    [0.75, 0.65, 0.60, 0.25, 0.10, 1.00]
  ];

  /* covariance from correlation and vols */
  var COV = ASSETS.map(function (a, i) {
    return ASSETS.map(function (b, j) {
      return CORR[i][j] * a.vol * b.vol;
    });
  });

  var state = { rf: 0.03, n: 20000, active: ASSETS.map(function () { return true; }) };

  function $(id) { return document.getElementById(id); }

  /* ---------- build asset checkboxes ---------- */
  var checksBox = $("asset-checks");
  ASSETS.forEach(function (a, i) {
    var row = document.createElement("label");
    row.className = "check-row";
    var cb = document.createElement("input");
    cb.id = "asset-" + a.key;
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", function () {
      var count = state.active.filter(Boolean).length;
      if (!cb.checked && count <= 2) { cb.checked = true; return; } /* keep >= 2 */
      state.active[i] = cb.checked;
      update();
    });
    row.appendChild(cb);
    row.appendChild(document.createTextNode(a.name + "  (μ " + QL.pct(a.mu, 1) + ", σ " + QL.pct(a.vol, 1) + ")"));
    checksBox.appendChild(row);
  });

  $("in-rf").addEventListener("input", function () {
    state.rf = parseFloat(this.value);
    $("val-rf").textContent = QL.pct(state.rf, 2);
    update();
  });
  $("in-n").addEventListener("input", function () {
    state.n = parseInt(this.value, 10);
    $("val-n").textContent = state.n.toLocaleString("en-US");
    update();
  });

  /* ---------- portfolio math over active subset ---------- */
  function portStats(w, idx) {
    var mu = 0, i, j;
    for (i = 0; i < idx.length; i++) mu += w[i] * ASSETS[idx[i]].mu;
    var v = 0;
    for (i = 0; i < idx.length; i++) {
      for (j = 0; j < idx.length; j++) {
        v += w[i] * w[j] * COV[idx[i]][idx[j]];
      }
    }
    return { mu: mu, vol: Math.sqrt(v) };
  }

  function update() {
    var idx = [];
    state.active.forEach(function (on, i) { if (on) idx.push(i); });
    var k = idx.length;
    var rng = QL.mulberry32(20260706); /* seeded: same cloud every load */

    var vols = new Array(state.n), rets = new Array(state.n), sharpes = new Array(state.n);
    var best = { sh: -Infinity, w: null, s: null };
    var minv = { vol: Infinity, w: null, s: null };

    for (var s = 0; s < state.n; s++) {
      /* Dirichlet(1,...,1): normalized unit exponentials */
      var w = new Array(k), tot = 0, i;
      for (i = 0; i < k; i++) {
        var u = rng();
        if (u < 1e-12) u = 1e-12;
        w[i] = -Math.log(u);
        tot += w[i];
      }
      for (i = 0; i < k; i++) w[i] /= tot;

      var st = portStats(w, idx);
      var sh = (st.mu - state.rf) / st.vol;
      vols[s] = 100 * st.vol; rets[s] = 100 * st.mu; sharpes[s] = sh;
      if (sh > best.sh) { best = { sh: sh, w: w.slice(), s: st }; }
      if (st.vol < minv.vol) { minv = { vol: st.vol, w: w.slice(), s: st }; }
    }

    /* equal weight */
    var ew = idx.map(function () { return 1 / k; });
    var ewSt = portStats(ew, idx);

    fillRow("mv", minv.s, state.rf);
    fillRow("ms", best.s, state.rf);
    fillRow("ew", ewSt, state.rf);

    drawFrontier(vols, rets, sharpes, minv, best, ewSt, idx);
    drawWeights(minv.w, best.w, ew, idx);
  }

  function fillRow(prefix, st, rf) {
    $(prefix + "-ret").textContent = QL.pct(st.mu, 1);
    $(prefix + "-vol").textContent = QL.pct(st.vol, 1);
    $(prefix + "-sh").textContent = QL.fmt((st.mu - rf) / st.vol, 2);
  }

  var sharpeScale = [
    [0.0, "#c9c2b4"], [0.4, "#a89a86"], [0.7, "#3e5c76"], [1.0, "#8a3033"]
  ];

  function drawFrontier(vols, rets, sharpes, minv, best, ewSt, idx) {
    /* Keep all samples in the optimization, but cap SVG markers for the basic
       Plotly bundle and a responsive browser payload. */
    var stride = Math.max(1, Math.ceil(vols.length / 3500));
    var displayVols = [], displayRets = [], displaySharpes = [];
    for (var sample = 0; sample < vols.length; sample += stride) {
      displayVols.push(vols[sample]);
      displayRets.push(rets[sample]);
      displaySharpes.push(sharpes[sample]);
    }
    var traces = [{
      x: displayVols, y: displayRets, mode: "markers", type: "scatter", name: "portfolios",
      marker: {
        size: 3, opacity: 0.5, color: displaySharpes, colorscale: sharpeScale,
        colorbar: { title: { text: "Sharpe", font: { size: 10 } }, thickness: 8, len: 0.6, outlinewidth: 0 }
      },
      hoverinfo: "skip"
    }, {
      x: idx.map(function (i) { return 100 * ASSETS[i].vol; }),
      y: idx.map(function (i) { return 100 * ASSETS[i].mu; }),
      text: idx.map(function (i) { return ASSETS[i].name; }),
      mode: "markers+text", name: "assets", textposition: "top center",
      textfont: { size: 9, color: "#6d675e" },
      marker: { symbol: "diamond", size: 8, color: "#1c1a17" }
    }, {
      x: [100 * minv.s.vol], y: [100 * minv.s.mu], mode: "markers", name: "min variance",
      marker: { size: 12, symbol: "circle-open", color: "#3e5c76", line: { width: 2.5 } }
    }, {
      x: [100 * best.s.vol], y: [100 * best.s.mu], mode: "markers", name: "max Sharpe",
      marker: { size: 12, symbol: "x", color: "#8a3033", line: { width: 2.5 } }
    }, {
      x: [100 * ewSt.vol], y: [100 * ewSt.mu], mode: "markers", name: "equal weight",
      marker: { size: 12, symbol: "triangle-up-open", color: "#1c1a17", line: { width: 2.5 } }
    }, {
      /* capital market line */
      x: [0, 100 * best.s.vol * 1.6],
      y: [100 * state.rf, 100 * (state.rf + (best.s.mu - state.rf) * 1.6)],
      mode: "lines", name: "CML",
      line: { width: 1, color: "#9a938a", dash: "dot" }
    }];
    Plotly.react("chart-frontier", traces, QL.layout({
      title: { text: "RISK-RETURN SPACE · " + vols.length.toLocaleString("en-US") + " SAMPLED · " + displayVols.length.toLocaleString("en-US") + " SHOWN", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "annualized volatility (%)" }, rangemode: "tozero" },
      yaxis: { title: { text: "annualized expected return (%)" } },
      showlegend: false
    }), QL.plotConfig);
  }

  function drawWeights(wMin, wMax, wEw, idx) {
    var names = idx.map(function (i) { return ASSETS[i].name; });
    function bar(w, label, color, shape) {
      return {
        x: names, y: w.map(function (x) { return 100 * x; }),
        name: label, type: "bar", marker: { color: color, pattern: { shape: shape } }
      };
    }
    var traces = [
      bar(wMin, "min variance", "#3e5c76", "/"),
      bar(wMax, "max Sharpe", "#8a3033", "x"),
      bar(wEw, "equal weight", "#c9c2b4", ".")
    ];
    Plotly.react("chart-weights", traces, QL.layout({
      title: { text: "PORTFOLIO WEIGHTS (%)", font: { size: 11 }, x: 0 },
      barmode: "group",
      yaxis: { title: { text: "weight (%)" } },
      xaxis: { tickfont: { size: 9 } }
    }), QL.plotConfig);

    var html = '<table><caption>Weights of highlighted portfolios, percent</caption><thead><tr><th scope="col">Asset</th><th scope="col">Minimum variance</th><th scope="col">Maximum Sharpe</th><th scope="col">Equal weight</th></tr></thead><tbody>';
    names.forEach(function (name, i) {
      html += '<tr><th scope="row">' + name + '</th><td>' + QL.fmt(100 * wMin[i], 2) + '%</td><td>' + QL.fmt(100 * wMax[i], 2) + '%</td><td>' + QL.fmt(100 * wEw[i], 2) + '%</td></tr>';
    });
    $("weights-data-table").innerHTML = html + '</tbody></table>';
  }

  /* ---------- assumptions table ---------- */
  function buildAssumptionsTable() {
    var html = '<table class="data"><caption>Illustrative annualized capital-market assumptions and correlations</caption><thead><tr><th scope="col">Asset class</th><th scope="col" class="num">E[return]</th><th scope="col" class="num">Volatility</th>';
    ASSETS.forEach(function (a) { html += '<th scope="col" class="num">ρ ' + a.key.toUpperCase() + "</th>"; });
    html += "</tr></thead><tbody>";
    ASSETS.forEach(function (a, i) {
      html += '<tr><th scope="row">' + a.name + '</th><td class="num">' + QL.pct(a.mu, 1) +
              '</td><td class="num">' + QL.pct(a.vol, 1) + "</td>";
      CORR[i].forEach(function (c) { html += '<td class="num">' + QL.fmt(c, 2) + "</td>"; });
      html += "</tr>";
    });
    html += "</tbody></table>";
    html += '<p class="figure-note">Annualized, illustrative. Not estimates from any specific sample period.</p>';
    $("assumptions-table").innerHTML = html;
  }

  buildAssumptionsTable();
  update();
})();
