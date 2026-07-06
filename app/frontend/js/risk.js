/* ============================================================
   Risk dashboard — seeded synthetic daily returns for an
   illustrative 60/40-with-alternatives portfolio.
   GARCH(1,1)-style variance recursion + occasional jumps,
   mulberry32 seed => identical series on every load.
   ============================================================ */
(function () {
  "use strict";

  var N_DAYS = 2520; /* ~10 years */
  var SEED = 90210;  /* chosen so the seeded decade looks like a plausible 60/40+alts history */

  /* ---------- generate returns ---------- */
  function generate() {
    var rng = QL.mulberry32(SEED);
    var gauss = QL.gaussian(rng);

    /* GARCH(1,1): sigma2_t = omega + alpha*eps^2 + beta*sigma2 */
    var targetVol = 0.095 / Math.sqrt(252);     /* ~9.5% annualized base */
    var alpha = 0.09, beta = 0.88;
    var omega = targetVol * targetVol * (1 - alpha - beta);
    var mu = 0.097 / 252;                        /* base drift; jump drag (~-3.2%/yr) nets it to ~6.5% */

    var rets = new Array(N_DAYS);
    var sigma2 = targetVol * targetVol;
    for (var t = 0; t < N_DAYS; t++) {
      var eps = Math.sqrt(sigma2) * gauss();
      /* rare jump component: ~4 per year, mean -0.8%, sd 1.6% */
      if (rng() < 4 / 252) {
        eps += (-0.008 + 0.016 * gauss());
      }
      rets[t] = mu + eps;
      sigma2 = omega + alpha * eps * eps + beta * sigma2;
    }
    return rets;
  }

  /* ---------- derived series ---------- */
  var rets = generate();
  var days = new Array(N_DAYS);
  var equity = new Array(N_DAYS);
  var v = 1;
  for (var i = 0; i < N_DAYS; i++) {
    v *= (1 + rets[i]);
    equity[i] = v;
    days[i] = i / 252; /* years */
  }

  var peak = -Infinity, drawdown = new Array(N_DAYS), maxDD = 0;
  for (i = 0; i < N_DAYS; i++) {
    if (equity[i] > peak) peak = equity[i];
    drawdown[i] = equity[i] / peak - 1;
    if (drawdown[i] < maxDD) maxDD = drawdown[i];
  }

  var W = 63;
  var rollVolX = [], rollVolY = [];
  for (i = W; i < N_DAYS; i++) {
    var s = 0, m = 0, j;
    for (j = i - W; j < i; j++) m += rets[j];
    m /= W;
    for (j = i - W; j < i; j++) s += (rets[j] - m) * (rets[j] - m);
    rollVolX.push(days[i]);
    rollVolY.push(100 * Math.sqrt(s / (W - 1)) * Math.sqrt(252));
  }

  /* ---------- stats ---------- */
  var meanD = QL.mean(rets), sdD = QL.stdev(rets);
  var annRet = Math.pow(equity[N_DAYS - 1], 252 / N_DAYS) - 1;
  var annVol = sdD * Math.sqrt(252);
  var sharpe = (meanD * 252 - 0.03) / annVol; /* vs 3% cash */

  var sorted = rets.slice().sort(function (a, b) { return a - b; });
  var var95 = -QL.quantile(sorted, 0.05);
  var tail = sorted.filter(function (x) { return x <= -var95; });
  var es95 = -QL.mean(tail);

  var skew = 0, kurt = 0;
  for (i = 0; i < N_DAYS; i++) {
    var z = (rets[i] - meanD) / sdD;
    skew += z * z * z;
    kurt += z * z * z * z;
  }
  skew /= N_DAYS;
  kurt = kurt / N_DAYS - 3; /* excess */

  /* ---------- stats grid ---------- */
  var statDefs = [
    ["Ann. return", QL.pct(annRet, 2), false],
    ["Ann. volatility", QL.pct(annVol, 2), false],
    ["Sharpe (rf 3%)", QL.fmt(sharpe, 2), false],
    ["Max drawdown", QL.pct(maxDD, 1), true],
    ["VaR 95 (1d)", QL.pct(var95, 2), true],
    ["ES 95 (1d)", QL.pct(es95, 2), true],
    ["Skewness", QL.fmt(skew, 2), false],
    ["Excess kurtosis", QL.fmt(kurt, 2), false]
  ];
  var grid = document.getElementById("stats");
  statDefs.forEach(function (d) {
    var el = document.createElement("div");
    el.className = "stat";
    el.innerHTML = '<span class="k">' + d[0] + '</span><span class="v' +
      (d[2] ? " accent" : "") + '">' + d[1] + "</span>";
    grid.appendChild(el);
  });

  /* ---------- charts ---------- */
  Plotly.newPlot("chart-equity", [{
    x: days, y: equity, mode: "lines", name: "equity",
    line: { width: 1.3, color: QL.colors.ink }
  }], QL.layout({
    title: { text: "GROWTH OF $1 · SIMULATED 60/40 + ALTERNATIVES", font: { size: 11 }, x: 0 },
    xaxis: { title: { text: "years" } },
    yaxis: { title: { text: "portfolio value" } },
    showlegend: false
  }), QL.plotConfig);

  Plotly.newPlot("chart-dd", [{
    x: days, y: drawdown.map(function (d) { return 100 * d; }),
    mode: "lines", fill: "tozeroy",
    line: { width: 1, color: QL.colors.accent },
    fillcolor: "rgba(138,48,51,0.12)"
  }], QL.layout({
    title: { text: "DRAWDOWN FROM PEAK (%)", font: { size: 11 }, x: 0 },
    xaxis: { title: { text: "years" } },
    yaxis: { title: { text: "drawdown (%)" } },
    showlegend: false
  }), QL.plotConfig);

  Plotly.newPlot("chart-vol", [{
    x: rollVolX, y: rollVolY, mode: "lines",
    line: { width: 1.2, color: QL.colors.slate }
  }], QL.layout({
    title: { text: "ROLLING 63-DAY VOL, ANNUALIZED (%)", font: { size: 11 }, x: 0 },
    xaxis: { title: { text: "years" } },
    yaxis: { title: { text: "volatility (%)" } },
    showlegend: false
  }), QL.plotConfig);

  /* histogram + normal overlay + VaR/ES lines */
  var pcts = rets.map(function (r) { return 100 * r; });
  var xs = QL.linspace(Math.min.apply(null, pcts), Math.max.apply(null, pcts), 160);
  var binW = 0.1; /* percent */
  var normY = xs.map(function (x) {
    var zz = (x / 100 - meanD) / sdD;
    /* scale pdf to counts: N * binWidth(in return units) * pdf */
    return N_DAYS * (binW / 100) * QL.normPdf(zz) / sdD;
  });

  Plotly.newPlot("chart-hist", [{
    x: pcts, type: "histogram", name: "daily returns",
    xbins: { size: binW },
    marker: { color: "rgba(62,92,118,0.55)", line: { width: 0 } }
  }, {
    x: xs, y: normY, mode: "lines", name: "matched Gaussian",
    line: { width: 1.4, color: QL.colors.ink, dash: "dot" }
  }], QL.layout({
    title: { text: "DAILY RETURN DISTRIBUTION (%)", font: { size: 11 }, x: 0 },
    xaxis: { title: { text: "daily return (%)" }, range: [-4.5, 4.5] },
    yaxis: { title: { text: "count" } },
    showlegend: false,
    shapes: [{
      type: "line", x0: -100 * var95, x1: -100 * var95, y0: 0, y1: 1,
      yref: "paper", line: { color: QL.colors.accent, width: 1.4, dash: "dash" }
    }, {
      type: "line", x0: -100 * es95, x1: -100 * es95, y0: 0, y1: 1,
      yref: "paper", line: { color: QL.colors.accent, width: 1.4 }
    }],
    annotations: [{
      x: -100 * var95, y: 0.98, yref: "paper", text: "VaR95",
      showarrow: false, xanchor: "left", font: { size: 9, color: QL.colors.accent }
    }, {
      x: -100 * es95, y: 0.88, yref: "paper", text: "ES95",
      showarrow: false, xanchor: "right", font: { size: 9, color: QL.colors.accent }
    }]
  }), QL.plotConfig);
})();
