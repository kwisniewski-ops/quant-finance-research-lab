/* ============================================================
   Factor exposures — static illustrative grouped bar chart.
   Loadings are stylized, not estimated; see repo notebook.
   ============================================================ */
(function () {
  "use strict";

  var FACTORS = ["Market", "Size", "Value", "Momentum", "Quality", "Low Vol"];

  var FUNDS = [
    { name: "SPY (S&P 500)",     color: "#1c1a17", betas: [1.00, -0.15, 0.00,  0.00,  0.05,  0.00] },
    { name: "IWM (Russell 2000)", color: "#8a3033", betas: [1.05,  0.85, 0.10, -0.05, -0.15, -0.20] },
    { name: "VTV (Value)",        color: "#3e5c76", betas: [0.95, -0.05, 0.45, -0.10,  0.10,  0.15] },
    { name: "MTUM (Momentum)",    color: "#a07c33", betas: [1.00, -0.10, -0.20, 0.55,  0.15, -0.05] }
  ];

  var traces = FUNDS.map(function (f) {
    return {
      x: FACTORS, y: f.betas, name: f.name, type: "bar",
      marker: { color: f.color, opacity: 0.85 }
    };
  });

  Plotly.newPlot("chart-factors", traces, QL.layout({
    title: { text: "ILLUSTRATIVE FACTOR LOADINGS BY FUND", font: { size: 11 }, x: 0 },
    barmode: "group",
    yaxis: { title: { text: "loading β" }, zeroline: true, zerolinewidth: 1.5 },
    legend: { orientation: "h", y: 1.1, x: 0 }
  }), QL.plotConfig);
})();
