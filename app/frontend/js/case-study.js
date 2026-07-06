/* ============================================================
   Case study — BS vs precomputed Heston prices across strikes.

   Heston call values computed OFFLINE (Python, Heston 1993
   characteristic function, "little trap" formulation, Simpson
   rule with 4,000 nodes on [0, 200]) and hardcoded below.
   Parameters: S0=100, r=0.02, q=0, T=0.5,
               v0=0.04, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7.
   Implied vols recovered by bisection against Black-Scholes.
   ============================================================ */
(function () {
  "use strict";

  var S0 = 100, R = 0.02, Q = 0, T = 0.5;

  var STRIKES = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130];
  var HESTON_CALLS = [30.9860, 26.2467, 21.6299, 17.1919, 13.0108, 9.1946,
                      5.8920, 3.2897, 1.5472, 0.6244, 0.2338, 0.0869, 0.0330];
  var HESTON_IVS = [0.2980, 0.2808, 0.2635, 0.2459, 0.2281, 0.2099, 0.1918,
                    0.1747, 0.1605, 0.1514, 0.1476, 0.1474, 0.1490];

  var state = { flatVol: 0.192 };

  function $(id) { return document.getElementById(id); }

  /* ---------- smile chart (section II) ---------- */
  function drawSmile() {
    var traces = [{
      x: STRIKES, y: HESTON_IVS.map(function (v) { return 100 * v; }),
      mode: "lines+markers", name: "smile-consistent implied vol",
      line: { width: 1.8, color: QL.colors.accent },
      marker: { size: 5, color: QL.colors.accent }
    }, {
      x: [STRIKES[0], STRIKES[STRIKES.length - 1]],
      y: [100 * state.flatVol, 100 * state.flatVol],
      mode: "lines", name: "Black-Scholes (flat σ)",
      line: { width: 1.2, color: QL.colors.ink, dash: "dash" }
    }];
    Plotly.react("chart-smile", traces, QL.layout({
      title: { text: "IMPLIED VOLATILITY vs STRIKE · T = 0.5Y (ILLUSTRATIVE)", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "strike K" } },
      yaxis: { title: { text: "implied volatility (%)" } }
    }), QL.plotConfig);
  }

  /* ---------- price comparison (section IV) ---------- */
  function bsAt(K) { return QL.bsPrice(S0, K, T, R, state.flatVol, Q, "call"); }

  function drawPrices() {
    var bs = STRIKES.map(bsAt);
    var traces = [{
      x: STRIKES, y: bs, mode: "lines", name: "Black-Scholes, flat σ",
      line: { width: 1.6, color: QL.colors.slate }
    }, {
      x: STRIKES, y: HESTON_CALLS, mode: "markers", name: "Heston (precomputed)",
      marker: { size: 7, symbol: "diamond", color: QL.colors.accent }
    }];
    Plotly.react("chart-prices", traces, QL.layout({
      title: { text: "6-MONTH CALL PRICES ACROSS STRIKES", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "strike K" } },
      yaxis: { title: { text: "call price" } }
    }), QL.plotConfig);

    var diff = STRIKES.map(function (K, i) { return bs[i] - HESTON_CALLS[i]; });
    Plotly.react("chart-diff", [{
      x: STRIKES, y: diff, type: "bar", name: "BS − Heston",
      marker: {
        color: diff.map(function (d) { return d >= 0 ? "rgba(62,92,118,0.7)" : "rgba(138,48,51,0.7)"; })
      }
    }], QL.layout({
      title: { text: "PRICING ERROR: BS(σ=" + (100 * state.flatVol).toFixed(1) + "%) − HESTON", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "strike K" } },
      yaxis: { title: { text: "price difference" } },
      showlegend: false
    }), QL.plotConfig);

    buildTable(bs);
  }

  function buildTable(bs) {
    var html = '<table class="data"><thead><tr>' +
      '<th class="num">Strike</th><th class="num">Heston call</th>' +
      '<th class="num">Heston impl. vol</th><th class="num">BS call (flat σ)</th>' +
      '<th class="num">BS − Heston</th></tr></thead><tbody>';
    STRIKES.forEach(function (K, i) {
      var d = bs[i] - HESTON_CALLS[i];
      html += "<tr><td class=\"num\">" + K + "</td>" +
        '<td class="num">' + QL.fmt(HESTON_CALLS[i], 4) + "</td>" +
        '<td class="num">' + QL.pct(HESTON_IVS[i], 2) + "</td>" +
        '<td class="num">' + QL.fmt(bs[i], 4) + "</td>" +
        '<td class="num" style="color:' + (Math.abs(d) > 0.15 ? "#8a3033" : "inherit") + '">' +
        (d >= 0 ? "+" : "") + QL.fmt(d, 4) + "</td></tr>";
    });
    html += "</tbody></table>";
    $("price-table").innerHTML = html;
  }

  /* ---------- wiring ---------- */
  $("in-flatvol").addEventListener("input", function () {
    state.flatVol = parseFloat(this.value);
    $("val-flatvol").textContent = QL.pct(state.flatVol, 1);
    drawSmile();
    drawPrices();
  });

  drawSmile();
  drawPrices();
})();
