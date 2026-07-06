/* ============================================================
   Asset pricing laboratory: Black-Scholes-Merton + CRR binomial
   ============================================================ */
(function () {
  "use strict";

  /* ---------- CRR binomial pricer ---------- */
  function crrPrice(S, K, T, r, sigma, q, steps, type, american) {
    var N = Math.max(2, Math.round(steps));
    var dt = T / N;
    var u = Math.exp(sigma * Math.sqrt(dt));
    var d = 1 / u;
    var disc = Math.exp(-r * dt);
    var p = (Math.exp((r - q) * dt) - d) / (u - d);
    p = Math.min(Math.max(p, 0), 1); /* guard extreme params */

    var isPut = type === "put";
    var i, j;

    /* terminal payoffs */
    var v = new Array(N + 1);
    for (i = 0; i <= N; i++) {
      var ST = S * Math.pow(u, N - i) * Math.pow(d, i);
      v[i] = isPut ? Math.max(K - ST, 0) : Math.max(ST - K, 0);
    }
    /* backward induction */
    for (j = N - 1; j >= 0; j--) {
      for (i = 0; i <= j; i++) {
        var cont = disc * (p * v[i] + (1 - p) * v[i + 1]);
        if (american) {
          var Snode = S * Math.pow(u, j - i) * Math.pow(d, i);
          var ex = isPut ? K - Snode : Snode - K;
          v[i] = Math.max(cont, ex);
        } else {
          v[i] = cont;
        }
      }
    }
    return v[0];
  }

  /* expose for node-based verification */
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { crrPrice: crrPrice };
  }
  if (typeof document === "undefined") return; /* running under node */

  /* ---------- state & DOM ---------- */
  var state = {
    S: 100, K: 100, T: 1, r: 0.05, sigma: 0.20, q: 0,
    type: "call", style: "european", steps: 200
  };

  function $(id) { return document.getElementById(id); }

  var sliders = [
    { id: "S", el: "in-S", out: "val-S", fmt: function (x) { return QL.fmt(x, 0); } },
    { id: "K", el: "in-K", out: "val-K", fmt: function (x) { return QL.fmt(x, 0); } },
    { id: "T", el: "in-T", out: "val-T", fmt: function (x) { return QL.fmt(x, 2); } },
    { id: "r", el: "in-r", out: "val-r", fmt: function (x) { return QL.pct(x, 2); } },
    { id: "sigma", el: "in-sigma", out: "val-sigma", fmt: function (x) { return QL.pct(x, 1); } },
    { id: "q", el: "in-q", out: "val-q", fmt: function (x) { return QL.pct(x, 2); } },
    { id: "steps", el: "in-steps", out: "val-steps", fmt: function (x) { return QL.fmt(x, 0); } }
  ];

  sliders.forEach(function (s) {
    $(s.el).addEventListener("input", function () {
      state[s.id] = parseFloat(this.value);
      $(s.out).textContent = s.fmt(state[s.id]);
      update();
    });
  });

  function wireToggle(containerId, attr, key) {
    var box = $(containerId);
    Array.prototype.forEach.call(box.querySelectorAll("button"), function (btn) {
      btn.addEventListener("click", function () {
        Array.prototype.forEach.call(box.querySelectorAll("button"), function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        state[key] = btn.getAttribute(attr);
        update();
      });
    });
  }
  wireToggle("type-toggle", "data-type", "type");
  wireToggle("style-toggle", "data-style", "style");

  /* ---------- update ---------- */
  function update() {
    var s = state;
    var bs = QL.bsPrice(s.S, s.K, s.T, s.r, s.sigma, s.q, s.type);
    var american = s.style === "american";
    var crr = crrPrice(s.S, s.K, s.T, s.r, s.sigma, s.q, s.steps, s.type, american);
    var g = QL.bsGreeks(s.S, s.K, s.T, s.r, s.sigma, s.q, s.type);

    $("out-bs").textContent = QL.fmt(bs, 4);
    $("out-crr").textContent = QL.fmt(crr, 4);
    $("out-diff").textContent = (crr - bs >= 0 ? "+" : "") + QL.fmt(crr - bs, 4);
    $("crr-label").textContent = "CRR Binomial (" + (american ? "American" : "European") + ", N=" + s.steps + ")";

    $("g-delta").textContent = QL.fmt(g.delta, 4);
    $("g-gamma").textContent = QL.fmt(g.gamma, 4);
    $("g-vega").textContent = QL.fmt(g.vega / 100, 4);
    $("g-theta").textContent = QL.fmt(g.theta / 365, 4);
    $("g-rho").textContent = QL.fmt(g.rho / 100, 4);
    var intrinsic = s.type === "put" ? Math.max(s.K - s.S, 0) : Math.max(s.S - s.K, 0);
    $("g-intrinsic").textContent = QL.fmt(intrinsic, 2);

    drawValueChart();
    drawVolChart();
  }

  /* ---------- chart 1: value vs underlying at several maturities ---------- */
  function drawValueChart() {
    var s = state;
    var Sgrid = QL.linspace(Math.max(5, 0.4 * s.K), 1.8 * s.K, 121);
    var fracs = [1.0, 0.5, 0.25, 0.05];
    var traces = fracs.map(function (f, i) {
      var T = Math.max(s.T * f, 0.01);
      return {
        x: Sgrid,
        y: Sgrid.map(function (x) { return QL.bsPrice(x, s.K, T, s.r, s.sigma, s.q, s.type); }),
        mode: "lines",
        name: "τ = " + QL.fmt(T, 2) + "y",
        line: { width: 1.6, color: QL.series[i] }
      };
    });
    traces.push({
      x: Sgrid,
      y: Sgrid.map(function (x) {
        return s.type === "put" ? Math.max(s.K - x, 0) : Math.max(x - s.K, 0);
      }),
      mode: "lines",
      name: "intrinsic",
      line: { width: 1, color: QL.colors.faint, dash: "dash" }
    });
    traces.push({
      x: [s.S], y: [QL.bsPrice(s.S, s.K, s.T, s.r, s.sigma, s.q, s.type)],
      mode: "markers", name: "current",
      marker: { size: 9, color: QL.colors.ink, symbol: "circle-open", line: { width: 1.5 } }
    });
    Plotly.react("chart-value", traces, QL.layout({
      title: { text: (state.type === "put" ? "PUT" : "CALL") + " VALUE vs UNDERLYING", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "underlying price S" } },
      yaxis: { title: { text: "option value" } }
    }), QL.plotConfig);
  }

  /* ---------- chart 2: price vs volatility ---------- */
  function drawVolChart() {
    var s = state;
    var vGrid = QL.linspace(0.02, 0.8, 100);
    var traces = [{
      x: vGrid.map(function (v) { return 100 * v; }),
      y: vGrid.map(function (v) { return QL.bsPrice(s.S, s.K, s.T, s.r, v, s.q, s.type); }),
      mode: "lines", name: "BS price",
      line: { width: 1.8, color: QL.colors.slate }
    }, {
      x: [100 * s.sigma],
      y: [QL.bsPrice(s.S, s.K, s.T, s.r, s.sigma, s.q, s.type)],
      mode: "markers", name: "current σ",
      marker: { size: 9, color: QL.colors.accent, symbol: "circle-open", line: { width: 1.5 } }
    }];
    Plotly.react("chart-vol", traces, QL.layout({
      title: { text: "PRICE vs VOLATILITY", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "volatility σ (%)" } },
      yaxis: { title: { text: "option value" } },
      showlegend: false
    }), QL.plotConfig);
  }

  update();
})();
