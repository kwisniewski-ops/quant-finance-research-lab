/* ============================================================
   Stochastic process explorer — GBM, OU, Merton, Heston,
   regime-switching GBM. Euler-Maruyama, seeded, 50 paths.
   ============================================================ */
(function () {
  "use strict";

  var N_PATHS = 50, N_STEPS = 252, T_YEARS = 1.0;

  function $(id) { return document.getElementById(id); }

  /* ---------- process definitions ---------- */
  var PROCESSES = {
    gbm: {
      label: "Geometric Brownian motion",
      yTitle: "price",
      params: [
        { key: "S0", label: "S₀", value: 100, min: 1, step: 1 },
        { key: "mu", label: "drift μ", value: 0.07, min: -0.5, step: 0.01 },
        { key: "sigma", label: "vol σ", value: 0.20, min: 0.01, step: 0.01 }
      ],
      simulate: function (p, gauss, rng, dt, path) {
        var x = p.S0;
        path[0] = x;
        for (var t = 1; t <= N_STEPS; t++) {
          x *= Math.exp((p.mu - 0.5 * p.sigma * p.sigma) * dt + p.sigma * Math.sqrt(dt) * gauss());
          path[t] = x;
        }
      }
    },
    ou: {
      label: "Ornstein-Uhlenbeck",
      yTitle: "level",
      params: [
        { key: "X0", label: "X₀", value: 2.0, step: 0.1 },
        { key: "theta", label: "mean θ", value: 4.0, step: 0.1 },
        { key: "kappa", label: "speed κ", value: 3.0, min: 0.01, step: 0.1 },
        { key: "sigma", label: "vol σ", value: 1.0, min: 0.01, step: 0.05 }
      ],
      simulate: function (p, gauss, rng, dt, path) {
        var x = p.X0;
        path[0] = x;
        for (var t = 1; t <= N_STEPS; t++) {
          x += p.kappa * (p.theta - x) * dt + p.sigma * Math.sqrt(dt) * gauss();
          path[t] = x;
        }
      }
    },
    merton: {
      label: "Merton jump-diffusion",
      yTitle: "price",
      params: [
        { key: "S0", label: "S₀", value: 100, min: 1, step: 1 },
        { key: "mu", label: "drift μ", value: 0.07, step: 0.01 },
        { key: "sigma", label: "diffusive σ", value: 0.15, min: 0.01, step: 0.01 },
        { key: "lambda", label: "jump intensity λ (/yr)", value: 3, min: 0, step: 0.5 },
        { key: "muJ", label: "jump mean μJ (log)", value: -0.05, step: 0.01 },
        { key: "sigJ", label: "jump vol σJ", value: 0.08, min: 0, step: 0.01 }
      ],
      simulate: function (p, gauss, rng, dt, path) {
        /* compensated drift so E[return] ≈ mu */
        var kbar = Math.exp(p.muJ + 0.5 * p.sigJ * p.sigJ) - 1;
        var x = p.S0;
        path[0] = x;
        for (var t = 1; t <= N_STEPS; t++) {
          var jump = 0;
          if (rng() < p.lambda * dt) jump = p.muJ + p.sigJ * gauss();
          x *= Math.exp((p.mu - p.lambda * kbar - 0.5 * p.sigma * p.sigma) * dt
                        + p.sigma * Math.sqrt(dt) * gauss() + jump);
          path[t] = x;
        }
      }
    },
    heston: {
      label: "Heston stochastic volatility",
      yTitle: "price",
      params: [
        { key: "S0", label: "S₀", value: 100, min: 1, step: 1 },
        { key: "mu", label: "drift μ", value: 0.05, step: 0.01 },
        { key: "v0", label: "v₀ (variance)", value: 0.04, min: 0.0001, step: 0.01 },
        { key: "kappa", label: "reversion κ", value: 2.0, min: 0.01, step: 0.1 },
        { key: "theta", label: "long-run θ", value: 0.05, min: 0.0001, step: 0.01 },
        { key: "xi", label: "vol-of-vol ξ", value: 0.6, min: 0, step: 0.05 },
        { key: "rho", label: "correlation ρ", value: -0.7, min: -0.99, max: 0.99, step: 0.05 }
      ],
      simulate: function (p, gauss, rng, dt, path) {
        /* full-truncation Euler: v enters drift & diffusion as max(v,0) */
        var x = Math.log(p.S0), v = p.v0;
        path[0] = p.S0;
        var sq1r = Math.sqrt(1 - p.rho * p.rho);
        for (var t = 1; t <= N_STEPS; t++) {
          var z1 = gauss(), z2 = p.rho * z1 + sq1r * gauss();
          var vp = Math.max(v, 0);
          x += (p.mu - 0.5 * vp) * dt + Math.sqrt(vp * dt) * z1;
          v += p.kappa * (p.theta - vp) * dt + p.xi * Math.sqrt(vp * dt) * z2;
          path[t] = Math.exp(x);
        }
      }
    },
    regime: {
      label: "Regime-switching GBM",
      yTitle: "price",
      params: [
        { key: "S0", label: "S₀", value: 100, min: 1, step: 1 },
        { key: "mu1", label: "calm drift μ₁", value: 0.10, step: 0.01 },
        { key: "sig1", label: "calm vol σ₁", value: 0.12, min: 0.01, step: 0.01 },
        { key: "mu2", label: "stress drift μ₂", value: -0.15, step: 0.01 },
        { key: "sig2", label: "stress vol σ₂", value: 0.35, min: 0.01, step: 0.01 },
        { key: "p12", label: "P(calm→stress) /day", value: 0.01, min: 0, max: 0.5, step: 0.005 },
        { key: "p21", label: "P(stress→calm) /day", value: 0.05, min: 0, max: 0.5, step: 0.005 }
      ],
      simulate: function (p, gauss, rng, dt, path) {
        var x = p.S0, regime = 0;
        path[0] = x;
        for (var t = 1; t <= N_STEPS; t++) {
          if (regime === 0 && rng() < p.p12) regime = 1;
          else if (regime === 1 && rng() < p.p21) regime = 0;
          var mu = regime === 0 ? p.mu1 : p.mu2;
          var sig = regime === 0 ? p.sig1 : p.sig2;
          x *= Math.exp((mu - 0.5 * sig * sig) * dt + sig * Math.sqrt(dt) * gauss());
          path[t] = x;
        }
      }
    }
  };

  var state = { proc: "gbm", seed: 7 };

  /* ---------- parameter inputs ---------- */
  function buildParams() {
    var box = $("param-box");
    box.innerHTML = "";
    PROCESSES[state.proc].params.forEach(function (prm) {
      var div = document.createElement("div");
      div.className = "ctrl";
      var lab = document.createElement("label");
      lab.textContent = prm.label;
      var inp = document.createElement("input");
      inp.type = "number";
      inp.value = prm.value;
      inp.step = prm.step;
      if (prm.min !== undefined) inp.min = prm.min;
      if (prm.max !== undefined) inp.max = prm.max;
      inp.setAttribute("data-key", prm.key);
      div.appendChild(lab);
      div.appendChild(inp);
      box.appendChild(div);
    });
  }

  function readParams() {
    var p = {};
    Array.prototype.forEach.call($("param-box").querySelectorAll("input"), function (inp) {
      p[inp.getAttribute("data-key")] = parseFloat(inp.value);
    });
    return p;
  }

  /* ---------- simulate & draw ---------- */
  function run() {
    var proc = PROCESSES[state.proc];
    var p = readParams();
    var rng = QL.mulberry32(state.seed * 1000003 + 17);
    var gauss = QL.gaussian(rng);
    var dt = T_YEARS / N_STEPS;

    var times = QL.linspace(0, T_YEARS, N_STEPS + 1);
    var traces = [];
    var terminals = new Array(N_PATHS);
    var path = new Array(N_STEPS + 1);

    for (var k = 0; k < N_PATHS; k++) {
      proc.simulate(p, gauss, rng, dt, path);
      terminals[k] = path[N_STEPS];
      traces.push({
        x: times, y: path.slice(), mode: "lines", hoverinfo: "skip",
        line: { width: 0.8, color: "rgba(62,92,118,0.38)" }
      });
    }
    /* mean path across simulations (recompute by re-running with same seed would be costly;
       approximate with pointwise mean of drawn traces) */
    var meanPath = new Array(N_STEPS + 1);
    for (var t = 0; t <= N_STEPS; t++) {
      var s = 0;
      for (k = 0; k < N_PATHS; k++) s += traces[k].y[t];
      meanPath[t] = s / N_PATHS;
    }
    traces.push({
      x: times, y: meanPath, mode: "lines", name: "cross-sectional mean",
      line: { width: 2, color: QL.colors.accent }
    });

    Plotly.react("chart-paths", traces, QL.layout({
      title: { text: proc.label.toUpperCase() + " · " + N_PATHS + " PATHS", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: "time (years)" } },
      yaxis: { title: { text: proc.yTitle } },
      showlegend: false
    }), QL.plotConfig);

    var meanT = QL.mean(terminals);
    Plotly.react("chart-terminal", [{
      x: terminals, type: "histogram", nbinsx: 24,
      marker: { color: "rgba(138,48,51,0.5)", line: { width: 0 } }
    }], QL.layout({
      title: { text: "TERMINAL VALUES AT T = 1", font: { size: 11 }, x: 0 },
      xaxis: { title: { text: proc.yTitle + " at T" } },
      yaxis: { title: { text: "paths" } },
      showlegend: false,
      shapes: [{
        type: "line", x0: meanT, x1: meanT, y0: 0, y1: 1, yref: "paper",
        line: { color: QL.colors.ink, width: 1.2, dash: "dash" }
      }],
      annotations: [{
        x: meanT, y: 1.02, yref: "paper", text: "mean " + QL.fmt(meanT, 1),
        showarrow: false, font: { size: 9, color: QL.colors.ink }
      }]
    }), QL.plotConfig);
  }

  /* ---------- prose visibility ---------- */
  function showProse() {
    ["gbm", "ou", "merton", "heston", "regime"].forEach(function (k) {
      var el = $("prose-" + k);
      if (el) el.hidden = (k !== state.proc);
    });
  }

  /* ---------- wiring ---------- */
  $("process-select").addEventListener("change", function () {
    state.proc = this.value;
    buildParams();
    showProse();
    run();
  });
  $("in-seed").addEventListener("input", function () {
    state.seed = parseInt(this.value, 10);
    $("val-seed").textContent = state.seed;
  });
  $("btn-sim").addEventListener("click", run);

  buildParams();
  showProse();
  run();
})();
