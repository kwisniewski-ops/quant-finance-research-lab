/* ============================================================
   Observatory hero — Monte Carlo path fan, drawn by hand.
   Geometric Brownian motion under the lab's seeded PRNG (QL),
   with a live convergence meter: the standard error of the
   terminal-mean estimate tightens as completed paths accumulate,
   the lab's own Error ∝ N^-1/2 result made visible.
   No libraries. Deterministic seed. Honors reduced motion.
   ============================================================ */

(function () {
  "use strict";

  if (typeof document === "undefined" || typeof QL === "undefined") return;

  var host = document.querySelector(".hero-field");
  var canvas = host && host.querySelector("canvas");
  if (!host || !canvas || !canvas.getContext) return;

  var meterN = document.getElementById("meter-paths");
  var meterSE = document.getElementById("meter-se");

  var SEED = 20260810;
  var N_PATHS = 280;
  var STEPS = 168;          /* weekly steps over the drawn horizon */
  var MU = 0.062;           /* drift, annualized */
  var SIGMA = 0.17;         /* volatility, annualized */
  var HORIZON = 3.2;        /* years across the canvas width */
  var DRAW_MS = 5200;       /* full sweep duration */
  var STAGGER = 0.5;        /* fraction of timeline over which paths begin */

  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var ctx = canvas.getContext("2d");
  var paths = null;         /* Float64Array per path of log-price */
  var terminals = [];
  var geom = { w: 0, h: 0, dpr: 1 };
  var colors = {};
  var startTime = null;
  var finishedCount = 0;
  var done = false;
  var rafId = 0;

  function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function readColors() {
    colors.faintLine = cssVar("--chart-2", "#2f5e93");
    colors.warmLine = cssVar("--chart-1", "#a06f1c");
    colors.meanLine = cssVar("--accent", "#7e6238");
    colors.grid = cssVar("--chart-grid", "#e7e2d6");
    colors.faint = cssVar("--faint", "#626c77");
  }

  function simulate() {
    var rng = QL.mulberry32(SEED);
    var gauss = QL.gaussian(rng);
    var dt = HORIZON / STEPS;
    var driftTerm = (MU - 0.5 * SIGMA * SIGMA) * dt;
    var diffTerm = SIGMA * Math.sqrt(dt);
    paths = [];
    for (var i = 0; i < N_PATHS; i++) {
      var p = new Float64Array(STEPS + 1);
      p[0] = 0;
      for (var k = 1; k <= STEPS; k++) {
        p[k] = p[k - 1] + driftTerm + diffTerm * gauss();
      }
      paths.push(p);
    }
  }

  /* map log-price to y: center 1.0 at 58% height, ±3.2 sigma-horizon spans the band */
  function yOf(logS) {
    var span = SIGMA * Math.sqrt(HORIZON) * 3.4;
    return geom.h * 0.58 - (logS / span) * geom.h * 0.44;
  }
  function xOf(step) { return (step / STEPS) * geom.w * 1.02 - geom.w * 0.01; }

  function resize() {
    var rect = host.getBoundingClientRect();
    geom.dpr = Math.min(window.devicePixelRatio || 1, 2);
    geom.w = Math.max(rect.width, 1);
    geom.h = Math.max(rect.height, 1);
    canvas.width = Math.round(geom.w * geom.dpr);
    canvas.height = Math.round(geom.h * geom.dpr);
    ctx.setTransform(geom.dpr, 0, 0, geom.dpr, 0, 0);
  }

  function pathStart(i) { return (i / N_PATHS) * STAGGER; }

  /* progress of path i's drawing head given global progress g in [0,1] */
  function headOf(i, g) {
    var s = pathStart(i);
    var local = (g - s) / (1 - STAGGER);
    return Math.max(0, Math.min(1, local));
  }

  function strokePath(i, fromStep, toStep) {
    var p = paths[i];
    if (toStep <= fromStep) return;
    ctx.beginPath();
    ctx.moveTo(xOf(fromStep), yOf(p[fromStep]));
    for (var k = fromStep + 1; k <= toStep; k++) ctx.lineTo(xOf(k), yOf(p[k]));
    var warm = i % 9 === 0;
    ctx.strokeStyle = warm ? colors.warmLine : colors.faintLine;
    ctx.globalAlpha = warm ? 0.34 : 0.13;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawGrid() {
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    for (var gy = 1; gy < 5; gy++) {
      var y = (gy / 5) * geom.h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(geom.w, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawMean() {
    ctx.beginPath();
    for (var k = 0; k <= STEPS; k++) {
      var t = (k / STEPS) * HORIZON;
      /* E[S_t] = e^{mu t}, so the mean sits at mu*t in log space */
      var y = yOf(MU * t);
      if (k === 0) ctx.moveTo(xOf(k), y); else ctx.lineTo(xOf(k), y);
    }
    ctx.strokeStyle = colors.meanLine;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 2;
    ctx.setLineDash([1, 0]);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawTerminalTicks() {
    if (!terminals.length) return;
    var sorted = terminals.slice().sort(function (a, b) { return a - b; });
    var qs = [0.1, 0.5, 0.9];
    ctx.strokeStyle = colors.faint;
    ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.85;
    for (var i = 0; i < qs.length; i++) {
      var v = QL.quantile(sorted, qs[i]);
      var y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(geom.w - 14, y);
      ctx.lineTo(geom.w - 2, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function updateMeter() {
    if (meterN) meterN.textContent = String(finishedCount);
    if (meterSE) {
      if (finishedCount > 8) {
        var se = QL.stdev(terminals) / Math.sqrt(finishedCount);
        meterSE.textContent = (100 * se).toFixed(2) + "%";
      } else {
        meterSE.textContent = "—";
      }
    }
  }

  function renderStatic() {
    resize();
    readColors();
    ctx.clearRect(0, 0, geom.w, geom.h);
    drawGrid();
    terminals = [];
    for (var i = 0; i < N_PATHS; i++) {
      strokePath(i, 0, STEPS);
      terminals.push(paths[i][STEPS]);
    }
    finishedCount = N_PATHS;
    drawMean();
    drawTerminalTicks();
    updateMeter();
  }

  var lastHeads = null;

  function frame(ts) {
    if (startTime === null) {
      startTime = ts;
      drawGrid();
      lastHeads = new Array(N_PATHS);
      for (var z = 0; z < N_PATHS; z++) lastHeads[z] = 0;
    }
    var g = Math.min(1, (ts - startTime) / DRAW_MS);
    var eased = 1 - Math.pow(1 - g, 2.2);
    for (var i = 0; i < N_PATHS; i++) {
      var head = Math.floor(headOf(i, eased) * STEPS);
      if (head > lastHeads[i]) {
        strokePath(i, lastHeads[i], head);
        if (head >= STEPS && lastHeads[i] < STEPS) {
          finishedCount++;
          terminals.push(paths[i][STEPS]);
        }
        lastHeads[i] = head;
      }
    }
    if ((finishedCount & 3) === 0 || g >= 1) updateMeter();
    if (g < 1) {
      rafId = window.requestAnimationFrame(frame);
    } else {
      done = true;
      drawMean();
      drawTerminalTicks();
      updateMeter();
    }
  }

  var resizeTimer = 0;
  window.addEventListener("resize", function () {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(function () {
      if (rafId) window.cancelAnimationFrame(rafId);
      renderStatic();
    }, 160);
  });

  simulate();
  if (reduced || document.hidden) {
    /* hidden tabs never tick rAF — present the finished fan instead */
    renderStatic();
  } else {
    resize();
    readColors();
    terminals = [];
    finishedCount = 0;
    rafId = window.requestAnimationFrame(frame);
    /* watchdog: if no frame ticks (throttled/hidden), settle to the static fan */
    window.setTimeout(function () {
      if (startTime === null && !done) {
        if (rafId) window.cancelAnimationFrame(rafId);
        renderStatic();
      }
    }, 900);
  }

  /* repaint with the other theme's colors when the scheme flips */
  if (window.matchMedia) {
    var scheme = window.matchMedia("(prefers-color-scheme: dark)");
    if (scheme.addEventListener) {
      scheme.addEventListener("change", function () {
        if (rafId) window.cancelAnimationFrame(rafId);
        renderStatic();
      });
    }
  }
})();
