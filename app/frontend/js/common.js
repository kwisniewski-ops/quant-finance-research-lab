/* ============================================================
   Quantitative Markets Research Lab — shared utilities
   Seeded PRNG, normal variates, normal CDF (erf-based),
   Black-Scholes core, Plotly theme helpers.
   ============================================================ */

var QL = (function () {
  "use strict";

  /* ---------- seeded PRNG: mulberry32 ---------- */
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Box-Muller: returns a function producing standard normals from rng */
  function gaussian(rng) {
    var spare = null;
    return function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var u = 0, v = 0;
      do { u = rng(); } while (u <= 1e-12);
      v = rng();
      var r = Math.sqrt(-2.0 * Math.log(u));
      var th = 2.0 * Math.PI * v;
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    };
  }

  /* ---------- error function: Abramowitz & Stegun 7.1.26 ---------- */
  /* |error| <= 1.5e-7, adequate for pricing display; symmetric via sign */
  function erf(x) {
    var sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    var t = 1.0 / (1.0 + p * x);
    var y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }

  function normCdf(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
  function normPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }

  /* ---------- Black-Scholes-Merton (continuous dividend yield q) ---------- */
  function bsPrice(S, K, T, r, sigma, q, type) {
    q = q || 0;
    if (T <= 0) {
      return type === "put" ? Math.max(K - S, 0) : Math.max(S - K, 0);
    }
    if (sigma <= 0) sigma = 1e-8;
    var sqT = Math.sqrt(T);
    var d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqT);
    var d2 = d1 - sigma * sqT;
    var dfq = Math.exp(-q * T), dfr = Math.exp(-r * T);
    if (type === "put") {
      return K * dfr * normCdf(-d2) - S * dfq * normCdf(-d1);
    }
    return S * dfq * normCdf(d1) - K * dfr * normCdf(d2);
  }

  /* Full Greeks. theta and rho returned per year; callers may scale. */
  function bsGreeks(S, K, T, r, sigma, q, type) {
    q = q || 0;
    if (T <= 0) T = 1e-8;
    if (sigma <= 0) sigma = 1e-8;
    var sqT = Math.sqrt(T);
    var d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqT);
    var d2 = d1 - sigma * sqT;
    var dfq = Math.exp(-q * T), dfr = Math.exp(-r * T);
    var pdf1 = normPdf(d1);
    var isPut = type === "put";

    var delta = isPut ? dfq * (normCdf(d1) - 1) : dfq * normCdf(d1);
    var gamma = dfq * pdf1 / (S * sigma * sqT);
    var vega  = S * dfq * pdf1 * sqT;               /* per 1.00 of vol */
    var theta;
    if (isPut) {
      theta = -S * dfq * pdf1 * sigma / (2 * sqT)
              + r * K * dfr * normCdf(-d2)
              - q * S * dfq * normCdf(-d1);
    } else {
      theta = -S * dfq * pdf1 * sigma / (2 * sqT)
              - r * K * dfr * normCdf(d2)
              + q * S * dfq * normCdf(d1);
    }
    var rho = isPut ? -K * T * dfr * normCdf(-d2)
                    :  K * T * dfr * normCdf(d2);   /* per 1.00 of rate */
    return { delta: delta, gamma: gamma, vega: vega, theta: theta, rho: rho };
  }

  /* ---------- basic stats ---------- */
  function mean(a) {
    var s = 0; for (var i = 0; i < a.length; i++) s += a[i];
    return s / a.length;
  }
  function stdev(a) {
    var m = mean(a), s = 0;
    for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return Math.sqrt(s / (a.length - 1));
  }
  function quantile(sorted, p) {
    var idx = p * (sorted.length - 1);
    var lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
  }

  /* ---------- Plotly theming ---------- */
  var colors = {
    ink: "#1c1a17", accent: "#8a3033", slate: "#3e5c76",
    ochre: "#a07c33", moss: "#5c6e4e", plum: "#6b4a6e",
    faint: "#9a938a", grid: "#e7e1d4"
  };
  var series = [colors.accent, colors.slate, colors.ochre, colors.moss, colors.plum, colors.ink];

  function layout(overrides) {
    var base = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "'IBM Plex Mono', Menlo, monospace", size: 11, color: "#6d675e" },
      margin: { l: 58, r: 20, t: 42, b: 46 },
      xaxis: { gridcolor: colors.grid, zerolinecolor: "#ddd6c8", linecolor: "#ddd6c8" },
      yaxis: { gridcolor: colors.grid, zerolinecolor: "#ddd6c8", linecolor: "#ddd6c8" },
      colorway: series,
      hoverlabel: {
        bgcolor: "#1c1a17", bordercolor: "#1c1a17",
        font: { family: "'IBM Plex Mono', Menlo, monospace", size: 11, color: "#f7f4ee" }
      },
      legend: { orientation: "h", y: 1.08, x: 0, font: { size: 10 } }
    };
    overrides = overrides || {};
    for (var k in overrides) {
      if (!overrides.hasOwnProperty(k)) continue;
      if (base[k] && typeof base[k] === "object" && !Array.isArray(base[k]) &&
          typeof overrides[k] === "object" && !Array.isArray(overrides[k])) {
        for (var kk in overrides[k]) {
          if (overrides[k].hasOwnProperty(kk)) base[k][kk] = overrides[k][kk];
        }
      } else {
        base[k] = overrides[k];
      }
    }
    return base;
  }

  var plotConfig = { displayModeBar: false, responsive: true };

  function fmt(x, d) {
    if (!isFinite(x)) return "—";
    return x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function pct(x, d) { return fmt(100 * x, d === undefined ? 1 : d) + "%"; }

  function linspace(a, b, n) {
    var out = new Array(n);
    for (var i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
    return out;
  }

  return {
    mulberry32: mulberry32, gaussian: gaussian,
    erf: erf, normCdf: normCdf, normPdf: normPdf,
    bsPrice: bsPrice, bsGreeks: bsGreeks,
    mean: mean, stdev: stdev, quantile: quantile,
    colors: colors, series: series, layout: layout, plotConfig: plotConfig,
    fmt: fmt, pct: pct, linspace: linspace
  };
})();

/* ---------- keyboard-accessible overflow regions ---------- */
(function () {
  "use strict";

  if (typeof document === "undefined") return;

  var scheduled = false;

  function nearestHeadingText(region) {
    var section = region.closest("section");
    var heading = section ? section.querySelector("h2, h3") : null;
    return heading ? heading.textContent.replace(/[#¶]/g, "").trim() : "research content";
  }

  function regionLabel(region) {
    var caption = region.querySelector("caption");
    var kind = region.classList.contains("math-block") ? "equation" : "table";
    return "Scrollable " + kind + ": " + (caption ? caption.textContent.trim() : nearestHeadingText(region));
  }

  function enhance(region) {
    if (region.dataset.scrollEnhanced === "true") return;

    var kind = region.classList.contains("math-block") ? "equation" : "table";
    var frame = document.createElement("div");
    var cue = document.createElement("span");
    var cueId = "scroll-cue-" + Math.random().toString(36).slice(2, 10);

    frame.className = "scroll-frame scroll-frame-" + kind;
    cue.className = "scroll-cue";
    cue.id = cueId;
    cue.hidden = true;
    cue.setAttribute("aria-hidden", "true");
    cue.textContent = "Scroll " + kind + " →";

    region.parentNode.insertBefore(frame, region);
    frame.appendChild(region);
    frame.appendChild(cue);
    region.classList.add("scroll-region");
    region.dataset.scrollEnhanced = "true";
    region.dataset.scrollCue = cueId;
  }

  function update(region) {
    var frame = region.parentElement;
    var cue = frame && frame.querySelector(".scroll-cue");
    if (!frame || !cue) return;

    var overflowing = region.scrollWidth > region.clientWidth + 2;
    frame.classList.toggle("is-overflowing", overflowing);
    cue.hidden = !overflowing;

    if (overflowing) {
      region.tabIndex = 0;
      region.setAttribute("role", "region");
      region.setAttribute("aria-label", regionLabel(region));
    } else {
      region.removeAttribute("tabindex");
      region.removeAttribute("role");
      region.removeAttribute("aria-label");
    }
  }

  function refresh() {
    scheduled = false;
    var regions = document.querySelectorAll(".math-block, .table-scroll");
    Array.prototype.forEach.call(regions, enhance);
    Array.prototype.forEach.call(regions, update);
  }

  function scheduleRefresh() {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(refresh);
  }

  function init() {
    refresh();
    window.addEventListener("load", scheduleRefresh);
    window.addEventListener("resize", scheduleRefresh);
    if (typeof MutationObserver !== "undefined") {
      new MutationObserver(scheduleRefresh).observe(document.body, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
