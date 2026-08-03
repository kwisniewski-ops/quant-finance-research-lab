/* Lazy Plotly loader.
   The site only uses bar, pie, and SVG scatter traces, so the official basic
   bundle replaces the full distribution. Plot calls are queued until the
   first chart approaches the viewport. */
(function () {
  "use strict";

  var queue = [];
  var loading = false;
  var proxy = {};
  ["newPlot", "react"].forEach(function (method) {
    proxy[method] = function () {
      queue.push({ method: method, args: Array.prototype.slice.call(arguments) });
    };
  });
  window.Plotly = proxy;

  function load() {
    if (loading || window.Plotly !== proxy) return;
    loading = true;
    var script = document.createElement("script");
    script.src = "https://cdn.plot.ly/plotly-basic-2.35.2.min.js";
    script.async = true;
    script.onload = function () {
      var realPlotly = window.Plotly;
      queue.splice(0).forEach(function (call) {
        realPlotly[call.method].apply(realPlotly, call.args);
      });
      document.dispatchEvent(new CustomEvent("ql:plotly-ready"));
    };
    script.onerror = function () {
      loading = false;
      document.querySelectorAll(".chart").forEach(function (chart) {
        chart.classList.add("chart-load-failed");
        chart.textContent = "Chart library unavailable. Use the accessible data table below.";
      });
    };
    document.head.appendChild(script);
  }

  function observe() {
    var chart = document.querySelector(".chart");
    if (!chart) return;
    if (!("IntersectionObserver" in window)) {
      load();
      return;
    }
    var observer = new IntersectionObserver(function (entries) {
      if (entries.some(function (entry) { return entry.isIntersecting; })) {
        observer.disconnect();
        load();
      }
    }, { rootMargin: "700px 0px" });
    observer.observe(chart);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", observe, { once: true });
  } else {
    observe();
  }
})();
