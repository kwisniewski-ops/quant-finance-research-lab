/* ============================================================
   Regime atlas — transmission plane wiring.
   The static SVG plane in the page is the visual instrument; the
   accessible state machinery remains the button matrix that
   regime-atlas.js owns. This module only:
     1. proxies quadrant clicks to the matching state button,
     2. mirrors the pressed state back onto the plane,
     3. animates the 2020–2023 trajectory when the plane scrolls
        into view (skipped under reduced motion).
   ============================================================ */

(function () {
  "use strict";

  if (typeof document === "undefined") return;

  var figure = document.querySelector(".atlas-plane-figure");
  if (!figure) return;

  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function stateButton(state) {
    return document.querySelector('[data-atlas-state="' + state + '"]');
  }

  function syncActive() {
    var quads = figure.querySelectorAll("[data-plane-state]");
    Array.prototype.forEach.call(quads, function (q) {
      var button = stateButton(q.getAttribute("data-plane-state"));
      var pressed = button && button.getAttribute("aria-pressed") === "true";
      q.classList.toggle("is-active", !!pressed);
    });
  }

  function initQuadrants() {
    var quads = figure.querySelectorAll("[data-plane-state]");
    Array.prototype.forEach.call(quads, function (q) {
      q.addEventListener("click", function () {
        var button = stateButton(q.getAttribute("data-plane-state"));
        if (button && !button.disabled) {
          button.click();
          syncActive();
        }
      });
    });
    /* mirror clicks made directly on the buttons */
    var buttons = document.querySelectorAll("[data-atlas-state]");
    Array.prototype.forEach.call(buttons, function (b) {
      b.addEventListener("click", function () {
        window.setTimeout(syncActive, 0);
      });
    });
    window.addEventListener("popstate", function () {
      window.setTimeout(syncActive, 0);
    });
    syncActive();
  }

  function initTrajectory() {
    var trail = figure.querySelector(".plane-trajectory");
    if (!trail || typeof trail.getTotalLength !== "function") return;
    var len = trail.getTotalLength();
    if (reduced || typeof IntersectionObserver === "undefined") return;

    trail.style.strokeDasharray = String(len);
    trail.style.strokeDashoffset = String(len);

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        trail.style.transition = "stroke-dashoffset 2400ms cubic-bezier(0.3, 0.6, 0.2, 1)";
        trail.style.strokeDashoffset = "0";
        io.disconnect();
      });
    }, { threshold: 0.35 });
    io.observe(figure);
  }

  function init() {
    initQuadrants();
    initTrajectory();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
