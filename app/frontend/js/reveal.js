/* ============================================================
   Scroll choreography — progressive enhancement only.
   1. Elements marked data-reveal fade/rise in as they enter the
      viewport (JS adds .reveal first, so nothing hides without JS).
   2. On finding pages the depth nav tracks the section in view.
   Honors prefers-reduced-motion.
   ============================================================ */

(function () {
  "use strict";

  if (typeof document === "undefined") return;

  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var hasIO = typeof IntersectionObserver !== "undefined";

  function initReveal() {
    var nodes = document.querySelectorAll("[data-reveal]");
    /* hidden tabs may never fire the observer — leave content visible */
    if (!nodes.length || !hasIO || reduced || document.hidden) return;

    var parents = [];
    var counts = [];
    Array.prototype.forEach.call(nodes, function (el) {
      var idx = parents.indexOf(el.parentNode);
      if (idx === -1) { parents.push(el.parentNode); counts.push(0); idx = parents.length - 1; }
      el.style.setProperty("--reveal-delay", Math.min(counts[idx] * 70, 420) + "ms");
      counts[idx]++;
      el.classList.add("reveal");
    });

    var pending = Array.prototype.slice.call(nodes);

    function show(el) {
      el.classList.add("is-in");
      var at = pending.indexOf(el);
      if (at !== -1) pending.splice(at, 1);
      io.unobserve(el);
    }

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) show(entry.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });

    pending.slice().forEach(function (el) { io.observe(el); });

    /* belt and braces: reveal anything on screen even if the observer
       never fires (prerender, bfcache, throttled background tabs) */
    function sweep() {
      if (!pending.length) return;
      var vh = window.innerHeight || document.documentElement.clientHeight;
      pending.slice().forEach(function (el) {
        var r = el.getBoundingClientRect();
        if (r.top < vh * 0.96 && r.bottom > 0) show(el);
      });
    }
    sweep();
    window.addEventListener("scroll", sweep, { passive: true });
    window.addEventListener("resize", sweep);
    document.addEventListener("visibilitychange", sweep);
  }

  function initDepthRail() {
    var nav = document.querySelector(".depth-nav");
    if (!nav || !hasIO) return;
    var links = nav.querySelectorAll('a[href^="#"]');
    if (!links.length) return;

    var map = {};
    var sections = [];
    Array.prototype.forEach.call(links, function (a) {
      var id = a.getAttribute("href").slice(1);
      var target = document.getElementById(id);
      if (target) {
        var section = target.closest("section") || target;
        map[section.id || id] = a;
        if (!section.id) section.id = id;
        sections.push(section);
      }
    });
    if (!sections.length) return;

    function setCurrent(link) {
      Array.prototype.forEach.call(links, function (a) {
        a.classList.toggle("is-current", a === link);
        if (a === link) a.setAttribute("aria-current", "true");
        else a.removeAttribute("aria-current");
      });
    }

    var io = new IntersectionObserver(function (entries) {
      var best = null;
      entries.forEach(function (entry) {
        if (entry.isIntersecting) best = entry.target;
      });
      if (best && map[best.id]) setCurrent(map[best.id]);
    }, { rootMargin: "-18% 0px -58% 0px", threshold: 0 });

    sections.forEach(function (s) { io.observe(s); });
  }

  function init() {
    initReveal();
    initDepthRail();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
