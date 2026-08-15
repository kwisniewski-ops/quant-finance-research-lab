/* ============================================================
   Results reel — center stage, circular.
   The current study sits centered with neighbors peeking at both
   edges. The reel loops: a clone of the last study sits before the
   first and a clone of the first sits after the last; when the
   scroller settles on a clone it teleports invisibly to the real
   slide, so both directions scroll forever. Arrows flank the
   stage; a dash rail below maps the set. Clicking a dimmed
   neighbor centers it instead of following its link.
   Progressive enhancement: without JS the cards simply stack.
   Honors reduced motion (no scale, instant jumps).
   ============================================================ */

(function () {
  "use strict";

  if (typeof document === "undefined") return;

  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function initReel(root) {
    var scroller = root.querySelector(".finding-card-grid");
    if (!scroller) return;
    var cards = Array.prototype.slice.call(scroller.children);
    if (cards.length < 2) return;
    var N = cards.length;

    root.classList.add("reel-ready");

    var stage = document.createElement("div");
    stage.className = "reel-stage";
    scroller.parentNode.insertBefore(stage, scroller);
    stage.appendChild(scroller);

    scroller.setAttribute("tabindex", "0");
    scroller.setAttribute("aria-roledescription", "carousel");
    scroller.setAttribute("aria-label", "Published results, one at a time");

    cards.forEach(function (card, i) {
      card.classList.remove("reveal", "is-in");
      card.removeAttribute("data-reveal");
      card.setAttribute("aria-roledescription", "slide");
      card.removeAttribute("aria-label");
      var position = document.createElement("span");
      position.className = "sr-only";
      position.textContent = "Result " + (i + 1) + " of " + N + ". ";
      card.insertBefore(position, card.firstChild);
    });

    /* circular seam: clone of the last study before the first,
       clone of the first after the last */
    function makeClone(source) {
      var c = source.cloneNode(true);
      c.setAttribute("aria-hidden", "true");
      c.setAttribute("data-reel-clone", "");
      c.removeAttribute("aria-label");
      if (c.matches("a, button, input, select, textarea, [tabindex]")) {
        c.setAttribute("tabindex", "-1");
      }
      Array.prototype.forEach.call(c.querySelectorAll("a, button"), function (el) {
        el.setAttribute("tabindex", "-1");
      });
      return c;
    }
    var headClone = makeClone(cards[N - 1]);
    var tailClone = makeClone(cards[0]);
    scroller.insertBefore(headClone, cards[0]);
    scroller.appendChild(tailClone);

    /* positions run 0..N+1; position p shows real study (p-1+N)%N */
    var slides = [headClone].concat(cards, [tailClone]);
    function realOf(pos) { return (pos - 1 + N) % N; }

    var pos = 1;          /* current position (may briefly be a clone) */
    var pending = null;
    var dragging = false;
    var settleTimer = 0;

    function centerOf(slide) { return slide.offsetLeft + slide.offsetWidth / 2; }

    function goToPos(p, behavior) {
      var target = Math.max(0, Math.min(slides.length - 1, p));
      pending = target;
      scroller.scrollTo({
        left: centerOf(slides[target]) - scroller.clientWidth / 2,
        behavior: behavior || (reduced ? "auto" : "smooth")
      });
    }

    function step(dir) {
      var from = pending === null ? pos : pending;
      goToPos(from + dir);
    }

    function makeFlank(dir, label, glyph, cls) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "reel-flank " + cls;
      b.setAttribute("aria-label", label);
      b.textContent = glyph;
      b.addEventListener("click", function () { step(dir); });
      stage.appendChild(b);
      return b;
    }
    makeFlank(-1, "Previous result", "←", "reel-flank--prev");
    makeFlank(1, "Next result", "→", "reel-flank--next");

    var rail = document.createElement("div");
    rail.className = "reel-rail";
    var dashes = cards.map(function (card, i) {
      var d = document.createElement("button");
      d.type = "button";
      d.className = "reel-dash";
      var title = card.querySelector("h3");
      d.setAttribute("aria-label", "Go to result " + (i + 1) + (title ? ": " + title.textContent : ""));
      d.addEventListener("click", function () { goToPos(i + 1); });
      rail.appendChild(d);
      return d;
    });
    var counter = document.createElement("span");
    counter.className = "reel-counter";
    counter.setAttribute("aria-hidden", "true");
    rail.appendChild(counter);
    stage.parentNode.insertBefore(rail, stage.nextSibling);

    /* clicking anything not centered brings it to center; clones
       always redirect the click into a recenter */
    slides.forEach(function (slide, p) {
      slide.addEventListener("click", function (e) {
        var isClone = slide.hasAttribute("data-reel-clone");
        if (p !== pos || isClone) {
          e.preventDefault();
          goToPos(p);
        }
      });
    });

    /* never teleport mid-gesture */
    ["pointerdown", "touchstart"].forEach(function (ev) {
      scroller.addEventListener(ev, function () { dragging = true; }, { passive: true });
    });
    ["pointerup", "pointercancel", "touchend", "touchcancel"].forEach(function (ev) {
      window.addEventListener(ev, function () {
        dragging = false;
        window.clearTimeout(settleTimer);
        settleTimer = window.setTimeout(onSettle, 160);
      }, { passive: true });
    });

    function paint(best) {
      var real = realOf(best);
      counter.textContent =
        (real + 1 < 10 ? "0" : "") + (real + 1) + " / " + (N < 10 ? "0" : "") + N;
      dashes.forEach(function (d, i) {
        if (i === real) d.setAttribute("aria-current", "true");
        else d.removeAttribute("aria-current");
      });
    }

    function sync() {
      var mid = scroller.scrollLeft + scroller.clientWidth / 2;
      var best = 0;
      var bestDist = Infinity;
      slides.forEach(function (slide, p) {
        var dist = Math.abs(centerOf(slide) - mid);
        if (dist < bestDist) { bestDist = dist; best = p; }
        slide.classList.toggle("is-away", dist > slide.offsetWidth * 0.25);
      });
      if (pending === best) pending = null;
      if (best !== pos || !counter.textContent) {
        pos = best;
        paint(best);
      }
    }

    /* seam repair: once scrolling truly stops on a clone, jump to its
       real twin in the same visual position */
    function jumpTo(p) {
      pos = p;
      pending = null;
      scroller.scrollLeft = centerOf(slides[p]) - scroller.clientWidth / 2;
      paint(p);
    }

    function onSettle() {
      if (dragging) return;
      sync();
      if (pos === 0) jumpTo(N);
      else if (pos === slides.length - 1) jumpTo(1);
    }

    var ticking = false;
    scroller.addEventListener("scroll", function () {
      if (!ticking) {
        ticking = true;
        var settle = function () {
          if (!ticking) return;
          ticking = false;
          sync();
        };
        window.requestAnimationFrame(settle);
        window.setTimeout(settle, 120);
      }
      window.clearTimeout(settleTimer);
      settleTimer = window.setTimeout(onSettle, 110);
    }, { passive: true });

    scroller.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    });

    window.addEventListener("resize", function () {
      goToPos(pos, "auto");
    });

    goToPos(1, "auto");
    sync();
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-reel]"), initReel);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
