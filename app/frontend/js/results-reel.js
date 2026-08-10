/* ============================================================
   Results reel — center stage.
   The current study sits centered with its neighbors peeking,
   dimmed, at both edges. Arrows flank the stage at mid-height;
   a dash rail below shows one dash per study. Clicking a dimmed
   neighbor brings it to center instead of following its link.
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

    root.classList.add("reel-ready");

    /* stage wrapper so the arrows can overlay the scroller */
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
      card.setAttribute("aria-label", "Result " + (i + 1) + " of " + cards.length);
    });

    var index = 0;
    var pending = null;

    function centerOf(card) {
      return card.offsetLeft + card.offsetWidth / 2;
    }

    function goTo(i, behavior) {
      var target = Math.max(0, Math.min(cards.length - 1, i));
      pending = target;
      scroller.scrollTo({
        left: centerOf(cards[target]) - scroller.clientWidth / 2,
        behavior: behavior || (reduced ? "auto" : "smooth")
      });
    }

    function step(dir) {
      var from = pending === null ? index : pending;
      goTo(from + dir);
    }

    /* flanking arrows */
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
    var prev = makeFlank(-1, "Previous result", "←", "reel-flank--prev");
    var next = makeFlank(1, "Next result", "→", "reel-flank--next");

    /* dash rail + counter */
    var rail = document.createElement("div");
    rail.className = "reel-rail";
    var dashes = cards.map(function (card, i) {
      var d = document.createElement("button");
      d.type = "button";
      d.className = "reel-dash";
      var title = card.querySelector("h3");
      d.setAttribute("aria-label", "Go to result " + (i + 1) + (title ? ": " + title.textContent : ""));
      d.addEventListener("click", function () { goTo(i); });
      rail.appendChild(d);
      return d;
    });
    var counter = document.createElement("span");
    counter.className = "reel-counter";
    counter.setAttribute("aria-hidden", "true");
    rail.appendChild(counter);
    stage.parentNode.insertBefore(rail, stage.nextSibling);

    /* a click on a dimmed neighbor centers it rather than leaving the page */
    cards.forEach(function (card, i) {
      card.addEventListener("click", function (e) {
        if (i !== index) {
          e.preventDefault();
          goTo(i);
        }
      });
    });

    function sync() {
      var mid = scroller.scrollLeft + scroller.clientWidth / 2;
      var best = 0;
      var bestDist = Infinity;
      cards.forEach(function (card, i) {
        var dist = Math.abs(centerOf(card) - mid);
        if (dist < bestDist) { bestDist = dist; best = i; }
        card.classList.toggle("is-away", dist > card.offsetWidth * 0.25);
      });
      if (pending === best) pending = null;
      if (best !== index || !counter.textContent) {
        index = best;
        counter.textContent =
          (index + 1 < 10 ? "0" : "") + (index + 1) + " / " + (cards.length < 10 ? "0" : "") + cards.length;
        dashes.forEach(function (d, i) {
          if (i === index) d.setAttribute("aria-current", "true");
          else d.removeAttribute("aria-current");
        });
        prev.disabled = index === 0;
        next.disabled = index === cards.length - 1;
      }
    }

    var ticking = false;
    scroller.addEventListener("scroll", function () {
      if (ticking) return;
      ticking = true;
      /* rAF for visible tabs; the timer covers hidden/throttled tabs where
         rAF never fires (whichever runs first does the work) */
      var settle = function () {
        if (!ticking) return;
        ticking = false;
        sync();
      };
      window.requestAnimationFrame(settle);
      window.setTimeout(settle, 120);
    }, { passive: true });

    scroller.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    });

    window.addEventListener("resize", function () {
      goTo(index, "auto");
    });

    sync();
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-reel]"), initReel);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
