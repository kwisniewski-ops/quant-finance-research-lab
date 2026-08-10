/* ============================================================
   Results reel — one finding at a time.
   Progressive enhancement over the stacked card list: with JS the
   list becomes a scroll-snap reel with arrows, a counter, keyboard
   support, and native swipe. Without JS the cards simply stack.
   Honors reduced motion (instant jumps, no smooth scrolling).
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
    scroller.setAttribute("tabindex", "0");
    scroller.setAttribute("aria-roledescription", "carousel");
    scroller.setAttribute("aria-label", "Published results, one at a time");

    cards.forEach(function (card, i) {
      /* slides manage their own entrance; the global reveal would hide
         off-axis cards indefinitely */
      card.classList.remove("reveal", "is-in");
      card.removeAttribute("data-reveal");
      card.setAttribute("aria-roledescription", "slide");
      card.setAttribute("aria-label", "Result " + (i + 1) + " of " + cards.length);
    });

    /* controls live beside the section heading */
    var headingRow = root.querySelector(".section-heading-row") ||
                     root.querySelector("h2") && root.querySelector("h2").parentNode;
    var controls = document.createElement("div");
    controls.className = "reel-controls";

    var counter = document.createElement("span");
    counter.className = "reel-counter";
    counter.setAttribute("aria-hidden", "true");

    function makeButton(dir, label, glyph) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "reel-arrow";
      b.setAttribute("aria-label", label);
      b.textContent = glyph;
      b.addEventListener("click", function () { step(dir); });
      return b;
    }

    var prev = makeButton(-1, "Previous result", "←");
    var next = makeButton(1, "Next result", "→");
    controls.appendChild(counter);
    controls.appendChild(prev);
    controls.appendChild(next);
    if (headingRow) headingRow.appendChild(controls);

    var index = 0;
    var pending = null;   /* target while a smooth scroll is in flight */

    function widthOf() { return scroller.clientWidth; }

    function step(dir) {
      var from = pending === null ? index : pending;
      var target = Math.max(0, Math.min(cards.length - 1, from + dir));
      pending = target;
      scroller.scrollTo({
        left: target * widthOf(),
        behavior: reduced ? "auto" : "smooth"
      });
    }

    function sync() {
      var i = Math.round(scroller.scrollLeft / Math.max(1, widthOf()));
      i = Math.max(0, Math.min(cards.length - 1, i));
      if (pending === i) pending = null;
      if (i === index && counter.textContent) return;
      index = i;
      counter.textContent =
        (i + 1 < 10 ? "0" : "") + (i + 1) + " / " + (cards.length < 10 ? "0" : "") + cards.length;
      prev.disabled = i === 0;
      next.disabled = i === cards.length - 1;
    }

    var ticking = false;
    scroller.addEventListener("scroll", function () {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () { ticking = false; sync(); });
    }, { passive: true });

    scroller.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    });

    window.addEventListener("resize", function () {
      /* re-align the current slide after a resize reflows widths */
      scroller.scrollTo({ left: index * widthOf(), behavior: "auto" });
    });

    sync();
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-reel]"), initReel);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
