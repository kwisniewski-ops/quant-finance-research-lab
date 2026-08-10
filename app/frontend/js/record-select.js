/* ============================================================
   Research record — one investigation at a time.
   The ledger index becomes the menu: choosing an entry shows only
   that record. Deep links (research-log.html#market-risk) and the
   browser's back button select the right record on arrival. A
   toggle restores the complete ledger. Without JS the full ledger
   renders exactly as before.
   ============================================================ */

(function () {
  "use strict";

  if (typeof document === "undefined") return;

  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function init() {
    var index = document.querySelector(".ledger-index");
    var ledger = document.querySelector(".ledger");
    if (!index || !ledger) return;
    var entries = Array.prototype.slice.call(ledger.querySelectorAll(".ledger-entry[id]"));
    var links = Array.prototype.slice.call(index.querySelectorAll('a[href^="#"]'));
    if (entries.length < 2 || !links.length) return;

    ledger.classList.add("record-focus");
    index.classList.add("record-menu");

    var showingAll = false;

    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "record-toggle";
    toggle.textContent = "View the complete ledger";
    index.appendChild(toggle);

    function entryFor(hash) {
      if (!hash) return null;
      var id = hash.charAt(0) === "#" ? hash.slice(1) : hash;
      if (!id) return null;
      var el = document.getElementById(id);
      if (!el) return null;
      return el.closest(".ledger-entry") || null;
    }

    function reveal(entry) {
      entry.hidden = false;
      if (entry.classList.contains("reveal")) entry.classList.add("is-in");
    }

    function paint(activeId) {
      links.forEach(function (a) {
        if (!showingAll && a.getAttribute("href") === "#" + activeId) a.setAttribute("aria-current", "true");
        else a.removeAttribute("aria-current");
      });
      toggle.textContent = showingAll ? "View one record at a time" : "View the complete ledger";
      toggle.setAttribute("aria-pressed", showingAll ? "true" : "false");
    }

    function showAll() {
      showingAll = true;
      entries.forEach(reveal);
      paint(null);
    }

    function select(entry, opts) {
      opts = opts || {};
      showingAll = false;
      entries.forEach(function (e) { e.hidden = e !== entry; });
      reveal(entry);
      paint(entry.id);
      if (opts.push && window.history && history.pushState) {
        history.pushState(null, "", "#" + entry.id);
      }
      if (opts.scroll) {
        var top = ledger.getBoundingClientRect().top + window.pageYOffset - 90;
        window.scrollTo({ top: top, behavior: opts.scroll === "smooth" && !reduced ? "smooth" : "auto" });
      }
    }

    links.forEach(function (a) {
      a.addEventListener("click", function (e) {
        var entry = entryFor(a.getAttribute("href"));
        if (!entry) return;
        e.preventDefault();
        select(entry, { push: true, scroll: "smooth" });
      });
    });

    toggle.addEventListener("click", function () {
      if (showingAll) select(entries[0], { push: false });
      else showAll();
    });

    function resolveLocation(scroll) {
      var entry = entryFor(window.location.hash);
      if (entry) {
        select(entry, { scroll: scroll });
        return;
      }
      select(entries[0], {});
      /* anchors outside the ledger (pipeline, protocol) stay reachable:
         re-anchor after the layout shrinks */
      if (window.location.hash) {
        var target = document.getElementById(window.location.hash.slice(1));
        if (target && !target.closest(".ledger-entry")) target.scrollIntoView();
      }
    }

    window.addEventListener("hashchange", function () { resolveLocation("auto"); });
    window.addEventListener("popstate", function () { resolveLocation(false); });

    resolveLocation(window.location.hash ? "auto" : false);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
