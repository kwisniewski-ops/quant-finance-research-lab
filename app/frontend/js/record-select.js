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
    if (entries.length < 2) return;

    ledger.classList.add("record-focus");

    /* the index is the no-JS fallback; the tabs replace it visually */
    index.hidden = true;

    var showingAll = false;

    var tabs = document.createElement("nav");
    tabs.className = "record-tabs";
    tabs.setAttribute("aria-label", "Choose an investigation");
    var tabButtons = entries.map(function (entry, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "record-tab";
      var no = i + 1 < 10 ? "0" + (i + 1) : String(i + 1);
      var title = entry.querySelector("h2");
      var status = entry.querySelector(".ledger-entry-head .status");
      b.innerHTML = '<span class="tab-kicker">Investigation</span><span class="tab-no">' + no + "</span>";
      if (status) {
        var chip = document.createElement("span");
        chip.className = "tab-status " + status.className.replace(/\bstatus\b/, "").trim();
        chip.textContent = status.textContent;
        b.appendChild(chip);
      }
      if (title) b.setAttribute("title", title.textContent.trim());
      b.setAttribute("aria-label", "Investigation " + (i + 1) + (title ? ": " + title.textContent.trim() : ""));
      b.addEventListener("click", function () {
        select(entry, { push: true, scroll: "smooth" });
      });
      tabs.appendChild(b);
      return b;
    });
    index.parentNode.insertBefore(tabs, index);

    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "record-toggle";
    toggle.textContent = "View the complete ledger";
    tabs.parentNode.insertBefore(toggle, index);

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
      entries.forEach(function (entry, i) {
        if (!showingAll && entry.id === activeId) tabButtons[i].setAttribute("aria-current", "true");
        else tabButtons[i].removeAttribute("aria-current");
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
