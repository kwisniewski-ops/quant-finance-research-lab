/* ============================================================
   Long-form reading tools: citation copying and compact TOCs.
   ============================================================ */
(function () {
  "use strict";

  if (typeof document === "undefined") return;

  function $(id) { return document.getElementById(id); }

  function fallbackCopy(text) {
    var input = document.createElement("textarea");
    input.value = text;
    input.setAttribute("readonly", "");
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.appendChild(input);
    input.select();
    var copied = document.execCommand("copy");
    document.body.removeChild(input);
    return copied ? Promise.resolve() : Promise.reject(new Error("Copy failed"));
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return fallbackCopy(text);
  }

  function citationFor(button) {
    var title = button.getAttribute("data-cite-title");
    var year = button.getAttribute("data-cite-year") || "2026";
    var target = button.getAttribute("data-cite-target");
    var url = window.location.href.split("#")[0] + (target ? "#" + target : "");
    return 'Wisniewski, Kyle. "' + title + '." Quantitative Markets Research Lab, ' + year + ". " + url;
  }

  function announce(message) {
    var status = $("reading-tools-status");
    if (status) status.textContent = message;
  }

  function initCitations() {
    Array.prototype.forEach.call(document.querySelectorAll(".cite-button"), function (button) {
      button.addEventListener("click", function () {
        var original = button.textContent;
        copyText(citationFor(button)).then(function () {
          button.textContent = "Citation copied";
          announce("Citation copied to clipboard.");
          window.setTimeout(function () { button.textContent = original; }, 1800);
        }).catch(function () {
          announce("Citation could not be copied. Copy the page address and section title manually.");
        });
      });
    });
  }

  function initContents() {
    Array.prototype.forEach.call(document.querySelectorAll(".local-toc a"), function (link) {
      link.addEventListener("click", function () {
        var contents = link.closest("details");
        if (contents && window.matchMedia("(max-width: 760px)").matches) contents.open = false;
      });
    });
  }

  function init() {
    initCitations();
    initContents();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
