/* Shared, privacy-preserving engagement event contract for the static lab. */
(function () {
  "use strict";

  if (typeof document === "undefined" || typeof window.CustomEvent !== "function") return;

  function emit(eventName, target) {
    window.dispatchEvent(new CustomEvent("kw:engagement", {
      detail: {
        event: eventName,
        target: target,
        source: "quantitative-markets-research-lab",
        path: window.location.pathname
      }
    }));
  }

  function classifyLink(link) {
    var href = link.getAttribute("href") || "";
    var explicitEvent = link.dataset.analyticsEvent;
    var explicitTarget = link.dataset.analyticsTarget;
    if (explicitEvent) return [explicitEvent, explicitTarget || href || "unspecified"];
    if (/\/contact(?:$|[/?#])|^mailto:/i.test(href)) return ["contact_action", explicitTarget || "lab-contact"];
    if (/kylewisniewski\.com\/projects\/quantitative-markets-research-lab/i.test(href)) {
      return ["featured_project_engagement", explicitTarget || "quant-lab-case-study"];
    }
    if (/github\.com\/kwisniewski-ops\/quant-finance-research-lab/i.test(href)) {
      return ["research_engagement", explicitTarget || "quant-lab-source"];
    }
    if (/finding-[^/]+\.html/i.test(href)) return ["research_engagement", explicitTarget || href.split("#")[0]];
    if (/^(?:asset-pricing|portfolio|risk|simulations|factors)\.html/i.test(href)) {
      return ["research_engagement", explicitTarget || "tool:" + href.split(/[?#]/)[0]];
    }
    return ["research_engagement", explicitTarget || href || "lab-navigation"];
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    var link = target && target.closest ? target.closest("a[href]") : null;
    if (!link) return;
    var classification = classifyLink(link);
    emit(classification[0], classification[1]);
  });

  var toolInteractionEmitted = false;
  document.addEventListener("change", function (event) {
    if (toolInteractionEmitted) return;
    var control = event.target;
    if (!control || !control.matches || !control.matches("main input, main select")) return;
    toolInteractionEmitted = true;
    emit("research_engagement", "tool-interaction:" + window.location.pathname.split("/").pop());
  });
})();
