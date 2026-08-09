"use strict";

(function initRegimeAtlas() {
  const workbench = document.querySelector(".atlas-workbench");
  const stateButtons = Array.from(document.querySelectorAll("[data-atlas-state]"));
  const panels = Array.from(document.querySelectorAll("[data-atlas-panel]"));
  const liquiditySelect = document.querySelector('[data-atlas-overlay="liquidity"]');
  const responseSelect = document.querySelector('[data-atlas-overlay="response"]');
  const liquidityOutput = document.querySelector("[data-atlas-liquidity]");
  const responseOutput = document.querySelector("[data-atlas-response]");
  const liveStatus = document.getElementById("atlas-live-status");

  if (!workbench || !stateButtons.length || !panels.length || !liquiditySelect ||
      !responseSelect || !liquidityOutput || !responseOutput || !liveStatus) return;

  const validStates = new Set(stateButtons.map((button) => button.dataset.atlasState));
  const scenarioHashPrefix = "#scenario-";
  function stateFromHash() {
    if (!window.location.hash.startsWith(scenarioHashPrefix)) return "unclassified";
    const candidate = window.location.hash.slice(scenarioHashPrefix.length);
    return validStates.has(candidate) ? candidate : "unclassified";
  }

  let selectedState = stateFromHash();

  function overlayDefinition(dimension, state) {
    const cell = document.querySelector(
      `[data-atlas-definition="${dimension}:${state}"] [data-atlas-definition-text]`
    );
    return cell ? cell.textContent.trim() : "Definition unavailable.";
  }

  function renderState(announce) {
    let selectedTitle = "Base state left unclassified";
    stateButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.atlasState === selectedState));
    });
    panels.forEach((panel) => {
      const selected = panel.dataset.atlasPanel === selectedState;
      panel.hidden = !selected;
      if (selected) selectedTitle = panel.querySelector("h3").textContent;
    });
    if (announce) liveStatus.textContent = `Scenario shown: ${selectedTitle}.`;
  }

  function renderOverlays(announce) {
    liquidityOutput.textContent = overlayDefinition("liquidity", liquiditySelect.value);
    responseOutput.textContent = overlayDefinition("response", responseSelect.value);
    if (announce) {
      const liquidityLabel = liquiditySelect.options[liquiditySelect.selectedIndex].text;
      const responseLabel = responseSelect.options[responseSelect.selectedIndex].text;
      liveStatus.textContent = `Overlays updated. Market liquidity: ${liquidityLabel}. Institutional response: ${responseLabel}.`;
    }
  }

  workbench.classList.add("atlas-workbench--enhanced");
  stateButtons.forEach((button) => { button.disabled = false; });
  liquiditySelect.disabled = false;
  responseSelect.disabled = false;
  stateButtons.forEach((button) => {
    button.addEventListener("click", () => {
      selectedState = button.dataset.atlasState;
      const nextHash = `#scenario-${selectedState}`;
      if (window.location.hash !== nextHash) {
        history.pushState({ atlasState: selectedState }, "", nextHash);
      }
      renderState(true);
    });
  });
  liquiditySelect.addEventListener("change", () => renderOverlays(true));
  responseSelect.addEventListener("change", () => renderOverlays(true));
  window.addEventListener("popstate", (event) => {
    if (event.state && validStates.has(event.state.atlasState)) {
      selectedState = event.state.atlasState;
    } else if (window.location.hash.startsWith(scenarioHashPrefix)) {
      selectedState = stateFromHash();
    } else if (!window.location.hash) {
      selectedState = "unclassified";
    } else {
      return;
    }
    renderState(false);
  });
  window.addEventListener("hashchange", () => {
    if (!window.location.hash.startsWith(scenarioHashPrefix)) return;
    selectedState = stateFromHash();
    renderState(false);
  });

  renderState(false);
  renderOverlays(false);
})();
