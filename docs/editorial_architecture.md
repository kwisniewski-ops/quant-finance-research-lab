# Editorial Architecture

This document governs how the Quantitative Markets & Institutions Lab presents research.
It protects technical precision while allowing a conclusion to be understood before the
reader reaches the complete empirical and mathematical record.

## Institutional proposition

The lab publishes independent, evidence-led research on financial models, market regimes,
institutions, and investor decisions. It examines how quantitative conclusions change when
estimates leave sample, correlations reverse, liquidity deteriorates, and institutional
constraints alter the transmission of shocks through markets.

Accessibility is achieved through ordered disclosure, not diluted language. Technical depth
reflects the evidentiary requirements of the claim.

## Primary information architecture

The global navigation is fixed to five collections:

1. **Overview** — mission, current corpus, and registered direction.
2. **Research** — executed investigations, syntheses, historical frameworks, and proposed programs.
3. **Models** — interactive implementations for assumption sensitivity and model behavior.
4. **Methods** — research design, validation, evidence classification, and reproducibility.
5. **Record** — versioned claims, evidence, failures, revisions, artifacts, and proposed studies.

Research is organized by analytical domain:

- Portfolio construction and diversification
- Market risk and liquidity
- Asset pricing, volatility, and factors
- Market regimes and institutional transmission

## Artifact taxonomy

Every public artifact declares its type. Approved types are:

- Empirical finding
- Numerical validation
- Empirical synthesis
- Historical framework
- Interactive model
- Technical note
- Proposed research

Artifact type must not imply an evidentiary status the work has not earned. A framework is not
an empirical investigation; an internal numerical check is not an independent replication; a
registered proposal is not a result.

## Title and voice standard

Published artifacts use declarative titles that name the model, mechanism, evidence, or inference
problem. Questions are reserved for falsifiable research design, competing explanations, and
registered protocols.

The public voice is institutional, impersonal, scientific, and precise. It does not use personal
pronouns, personal names, promotional self-assertion, or generalized content-marketing language.
Terms such as “plain language,” “simple first,” “short answer,” “start with a question,” and
“continue exploring” do not describe the lab's research architecture.

## Publication architecture

Each executed investigation follows five progressively deeper layers:

1. **Conclusion** — the bounded result and its decision relevance.
2. **Evidence** — data, estimates, validation tests, and uncertainty.
3. **Decision consequences** — implications for allocation, pricing, risk governance, or model use.
4. **Identification and limits** — alternative explanations, unsupported claims, and failure boundaries.
5. **Methods and reproduction** — mathematics, data vintage, source commit, executable artifact, and record.

The opening conclusion must remain scientifically bounded. Later layers add scrutiny and detail;
they do not repair an overclaimed headline.

## Evidence classification

Public claims use the following labels consistently:

- **Lab measurement** — a value generated from declared data, transformations, estimators, and uncertainty procedures.
- **Numerical validation** — agreement with an identity, limiting case, convergence rate, or independently implemented method.
- **External empirical evidence** — a measured relationship reported by an identified study or dataset outside the lab.
- **Institutional record** — a dated decision, release, legal instrument, or official account establishing chronology and stated action, not causality by itself.
- **Interpretive synthesis** — a bounded connection among measurements, records, mechanisms, and decision consequences.
- **Proposed research** — a pre-declared question, baseline, competing explanation, and publication gate without a supported result.

Scenario language must state that it is not a forecast. Historical analogy does not supply a
probability, identify a cause, or classify the present.

## Research-status vocabulary

The research record uses these statuses:

- Proposed
- Running
- Validated — numerical or theoretical verification
- Supported — an empirical claim met its declared criterion
- Revised
- Rejected
- Independently replicated — reserved for a genuinely independent repetition

## Current corpus boundary

The current executed corpus contains six investigations concentrated on model validation,
portfolio construction, factors, derivatives, simulation, and market risk. Regime and institutional
work currently comprises one historical framework, one empirical synthesis, and four registered
programs. The lab does not presently publish a live regime classifier, regime probability, or
forecast of the current market environment.

## Release gate

An editorial release is complete only when:

- primary navigation, metadata, breadcrumbs, feed titles, and page headings share the same taxonomy;
- declarative claims remain bounded by their sample and design;
- executed and proposed research cannot be mistaken for one another;
- evidence classes remain visible and distinct;
- revisions preserve frozen data, source commits, and prior failure records;
- the global-shell normalizer is idempotent; and
- automated publishing regressions pass with the quantitative test suite.
