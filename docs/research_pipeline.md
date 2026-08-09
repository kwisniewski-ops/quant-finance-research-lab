# Regime Research Pipeline

This document registers the next empirical program before data selection, feature
screening, or model comparison. Every item is **proposed**. None is a finding,
present-state classification, forecast, probability estimate, or investment
recommendation.

The public scenario framework is
[`regime-atlas.html`](../app/frontend/regime-atlas.html). The atlas organizes
questions. A result enters the research record only after it passes the gates below.

## Shared acceptance gate

A regime study is publishable as empirical evidence only when it has:

1. a named decision and a falsifiable claim;
2. a baseline defined before model comparison;
3. source, series identifier, geography, observation date, release vintage,
   transformation, revision policy, and dataset hash;
4. a classifier or conditioning rule declared before the evaluation window;
5. chronological holdouts or another design that prevents look-ahead;
6. uncertainty around estimates and state assignments;
7. multiplicity control for screened variables and specifications;
8. at least one competing explanation and a test that can distinguish it;
9. sensitivity to windows, frequency, geography, and revised versus real-time data;
10. a rejection or revision rule that remains visible when the claim fails.

Historical chronology alone cannot identify a cause. No composite “institutional
credibility” score is permitted unless its components, weights, uncertainty, and
falsification behavior are independently defensible.

## Program A — Discount-rate transmission and valuation

**Question.** When does discount-rate repricing dominate improving cash flows?

**Decision.** Whether an equity or long-duration allocation is robust to a change
in the mix of earnings news, inflation compensation, real rates, and risk premia.

**Baseline.** A chronological valuation-sensitivity model using market and yield
inputs without regime labels.

**Candidate evidence.** Real-time activity and inflation releases; forecast
surprises; expectation measures; real and nominal yields; term-premium estimates;
valuation-duration measures; earnings revisions; and dated policy decisions.

**Competing explanations.** Risk-premium changes, earnings news, market
concentration, positioning, and liquidity can produce the same price outcome.

**Falsification and publication gate.** A pre-declared conditioning rule must
improve decision-relevant error on an untouched chronological holdout after
uncertainty and multiple comparisons. Otherwise the regime claim is rejected and
the baseline remains preferred.

## Program B — Liquidity and risk-model failure

**Question.** Do liquidity measures explain risk-forecast failures beyond
volatility filtering?

**Decision.** Whether market-risk monitoring needs explicit funding and
market-function indicators in addition to return volatility.

**Baseline.** The existing rejected 500-observation historical VaR model and a
pre-specified volatility-filtered replacement.

**Candidate evidence.** Bid–ask and depth proxies; secured and unsecured funding
spreads; dealer or intermediary capacity where available; Office of Financial
Research stress measures; New York Fed market data; volatility states; and the
existing breach sequence.

**Competing explanations.** Breach clusters may follow volatility persistence,
heavy tails, jump risk, or an inadequate quantile estimator without an incremental
liquidity channel.

**Falsification and publication gate.** Coverage and independence must improve on
untouched data, the selected measures must add information beyond volatility, and
the result must survive omission of any one crisis window. A post hoc proxy menu
does not qualify.

## Program C — Policy constraints and stock–bond correlation

**Question.** Can observable policy constraints improve forecasts of stock–bond
correlation sign?

**Decision.** Whether a nominal-duration sleeve deserves to be modeled as a stable
equity hedge or as conditional protection against a named shock.

**Baseline.** Rolling historical correlation with no macroeconomic classifier.

**Candidate evidence.** Inflation and growth surprises; policy-rate expectations;
inflation expectations and risk compensation; real yields; term-premium estimates;
sovereign debt supply; central-bank balance-sheet direction; communication; and
duration-consistent stock and bond total returns.

**Competing explanations.** Risk premia, bond duration, positioning, fiscal news,
sampling frequency, and rolling-window choice can move the observed correlation
without the proposed policy mechanism.

**Falsification and publication gate.** The classifier must be frozen before the
holdout, outperform the historical baseline on out-of-sample sign or loss criteria,
and report uncertainty around each assignment. Failure leaves the current atlas
unclassified.

## Program D — Institutions, currency mismatch, and capital flows

**Question.** When do institutional constraints alter currency and funding
transmission?

**Decision.** Whether external funding, exchange-rate, and liquidity scenarios
adequately represent a country's balance-sheet and policy architecture.

**Baseline.** Country-specific market and balance-sheet variables without a
composite institutional score.

**Candidate evidence.** Foreign-currency liabilities; maturity structure; reserves;
cross-border banking and portfolio flows; exchange-rate arrangements; bank and
corporate leverage; legal authority; conditional facilities; and dated domestic and
multilateral decision records.

**Competing explanations.** Global risk appetite, commodity exposure, private
leverage, external demand, political events, and data quality can produce similar
price and flow outcomes.

**Falsification and publication gate.** Country heterogeneity and real-time data
availability must remain explicit. Any institutional variable must add stable,
out-of-sample information after market and balance-sheet baselines. A cross-country
narrative without a declared comparison does not qualify.

## Evidence status

| Program | Status | New empirical estimate | Present classifier | Forecast |
|---|---|---:|---:|---:|
| Discount-rate transmission | Proposed | No | No | No |
| Liquidity and risk-model failure | Proposed | No | No | No |
| Policy constraints and stock–bond correlation | Proposed | No | No | No |
| Institutions, currency mismatch, and capital flows | Proposed | No | No | No |

Last registered: 2026-08-09.
