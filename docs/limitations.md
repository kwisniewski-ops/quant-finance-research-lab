# Limitations

This is the honest page. Everything in this repository works as specified — 201
tests pass, four pricing methods agree to fractions of a cent, the backtester
provably cannot peek — and none of that makes the outputs *true*. Below are the
limitations that matter, stated specifically enough to be checked, with pointers to
where the notebooks demonstrate them on real data.

**This is research tooling for studying quantitative methods. It is not investment
advice, and nothing in this repository is fit for making investment decisions.**

## 1. Estimation error dominates optimization error

The optimizers solve their mathematical problems essentially exactly. The inputs are
where the damage lives. The standard error of an annualized mean return is
$\sigma/\sqrt{Y}$ — for an 18%-vol asset over seven years, about ±7% per year, the
same order as the quantity itself. Concretely, in this repository (notebook 02): the
max-Sharpe portfolio fitted on 2015–2021 posted an in-sample Sharpe of **1.12** and
an out-of-sample (2022–2026) Sharpe of **0.44**. That 0.7 gap is not a bug or bad
luck; it is the expected cost of optimizing on estimates. No refinement of the
optimizer can close it, because the optimizer is not where the error is. Corollary:
differences of ±0.2 in reported Sharpe ratios over a few years are *inside sampling
noise* and should never be used to rank methods.

## 2. Parameters assumed constant are not

Every model here freezes something the world insists on moving:

- Black–Scholes freezes $\sigma$; SPY's 21-day realized vol ran from **3.4% to 93%**
  in our sample (notebook 03).
- Heston freezes $(\kappa, \theta, \xi, \rho)$; desks recalibrate them daily, which
  is a running confession that they are state variables, not constants.
- Factor models freeze betas; QQQ's momentum loading swung from **−0.14 to +0.26**
  across rolling windows — it changes *sign* (notebook 05).
- Portfolio optimizers freeze $\Sigma$; the SPY–TLT correlation moved from an
  average of **−0.35** (2015–2019) to **+0.37** (2022–23), a regime break that
  invalidated the design premise of every 60/40-style allocation simultaneously
  (notebook 06).

The pattern is recursive: each model fixes the previous model's moving part and
introduces its own. There is no top to this ladder; there is only knowing which rung
you are standing on.

## 3. Normality fails exactly when it matters

Daily portfolio returns in notebook 06 show excess kurtosis around **13**. At the
95% confidence level, Gaussian VaR is nearly harmless (the four VaR estimators agree
within ~10%). At 99%, the Gaussian understates historical VaR and ES visibly — the
error is small where risk numbers are easy and large where they are consulted.
SPY's worst day in the sample is a ~10σ event under normality (probability
~$10^{-23}$; notebook 03). Cornish–Fisher patches the moments but is an expansion,
not a law, and historical estimators can only replay tails they have seen. Rule of
thumb encoded in the notebooks: any Gaussian-based number beyond the 95th percentile
should be treated as a lower bound.

## 4. Costs and liquidity are modeled crudely

`ProportionalCost` (flat bps on turnover) and `FixedSlippage` capture the first-order
drag for a small investor trading liquid ETFs at the close — the 60/40 backtest's
total cost of ~0.2% of NAV over a decade is probably realistic *for that strategy at
small size*. Everything harder is absent: market impact (costs growing with size),
bid–ask dynamics in stress (spreads widen exactly when rebalancing rules fire),
financing and shorting costs, taxes, and the impossibility of trading at the close
you observed. High-turnover strategies evaluated with this cost model will look
better than they are, roughly in proportion to their turnover. The barrier-option
and hedging discussions similarly ignore that continuous rehedging is a fiction.

## 5. Single-regime calibrations

Nearly every empirical number in this repository is estimated from 2015–2026: one
secular-growth decade with two short crashes and one inflation shock, experienced by
US-listed instruments during a historically unusual mega-cap run. The factor premia,
the covariance structure, the Heston-like parameter magnitudes, the scenario
calibrations — all are samples from this one regime. The 2022 stock–bond episode is
the in-sample proof of what that costs: 2,500 daily observations of negative
correlation provided zero warning of the sign flip, because all 2,500 came from the
regime that was about to end. Models fitted here should be expected to fail at the
next regime boundary, in a direction the fit cannot indicate.

## 6. Past scenarios do not bound future ones

The stress library replays four historical episodes (GFC, Covid, 2022 rates,
dot-com) as instantaneous shocks to static weights. This is useful and strictly
backward-looking. Three specific understatements: (i) real crises unfold over months
with feedback — forced deleveraging, correlation spirals, liquidity holes — that a
one-shot $w^\top s$ cannot represent; (ii) the scenario set is small and famous,
and famous scenarios are the ones portfolios are already braced for; (iii) nothing
in the arithmetic prevents the next episode from being worse than the worst entry
in the table — 2022 itself was outside the pre-2022 scenario library of most
practitioners. Scenario tables are a floor for imagination, not a ceiling for loss.

## 7. The universe is survivorship-tilted

The fifteen ETFs were selected in 2026, which means every one of them survived,
stayed liquid, and kept tracking its index. Funds that closed, strategies that
stopped working, and asset classes that became untradable are absent by
construction. This biases *every* backtest and estimate in the repository upward by
an unknowable amount. The same applies at one remove to the factor data: the Ken
French portfolios are paper constructions ignoring implementation costs, and the
published factor menu itself survived a publication filter (Harvey–Liu–Zhu count
hundreds of proposed factors; a handful replicate).

## 8. Statistical inference is optimistic as reported

OLS $t$-statistics in the factor notebooks assume i.i.d. errors; volatility
clustering (demonstrated in notebook 03) inflates them. Sharpe ratios are reported
without deflation for the number of strategy variants examined. Rolling statistics
induce serial correlation between adjacent windows that makes trends look smoother
than the underlying information warrants. Where a marginal result appears (QQQ's
$t = 2.3$ alpha in notebook 05), the notebooks explicitly decline to treat it as a
discovery — after honest multiple-testing accounting, it is what noise looks like.

## 9. Numerical honesty has edges too

Monte Carlo results carry reported standard errors — but discretization bias (CIR,
Heston full-truncation Euler) is *not* included in those error bars and was not
systematically budgeted by step-halving. The Hagan SABR formula degrades for long
maturities and extreme strikes and can imply negative densities in the far wings.
Cornish–Fisher quantiles can be non-monotone at extreme skew/kurtosis. The
`nearest_psd` repair changes the matrix it repairs. Each is documented at the call
site; none is eliminated.

## A checklist for reading any single number from this repository

Before quoting a result, run it through five questions:

1. **What was estimated, and what is its standard error?** If the answer involves a
   mean return, the standard error is probably comparable to the estimate.
2. **What window produced it, and what regime was that window?** A statistic from
   2015–2026 is a statistic *about* 2015–2026.
3. **Was the out-of-sample period touched more than once?** If a parameter was tuned
   after seeing the evaluation window, the evaluation window is gone.
4. **What would this number look like under pure noise?** (Sharpe differences of
   ±0.2, $t$-statistics of 2.3 among six tests, and one good backtest window all
   look like signal and are usually noise.)
5. **Which frozen parameter, if it moved, would break the conclusion?** Section 2
   above is the menu; there is always at least one.

A number that survives all five is worth discussing. Most numbers do not survive
all five, and saying so is the job.

## What this repository is for

Given all of the above, what survives? The *relationships* survive better than the
*levels*: that MVO concentrates and degrades out-of-sample, that ES exceeds VaR under
fat tails, that smiles steepen as maturity shortens, that diversification decays
under correlation stress — these are structural results the notebooks demonstrate
mechanically, and they replicate across regimes far better than any particular
Sharpe ratio or VaR number will. Use the library to understand mechanisms, to check
intuitions against honest arithmetic, and to practice saying precisely what a model
cannot see.

Models clarify uncertainty; they do not eliminate it. Risk models are lenses, not
shields. And nothing here is investment advice.
