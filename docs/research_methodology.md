# Research Methodology

This page documents how research in this repository is *conducted*, as opposed to
what the models assume ([`model_assumptions.md`](model_assumptions.md)) or where they
break ([`limitations.md`](limitations.md)). The organizing principle: **a result that
cannot be reproduced from a clean clone is an anecdote**, and most quantitative
finance "findings" die of one of three preventable diseases — look-ahead, unseeded
randomness, or silent data revision. The practices below are aimed at those three.

## 1. Reproducibility mechanics

**Seeded randomness, no global state.** Every simulator in the library accepts an
explicit `rng: np.random.Generator` and never touches NumPy's global seed. Notebooks
declare `SEED = 42` in their first code cell and construct generators locally
(`np.random.default_rng(SEED)`), so any figure can be regenerated bit-for-bit, and
two experiments in the same notebook cannot contaminate each other through shared
generator state. Where an experiment needs *independent* replications (e.g., the
Monte Carlo convergence study in notebook 04), replicate seeds are derived
deterministically (`default_rng(7000 + rep)`), so "independent" is also auditable.

**Cached data snapshots.** All empirical work reads frozen CSVs in
`data/snapshots/` — `prices.csv` (15 ETFs, adjusted daily closes, 2015→2026) and
`ff_factors.csv` (Fama–French 5 + momentum, daily, decimal units, 1963→2026). The
loaders (`load_prices`, `load_ff_factors`) read the cache by default and only touch
the network under an explicit `refresh=True` or the CLI
(`python -m src.data.market_data_loader --refresh --tickers ...`). This is a
deliberate inversion of the usual default: *live data is the opt-in*, because
vendor-side revisions (dividend adjustments, split restatements) silently change
"historical" prices and make yesterday's results unreproducible today. The snapshot
is part of the experiment.

**Pinned dependencies.** The stack is NumPy/SciPy/pandas/matplotlib with versions
pinned in `requirements.txt`. Numerical results in the notebooks are tolerant of
minor-version drift (tolerances in tests are explicit, not exact-match), but the
policy is to upgrade deliberately and re-run the full test suite and notebooks, never
implicitly.

## 2. How data flows

```
yfinance / Ken French  --(refresh only)-->  data/snapshots/*.csv
                                                  |
                              load_prices / load_ff_factors
                                                  |
                                    validate_prices  ->  ValidationReport
                                                  |
                          to_returns  ->  analysis / backtests / notebooks
```

One direction, three stages. The validation stage (`src/data/data_validation.py`)
checks missing-value fractions, stale runs (repeated identical prices, a classic
symptom of a dead feed), z-score outliers, and index monotonicity, and returns a
structured `ValidationReport` rather than printing warnings that scroll away.
Analysis code consumes returns, not prices, except where the contract requires
prices (the backtester, which must compute drift and turnover in price space).
No notebook reaches around the loaders to read CSVs directly.

## 3. Testing philosophy: cross-model anchors

The test suite (201 tests) leans on a principle that suits quantitative libraries
unusually well: **independent implementations of the same mathematical object must
agree**, so correctness can be tested without trusting any single implementation.
The anchors, from `docs/api_contract.md`:

- put–call parity (an arbitrage identity — holds regardless of model correctness);
- binomial → Black–Scholes convergence as steps grow;
- Crank–Nicolson ≈ closed form to grid tolerance;
- Merton with `lam=0` collapses to Black–Scholes exactly;
- Heston Monte Carlo ≈ Heston semi-analytic within loose (statistical) tolerance;
- ES ≥ VaR always; risk-parity risk contributions equal; every optimizer's weights
  sum to one within bounds;
- the backtester's `weight_fn` provably receives only data strictly before each
  rebalance (asserted in the engine *and* tested from outside).

Point-value regression tests ("the price is 10.4506") exist too, with explicit
tolerances, but the anchors carry more weight: they fail loudly when a refactor
breaks the *mathematics* rather than merely shifting a constant. Notebook 01 is the
narrative version of this test philosophy, run on real figures.

## 4. Backtest design: no look-ahead, by construction

The single most common fatal flaw in published backtests is information leakage.
The engine's countermeasures are structural, not procedural:

- At rebalance date $t$, `weight_fn` receives a trailing window whose last row is
  **strictly before** $t$ — enforced with an assertion in the hot loop
  (`window.index.max() < date`), so a leak is a crash, not a great Sharpe.
- Weights **drift** with realized returns between rebalances; turnover — and
  therefore cost — is measured against the *drifted* portfolio, not last month's
  targets. Ignoring drift understates turnover systematically.
- Costs are deducted on the rebalance day's return, so cost drag compounds through
  the equity curve exactly as it would in an account.
- The first trade occurs only when a full lookback window exists: no partially
  informed warm-up trades.

What the engine does *not* protect against — because no engine can — are the leaks
upstream of it: choosing the universe with hindsight, tuning hyperparameters on the
"out-of-sample" period, or running many strategy variants and reporting the best
(backtest overfitting; see Bailey & López de Prado, 2014, on deflated Sharpe ratios).
Those require discipline, not code, which is why they are named here.

## 5. Out-of-sample discipline

The convention across notebooks: estimation windows and evaluation windows are
declared *before* results are shown, split at a stated date (notebook 02 fits on
2015–2021 and evaluates on 2022–2026), and the evaluation window is touched exactly
once. Where a result is a single out-of-sample window, the notebooks say so and
refuse to generalize — one window is one draw from the distribution of histories.
In-sample and out-of-sample metrics are always shown side by side, because the *gap*
between them is the most informative number in the comparison: it is a direct
estimate of how much of the in-sample performance was estimation noise being
harvested.

Multiple-testing awareness is part of the same discipline: notebook 05 finds one
marginally significant alpha among six regressions and treats it as expected noise
under the null rather than a discovery (Harvey, Liu & Zhu, 2016).

## 6. Reproducing everything from a clean clone

```bash
git clone <repo> && cd Quantitative_Markets_Research_Lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. verify the library against its contract
pytest                                    # 201 tests, seeds fixed, ~<1 min

# 2. re-execute the notebooks against the cached snapshots
pip install nbclient nbformat ipykernel
for nb in notebooks/0*.ipynb; do
  jupyter execute "$nb"                   # or: python notebooks/_build/execute.py <name>
done
```

Every notebook (a) locates the repo root and inserts it on `sys.path`, (b) reads only
cached snapshots, and (c) seeds every generator — so re-execution reproduces every
number and figure in this repository exactly, modulo minor-version numerical drift.
Refreshing the data snapshots (`--refresh`) is the *one* step that changes results,
and it is therefore manual, deliberate, and diff-able (the snapshots are plain CSV
under version control).

## 7. Notebook conventions

- Markdown states the question and the interpretation; code cells are short and
  produce one table or one figure each.
- Every figure is followed (or preceded) by prose interpreting it — an uninterpreted
  figure is treated as a bug.
- Every notebook ends with an explicit limitations section. The recurring theme is
  intellectual humility as a method: models clarify uncertainty; they do not
  eliminate it.

## References

- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *JPM* 40(5).
- Harvey, C., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected
  Returns." *RFS* 29(1).
- Sandve et al. (2013). "Ten Simple Rules for Reproducible Computational Research."
  *PLoS Comput. Biol.* 9(10).
