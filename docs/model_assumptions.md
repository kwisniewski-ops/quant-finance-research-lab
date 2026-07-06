# Model Assumptions

Every model in this library is a bundle of assumptions wearing an API. This page
makes the bundles explicit, module by module: first a summary table, then prose on
the assumptions that most often bite. The companion page,
[`limitations.md`](limitations.md), covers what goes wrong when they fail;
the notebooks demonstrate several of the failures on real data.

A useful habit when reading any table below: for each assumption, ask *"what
observable would tell me this is false?"* — most have cheap empirical fingerprints,
and notebooks 03 and 06 compute several of them.

## Summary table

| Module | Assumes | Notably does **not** assume |
|---|---|---|
| `models.black_scholes` | GBM; constant $\sigma$, $r$, $q$; continuous frictionless trading; no jumps | anything about the real-world drift $\mu$ (priced out by hedging) |
| `models.binomial_tree` | Same market as BS; CRR up/down factors; discrete exercise dates approximate continuous ones | European exercise (handles American) |
| `math.pde_solvers` | Same market as BS; truncated spatial domain (`s_max_mult`); smooth-enough payoff for CN | — |
| `models.heston_model` | CIR variance; constant $\kappa, \theta, \xi, \rho$; Feller condition desirable, not enforced; one vol factor | constant volatility; Gaussian returns; independence of price and vol |
| `models.jump_diffusion` (Merton) | Poisson jump arrivals; lognormal jump sizes; jump risk diversifiable (not priced) | path continuity |
| `models.stochastic_volatility` (SABR) | Hagan asymptotic expansion valid (short-to-moderate $T$, strikes not extreme); parameters per-expiry | that SABR describes dynamics — used as a smile interpolator |
| `models.factor_models` | Linear factor structure; exogenous factors; i.i.d. homoskedastic errors for the $t$-stats | that betas are constant (see `rolling_betas`) |
| `portfolio.mean_variance` | $\mu, \Sigma$ **known exactly**; quadratic utility (or Gaussian returns); one-period horizon; no costs | — |
| `portfolio.black_litterman` | CAPM equilibrium prior ($\pi = \delta \Sigma w_{mkt}$); Gaussian views; scalar $\tau$ | that the investor has views on every asset |
| `portfolio.risk_parity` | $\Sigma$ estimable and stable; volatility a sufficient risk proxy | anything about $\mu$ (deliberately) |
| `portfolio.hierarchical_risk_parity` | Correlation distances meaningful; single-linkage tree reflects real structure | invertibility/conditioning of $\Sigma$; anything about $\mu$ |
| `portfolio.robust_optimization` | Box uncertainty with known radius $u$; worst case is the right objective | that $\hat{\mu}$ is exact — but does assume its *ranking* has meaning |
| `risk.value_at_risk` / `expected_shortfall` | Stationarity of the return distribution over the estimation window; window represents the future | normality (historical/CF variants); linear P&L is implicit throughout |
| `risk.stress_testing` | Scenario shocks instantaneous and simultaneous; past episodes span relevant futures; correlation blend preserves vols | — |
| `backtesting.engine` | Fills at close prices; proportional costs + fixed slippage; no market impact; fractional shares; dividends via adjusted closes | foreknowledge of any kind (enforced no-look-ahead) |
| `data` loaders | Snapshot CSVs are correct; adjusted closes proxy total return; chosen ETF universe (a survivorship-tilted choice) | — |

## Pricing models

**Black–Scholes** assumes the stock follows $dS = \mu S\,dt + \sigma S\,dW$ with
$\sigma$ a constant of nature, markets frictionless and continuous, and paths free of
jumps. Its deepest assumption is the *consequence* of these: the option is redundant,
perfectly replicable by continuous delta hedging, so its price contains no risk
premium. Empirical fingerprints of failure: realized volatility varied by a factor of
~27 in our 2015–2026 SPY sample, daily returns show excess kurtosis ≈ 14, and
absolute returns are autocorrelated for weeks (notebook 03, §2). The binomial tree
and Crank–Nicolson solver inherit the same market assumptions and add only numerical
ones (lattice/grid resolution, domain truncation at `s_max_mult` times spot,
Rannacher-style care near payoff kinks).

**Heston** promotes variance to a CIR process — mean-reverting, correlated with
price — which repairs the smile, the clustering, and much of the kurtosis. What it
still assumes: the five parameters $(v_0, \kappa, \theta, \xi, \rho)$ are *constants*;
one volatility factor drives the whole surface; variance follows a diffusion (no vol
jumps). Real desks recalibrate Heston daily, and term-structure evidence generally
wants at least two factors. The Feller condition $2\kappa\theta \ge \xi^2$ is exposed
as a property (`feller_satisfied`), not enforced, because market-implied calibrations
routinely violate it; when violated, the variance process touches zero and simulation
bias concentrates there (notebook 04, §4).

**Merton jump diffusion** adds compound-Poisson jumps with lognormal sizes,
$\ln(1+J) \sim \mathcal{N}(\mu_j, \sigma_j^2)$, arriving at rate $\lambda$. Its
load-bearing economic assumption is that **jump risk is diversifiable and therefore
unpriced** — the hedging argument only removes diffusion risk, and Merton closes the
model by assuming the residual jump risk earns no premium. For single names this is
defensible; for index options (where jumps are systematic by construction) it is
clearly wrong, and market crash premia are the measurable gap. The series solution
also assumes the sum over jump counts converges quickly (`n_terms=60` is generous).

**SABR** is best understood as a *quoting convention with dynamics-flavored
parameters*. The Hagan (2002) formula is an asymptotic expansion valid for
short-to-moderate maturities and strikes not too far from the forward; it degrades
for long-dated or extreme-strike options and can imply negative densities in the far
wings. The library uses it the way practitioners do — per-expiry smile
interpolation — and makes no claim that a single SABR parameter set describes the
evolution of the surface through time.

## Portfolio construction

**Mean–variance** assumes the first two moments are *known*, not estimated — the
entire optimization is conditional on inputs whose sampling error it never sees. It
further assumes either quadratic utility or Gaussian returns (so that variance is a
sufficient risk statistic; skew and kurtosis are invisible to it), a single-period
horizon, and costless trading. Notebook 02 quantifies the damage: the in-sample max
Sharpe of ~1.1 degrades to ~0.4 out of sample.

**Black–Litterman** replaces the raw $\hat{\mu}$ with a posterior anchored on the
CAPM equilibrium: the prior mean is the return vector that would make observed market
weights optimal, $\pi = \delta \Sigma w_{mkt}$. This imports CAPM's assumptions
(mean–variance investors, a representative market portfolio, known $\delta$) as a
*prior*, which is much weaker than importing them as truth — but the posterior is
only as reasonable as the equilibrium anchor, the scalar $\tau$, and the view
uncertainty $\Omega$ (here Idzorek-style, proportional to view variance).

**Risk parity and HRP** are interesting for what they *refuse* to assume: nothing
about expected returns. That refusal is their entire risk-management appeal and their
entire performance risk — equalizing risk contributions implicitly bets that
Sharpe ratios are roughly equal across assets, and structurally overweights low-vol
assets (bonds), which is a regime bet on stock–bond correlation (2022 called it;
notebooks 02 and 06). Both still assume $\Sigma$ is estimable and reasonably stable;
HRP additionally assumes single-linkage correlation clustering finds structure that
is real rather than sampled. Robust max-Sharpe assumes the uncertainty box radius is
known — and notebook 02 shows that haircutting $\hat\mu$ symmetrically preserves its
*ranking*, so robustness in the box sense need not diversify at all.

## Risk measurement

All estimators in `risk.value_at_risk` and `risk.expected_shortfall` assume
**stationarity**: the window you estimated from is distributionally the future you
are exposed to. That assumption fails on schedule — precisely at regime changes,
which is when risk numbers are consulted. The parametric variants add normality
(wrong in the far tail, per notebook 06's 95%-vs-99% comparison); Cornish–Fisher
assumes the moment expansion is valid (it can misbehave at extreme skew/kurtosis);
Monte Carlo VaR assumes the fitted multivariate normal — its flexibility is
architectural, not distributional. Historical VaR assumes the sample *contains* the
tail you care about: a 99% daily VaR estimated on one calm year has seen roughly two
or three relevant observations.

`stress_testing` assumes shocks are instantaneous, simultaneous, and applied to
static weights — no intra-crisis rebalancing, no flight-to-quality timing, no
liquidity spiral. The correlation-stress tool assumes vols are unchanged while
correlations blend toward one, then projects back to the PSD cone; it is a
sensitivity analysis, not a forecast of any particular crisis geometry.

## Backtesting and data

The `Backtester` fills at the daily close with proportional costs (default 10bp) and
optional fixed slippage — no market impact, no bid–ask dynamics, no financing or
borrow costs, fractional shares allowed, and dividends folded in through adjusted
closes. Its one *strong* guarantee is negative: no look-ahead, enforced by
construction and asserted at every rebalance. The data layer assumes the cached
snapshots are accurate and — the quietest assumption in the whole repository — that
the fifteen chosen ETFs constitute a fair universe. They do not, exactly: every one
of them survived to 2026 and was liquid enough to select, which tilts every backtest
run on them. See [`limitations.md`](limitations.md).

## Empirical fingerprints: how each assumption announces its failure

For the assumptions that matter most, the cheap diagnostic that detects the failure —
all computable from the cached snapshots, most computed in the notebooks:

| Assumption | Fingerprint of failure | Where shown |
|---|---|---|
| Constant volatility | rolling realized vol varies by >2×; ACF of \|returns\| significant for weeks | notebook 03, §2 |
| Gaussian returns | excess kurtosis ≫ 0; >4σ days present in a decade of data | notebooks 03, 06 |
| Flat implied-vol surface | market smiles/skews at every expiry | notebook 03, §3 |
| Diversifiable jump risk | index put premia exceed Merton-with-unpriced-jumps values | (literature; not testable from prices alone here) |
| Known $\mu$ | in-sample vs out-of-sample Sharpe gap | notebook 02, §5 |
| Stable $\Sigma$ / correlations | rolling correlation regime breaks (SPY–TLT 2022) | notebook 06, §7 |
| Constant betas | rolling betas drift and change sign | notebook 05, §2 |
| Stationary risk distribution | VaR backtest breach clustering after regime shifts | notebook 06 |
| Costless / frictionless trading | turnover × spread arithmetic vs backtest net returns | notebook 06, §1 |

## How to use this page

Before trusting any notebook result, find its modules in the table above and ask
which assumption is most strained by the question being asked. The library's design
philosophy is that assumptions are cheapest when written down: a model whose
assumptions you can enumerate is a lens with a known distortion, and models clarify
uncertainty — they do not eliminate it.
