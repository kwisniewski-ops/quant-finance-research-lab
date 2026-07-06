# Mathematical Foundations

This document collects the mathematics that the library implements, at the level of
detail a reader needs to audit the code rather than re-derive the field. The
stochastic-calculus core — Brownian motion, quadratic variation, Itô's lemma, the SDE
zoo, Girsanov, Feynman–Kac, and discretization orders — is developed carefully in
[`src/math/stochastic_calculus_notes.md`](../src/math/stochastic_calculus_notes.md);
sections 1–4 below give only the working summary needed to read the pricing modules,
then this document covers the ground those notes deliberately leave out: the
statistics and optimization theory behind the portfolio, factor, and risk modules.

---

## 1. Probability spaces and filtrations (the two-paragraph version)

All randomness lives on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$
equipped with a filtration $\{\mathcal{F}_t\}_{t \ge 0}$ — an increasing family of
$\sigma$-algebras representing *information available at time $t$*. A process $X_t$
is *adapted* if $X_t$ is $\mathcal{F}_t$-measurable: you may not use tomorrow's
prices today. This is not an ornament. It is the mathematical form of the
**no-look-ahead rule**, and it survives all the way down to code: the backtester's
`weight_fn` receives a price window whose last row is *strictly before* the rebalance
date, which is nothing more than adaptedness enforced with an `assert`.

A *martingale* is an adapted process with $\mathbb{E}[X_t \mid \mathcal{F}_s] = X_s$
for $s \le t$ — a game that is fair given current information. The first fundamental
theorem of asset pricing says, loosely, that a market admits no arbitrage if and only
if there exists an equivalent measure $\mathbb{Q}$ under which discounted asset
prices are martingales. Pricing is then computing conditional expectations under
$\mathbb{Q}$.

## 2. Brownian motion and Itô calculus (summary; full treatment in the notes)

Standard Brownian motion $W_t$ has independent Gaussian increments,
$W_t - W_s \sim \mathcal{N}(0, t-s)$, continuous paths, and — the property from which
everything follows — nonvanishing quadratic variation: $\langle W \rangle_t = t$,
informally $(dW_t)^2 = dt$. Because paths have infinite total variation, ordinary
calculus fails and the chain rule acquires a second-order term. For
$dX_t = \mu_t\,dt + \sigma_t\,dW_t$ and smooth $f$, **Itô's lemma** reads

$$
df(t, X_t) = \left(\frac{\partial f}{\partial t}
+ \mu_t \frac{\partial f}{\partial x}
+ \tfrac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right) dt
+ \sigma_t \frac{\partial f}{\partial x}\, dW_t .
$$

Applied to $\ln S_t$ under GBM, it produces the $-\tfrac{1}{2}\sigma^2$ correction
that makes `simulate_gbm`'s log-Euler scheme *exact in distribution*, and applied to
a discounted option value it produces the pricing PDE below.

## 3. Risk-neutral valuation and the pricing PDE

Under $\mathbb{Q}$ (constructed via Girsanov's theorem — see the notes, §5), the
stock's drift becomes the risk-free rate and any European claim with payoff
$\Phi(S_T)$ is priced by

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\Phi(S_T)\right].
$$

The Feynman–Kac theorem converts this expectation into the terminal-value problem

$$
\frac{\partial V}{\partial t}
+ \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ (r - q)\,S\,\frac{\partial V}{\partial S} - rV = 0,
\qquad V(S, T) = \Phi(S),
$$

which is what `src/math/pde_solvers.py` discretizes. The library's four pricing
routes — closed form (`black_scholes`), lattice (`binomial_tree`), PDE
(`pde_solvers`), and simulation (`monte_carlo_methods`) — are four numerical
treatments of this one object; notebook 01 verifies their agreement.

## 4. Discretization in one paragraph

When exact transition densities exist (GBM, OU) the library samples them directly,
so simulation error is purely statistical. When they do not (CIR, Heston), it uses
full-truncation Euler, which has weak-order-1 bias of size $O(\Delta t)$ on top of
the $O(1/\sqrt{N})$ Monte Carlo error; the two budgets shrink at different rates and
must be tracked separately. Crank–Nicolson is second-order in both $\Delta S$ and
$\Delta t$ and unconditionally stable; American exercise is handled by projected SOR
on the resulting linear complementarity problem. Strong/weak order definitions and
the derivations are in the stochastic-calculus notes, §7.

---

## 5. Statistics of returns: what is estimable and what is not

Portfolio construction consumes two estimates with wildly different reliability.

**Means are nearly hopeless.** For i.i.d. returns with annualized volatility
$\sigma$, the sample mean over $Y$ years has standard error

$$
\operatorname{SE}(\hat{\mu}) = \frac{\sigma}{\sqrt{Y}},
$$

*independent of the sampling frequency* — daily data does not help, only calendar
time does (Merton, 1980). With $\sigma = 16\%$ and 10 years, the 95% confidence band
on an annual equity mean is roughly $\pm 10$ percentage points: wider than the
quantity being estimated.

**Covariances are merely hard.** Element-wise, $\hat{\Sigma}$ converges at rate
$\sqrt{n}$ in the number of observations, so high-frequency data genuinely helps. The
difficulty is dimensionality: an $N$-asset covariance has $N(N+1)/2$ free parameters,
and when $N$ is not small relative to $n$ the sample eigenvalue spectrum is badly
distorted — largest eigenvalues biased up, smallest biased down — which is
catastrophic for any procedure that *inverts* $\Sigma$. Two remedies appear in the
library:

- **Shrinkage** (`ledoit_wolf_shrinkage`): replace $\hat{\Sigma}$ by the convex
  combination $\delta F + (1-\delta)\hat{\Sigma}$ with $F$ a structured
  (constant-correlation) target and $\delta$ chosen to minimize expected Frobenius
  loss (Ledoit & Wolf, 2004). Bias is accepted to buy variance reduction.
- **PSD repair** (`nearest_psd`, `safe_cholesky`): stressed or exponentially-weighted
  covariance estimates can leave the positive-semidefinite cone; Higham-style
  eigenvalue clipping projects them back, so downstream Cholesky factorizations and
  quadratic forms remain meaningful.

**Higher moments enter through risk measures.** The Cornish–Fisher expansion adjusts
a Gaussian quantile $z_\alpha$ for sample skewness $s$ and excess kurtosis $k$:

$$
\tilde{z}_\alpha = z_\alpha + \frac{s}{6}(z_\alpha^2 - 1)
+ \frac{k}{24}(z_\alpha^3 - 3z_\alpha) - \frac{s^2}{36}(2z_\alpha^3 - 5z_\alpha),
$$

used by `cornish_fisher_var`. It is an asymptotic expansion, not a distribution:
for extreme $|s|, k$ it can produce non-monotone quantiles, one reason the library
always computes historical VaR alongside it.

## 6. Optimization foundations for portfolio construction

### 6.1 Convexity, and why mean–variance is tractable

A set is convex if it contains the segment between any two of its points; a function
$f$ is convex if its epigraph is. The two facts that make portfolio optimization a
solved *numerical* problem: (i) the feasible set
$\{w : \mathbf{1}^\top w = 1,\ l \le w \le u\}$ is a convex polytope, and (ii) the
variance $w^\top \Sigma w$ is a convex quadratic whenever $\Sigma \succeq 0$ — which
is exactly why the library insists on PSD covariance inputs. For convex problems
every local minimum is global, and first-order (KKT) conditions are sufficient, not
merely necessary. Minimum-variance and mean–variance-with-target-return are convex
QPs; max-Sharpe is not convex as stated (a ratio), but is quasi-concave and solvable
either by SLSQP directly (the library's choice, with a feasibility fallback) or by
the classical homogenization to a convex QP.

### 6.2 KKT in one paragraph

For $\min_w f(w)$ subject to $g_i(w) \le 0$ and $h_j(w) = 0$, the Karush–Kuhn–Tucker
conditions require a stationary Lagrangian
$\nabla f + \sum_i \lambda_i \nabla g_i + \sum_j \nu_j \nabla h_j = 0$ with
$\lambda_i \ge 0$, feasibility, and complementary slackness
$\lambda_i\, g_i(w) = 0$. In the portfolio context the multiplier on the budget
constraint is the shadow price of capital, and complementary slackness is why
long-only optimal portfolios are *sparse*: assets whose marginal Sharpe contribution
is dominated sit exactly at the bound $w_i = 0$ with a strictly positive multiplier.
The concentrated MVO weights in notebook 02 are KKT conditions doing precisely what
they promise.

### 6.3 The unconstrained solution and its fragility

Without bounds, the tangency portfolio is $w^* \propto \Sigma^{-1}(\mu - r_f\mathbf{1})$.
The $\Sigma^{-1}$ is the villain: inversion amplifies the smallest (worst-estimated)
eigenvalue directions, and errors in $\mu$ are projected through it, so estimation
noise is *leveraged*, not averaged away (Michaud, 1989). Every alternative
construction in `src/portfolio` is a different way of declining to compute
$\Sigma^{-1}\hat{\mu}$ at face value:

- **Black–Litterman** replaces $\hat{\mu}$ with a Bayesian posterior that shrinks
  views toward the market-implied equilibrium $\pi = \delta \Sigma w_{mkt}$.
- **Robust max-Sharpe** optimizes the worst case over a box
  $\mu \in [\hat{\mu} - u, \hat{\mu} + u]$, equivalent to an $\ell_1$-style penalty
  $u^\top |w|$ on the return term.
- **Risk parity** discards $\mu$ and solves
  $w_i (\Sigma w)_i = w_j (\Sigma w)_j\ \forall i,j$ — a nonconvex-looking condition
  that admits a convex reformulation and, in the library, a well-behaved SLSQP
  formulation on squared deviations of risk contributions.
- **HRP** discards inversion entirely: correlation distance
  $d_{ij} = \sqrt{\tfrac{1}{2}(1 - \rho_{ij})}$, single-linkage clustering,
  quasi-diagonalization, and recursive inverse-variance bisection
  (López de Prado, 2016).

### 6.4 Risk contributions

Portfolio volatility $\sigma_p(w) = \sqrt{w^\top \Sigma w}$ is homogeneous of degree
one, so Euler's theorem gives an exact decomposition

$$
\sigma_p = \sum_i w_i \frac{\partial \sigma_p}{\partial w_i}
= \sum_i \frac{w_i\,(\Sigma w)_i}{\sigma_p},
$$

and the *fractional risk contribution* $\text{RC}_i = w_i (\Sigma w)_i / w^\top \Sigma w$
sums to one. This is the quantity `risk_contributions` reports and risk parity
equalizes; it is also the honest answer to "what does this portfolio actually own,"
since weights and risk can disagree by an order of magnitude (notebook 02, §3).

## 7. Linear regression as used by the factor models

`fit_factor_model` estimates $r_t - r_{f,t} = \alpha + \beta^\top F_t + \varepsilon_t$
by OLS: $\hat{\theta} = (X^\top X)^{-1} X^\top y$ via `numpy.lstsq`, with
$\operatorname{Var}(\hat{\theta}) = s^2 (X^\top X)^{-1}$ and plain $t$-statistics.
Under the classical assumptions (exogeneity, homoskedastic serially-uncorrelated
errors) these are exact-\(t\); daily financial returns violate the homoskedasticity
assumption conspicuously (volatility clustering), so reported $t$-statistics should be
read as modestly optimistic — the docs and notebook 05 flag this rather than hide it.
$R^2$ retains its usual interpretation as fraction of variance explained. Rolling
estimation (`rolling_betas`) re-fits on trailing windows; the resulting beta paths are
themselves estimates with sampling error of order $\sigma_\varepsilon / (\sigma_F \sqrt{n})$
per window, so slow drift is meaningful while high-frequency wiggle largely is not.

## 8. Coherence of risk measures, briefly

A risk measure $\rho$ is *coherent* (Artzner et al., 1999) if it is monotone,
translation-invariant, positively homogeneous, and subadditive
($\rho(A+B) \le \rho(A) + \rho(B)$ — diversification never hurts). Value-at-Risk
fails subadditivity in general; expected shortfall satisfies it. This is not academic
tidiness: a VaR-based limit system can reward concentrating tail risk into rarely-hit
buckets, which is one reason `expected_shortfall` sits beside `value_at_risk` in the
library and regulation moved to ES (FRTB). The empirical comparison of the two under
fat tails is the subject of notebook 06, §2.

## References

- Øksendal, B. *Stochastic Differential Equations*, 6th ed. — filtrations, Itô calculus.
- Shreve, S. *Stochastic Calculus for Finance II* — risk-neutral valuation, Girsanov.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization* — convexity, KKT.
- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1).
- Merton, R. C. (1980). "On Estimating the Expected Return on the Market." *JFE* 8.
- Michaud, R. (1989). "The Markowitz Optimization Enigma." *FAJ* 45(1).
- Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices." *Journal of Multivariate Analysis* 88.
- López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
  Out-of-Sample." *Journal of Portfolio Management* 42(4).
- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). "Coherent Measures of
  Risk." *Mathematical Finance* 9(3).
- Higham, N. (2002). "Computing the Nearest Correlation Matrix." *IMA J. Numer. Anal.* 22.
