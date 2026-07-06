# Stochastic Calculus Notes

Working notes on the mathematics underlying the models in this repository:
Brownian motion, Itô calculus, the SDEs we simulate, the change of measure
that justifies risk-neutral pricing, the Feynman–Kac bridge to PDEs, and the
discretization schemes used in `src/math/monte_carlo_methods.py` and
`src/models/`. The aim is to be precise but readable — every result here is
used somewhere in the codebase.

---

## 1. Brownian motion

A standard **Brownian motion** (Wiener process) $W = (W_t)_{t \ge 0}$ is a
process with:

1. $W_0 = 0$;
2. independent increments: $W_t - W_s \perp \mathcal{F}_s$ for $t > s$;
3. Gaussian increments: $W_t - W_s \sim \mathcal{N}(0,\, t - s)$;
4. continuous paths (almost surely).

Two properties matter most for what follows. First, the **scaling**: over a
short interval $\Delta t$, the increment $\Delta W \sim \mathcal{N}(0, \Delta t)$
has typical size $\sqrt{\Delta t}$, not $\Delta t$. Randomness dominates drift
at short horizons. Second, paths are continuous but **nowhere differentiable**
and have infinite first-order variation — which is exactly why ordinary
calculus fails and a new chain rule is needed.

## 2. Quadratic variation

For a partition $0 = t_0 < t_1 < \dots < t_n = t$ with mesh $\to 0$,

$$
[W]_t \;=\; \lim \sum_{i} \left(W_{t_{i+1}} - W_{t_i}\right)^2 \;=\; t
\qquad \text{(in probability).}
$$

The intuition: each squared increment has mean $\Delta t$ and variance
$2\Delta t^2$; summing $n = t/\Delta t$ of them gives mean $t$ and vanishing
variance. The sum of squares does **not** vanish — it converges to a
deterministic clock.

This is the single most important fact in the subject. It is compactly
written as the multiplication table

$$
(dW_t)^2 = dt, \qquad dW_t \, dt = 0, \qquad (dt)^2 = 0,
$$

and it is where every "extra term" in Itô calculus comes from. For two
correlated Brownian motions with $d\langle W^1, W^2\rangle_t = \rho\,dt$
(used in Heston), the cross-variation replaces $(dW)^2$ accordingly.

## 3. Itô's lemma

Let $X_t$ follow the Itô process

$$
dX_t = \mu(t, X_t)\,dt + \sigma(t, X_t)\,dW_t,
$$

and let $f(t, x)$ be $C^{1,2}$. Then

$$
df(t, X_t) =
\left( \frac{\partial f}{\partial t}
     + \mu \frac{\partial f}{\partial x}
     + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt
+ \sigma \frac{\partial f}{\partial x}\, dW_t .
$$

**Derivation sketch.** Taylor-expand $f$ to second order over a small step:

$$
\Delta f \approx f_t\,\Delta t + f_x\,\Delta X
           + \tfrac{1}{2} f_{xx} (\Delta X)^2 + \dots
$$

For a differentiable path, $(\Delta X)^2 = O(\Delta t^2)$ and the last term
dies. For an Itô process, $\Delta X \approx \mu\,\Delta t + \sigma\,\Delta W$,
so $(\Delta X)^2 \approx \sigma^2 (\Delta W)^2 \to \sigma^2\,\Delta t$ by
quadratic variation — a **first-order** contribution that survives the limit.
Summing over the partition and letting the mesh go to zero yields the lemma;
the higher-order terms vanish by the multiplication table above.

**Canonical example.** For GBM $dS = \mu S\,dt + \sigma S\,dW$, apply Itô to
$f = \ln S$:

$$
d\ln S_t = \left(\mu - \tfrac{1}{2}\sigma^2\right)dt + \sigma\,dW_t
\;\;\Longrightarrow\;\;
S_t = S_0 \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t\right].
$$

The $-\tfrac{1}{2}\sigma^2$ is not a convention; it is quadratic variation
showing up in the exponent (volatility drag). This exact solution is why
`simulate_gbm` is bias-free at the grid points.

## 4. The SDE zoo used in this repository

**Geometric Brownian motion** — the Black–Scholes underlying
(`simulate_gbm`, `bs_price`):

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t .
$$

Lognormal marginals, constant volatility, no memory. Wrong in the details
(no fat tails, no vol clustering) but the indispensable baseline.

**Ornstein–Uhlenbeck** — mean reversion with Gaussian noise
(`simulate_ou`; the classical model for spreads and log-vol):

$$
dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t .
$$

Linear SDE, solvable with the integrating factor $e^{\kappa t}$:

$$
X_t = \theta + (X_0 - \theta)e^{-\kappa t}
      + \sigma \int_0^t e^{-\kappa (t-s)}\, dW_s ,
$$

so $X_t \sim \mathcal{N}\!\big(\theta + (X_0-\theta)e^{-\kappa t},\;
\tfrac{\sigma^2}{2\kappa}(1 - e^{-2\kappa t})\big)$. The transition law is
Gaussian and known in closed form — hence the **exact discretization** in
`simulate_ou`.

**Cox–Ingersoll–Ross (CIR)** — mean reversion with square-root noise
(`simulate_cir`; short rates, and the variance process inside Heston):

$$
dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t .
$$

The diffusion coefficient vanishes at zero, which keeps the process
non-negative. If the **Feller condition** $2\kappa\theta \ge \sigma^2$ holds,
the origin is unattainable (paths stay strictly positive); if it fails, paths
touch zero and reflect. Discretization must respect this — see §7.

**Heston** — stochastic volatility (`heston_model.py`):

$$
dS_t = r S_t\,dt + \sqrt{v_t}\, S_t\, dW_t^S, \qquad
dv_t = \kappa(\theta - v_t)\,dt + \xi \sqrt{v_t}\, dW_t^v, \qquad
d\langle W^S, W^v\rangle_t = \rho\,dt .
$$

Variance is a CIR process; negative $\rho$ produces the equity skew
(spot down → variance up → left tail fattens). The log-price characteristic
function is known in closed form, giving semi-analytic European prices by
Fourier inversion (Heston 1993; we use the numerically stable formulation of
Albrecher et al. 2007).

**Merton jump-diffusion** (`jump_diffusion.py`):

$$
\frac{dS_t}{S_{t^-}} = (\mu - \lambda \bar{k})\,dt + \sigma\,dW_t
+ \left(e^{Y} - 1\right) dN_t ,
$$

with $N_t \sim \text{Poisson}(\lambda t)$, jump sizes
$Y \sim \mathcal{N}(\mu_J, \sigma_J^2)$, and compensator
$\bar{k} = e^{\mu_J + \sigma_J^2/2} - 1$ chosen so the expected growth rate is
$\mu$. Jumps add genuinely new risk: the market is incomplete (you cannot
hedge a jump with a position in the stock), and short-dated smiles become
steep — something diffusive models struggle to produce.

## 5. Girsanov and the risk-neutral measure

Pricing does not use the real-world drift. **Girsanov's theorem** says that
changing the probability measure with the exponential martingale density

$$
\left.\frac{d\mathbb{Q}}{d\mathbb{P}}\right|_{\mathcal{F}_t}
= \exp\!\left( -\int_0^t \lambda_s\, dW_s
               - \tfrac{1}{2}\int_0^t \lambda_s^2\, ds \right)
$$

turns $W_t^{\mathbb{Q}} = W_t + \int_0^t \lambda_s\,ds$ into a Brownian
motion under $\mathbb{Q}$: a measure change **shifts drifts and leaves
volatilities untouched**. Choosing the market price of risk
$\lambda = (\mu - r)/\sigma$ moves the GBM drift from $\mu$ to $r$.

The economic content: under $\mathbb{Q}$, the discounted price
$e^{-rt} S_t$ is a martingale, so "expected return" arguments cannot generate
arbitrage. Any payoff $H$ deliverable at $T$ is then priced as

$$
V_0 = e^{-rT}\, \mathbb{E}^{\mathbb{Q}}[H].
$$

Intuition: $\mathbb{Q}$ reweights scenarios by marginal utility — bad states
(crashes) get more weight than their real-world frequency, which is exactly
why implied vol exceeds realized vol on average and why the drift $\mu$
drops out of every pricing formula in this repo. This is also why our Monte
Carlo pricers are run with `mu = r`.

## 6. Feynman–Kac and the pricing PDE

The **Feynman–Kac theorem** links conditional expectations of diffusions to
parabolic PDEs. If

$$
u(t, x) = \mathbb{E}\!\left[ e^{-r(T-t)}\, h(X_T) \,\middle|\, X_t = x \right],
\qquad dX_s = \mu(s, X_s)\,ds + \sigma(s, X_s)\,dW_s,
$$

then $u$ solves

$$
\frac{\partial u}{\partial t}
+ \mu(t,x) \frac{\partial u}{\partial x}
+ \frac{1}{2}\sigma^2(t,x) \frac{\partial^2 u}{\partial x^2}
- r u = 0, \qquad u(T, x) = h(x).
$$

**Why:** by construction, $e^{-rt} u(t, X_t)$ is a martingale (it is a
conditional expectation of a fixed terminal quantity). Apply Itô to it and
set the $dt$-term to zero — the PDE is precisely the statement "no drift".

Specializing to risk-neutral GBM ($\mu = rS$, $\sigma = \sigma S$) gives the
**Black–Scholes equation**

$$
\frac{\partial V}{\partial t}
+ \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ r S \frac{\partial V}{\partial S} - r V = 0 .
$$

So the two pricing routes in this repository are the same object seen from
the two sides of Feynman–Kac: `mc_price` estimates the expectation directly;
`cn_bs_price` solves the PDE by finite differences. They must agree, and the
test suite checks that they do (against the closed form, which solves the
PDE analytically). American options replace the PDE by a **linear
complementarity problem** — the value can never fall below intrinsic, and
where it is above intrinsic the PDE holds with equality — solved in
`pde_solvers.py` with projected SOR.

## 7. Discretization schemes: strong and weak order

A scheme has **strong order** $\gamma$ if
$\mathbb{E}\,|X_T^{\Delta} - X_T| = O(\Delta t^{\gamma})$ (pathwise
closeness — matters for hedging simulations and multilevel MC), and **weak
order** $\beta$ if
$|\mathbb{E}\,f(X_T^{\Delta}) - \mathbb{E}\,f(X_T)| = O(\Delta t^{\beta})$
for smooth $f$ (distributional closeness — what matters for pricing).

**Euler–Maruyama**

$$
X_{t+\Delta t} = X_t + \mu(t, X_t)\,\Delta t + \sigma(t, X_t)\sqrt{\Delta t}\,Z
$$

has strong order $1/2$ and weak order $1$ under Lipschitz coefficients.

**Milstein** adds the Itô correction
$\tfrac{1}{2}\sigma \sigma' \left[(\Delta W)^2 - \Delta t\right]$
and lifts the strong order to $1$ (weak order stays $1$).

**Exact schemes.** When the transition density is known, sample it directly
and eliminate discretization bias entirely: GBM (lognormal — `simulate_gbm`)
and OU (Gaussian — `simulate_ou`). For these, `n_steps` controls only how
finely you *observe* the path, not accuracy.

**Square-root processes and full truncation.** Naive Euler applied to CIR or
the Heston variance can go negative, and $\sqrt{v}$ then fails. Among the
standard fixes (absorption, reflection, partial/full truncation), **full
truncation** (Lord, Koekkoek & van Dijk 2010) applies the positive part
inside both the drift and the diffusion,

$$
\tilde{v}_{t+\Delta t} = \tilde{v}_t
+ \kappa\left(\theta - \tilde{v}_t^{+}\right)\Delta t
+ \xi \sqrt{\tilde{v}_t^{+}} \sqrt{\Delta t}\, Z, \qquad
v_t = \tilde{v}_t^{+},
$$

and has the smallest bias in practice. It is what `simulate_cir` and
`simulate_heston` implement — the auxiliary process $\tilde{v}$ may dip
below zero, but the variance fed to the spot never does. Higham (2002) is
the standard reference for why naive schemes fail on such SDEs and how to
verify convergence experimentally.

**Variance reduction.** Antithetic variates (`simulate_gbm` with
`antithetic=True`) pair each draw $Z$ with $-Z$: the pair average has
variance $\tfrac{1}{2}(1 + \rho_{\text{anti}})\,\mathrm{Var}$, with
$\rho_{\text{anti}} < 0$ for monotone payoffs — same accuracy, roughly half
the paths. Control variates (regressing the payoff on a quantity with known
expectation, e.g. $S_T$) compound the effect; see Glasserman (2004).

**Rule of thumb.** Monte Carlo error has two parts:
statistical $O(n_{\text{paths}}^{-1/2})$ and bias
$O(\Delta t^{\beta})$. There is no point simulating $10^7$ paths on a grid
whose bias dwarfs the standard error — balance them (for Euler weak order 1:
$\Delta t \propto n_{\text{paths}}^{-1/2}$).

## 8. References

- Øksendal, B. (2003). *Stochastic Differential Equations*, 6th ed. Springer.
- Shreve, S. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Higham, D. J. (2002). "An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations." *SIAM Review*, 43(3), 525–546.
- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*, 6(2), 327–343.
- Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). "The Little Heston Trap." *Wilmott Magazine*, Jan, 83–92.
- Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous." *Journal of Financial Economics*, 3, 125–144.
- Lord, R., Koekkoek, R., van Dijk, D. (2010). "A Comparison of Biased Simulation Schemes for Stochastic Volatility Models." *Quantitative Finance*, 10(2), 177–194.
- Hagan, P. S., Kumar, D., Lesniewski, A., Woodward, D. (2002). "Managing Smile Risk." *Wilmott Magazine*, Sep, 84–108.
