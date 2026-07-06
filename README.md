# Quantitative Markets Research Lab

A reproducible research engine for testing, visualizing, and explaining quantitative
finance models across asset pricing, portfolio optimization, derivatives, risk, and
market microstructure.

> Financial markets can be studied as complex adaptive systems where uncertainty,
> volatility, human behavior, capital allocation, and mathematical structure converge.
> This project explores how quantitative models can clarify market behavior while
> remaining honest about the limits of prediction.

**Models clarify uncertainty; they do not eliminate it.** Every module here ships with
its assumptions documented, its limitations stated, and tests that anchor it to known
results. The companion website presents the same research interactively:
[`app/frontend/`](app/frontend/) (deployable to GitHub Pages — see [DEPLOY.md](DEPLOY.md)).

---

## What's inside

| Layer | Contents |
|---|---|
| **Asset pricing** (`src/models/`) | Black–Scholes (vectorized, Greeks, implied vol), CRR binomial trees, Heston stochastic volatility (semi-analytic characteristic function + full-truncation Euler simulation), Merton jump-diffusion, SABR implied vol |
| **Portfolio construction** (`src/portfolio/`) | Mean–variance & efficient frontier, Black–Litterman, equal-risk-contribution risk parity (damped Newton, exact to ~1e-14), hierarchical risk parity, robust/resampled optimization |
| **Risk analytics** (`src/risk/`) | Historical / parametric / Cornish–Fisher / Monte Carlo VaR, expected shortfall, drawdown analysis, historical + correlation stress testing |
| **Backtesting** (`src/backtesting/`) | Look-ahead-free engine (verified by test), transaction costs, slippage, full performance metrics |
| **Numerical methods** (`src/math/`) | Monte Carlo (GBM, OU, CIR, regime-switching, antithetic variates), Crank–Nicolson PDE solver with Rannacher start-up, PSD repair, Ledoit–Wolf shrinkage — plus [stochastic calculus notes](src/math/stochastic_calculus_notes.md) |
| **Data** (`src/data/`) | Cache-first loaders (15-ETF universe 2015→present, Fama–French 5 factors + momentum), data validation with outlier/staleness detection |
| **Research notebooks** (`notebooks/`) | Six executed notebooks — pricing model cross-validation, optimization under estimation error, the flagship Black-Scholes→Heston case study, Monte Carlo methods, factor regressions on real ETFs, a full market-risk workup |
| **Documentation** (`docs/`) | [Mathematical foundations](docs/mathematical_foundations.md) · [Model assumptions](docs/model_assumptions.md) · [Research methodology](docs/research_methodology.md) · [Limitations](docs/limitations.md) · [API contract](docs/api_contract.md) |
| **Website** (`app/frontend/`) | Static research interface: interactive option pricer, efficient frontier explorer, stochastic process simulator, risk dashboard, factor research, mathematical notes, and the flagship case study |

## Selected findings from the notebooks

- **Estimation error dominates optimization.** Max-Sharpe MVO fit on 2015–2021 drops from
  Sharpe 1.12 in-sample to 0.44 out-of-sample (2022–2026); risk-based allocations (risk
  parity, HRP) halve out-of-sample volatility and drawdown. Notably, box-uncertainty
  "robust" optimization concentrated *harder* than MVO — uncertainty sets haircut the
  level of expected returns, not their ranking. ([notebook 02](notebooks/02_portfolio_optimization.ipynb))
- **Black–Scholes assumptions fail empirically in measurable ways.** SPY 21-day realized
  volatility spans 3.4%→93%; excess kurtosis ≈ 14; the worst day is a ~10σ event under
  the Gaussian assumption. A Feller-satisfying Heston calibration reproduces a 25%→16%
  implied-vol skew at three months. ([notebook 03](notebooks/03_derivatives_pricing.ipynb))
- **Style ETFs load on their advertised factors** (MTUM momentum β=0.34, t=49; VLUE value
  β=0.31, t=29), and five of six alphas are statistically indistinguishable from zero once
  factor exposure is priced. ([notebook 05](notebooks/05_factor_investing.ipynb))
- **Diversification decays exactly when needed.** SPY–TLT correlation flipped from −0.35
  (2015–19) to +0.37 (2022–23); stressing correlations toward 1 raises a 60/40+alts
  portfolio's volatility from 11.6% to 15.8%. ([notebook 06](notebooks/06_market_risk_dashboard.ipynb))

## Quick start

```bash
git clone https://github.com/kylewisniewski/quant-research-lab.git
cd quant-research-lab
pip install -e ".[all]"
pytest            # 201 tests, ~1 minute
```

Cached market-data snapshots ship with the repo (`data/snapshots/`), so every notebook
runs offline and reproduces exactly. To refresh with current data:

```bash
python -m src.data.market_data_loader --refresh --tickers SPY QQQ IWM EFA EEM AGG TLT LQD GLD DBC VNQ USMV MTUM VLUE QUAL
python -m src.data.factor_data_loader --refresh
```

Minimal example:

```python
from src.models.black_scholes import bs_price, bs_greeks
from src.models.heston_model import HestonParams, heston_price

bs_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)          # 10.4506
params = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7)
heston_price(S=100, K=100, T=1.0, r=0.05, params=params)   # smile-consistent price
```

## Design principles

**Reproducibility.** Seeded random generators everywhere, cached data snapshots committed
to the repo, executed notebooks with builder scripts (`notebooks/_build/`), pinned
dependency floors.

**Tested against theory.** Cross-model anchors, not just unit tests: put–call parity to
1e-10; binomial and Crank–Nicolson converge to Black–Scholes; Merton with zero jump
intensity *is* Black–Scholes; Heston Monte Carlo agrees with the semi-analytic price;
ES ≥ VaR; the backtester provably cannot see the future.

**Honest about limits.** Read [docs/limitations.md](docs/limitations.md) first. Parameters
assumed constant are not; normality fails exactly when it matters; past stress scenarios
do not bound future ones.

## Repository structure

```
quant-research-lab/
├── src/                  # library code (models, portfolio, risk, backtesting, math, data, visualization)
├── notebooks/            # six executed research notebooks + build scripts
├── tests/                # 201 tests: cross-model anchors, no-look-ahead proofs, analytic checks
├── docs/                 # foundations, assumptions, methodology, limitations, API contract
├── app/frontend/         # static research website (GitHub Pages ready)
└── data/snapshots/       # cached market data for offline reproducibility
```

## Disclaimer

This is research tooling for studying models, not investment advice. Nothing here
constitutes a recommendation to buy or sell any security.

## Author

**Kyle Wisniewski** — data & AI strategist; graduate study in applied quantitative
finance and global economic affairs, University of Denver.
kywisniewski@gmail.com

## License

MIT — see [LICENSE](LICENSE).
