# API Contract — Quantitative Markets Research Lab

Internal specification. Every module implements these exact signatures so notebooks,
tests, and the website data-export scripts interlock. Style: NumPy docstrings, full
type hints, `dataclasses` for structured results, vectorized NumPy where natural.

## Global conventions

- Import style: `from src.models.black_scholes import bs_price` (repo root on `sys.path`; run everything from repo root).
- All rates/vols/returns are **annualized decimals**; continuous compounding for pricing.
- Time `T` in years. `option_type` is `"call" | "put"`.
- Randomness: every simulation accepts `rng: np.random.Generator | None = None` and seeds via `np.random.default_rng(seed)` — deterministic tests use `default_rng(42)`.
- Prices `pd.DataFrame` indexed by `DatetimeIndex`, columns = tickers. Returns are simple unless `log=True`.
- No global state. No prints in library code (use `logging`).

## src/models

### black_scholes.py
- `bs_price(S, K, T, r, sigma, option_type="call", q=0.0) -> float | np.ndarray` (vectorized)
- `bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> dict[str, float]` keys: delta, gamma, vega, theta, rho
- `implied_vol(price, S, K, T, r, option_type="call", q=0.0, tol=1e-8) -> float` (Brent; return `np.nan` if no root)

### binomial_tree.py
- `binomial_price(S, K, T, r, sigma, n_steps=500, option_type="call", american=False, q=0.0) -> float` (CRR)

### heston_model.py
- `@dataclass HestonParams: v0, kappa, theta, xi, rho` with `.feller_satisfied` property
- `heston_price(S, K, T, r, params: HestonParams, option_type="call") -> float` — semi-analytic (Heston 1993 characteristic function, Gauss-Legendre integration; numerically stable "little trap" formulation)
- `simulate_heston(S0, params, r, T, n_steps, n_paths, rng=None) -> tuple[np.ndarray, np.ndarray]` — (S_paths, v_paths), shape (n_paths, n_steps+1), full-truncation Euler

### jump_diffusion.py (Merton)
- `merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type="call", n_terms=60) -> float` (series solution)
- `simulate_merton(S0, mu, sigma, lam, mu_j, sigma_j, T, n_steps, n_paths, rng=None) -> np.ndarray`

### stochastic_volatility.py
- `sabr_implied_vol(F, K, T, alpha, beta, rho, nu) -> float` (Hagan 2002; handle ATM branch)
- `implied_vol_surface(S, r, strikes, maturities, price_fn) -> pd.DataFrame` (rows=maturities, cols=strikes)

### factor_models.py
- `@dataclass FactorModelResult: alpha, alpha_tstat, betas: pd.Series, tstats: pd.Series, r_squared, resid: pd.Series, n_obs`
- `fit_factor_model(asset_returns: pd.Series, factor_returns: pd.DataFrame, rf: pd.Series | float = 0.0) -> FactorModelResult` (OLS via numpy lstsq; t-stats with plain SE)
- `rolling_betas(asset_returns, factor_returns, window=252) -> pd.DataFrame`

## src/math

### monte_carlo_methods.py
- `simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, rng=None, antithetic=False) -> np.ndarray` (n_paths, n_steps+1)
- `simulate_ou(x0, kappa, theta, sigma, T, n_steps, n_paths, rng=None) -> np.ndarray` (exact discretization)
- `simulate_cir(x0, kappa, theta, sigma, T, n_steps, n_paths, rng=None) -> np.ndarray` (full truncation)
- `simulate_regime_switching(S0, mus, sigmas, transition_matrix, T, n_steps, n_paths, rng=None) -> tuple[np.ndarray, np.ndarray]` (paths, regime_paths)
- `mc_price(payoff: np.ndarray, r, T) -> tuple[float, float]` — (price, standard_error)

### pde_solvers.py
- `cn_bs_price(S0, K, T, r, sigma, option_type="call", american=False, n_s=200, n_t=200, s_max_mult=4.0) -> float` (Crank–Nicolson; PSOR for American)
- `cn_bs_grid(...) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — (s_grid, t_grid, value_surface)

### numerical_linear_algebra.py
- `nearest_psd(A: np.ndarray, eps=1e-10) -> np.ndarray` (Higham-style eigenvalue clipping, preserve unit diagonal option `corr=False`)
- `is_psd(A, tol=1e-10) -> bool`
- `ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]` — (shrunk cov, shrinkage intensity), constant-correlation target
- `safe_cholesky(A) -> np.ndarray` (falls back to nearest_psd)

## src/portfolio

Weights are `pd.Series` indexed by asset, long-only by default via `bounds=(0.0, 1.0)`, fully invested (`sum == 1`). `mu: pd.Series`, `cov: pd.DataFrame` — annualized.

### mean_variance.py
- `@dataclass PortfolioResult: weights: pd.Series, expected_return: float, volatility: float, sharpe: float`
- `min_variance(cov, bounds=(0.0, 1.0)) -> PortfolioResult` (mu optional → ret/sharpe nan-safe)
- `max_sharpe(mu, cov, rf=0.0, bounds=(0.0, 1.0)) -> PortfolioResult`
- `efficient_frontier(mu, cov, n_points=50, rf=0.0, bounds=(0.0, 1.0)) -> pd.DataFrame` cols: expected_return, volatility, sharpe + one col per asset weight

### black_litterman.py
- `implied_equilibrium_returns(cov, market_weights, delta=2.5) -> pd.Series`
- `bl_posterior(cov, market_weights, P: np.ndarray, Q: np.ndarray, tau=0.05, omega=None, delta=2.5) -> tuple[pd.Series, pd.DataFrame]` (posterior mu, posterior cov; omega=None → Idzorek-style diag(tau·P·Σ·Pᵀ))

### risk_parity.py
- `risk_contributions(weights, cov) -> pd.Series` (fractional, sums to 1)
- `risk_parity_weights(cov, budget=None) -> pd.Series` (SLSQP on squared RC deviations)

### hierarchical_risk_parity.py
- `hrp_weights(returns: pd.DataFrame) -> pd.Series` (corr-distance, single linkage, quasi-diag, recursive bisection — Lopez de Prado 2016)

### robust_optimization.py
- `robust_max_sharpe(mu, cov, mu_uncertainty: pd.Series | float, rf=0.0, bounds=(0.0, 1.0)) -> PortfolioResult` (worst-case mu in box)
- `resampled_frontier(mu, cov, n_samples=200, n_points=25, n_obs=252, rng=None, bounds=(0.0, 1.0)) -> pd.DataFrame` (Michaud-style average weights)

## src/risk

`returns` = `pd.Series` (single portfolio) unless noted. `alpha=0.95` = confidence. VaR/ES returned as **positive loss numbers**.

### value_at_risk.py
- `historical_var(returns, alpha=0.95) -> float`
- `parametric_var(returns, alpha=0.95) -> float` (normal)
- `cornish_fisher_var(returns, alpha=0.95) -> float`
- `monte_carlo_var(mu, cov, weights, alpha=0.95, n_sims=100_000, rng=None) -> float` (daily params)

### expected_shortfall.py
- `historical_es(returns, alpha=0.95) -> float`
- `parametric_es(returns, alpha=0.95) -> float`

### drawdown_analysis.py
- `drawdown_series(returns) -> pd.Series`
- `max_drawdown(returns) -> float` (positive number)
- `drawdown_stats(returns) -> pd.DataFrame` (top-5 drawdowns: depth, start, trough, recovery, duration_days)

### stress_testing.py
- `@dataclass Scenario: name: str, shocks: dict[str, float]` (asset → return shock)
- `apply_scenario(weights, scenario) -> float` (portfolio P&L)
- `run_scenarios(weights, scenarios: list[Scenario]) -> pd.DataFrame`
- `correlation_stress(cov, stress_factor: float) -> pd.DataFrame` (blend corr toward 1, rebuild cov, ensure PSD via nearest_psd)
- `HISTORICAL_SCENARIOS: list[Scenario]` (GFC-2008, Covid-2020, 2022 rate shock, dot-com — approximate published asset-class moves)

## src/backtesting

### engine.py
- `@dataclass BacktestResult: returns: pd.Series, equity_curve: pd.Series, weights: pd.DataFrame, turnover: pd.Series, costs: pd.Series`
- `class Backtester(cost_model=None, slippage_model=None, rebalance="M")` with `.run(prices: pd.DataFrame, weight_fn: Callable[[pd.DataFrame], pd.Series], lookback=252) -> BacktestResult` — vectorized loop over rebalance dates; weight_fn sees only trailing window (no look-ahead); weights drift between rebalances.

### transaction_costs.py
- `class ProportionalCost(bps: float = 10.0)` with `.cost(turnover: float) -> float` (fraction of NAV)

### slippage.py
- `class FixedSlippage(bps: float = 5.0)` with `.cost(turnover: float) -> float`

### performance_metrics.py
- `annualized_return(returns, periods=252) -> float`, `annualized_vol(...)`, `sharpe_ratio(returns, rf=0.0, periods=252)`, `sortino_ratio(...)`, `calmar_ratio(...)`, `hit_rate(returns)`
- `summary(returns, rf=0.0, periods=252) -> pd.Series` (all of the above + max_drawdown, skew, kurtosis, VaR95, ES95)

## src/data

### market_data_loader.py
- `load_prices(tickers: list[str], start="2015-01-01", end=None, cache_dir="data/snapshots", refresh=False) -> pd.DataFrame` — adjusted closes; reads `{cache_dir}/prices.csv` when present and `refresh=False`; on `refresh=True` fetches via `yfinance` and rewrites cache. Raise `FileNotFoundError` with helpful message if no cache and yfinance unavailable.
- `to_returns(prices, log=False) -> pd.DataFrame`
- CLI: `python -m src.data.market_data_loader --refresh --tickers SPY QQQ ...`

### factor_data_loader.py
- `load_ff_factors(cache_dir="data/snapshots", refresh=False) -> pd.DataFrame` — Fama-French 5 factors + momentum, daily, decimal units; cached as `ff_factors.csv`; direct download of Ken French zip when refresh.

### data_validation.py
- `@dataclass ValidationReport: n_rows, n_missing: pd.Series, stale_runs: pd.Series, outliers: pd.Series, index_monotonic: bool, issues: list[str]` with `.passed` property
- `validate_prices(prices, max_missing_frac=0.05, outlier_z=8.0, stale_len=10) -> ValidationReport`

## src/visualization

Matplotlib figures returned (never shown/saved inside the function).

- `efficient_frontier.py`: `plot_efficient_frontier(frontier_df, assets_mu=None, assets_vol=None, highlight: dict[str, PortfolioResult] | None = None) -> plt.Figure`
- `volatility_surface.py`: `plot_volatility_surface(surface_df) -> plt.Figure` (3D)
- `risk_dashboard.py`: `plot_risk_dashboard(returns: pd.Series, benchmark: pd.Series | None = None) -> plt.Figure` (2×2: equity curve, drawdown, rolling vol, return histogram with VaR/ES lines)

## Testing conventions

- `pytest` from repo root; seeds fixed; numeric checks with explicit tolerances.
- Cross-model sanity anchors: put-call parity; binomial → BS convergence; CN-PDE ≈ BS; Merton with `lam=0` → BS; Heston MC ≈ Heston semi-analytic (loose tol); ES ≥ VaR; risk-parity RCs equal; all weight vectors sum to 1 and respect bounds; backtest has no look-ahead (weight_fn receives data strictly before rebalance date).
