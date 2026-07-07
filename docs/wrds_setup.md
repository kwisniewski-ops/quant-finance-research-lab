# WRDS Data Layer — Setup and Usage

The lab's professional empirical modules (Fama-French replication, anomaly
research, fundamentals-driven portfolios) run on CRSP and Compustat via WRDS.
This document covers setup, the pull, and — most importantly — what may and
may not be published.

## Licensing: the one rule that matters

**CRSP and Compustat data are licensed to your institution and may not be
redistributed.** Concretely, for this repo:

- Raw or security-level extracts live only in `data/wrds/`, which is
  **gitignored**. Never commit them, never upload them, never serve them from
  the website.
- What *can* be published — consistent with how academic papers handle the
  same data — is the **derived, aggregated layer**: factor portfolio return
  series you construct, regression coefficients and t-statistics, decile
  spreads, summary statistics, and charts thereof.
- If a specific output feels borderline (e.g., a long time series for a
  single named stock), it probably is. Confirm with WRDS support under your
  institution's license before publishing it.

## One-time setup

```bash
pip install wrds pyarrow
```

The first connection prompts for your WRDS username and password and offers
to write a `~/.pgpass` entry so future connections are automatic. Credentials
are handled by the `wrds` package; nothing in this repo stores them.

## The pull

```bash
# everything (recommended first run; ~10-25 min, a few GB with daily data)
python -m src.data.wrds_loader --all --username YOUR_WRDS_USER

# or piecewise
python -m src.data.wrds_loader --crsp-monthly --compustat-annual --ccm-links --ff-factors
python -m src.data.wrds_loader --crsp-daily --daily-start 1990   # the big one
```

| File | Source | Approx. size | Feeds |
|---|---|---|---|
| `crsp_monthly.parquet` | crsp.msf/msenames/msedelist | ~150 MB | Everything: sorts, factor construction |
| `compustat_annual.parquet` | comp.funda | ~100 MB | Book equity (HML), quality, accruals |
| `compustat_quarterly.parquet` | comp.fundq | ~150 MB | SUE / PEAD (needs `rdq`) |
| `ccm_links.parquet` | crsp.ccmxpf_lnkhist | < 1 MB | GVKEY↔PERMNO merge |
| `ff_factors_monthly.parquet` | ff.factors_monthly | < 1 MB | Validation target for replication |
| `crsp_daily_{year}.parquet` | crsp.dsf | ~50-120 MB/yr | Rolling betas, event studies |

## Methodological conventions baked into the loader

- **Universe**: ordinary common shares (SHRCD 10/11), NYSE/AMEX/NASDAQ
  (EXCHCD 1/2/3) — the standard Fama-French universe.
- **Delisting bias**: delisting returns compounded into the final month;
  missing performance-related delisting returns imputed at −30%
  (Shumway 1997).
- **Book equity**: SEQ → CEQ+PSTK → AT−LT fallback, plus TXDITC, minus
  preferred (PSTKRV → PSTKL → PSTK); non-positive BE excluded.
- **CCM merge**: researched links only (LC/LU), primary securities (P/C),
  link validity windows enforced via `link_compustat_crsp()`.
- **Look-ahead discipline** is the *consumer's* job: annual fundamentals for
  fiscal year *t* are usable from July of *t+1* (FF timing); quarterly data
  usable only after `rdq`.

## Using the data

```python
from src.data.wrds_loader import load_wrds, load_crsp_daily, link_compustat_crsp

crsp = load_wrds("crsp_monthly")
funda = load_wrds("compustat_annual")
links = load_wrds("ccm_links")
funda_linked = link_compustat_crsp(funda, links)
daily_2020s = load_crsp_daily(range(2020, 2026))
```

## Research roadmap built on this layer

1. **Fama-French replication** (notebook 07): construct SMB/HML/UMD from the
   2×3 sorts, validate against `ff_factors_monthly` (target: ρ ≥ 0.98),
   then Fama-MacBeth cross-sectional regressions.
2. **Anomaly program** (notebook 08): momentum, Sloan accruals,
   profitability, low-beta — decile long-shorts with proper lags and
   delisting handling.
3. **Fundamentals portfolio lab** (notebook 09): Compustat composite scores
   feeding the existing optimizers and the look-ahead-free backtester on a
   real equity universe.
