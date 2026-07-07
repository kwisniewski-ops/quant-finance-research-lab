"""WRDS data layer: CRSP, Compustat, CCM links, and Fama-French factors.

Pulls the research-grade inputs for the lab's empirical modules — factor
construction, anomaly studies, and fundamentals-driven portfolios — from
WRDS PostgreSQL and caches them locally as parquet.

LICENSING — READ FIRST
----------------------
CRSP and Compustat data are licensed and may NOT be redistributed. Everything
this module writes lands in ``data/wrds/``, which is **gitignored**. Never
commit these files, never publish security-level extracts on the website.
Derived, aggregated results (factor portfolio returns you construct,
regression coefficients, summary statistics) are the publishable layer,
consistent with standard academic practice. When in doubt, confirm with WRDS
support under your institution's license.

Setup
-----
1. ``pip install wrds pyarrow``
2. First connection prompts for your WRDS username/password and offers to
   create a ``~/.pgpass`` entry so you are never prompted again. Credentials
   are handled entirely by the ``wrds`` package — never hardcode them.
3. Pull everything::

       python -m src.data.wrds_loader --all --username YOUR_WRDS_USER

   or individual pieces with ``--crsp-monthly --compustat-annual ...``.

What gets pulled (the manifest)
-------------------------------
===============================  ============================  ==========================================
File                             WRDS source                   Purpose
===============================  ============================  ==========================================
crsp_monthly.parquet             crsp.msf + msenames +         Monthly returns w/ delisting adjustment,
                                 msedelist                     market equity; universe for sorts (1963-)
crsp_daily_{year}.parquet        crsp.dsf                      Daily returns (rolling betas, PEAD);
                                                               chunked by year, optional (large)
compustat_annual.parquet         comp.funda                    Point-in-time annual fundamentals +
                                                               Fama-French book equity
compustat_quarterly.parquet      comp.fundq                    Quarterly fundamentals + earnings
                                                               announcement dates (rdq) for PEAD
ccm_links.parquet                crsp.ccmxpf_lnkhist           GVKEY↔PERMNO link table with validity
                                                               windows (LC/LU, primary links only)
ff_factors_monthly.parquet       ff.factors_monthly            Published FF factors — validation target
                                                               for our from-scratch replication
===============================  ============================  ==========================================

References
----------
Fama & French (1993); Shumway (1997) on delisting bias; standard CCM merge
conventions per WRDS documentation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WRDS_DIR = Path("data/wrds")

# Performance-related delisting codes for which a missing delisting return is
# imputed at -30% (Shumway 1997 convention, documented deviation: applied to
# both NYSE/AMEX and NASDAQ for simplicity).
_PERF_DELIST_CODES = (500, *range(520, 585))


def _connect(username: str | None = None):
    """Open a WRDS connection (imports lazily so the repo works without wrds)."""
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'wrds' package is required for WRDS pulls: pip install wrds pyarrow"
        ) from exc
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def _write(df: pd.DataFrame, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_parquet(path, index=False)
    logger.info("wrote %s: %d rows, %.1f MB", path, len(df), path.stat().st_size / 1e6)
    return path


# --------------------------------------------------------------------------
# CRSP
# --------------------------------------------------------------------------

def pull_crsp_monthly(
    conn,
    start: str = "1963-01-01",
    out_dir: Path = WRDS_DIR,
) -> pd.DataFrame:
    """Monthly CRSP stock file with delisting-adjusted returns and market equity.

    Universe: ordinary common shares (SHRCD 10, 11) on NYSE/AMEX/NASDAQ
    (EXCHCD 1, 2, 3). Delisting returns are compounded into the final month
    per Shumway (1997); missing performance-related delisting returns are
    imputed at -30%.

    Columns: permno, date, ret (delist-adjusted), retx, prc, shrout,
    me (|prc|*shrout, $ thousands), exchcd, shrcd, siccd, hexcd.
    """
    query = f"""
        select a.permno, a.date, a.ret, a.retx, a.prc, a.shrout,
               b.exchcd, b.shrcd, b.siccd,
               c.dlret, c.dlstcd
        from crsp.msf a
        join crsp.msenames b
          on a.permno = b.permno
         and b.namedt <= a.date and a.date <= b.nameendt
        left join crsp.msedelist c
          on a.permno = c.permno
         and date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
        where a.date >= '{start}'
          and b.shrcd in (10, 11)
          and b.exchcd in (1, 2, 3)
    """
    df = conn.raw_sql(query, date_cols=["date"])
    df["permno"] = df["permno"].astype(int)

    # Impute missing performance-related delisting returns, then compound.
    perf = df["dlstcd"].isin(_PERF_DELIST_CODES) & df["dlret"].isna()
    df.loc[perf, "dlret"] = -0.30
    has_dl = df["dlret"].notna()
    df.loc[has_dl, "ret"] = (1 + df.loc[has_dl, "ret"].fillna(0)) * (
        1 + df.loc[has_dl, "dlret"]
    ) - 1

    df["me"] = df["prc"].abs() * df["shrout"]
    df = df.drop(columns=["dlret", "dlstcd"]).sort_values(["permno", "date"])
    _write(df, "crsp_monthly.parquet", out_dir)
    return df


def pull_crsp_daily(
    conn,
    start_year: int = 1990,
    end_year: int | None = None,
    out_dir: Path = WRDS_DIR,
) -> list[Path]:
    """Daily CRSP returns, chunked one parquet per year (the table is large).

    Used for rolling-beta estimation (low-beta anomaly) and event studies
    (PEAD). Restricting to 1990+ keeps the pull to a manageable size; widen
    if a study needs it. Columns: permno, date, ret, prc, shrout, vol.
    """
    end_year = end_year or pd.Timestamp.today().year
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        query = f"""
            select a.permno, a.date, a.ret, a.prc, a.shrout, a.vol
            from crsp.dsf a
            join crsp.msenames b
              on a.permno = b.permno
             and b.namedt <= a.date and a.date <= b.nameendt
            where a.date between '{year}-01-01' and '{year}-12-31'
              and b.shrcd in (10, 11)
              and b.exchcd in (1, 2, 3)
        """
        df = conn.raw_sql(query, date_cols=["date"])
        if df.empty:
            logger.warning("crsp daily %d: empty, skipping", year)
            continue
        df["permno"] = df["permno"].astype(int)
        paths.append(_write(df, f"crsp_daily_{year}.parquet", out_dir))
    return paths


# --------------------------------------------------------------------------
# Compustat
# --------------------------------------------------------------------------

def _book_equity(df: pd.DataFrame) -> pd.Series:
    """Fama-French book equity.

    BE = stockholders' equity + deferred taxes & ITC - preferred stock, where
    stockholders' equity is SEQ, else CEQ + PSTK, else AT - LT; preferred is
    PSTKRV, else PSTKL, else PSTK, else 0; TXDITC missing -> 0.
    """
    se = df["seq"].fillna(df["ceq"] + df["pstk"].fillna(0)).fillna(df["at"] - df["lt"])
    pref = df["pstkrv"].fillna(df["pstkl"]).fillna(df["pstk"]).fillna(0)
    be = se + df["txditc"].fillna(0) - pref
    return be.where(be > 0)  # non-positive BE excluded per FF convention


def pull_compustat_annual(
    conn,
    start: str = "1962-01-01",
    out_dir: Path = WRDS_DIR,
) -> pd.DataFrame:
    """Annual Compustat fundamentals with Fama-French book equity.

    Standard filters: industrial format, standardized, domestic, consolidated.
    Fields cover the value/quality/profitability/accrual constructions:
    balance sheet (at, lt, seq, ceq, pstk*, txditc, act, lct, che, dlc, dltt),
    income (ib, ni, sale, cogs, xsga, xint, dp), cash flow (oancf), and
    per-share (prcc_f, csho). Adds ``be`` (book equity).
    """
    query = f"""
        select gvkey, datadate, fyear,
               at, lt, seq, ceq, pstk, pstkrv, pstkl, txditc,
               act, lct, che, dlc, dltt, ib, ni, sale, cogs, xsga, xint, dp,
               oancf, prcc_f, csho
        from comp.funda
        where indfmt = 'INDL' and datafmt = 'STD'
          and popsrc = 'D' and consol = 'C'
          and datadate >= '{start}'
    """
    df = conn.raw_sql(query, date_cols=["datadate"])
    df["be"] = _book_equity(df)
    df = df.sort_values(["gvkey", "datadate"])
    _write(df, "compustat_annual.parquet", out_dir)
    return df


def pull_compustat_quarterly(
    conn,
    start: str = "1971-01-01",
    out_dir: Path = WRDS_DIR,
) -> pd.DataFrame:
    """Quarterly Compustat fundamentals including earnings announcement dates.

    ``rdq`` (report date of quarterly earnings) is the event anchor for
    post-earnings-announcement-drift studies; ``ibq``/``epspxq`` feed SUE.
    """
    query = f"""
        select gvkey, datadate, fyearq, fqtr, rdq,
               ibq, atq, ltq, seqq, ceqq, saleq, niq, epspxq, cshprq, prccq
        from comp.fundq
        where indfmt = 'INDL' and datafmt = 'STD'
          and popsrc = 'D' and consol = 'C'
          and datadate >= '{start}'
    """
    df = conn.raw_sql(query, date_cols=["datadate", "rdq"])
    df = df.sort_values(["gvkey", "datadate"])
    _write(df, "compustat_quarterly.parquet", out_dir)
    return df


# --------------------------------------------------------------------------
# Links and factors
# --------------------------------------------------------------------------

def pull_ccm_links(conn, out_dir: Path = WRDS_DIR) -> pd.DataFrame:
    """CRSP/Compustat merged link table (GVKEY <-> PERMNO with date windows).

    Keeps researched links (LC, LU) and primary securities (P, C). Open-ended
    links get today's date as the window end.
    """
    query = """
        select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where linktype in ('LC', 'LU')
          and linkprim in ('P', 'C')
          and lpermno is not null
    """
    df = conn.raw_sql(query, date_cols=["linkdt", "linkenddt"])
    df["permno"] = df["permno"].astype(int)
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp.today().normalize())
    _write(df, "ccm_links.parquet", out_dir)
    return df


def pull_ff_factors(conn, out_dir: Path = WRDS_DIR) -> pd.DataFrame:
    """Published Fama-French monthly factors from WRDS (validation target)."""
    query = "select date, mktrf, smb, hml, rf, umd from ff.factors_monthly"
    df = conn.raw_sql(query, date_cols=["date"]).sort_values("date")
    _write(df, "ff_factors_monthly.parquet", out_dir)
    return df


# --------------------------------------------------------------------------
# Merge helper
# --------------------------------------------------------------------------

def link_compustat_crsp(
    compustat: pd.DataFrame,
    links: pd.DataFrame,
    date_col: str = "datadate",
) -> pd.DataFrame:
    """Attach PERMNOs to Compustat rows using CCM link validity windows.

    A row is linked when ``linkdt <= datadate <= linkenddt``. Rows without a
    valid link are dropped (they have no CRSP counterpart).
    """
    merged = compustat.merge(
        links[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner"
    )
    valid = (merged["linkdt"] <= merged[date_col]) & (
        merged[date_col] <= merged["linkenddt"]
    )
    return merged.loc[valid].drop(columns=["linkdt", "linkenddt"])


# --------------------------------------------------------------------------
# Local access (post-pull)
# --------------------------------------------------------------------------

def load_wrds(name: str, wrds_dir: Path = WRDS_DIR) -> pd.DataFrame:
    """Load a previously pulled WRDS parquet by short name.

    Names: 'crsp_monthly', 'compustat_annual', 'compustat_quarterly',
    'ccm_links', 'ff_factors_monthly'. Daily files load via
    ``load_crsp_daily(years)``.
    """
    path = wrds_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data.wrds_loader --all "
            f"(requires WRDS credentials; see module docstring)."
        )
    return pd.read_parquet(path)


def load_crsp_daily(years: range | list[int], wrds_dir: Path = WRDS_DIR) -> pd.DataFrame:
    """Concatenate yearly daily-CRSP parquets for the requested years."""
    frames = []
    for year in years:
        path = wrds_dir / f"crsp_daily_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Pull with: python -m src.data.wrds_loader "
                f"--crsp-daily --daily-start {min(years)} --daily-end {max(years)}"
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Pull WRDS data into data/wrds/ (gitignored — do not redistribute)."
    )
    parser.add_argument("--username", default=None, help="WRDS username")
    parser.add_argument("--all", action="store_true", help="pull everything below")
    parser.add_argument("--crsp-monthly", action="store_true")
    parser.add_argument("--crsp-daily", action="store_true")
    parser.add_argument("--compustat-annual", action="store_true")
    parser.add_argument("--compustat-quarterly", action="store_true")
    parser.add_argument("--ccm-links", action="store_true")
    parser.add_argument("--ff-factors", action="store_true")
    parser.add_argument("--start", default="1963-01-01", help="CRSP monthly start date")
    parser.add_argument("--daily-start", type=int, default=1990)
    parser.add_argument("--daily-end", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    conn = _connect(args.username)
    try:
        if args.all or args.crsp_monthly:
            pull_crsp_monthly(conn, start=args.start)
        if args.all or args.compustat_annual:
            pull_compustat_annual(conn)
        if args.all or args.compustat_quarterly:
            pull_compustat_quarterly(conn)
        if args.all or args.ccm_links:
            pull_ccm_links(conn)
        if args.all or args.ff_factors:
            pull_ff_factors(conn)
        if args.all or args.crsp_daily:
            pull_crsp_daily(conn, start_year=args.daily_start, end_year=args.daily_end)
    finally:
        conn.close()
    logging.info("Done. Files in %s — gitignored; do not commit or redistribute.", WRDS_DIR)


if __name__ == "__main__":  # pragma: no cover
    main()
