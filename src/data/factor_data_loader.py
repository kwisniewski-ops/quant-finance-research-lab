"""Fama-French factor data loading (cache-first).

Downloads the daily *Fama/French 5 Factors (2x3)* and *Momentum*
research files from Ken French's data library, converts them from
percent to **decimal** units, merges them into a single frame, and
caches the result as ``data/snapshots/ff_factors.csv``. As with the
price loader, the network branch runs only when ``refresh=True``.

Columns: ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``, ``RF``,
``Mom``.

CLI
---
::

    python -m src.data.factor_data_loader --refresh

References
----------
Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model".
*Journal of Financial Economics*, 116(1), 1-22.

Carhart, M. (1997). "On Persistence in Mutual Fund Performance".
*Journal of Finance*, 52(1), 57-82.
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

__all__ = ["load_ff_factors"]

logger = logging.getLogger(__name__)

_CACHE_FILE = "ff_factors.csv"
_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FF5_ZIP = _BASE + "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOM_ZIP = _BASE + "F-F_Momentum_Factor_daily_CSV.zip"


def _parse_french_csv(raw: bytes) -> pd.DataFrame:
    """Parse a Ken French daily CSV (percent units, ragged footer)."""
    text = raw.decode("latin-1")
    lines = text.splitlines()
    # Header row: first line whose first comma-separated field is empty
    # (the date column is unlabeled) and that has named factor columns.
    start = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == "" and any(parts[1:]):
            start = i
            break
    if start is None:
        raise ValueError("could not locate the header row in the Ken French CSV")
    header_parts = [p.strip() for p in lines[start].split(",")]
    # Keep only named columns (files often carry trailing empty fields).
    col_pos = [i for i, p in enumerate(header_parts) if i > 0 and p != ""]
    header = [header_parts[i] for i in col_pos]
    rows = []
    for line in lines[start + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if (
            len(parts) <= max(col_pos)
            or not parts[0].isdigit()
            or len(parts[0]) != 8
            or any(parts[i] == "" for i in col_pos)
        ):
            # Blank line, annual table, or copyright footer — stop at
            # the first non-daily row after data started.
            if rows:
                break
            continue
        rows.append([parts[0]] + [parts[i] for i in col_pos])
    df = pd.DataFrame(rows, columns=["Date"] + header)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").astype(float)
    df = df.replace([-99.99, -999.0], pd.NA).astype(float)
    return df / 100.0  # percent -> decimal


def _download_zip_csv(url: str) -> pd.DataFrame:
    import requests  # noqa: PLC0415 — lazy by design (cache-first)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(name)
    return _parse_french_csv(raw)


def load_ff_factors(cache_dir: str = "data/snapshots", refresh: bool = False) -> pd.DataFrame:
    """Load daily Fama-French 5 factors + momentum, in decimal units.

    Parameters
    ----------
    cache_dir : str, default "data/snapshots"
        Snapshot directory containing/receiving ``ff_factors.csv``.
    refresh : bool, default False
        Download from Ken French's library and rewrite the cache.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns ``Mkt-RF, SMB, HML, RMW, CMA, RF, Mom``,
        decimal daily returns.

    Raises
    ------
    FileNotFoundError
        No cache and ``refresh=False``.

    References
    ----------
    Fama & French (2015); Carhart (1997). Data: Ken French Data
    Library, Tuck School of Business at Dartmouth.
    """
    cache_path = Path(cache_dir) / _CACHE_FILE

    if not refresh:
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True).astype(float)
            df.index.name = "Date"
            logger.debug("loaded factor cache %s (%d rows)", cache_path, len(df))
            return df.sort_index()
        raise FileNotFoundError(
            f"No factor cache at {cache_path}. Build it with:\n"
            "  python -m src.data.factor_data_loader --refresh\n"
            "or pass refresh=True to load_ff_factors()."
        )

    ff5 = _download_zip_csv(_FF5_ZIP)
    mom = _download_zip_csv(_MOM_ZIP)
    mom.columns = ["Mom" for _ in mom.columns]  # file labels it 'Mom   '
    merged = ff5.join(mom, how="inner").sort_index()
    merged = merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, float_format="%.6f")
    logger.info("wrote %d rows to %s (%s -> %s)", len(merged), cache_path,
                merged.index[0].date(), merged.index[-1].date())
    return merged


def _main() -> None:  # pragma: no cover — exercised via CLI
    parser = argparse.ArgumentParser(description="Refresh the Fama-French factor snapshot.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", default="data/snapshots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = load_ff_factors(cache_dir=args.cache_dir, refresh=args.refresh)
    logger.info("factors: %d rows, %s -> %s", len(df), df.index[0].date(), df.index[-1].date())


if __name__ == "__main__":  # pragma: no cover
    _main()
