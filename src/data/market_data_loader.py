"""Market data loading with a cache-first design.

``load_prices`` reads the CSV snapshot in ``data/snapshots/`` when
present so that notebooks and tests run offline and reproducibly;
``yfinance`` is imported *only* inside the ``refresh=True`` branch. If
no real snapshot exists, a clearly-labeled synthetic demo file
(``prices_synthetic_demo.csv``) is used as a last resort with a logged
warning.

CLI
---
Refresh the snapshot from the repo root::

    python -m src.data.market_data_loader --refresh \\
        --tickers SPY QQQ IWM EFA EEM AGG TLT LQD GLD DBC VNQ \\
                  USMV MTUM VLUE QUAL \\
        --start 2015-01-01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_prices", "to_returns", "DEFAULT_TICKERS"]

logger = logging.getLogger(__name__)

DEFAULT_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "LQD",
    "GLD", "DBC", "VNQ", "USMV", "MTUM", "VLUE", "QUAL",
]

_CACHE_FILE = "prices.csv"
_SYNTHETIC_FILE = "prices_synthetic_demo.csv"


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()
    return df.astype(float)


def _fetch_yfinance(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Download adjusted daily closes. Only called on refresh."""
    try:
        import yfinance as yf  # noqa: PLC0415 — lazy by design (cache-first)
    except ImportError as exc:  # pragma: no cover
        raise FileNotFoundError(
            "No price cache found and yfinance is not installed. "
            "Run `pip install yfinance`, then "
            "`python -m src.data.market_data_loader --refresh --tickers "
            + " ".join(tickers)
            + "` from the repo root to build data/snapshots/prices.csv."
        ) from exc

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {tickers}")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(
        columns={"Close": tickers[0]}
    )
    close = close.reindex(columns=tickers)
    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    return close.astype(float)


def load_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: str = "data/snapshots",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load adjusted daily close prices, cache first.

    Resolution order:

    1. ``refresh=False`` and ``{cache_dir}/prices.csv`` exists — read
       the snapshot (no network, no yfinance import).
    2. ``refresh=True`` — fetch via yfinance and rewrite the snapshot.
    3. No snapshot, no refresh — fall back to
       ``prices_synthetic_demo.csv`` **with a logged warning** if it
       exists, else raise ``FileNotFoundError`` with instructions.

    Parameters
    ----------
    tickers : list of str
        Tickers to return (must be a subset of the cached columns when
        reading from cache).
    start, end : str
        ISO date bounds applied to the returned frame (``end=None`` =
        through the last cached date).
    cache_dir : str, default "data/snapshots"
        Snapshot directory, relative to the repo root.
    refresh : bool, default False
        Force a network fetch and rewrite the cache.

    Returns
    -------
    pd.DataFrame
        Adjusted closes, DatetimeIndex ascending, columns = tickers.

    Raises
    ------
    FileNotFoundError
        No cache, no synthetic fallback, and ``refresh=False``.
    KeyError
        Requested tickers missing from the cached snapshot.
    """
    if not tickers:
        raise ValueError("tickers list is empty")
    tickers = [str(t).upper() for t in tickers]
    cache_path = Path(cache_dir) / _CACHE_FILE

    if refresh:
        prices = _fetch_yfinance(tickers, start, end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path, float_format="%.6f")
        logger.info("wrote %d rows x %d tickers to %s", *prices.shape, cache_path)
        df = prices
    elif cache_path.exists():
        df = _read_cache(cache_path)
        logger.debug("loaded price cache %s (%d rows)", cache_path, len(df))
    else:
        synthetic_path = Path(cache_dir) / _SYNTHETIC_FILE
        if synthetic_path.exists():
            logger.warning(
                "No real price snapshot at %s — falling back to SYNTHETIC demo data "
                "(%s). Results are illustrative only. Build the real cache with: "
                "python -m src.data.market_data_loader --refresh --tickers %s",
                cache_path, synthetic_path, " ".join(tickers),
            )
            df = _read_cache(synthetic_path)
        else:
            raise FileNotFoundError(
                f"No price cache at {cache_path}. Build it with:\n"
                "  python -m src.data.market_data_loader --refresh --tickers "
                + " ".join(tickers)
                + "\nor pass refresh=True to load_prices()."
            )

    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise KeyError(
            f"tickers {missing} not in cached snapshot (has {list(df.columns)}); "
            "re-run with refresh=True to fetch them"
        )
    out = df.loc[:, tickers]
    out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    if out.empty:
        raise ValueError(f"no cached data in requested range [{start}, {end}]")
    return out


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Convert a price panel to simple (default) or log returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices, DatetimeIndex, columns = tickers.
    log : bool, default False
        If True, return :math:`\\ln(P_t / P_{t-1})`; otherwise
        :math:`P_t / P_{t-1} - 1`.

    Returns
    -------
    pd.DataFrame
        Returns with the first (undefined) row dropped.
    """
    if not isinstance(prices, (pd.DataFrame, pd.Series)):
        raise TypeError(f"prices must be a pd.DataFrame, got {type(prices).__name__}")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive to compute returns")
    if log:
        return np.log(prices / prices.shift(1)).iloc[1:]
    return prices.pct_change(fill_method=None).iloc[1:]


def _main() -> None:  # pragma: no cover — exercised via CLI
    parser = argparse.ArgumentParser(description="Refresh the price snapshot cache.")
    parser.add_argument("--refresh", action="store_true", help="fetch from yfinance and rewrite the cache")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="tickers to fetch")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache-dir", default="data/snapshots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prices = load_prices(
        args.tickers, start=args.start, end=args.end,
        cache_dir=args.cache_dir, refresh=args.refresh,
    )
    logger.info(
        "prices: %d rows x %d tickers, %s -> %s",
        len(prices), prices.shape[1], prices.index[0].date(), prices.index[-1].date(),
    )


if __name__ == "__main__":  # pragma: no cover
    _main()
