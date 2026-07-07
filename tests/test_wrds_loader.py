"""Tests for the WRDS loader's pure transformation logic.

No WRDS connection required — these exercise the book-equity construction
and the CCM link-window merge on synthetic frames.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.wrds_loader import _book_equity, link_compustat_crsp


def test_book_equity_prefers_seq_then_ceq_pstk_then_at_lt():
    df = pd.DataFrame(
        {
            "seq": [100.0, np.nan, np.nan],
            "ceq": [np.nan, 80.0, np.nan],
            "pstk": [np.nan, 5.0, np.nan],
            "at": [np.nan, np.nan, 200.0],
            "lt": [np.nan, np.nan, 120.0],
            "pstkrv": [10.0, np.nan, np.nan],
            "pstkl": [np.nan, 4.0, np.nan],
            "txditc": [2.0, np.nan, np.nan],
        }
    )
    be = _book_equity(df)
    # seq + txditc - pstkrv = 100 + 2 - 10
    assert be.iloc[0] == pytest.approx(92.0)
    # (ceq + pstk) + 0 - pstkl = 85 - 4
    assert be.iloc[1] == pytest.approx(81.0)
    # (at - lt) + 0 - 0 = 80
    assert be.iloc[2] == pytest.approx(80.0)


def test_book_equity_nonpositive_is_nan():
    df = pd.DataFrame(
        {
            "seq": [5.0],
            "ceq": [np.nan],
            "pstk": [np.nan],
            "at": [np.nan],
            "lt": [np.nan],
            "pstkrv": [10.0],
            "pstkl": [np.nan],
            "txditc": [0.0],
        }
    )
    assert _book_equity(df).isna().iloc[0]


def test_ccm_link_respects_date_windows():
    comp = pd.DataFrame(
        {
            "gvkey": ["001", "001", "002"],
            "datadate": pd.to_datetime(["2000-12-31", "2010-12-31", "2005-12-31"]),
            "be": [1.0, 2.0, 3.0],
        }
    )
    links = pd.DataFrame(
        {
            "gvkey": ["001", "002"],
            "permno": [10001, 10002],
            "linkdt": pd.to_datetime(["1995-01-01", "2006-01-01"]),
            "linkenddt": pd.to_datetime(["2005-12-31", "2020-12-31"]),
        }
    )
    out = link_compustat_crsp(comp, links)
    # gvkey 001 in 2000 links; 001 in 2010 is outside the window; 002 in 2005
    # predates its link start.
    assert len(out) == 1
    assert out.iloc[0]["permno"] == 10001
    assert out.iloc[0]["be"] == 1.0
