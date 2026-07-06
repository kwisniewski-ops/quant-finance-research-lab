"""Drawdown analysis.

Given returns :math:`r_t`, the equity curve is
:math:`E_t = \\prod_{s \\le t} (1 + r_s)` and the drawdown series is

.. math::

    DD_t = \\frac{E_t}{\\max_{s \\le t} E_s} - 1 \\le 0 .

``max_drawdown`` reports :math:`\\max_t |DD_t|` as a positive number.
``drawdown_stats`` decomposes the path into distinct drawdown episodes
and tabulates the five deepest.

References
----------
Magdon-Ismail, M. & Atiya, A. (2004). "Maximum Drawdown". *Risk*,
17(10), 99-102.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["drawdown_series", "max_drawdown", "drawdown_stats"]


def _validate_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns).__name__}")
    r = returns.dropna()
    if r.empty:
        raise ValueError("returns has no non-NaN observations")
    if (r <= -1.0).any():
        raise ValueError("returns contains values <= -100%; equity curve would be non-positive")
    return r.astype(float)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown path from a return series.

    Parameters
    ----------
    returns : pd.Series
        Period returns (decimal).

    Returns
    -------
    pd.Series
        Drawdown at each date, in ``[-1, 0]`` (0 at running highs).
    """
    r = _validate_returns(returns)
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive number.

    Parameters
    ----------
    returns : pd.Series
        Period returns.

    Returns
    -------
    float
        ``max |drawdown|`` in decimal units (e.g. 0.25 = -25%).
    """
    return float(-drawdown_series(returns).min())


def drawdown_stats(returns: pd.Series) -> pd.DataFrame:
    """Table of the five deepest drawdown episodes.

    An episode starts at the first date below the prior peak and ends
    at the first date the peak is regained (``recovery`` is ``NaT`` if
    the drawdown is still open at the end of the sample).

    Parameters
    ----------
    returns : pd.Series
        Period returns indexed by date.

    Returns
    -------
    pd.DataFrame
        Up to five rows sorted by depth, columns: ``depth`` (positive
        decimal), ``start`` (last peak date before the episode),
        ``trough`` (date of maximum drawdown), ``recovery`` (date the
        prior peak is regained or ``NaT``), ``duration_days``
        (observations from start through recovery, or through sample
        end if unrecovered).
    """
    dd = drawdown_series(returns)
    in_dd = dd.to_numpy() < 0
    idx = dd.index

    episodes: list[dict] = []
    i = 0
    n = len(dd)
    while i < n:
        if not in_dd[i]:
            i += 1
            continue
        start_pos = i - 1 if i > 0 else 0  # last date at the peak
        j = i
        while j < n and in_dd[j]:
            j += 1
        segment = dd.iloc[i:j]
        trough_date = segment.idxmin()
        recovery = idx[j] if j < n else pd.NaT
        end_pos = j if j < n else n - 1
        episodes.append(
            {
                "depth": float(-segment.min()),
                "start": idx[start_pos],
                "trough": trough_date,
                "recovery": recovery,
                "duration_days": int(end_pos - start_pos),
            }
        )
        i = j

    cols = ["depth", "start", "trough", "recovery", "duration_days"]
    if not episodes:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(episodes, columns=cols)
    out = out.sort_values("depth", ascending=False).head(5).reset_index(drop=True)
    return out
