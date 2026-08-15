"""
Relative-value signal for curve spreads: given a history of par yield curves,
compute the z-score of today's spread between two tenors against its own
historical distribution, and flag rich/cheap when that z-score is extreme.

This is a curve-shape (steepener/flattener) signal, not an outright-yield
signal: it's asking "is this SPREAD wide/narrow relative to where it's
normally been", not "are rates high or low".

Convention: spread(A, B) = yield(B) - yield(A) for A shorter than B, e.g.
spread("2Y", "10Y") is the classic "2s10s" steepness in bps. A positive
z-score means the spread is WIDE (steep) relative to history; very positive
-> curve unusually steep -> flag as "cheap steepener / rich flattener"
depending on which leg you're asking about. This module reports the z-score
and a plain rich/cheap direction on the SPREAD itself; translating that into
a specific trade (buy the 10Y, sell the 2Y, etc.) is left to the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RICH_CHEAP_Z_THRESHOLD = 1.5  # |z| beyond this is flagged, not just reported


def spread_bps(tenor_short: str, tenor_long: str, par_yields_pct: dict[str, float] | pd.Series) -> float:
    """Spread in bps: yield(tenor_long) - yield(tenor_short). Positive = normal/steep."""
    return (par_yields_pct[tenor_long] - par_yields_pct[tenor_short]) * 100


def spread_history_bps(tenor_short: str, tenor_long: str, curve_history: pd.DataFrame) -> pd.Series:
    """
    curve_history: DataFrame indexed by date, columns = tenors (as produced by
    data.curve_fetcher.fetch_curve_history), values = par yields in %.
    Returns a Series of the spread in bps at each historical date.
    """
    return (curve_history[tenor_long] - curve_history[tenor_short]) * 100


def relative_value_signal(tenor_short: str, tenor_long: str,
                           par_yields_pct: dict[str, float] | pd.Series,
                           curve_history: pd.DataFrame,
                           z_threshold: float = RICH_CHEAP_Z_THRESHOLD) -> dict:
    """
    Computes today's spread, its historical mean/std (from curve_history,
    EXCLUDING today so the z-score isn't measured against itself), the
    z-score, and a rich/cheap/neutral flag.
    """
    today_spread = spread_bps(tenor_short, tenor_long, par_yields_pct)
    hist_spreads = spread_history_bps(tenor_short, tenor_long, curve_history)

    mean = hist_spreads.mean()
    std = hist_spreads.std()

    if std < 1e-6 or np.isnan(std):
        # Guard against near-degenerate history (e.g. a mathematically constant
        # series that floating-point arithmetic leaves at ~1e-15 std instead of
        # exactly 0.0). An exact "std == 0" check does NOT catch this -- it
        # slips through and divides near-zero by near-zero, producing garbage
        # z-scores instead of the correct, well-defined z=0.
        z = 0.0
    else:
        z = (today_spread - mean) / std

    if z > z_threshold:
        flag = "steep vs history (spread rich/wide)"
    elif z < -z_threshold:
        flag = "flat vs history (spread cheap/narrow)"
    else:
        flag = "neutral"

    return {
        "pair": f"{tenor_short}{tenor_long}",
        "spread_bps": today_spread,
        "hist_mean_bps": float(mean),
        "hist_std_bps": float(std),
        "z_score": float(z),
        "flag": flag,
    }


def scan_curve_pairs(tenor_pairs: list[tuple[str, str]],
                      par_yields_pct: dict[str, float] | pd.Series,
                      curve_history: pd.DataFrame,
                      z_threshold: float = RICH_CHEAP_Z_THRESHOLD) -> pd.DataFrame:
    """
    Runs relative_value_signal across a list of (short, long) tenor pairs
    and returns a summary DataFrame, sorted by |z-score| descending so the
    most extreme dislocations surface first.
    """
    rows = [
        relative_value_signal(a, b, par_yields_pct, curve_history, z_threshold)
        for a, b in tenor_pairs
    ]
    df = pd.DataFrame(rows).set_index("pair")
    return df.reindex(df["z_score"].abs().sort_values(ascending=False).index)