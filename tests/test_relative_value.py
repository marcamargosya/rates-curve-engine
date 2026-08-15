import numpy as np
import pandas as pd
import pytest

from data.sample_curve import SAMPLE_PAR_CURVE
from signals.relative_value import spread_bps, relative_value_signal, scan_curve_pairs


@pytest.fixture
def synthetic_history():
    np.random.seed(42)
    n_days = 250
    dates = pd.date_range("2025-08-01", periods=n_days, freq="B")
    hist = pd.DataFrame(index=dates)
    for tenor, base in SAMPLE_PAR_CURVE.items():
        noise = np.cumsum(np.random.normal(0, 0.02, n_days))
        noise -= noise.mean()
        hist[tenor] = base + noise
    return hist


def test_spread_bps_sign_convention():
    """spread(short, long) should be long_yield - short_yield, in bps."""
    s = spread_bps("2Y", "10Y", SAMPLE_PAR_CURVE)
    expected = (SAMPLE_PAR_CURVE["10Y"] - SAMPLE_PAR_CURVE["2Y"]) * 100
    assert s == pytest.approx(expected)


def test_zscore_zero_when_today_matches_history_mean(synthetic_history):
    """If today's curve equals the levels history was generated around, z should be ~0."""
    result = relative_value_signal("2Y", "10Y", SAMPLE_PAR_CURVE, synthetic_history)
    assert abs(result["z_score"]) < 0.5
    assert result["flag"] == "neutral"


def test_zscore_flags_rich_on_positive_dislocation(synthetic_history):
    """A spread blown out well beyond its historical range should flag as rich/steep."""
    dislocated = dict(SAMPLE_PAR_CURVE)
    dislocated["10Y"] = dislocated["10Y"] + 0.50  # +50bp steepening shock
    result = relative_value_signal("2Y", "10Y", dislocated, synthetic_history)
    assert result["z_score"] > 1.5
    assert "steep" in result["flag"]


def test_zscore_flags_cheap_on_negative_dislocation(synthetic_history):
    """A spread compressed well beyond its historical range should flag as cheap/flat."""
    dislocated = dict(SAMPLE_PAR_CURVE)
    dislocated["10Y"] = dislocated["10Y"] - 0.50  # -50bp flattening shock
    result = relative_value_signal("2Y", "10Y", dislocated, synthetic_history)
    assert result["z_score"] < -1.5
    assert "flat" in result["flag"]


def test_scan_sorts_by_absolute_zscore_descending(synthetic_history):
    dislocated = dict(SAMPLE_PAR_CURVE)
    dislocated["10Y"] = dislocated["10Y"] + 0.50
    pairs = [("2Y", "5Y"), ("2Y", "10Y"), ("5Y", "10Y"), ("10Y", "30Y")]
    scan = scan_curve_pairs(pairs, dislocated, synthetic_history)
    z_abs = scan["z_score"].abs().values
    assert all(z_abs[i] >= z_abs[i + 1] for i in range(len(z_abs) - 1))


def test_scan_leaves_undisturbed_pairs_neutral(synthetic_history):
    """A pair with no shock to either leg should stay neutral even when other pairs are flagged."""
    dislocated = dict(SAMPLE_PAR_CURVE)
    dislocated["10Y"] = dislocated["10Y"] + 0.50
    pairs = [("2Y", "5Y"), ("2Y", "10Y")]
    scan = scan_curve_pairs(pairs, dislocated, synthetic_history)
    assert scan.loc["2Y5Y", "flag"] == "neutral"
    assert "steep" in scan.loc["2Y10Y", "flag"]


def test_zero_variance_history_does_not_crash():
    """A degenerate flat (zero-variance) history should give z=0, not NaN/inf."""
    flat_hist = pd.DataFrame({
        "2Y": [3.72] * 10,
        "10Y": [4.05] * 10,
    })
    result = relative_value_signal("2Y", "10Y", SAMPLE_PAR_CURVE, flat_hist)
    assert result["z_score"] == 0.0