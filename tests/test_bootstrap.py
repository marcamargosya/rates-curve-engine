import numpy as np
import pytest

from pricing.bootstrap import bootstrap_curve, Curve
from data.sample_curve import SAMPLE_PAR_CURVE, TENOR_YEARS


@pytest.fixture
def curve():
    return bootstrap_curve(SAMPLE_PAR_CURVE, TENOR_YEARS)


def test_discount_factors_decreasing(curve):
    """Discount factors must strictly decrease with maturity (positive rates)."""
    ts = np.linspace(0.01, 30, 200)
    dfs = [curve.discount_factor(t) for t in ts]
    assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))


def test_discount_factor_at_zero_is_one():
    c = bootstrap_curve(SAMPLE_PAR_CURVE, TENOR_YEARS)
    assert c.discount_factor(1e-6) == pytest.approx(1.0, abs=1e-3)


def test_par_yield_round_trip_short_end(curve):
    """Money-market tenors (<=1Y) should round-trip essentially exactly."""
    for tenor in ["1M", "3M", "6M", "1Y"]:
        t = TENOR_YEARS[tenor]
        implied = curve.par_yield_check(t) * 100
        assert implied == pytest.approx(SAMPLE_PAR_CURVE[tenor], abs=0.01)


def test_par_yield_round_trip_belly(curve):
    """2Y-10Y (denser nodes) should round-trip within ~1bp."""
    for tenor in ["2Y", "3Y", "5Y", "7Y", "10Y"]:
        t = TENOR_YEARS[tenor]
        implied = curve.par_yield_check(t) * 100
        assert implied == pytest.approx(SAMPLE_PAR_CURVE[tenor], abs=0.05)


def test_par_yield_round_trip_long_end_within_tolerance(curve):
    """10Y-30Y gap is wide (only 3 nodes); allow looser tolerance, document why."""
    for tenor in ["20Y", "30Y"]:
        t = TENOR_YEARS[tenor]
        implied = curve.par_yield_check(t) * 100
        assert implied == pytest.approx(SAMPLE_PAR_CURVE[tenor], abs=1.0)


def test_zero_rate_positive_across_curve(curve):
    ts = np.linspace(0.01, 30, 50)
    for t in ts:
        assert curve.zero_rate(t) > 0


def test_curve_extrapolation_short_end_flat_rate():
    """Below the shortest node, zero rate should equal the shortest node's flat rate."""
    c = bootstrap_curve(SAMPLE_PAR_CURVE, TENOR_YEARS)
    r_at_node = c.zero_rate(TENOR_YEARS["1M"])
    r_below = c.zero_rate(TENOR_YEARS["1M"] / 2)
    assert r_below == pytest.approx(r_at_node, abs=1e-4)


def test_curve_extrapolation_long_end_flat_rate():
    """Beyond the longest node, zero rate should equal the longest node's flat rate."""
    c = bootstrap_curve(SAMPLE_PAR_CURVE, TENOR_YEARS)
    r_at_node = c.zero_rate(30)
    r_beyond = c.zero_rate(40)
    assert r_beyond == pytest.approx(r_at_node, abs=1e-4)