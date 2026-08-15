import pytest

from data.sample_curve import SAMPLE_PAR_CURVE, TENOR_YEARS
from signals.carry_rolldown import carry, rolldown, yield_rolldown_bps, carry_and_rolldown

FLAT_CURVE = {k: 4.0 for k in SAMPLE_PAR_CURVE}


def test_carry_scales_linearly_with_holding_period():
    c1 = carry("10Y", SAMPLE_PAR_CURVE, holding_years=1.0)
    c2 = carry("10Y", SAMPLE_PAR_CURVE, holding_years=2.0)
    assert c2 == pytest.approx(2 * c1)


def test_carry_equals_coupon_times_holding():
    c = carry("10Y", SAMPLE_PAR_CURVE, holding_years=0.5)
    assert c == pytest.approx(SAMPLE_PAR_CURVE["10Y"] * 0.5)


def test_rolldown_zero_on_flat_curve_at_any_holding_period():
    """
    Regression test for the bug found during development: a flat curve must
    produce ~zero rolldown at every holding period, including sub-coupon-period
    holdings (e.g. 3M on a semiannual bond), where an earlier version leaked
    accrued interest into the rolldown figure via an unstripped dirty price.
    """
    for h in [0.01, 0.1, 0.25, 0.4, 0.49, 0.5, 0.75, 1.0]:
        r = rolldown("10Y", TENOR_YEARS, FLAT_CURVE, holding_years=h)
        assert abs(r) < 0.01


def test_rolldown_scales_roughly_linearly_with_small_holding_periods():
    """Rolldown/holding_years should be roughly stable for small h (near-linear regime)."""
    ratios = []
    for h in [0.1, 0.25, 0.5]:
        r = rolldown("10Y", TENOR_YEARS, SAMPLE_PAR_CURVE, holding_years=h)
        ratios.append(r / h)
    # allow some drift for convexity, but not order-of-magnitude swings
    assert max(ratios) / min(ratios) < 2.0


def test_rolldown_sign_matches_yield_rolldown_direction():
    """Price-space rolldown and yield-space rolldown should agree in sign."""
    for tenor in ["2Y", "5Y", "10Y", "30Y"]:
        r = rolldown(tenor, TENOR_YEARS, SAMPLE_PAR_CURVE, holding_years=0.25)
        yr = yield_rolldown_bps(tenor, TENOR_YEARS, SAMPLE_PAR_CURVE, holding_years=0.25)
        # both ~0 is fine; otherwise signs must agree
        if abs(r) > 1e-3 and abs(yr) > 1e-2:
            assert (r > 0) == (yr > 0)


def test_carry_and_rolldown_total_return_is_sum():
    result = carry_and_rolldown("10Y", TENOR_YEARS, SAMPLE_PAR_CURVE, holding_years=0.25)
    assert result["total_return"] == pytest.approx(result["carry"] + result["rolldown"])


def test_flat_curve_total_return_approx_equals_running_yield():
    """On a flat curve, carry+rolldown should collapse to ~just the running coupon yield."""
    result = carry_and_rolldown("10Y", TENOR_YEARS, FLAT_CURVE, holding_years=0.5)
    expected = 4.0 * 0.5  # coupon rate * holding period
    assert result["total_return"] == pytest.approx(expected, abs=0.02)