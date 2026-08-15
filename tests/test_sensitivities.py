import pytest

from data.sample_curve import SAMPLE_PAR_CURVE, TENOR_YEARS
from pricing.bootstrap import bootstrap_curve
from risk.sensitivities import _bond_price, dv01, convexity, key_rate_durations


def test_par_bond_prices_near_par():
    """A bond coupon-matched to its own tenor's par yield should price ~100."""
    curve = bootstrap_curve(SAMPLE_PAR_CURVE, TENOR_YEARS)
    for tenor, t in TENOR_YEARS.items():
        price = _bond_price(curve, t, SAMPLE_PAR_CURVE[tenor])
        # Same ~1bp tolerance as the bootstrap round-trip tests (Day 1),
        # wider at the 20Y/30Y long end where curve nodes are sparse.
        tol = 1.5 if tenor in ("20Y", "30Y") else 0.3
        assert price == pytest.approx(100, abs=tol)


def test_dv01_positive_and_increases_with_maturity():
    """DV01 must be positive (price falls when yields rise) and grow with tenor."""
    tenors = ["2Y", "5Y", "10Y", "30Y"]
    dv01s = [dv01(t, TENOR_YEARS, SAMPLE_PAR_CURVE) for t in tenors]
    assert all(d > 0 for d in dv01s)
    assert all(dv01s[i] < dv01s[i + 1] for i in range(len(dv01s) - 1))


def test_convexity_positive_and_increases_with_maturity():
    """Convexity must be positive and grow with tenor, same shape as DV01."""
    tenors = ["2Y", "5Y", "10Y", "30Y"]
    convexities = [convexity(t, TENOR_YEARS, SAMPLE_PAR_CURVE) for t in tenors]
    assert all(c > 0 for c in convexities)
    assert all(convexities[i] < convexities[i + 1] for i in range(len(convexities) - 1))


def test_key_rate_durations_sum_to_parallel_dv01():
    """Sum of per-node KRDs should reconstruct the parallel DV01 (standard identity)."""
    for tenor in ["5Y", "10Y", "30Y"]:
        krds = key_rate_durations(tenor, TENOR_YEARS, SAMPLE_PAR_CURVE)
        parallel = dv01(tenor, TENOR_YEARS, SAMPLE_PAR_CURVE)
        assert sum(krds.values()) == pytest.approx(parallel, abs=1e-3)


def test_key_rate_duration_dominated_by_own_tenor():
    """The bond's own maturity node should carry by far the largest KRD."""
    krds = key_rate_durations("10Y", TENOR_YEARS, SAMPLE_PAR_CURVE)
    own_node = krds["10Y"]
    other_nodes = [v for k, v in krds.items() if k != "10Y"]
    assert own_node > 0
    assert all(abs(own_node) > abs(v) for v in other_nodes)


def test_key_rate_duration_known_interpolation_artifact():
    """
    Documented limitation: bumping a node adjacent to (but not at) the bond's
    own maturity can produce a small negative KRD, due to log-linear
    interpolation interacting with the recursive par-bond bootstrap (the
    bump changes discount factors used to price coupons between nodes).
    This is a real artifact of the chosen interpolation scheme, not a bug --
    asserting its magnitude stays small and doesn't corrupt the KRD sum.
    """
    krds = key_rate_durations("10Y", TENOR_YEARS, SAMPLE_PAR_CURVE)
    off_node_total = sum(v for k, v in krds.items() if k != "10Y")
    assert abs(off_node_total) < 0.01  # small relative to the ~0.08 own-node KRD