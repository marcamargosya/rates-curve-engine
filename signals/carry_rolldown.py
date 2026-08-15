"""
Carry and rolldown for a par bond position, over a specified holding period,
assuming the curve is static (unchanged) over that period -- the standard
"curve-unchanged" scenario used to decompose expected bond return.

Definitions used here:
  - Carry: coupon income accrued over the holding period, per $100 face.
      carry = coupon_rate(%) * holding_years
    (Ignores repo/financing cost -- there's no funding curve in this engine,
    so this is running yield carry, not net-of-financing carry. Documented
    as a scope limitation, same as the money-market bootstrap convention.)
  - Rolldown: price appreciation purely from the bond aging by the holding
    period (maturity shortens from T to T - h) while the curve itself stays
    fixed. Computed by repricing the SAME fixed coupon at the SHORTER
    maturity off TODAY's curve, and comparing to today's price.
      rolldown = P(T - h, coupon, curve_today) - P(T, coupon, curve_today)
  - Yield rolldown (bps): y(T) - y(T - h) read directly off today's curve
    -- the yield pickup from rolling down a (typically upward-sloping)
    curve. Simple, curve-native alternative to the price-space number above.
  - Total expected return (curve-unchanged) = carry + rolldown (price-space).
"""

from __future__ import annotations

from pricing.bootstrap import bootstrap_curve, Curve
from risk.sensitivities import _bond_price


def yield_rolldown_bps(tenor: str, tenor_years: dict[str, float],
                        par_yields_pct: dict[str, float], holding_years: float) -> float:
    """
    Yield pickup (in bps) from rolling down the curve: y(T) - y(T - holding_years),
    read directly off today's curve. Positive on a normal upward-sloping curve
    (shorter maturity = lower yield = bond "rolls down" to a lower yield).
    """
    curve = bootstrap_curve(par_yields_pct, tenor_years)
    t = tenor_years[tenor]
    t_rolled = max(t - holding_years, 1e-6)

    y_now = curve.zero_rate(t) * 100
    y_rolled = curve.zero_rate(t_rolled) * 100
    return (y_now - y_rolled) * 100  # convert % to bps


def carry(tenor: str, par_yields_pct: dict[str, float], holding_years: float) -> float:
    """
    Running yield carry: coupon income accrued over the holding period,
    per $100 face. No financing/repo cost netted out (see module docstring).
    """
    coupon_rate_pct = par_yields_pct[tenor]
    return coupon_rate_pct * holding_years


def _bond_price_aged(curve: Curve, original_maturity: float, coupon_pct: float,
                      holding_years: float, freq: int = 2) -> float:
    """
    Prices the SAME bond's remaining cash flows after `holding_years` have
    elapsed, off `curve` (assumed unchanged/static), as a CLEAN price (with
    accrued interest stripped out). Ages the bond's ACTUAL original payment
    schedule rather than regenerating a fresh grid rounded to the nearest
    coupon date -- regenerating fresh is what silently broke rolldown for
    holding periods shorter than one coupon spacing (e.g. a 3M holding on a
    semiannual bond: round(9.75*2) == round(10*2) == 20, so the "rolled"
    schedule was identical to the original and rolldown came out exactly
    zero).

    Accrued interest must be stripped for any holding period that lands
    mid-coupon-period: the raw discounted-cashflow ("dirty") price rises
    smoothly toward the next coupon date and then drops by the coupon
    amount right after payment -- the standard dirty-price sawtooth. Left
    in, that sawtooth silently leaks coupon income into rolldown(), which
    double-counts against carry() (already the full coupon accrual for the
    period). Confirmed by testing on a FLAT curve, where true rolldown must
    be exactly zero at every holding period: without this fix, rolldown
    incorrectly came out as large as ~2.0 (price points) for holding
    periods just under one coupon spacing.
    """
    if original_maturity <= 1.0:
        # Money-market bond: single terminal payment, no coupon schedule to age.
        t_remaining = max(original_maturity - holding_years, 1e-6)
        return (100 + coupon_pct * original_maturity) * curve.discount_factor(t_remaining)

    n_pmts = max(int(round(original_maturity * freq)), 1)
    original_times = [i / freq for i in range(1, n_pmts + 1)]
    remaining_times = [t - holding_years for t in original_times if t > holding_years]
    if not remaining_times:
        remaining_times = [1e-6]

    coupon_pmt = coupon_pct / freq
    dfs = [curve.discount_factor(t) for t in remaining_times]
    dirty_price = coupon_pmt * sum(dfs[:-1]) + (100 + coupon_pmt) * dfs[-1]

    period = 1 / freq
    elapsed_in_period = holding_years % period
    accrued = coupon_pmt * (elapsed_in_period / period)

    return float(dirty_price - accrued)


def rolldown(tenor: str, tenor_years: dict[str, float],
             par_yields_pct: dict[str, float], holding_years: float) -> float:
    """
    Price-space rolldown: age the bond's ACTUAL cash-flow schedule by the
    holding period (see _bond_price_aged), reprice off today's curve, minus
    today's price. Positive on a normal upward-sloping curve (shorter
    remaining maturity prices higher off a curve with lower short yields).
    """
    curve = bootstrap_curve(par_yields_pct, tenor_years)
    t = tenor_years[tenor]
    coupon = par_yields_pct[tenor]  # fixed coupon, bond doesn't change

    price_now = _bond_price(curve, t, coupon)
    price_aged = _bond_price_aged(curve, t, coupon, holding_years)
    return price_aged - price_now


def carry_and_rolldown(tenor: str, tenor_years: dict[str, float],
                        par_yields_pct: dict[str, float],
                        holding_years: float) -> dict[str, float]:
    """
    Full breakdown: carry, price-space rolldown, their sum (curve-unchanged
    total expected return per $100 face), and yield-space rolldown in bps
    as a curve-native cross-check.
    """
    c = carry(tenor, par_yields_pct, holding_years)
    r = rolldown(tenor, tenor_years, par_yields_pct, holding_years)
    yr_bps = yield_rolldown_bps(tenor, tenor_years, par_yields_pct, holding_years)

    return {
        "carry": c,
        "rolldown": r,
        "total_return": c + r,
        "yield_rolldown_bps": yr_bps,
    }