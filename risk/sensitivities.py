"""
DV01, convexity, and key-rate durations via bump-and-reprice.

Convention: DV01 and KRDs are reported as the dollar (or price-point) change
in the value of a $100-face par bond of the given tenor, per 1bp (0.0001)
parallel or key-rate shift in yield. Sign convention: DV01 is positive
(price falls when yields rise, so we report the magnitude of that fall).

All bumps are applied directly on the *curve nodes* (i.e. the par yields
that were fed into bootstrap_curve), not on the zero curve, so results are
directly interpretable against the same market quotes a trader watches.
"""

from __future__ import annotations

import numpy as np

from pricing.bootstrap import bootstrap_curve, Curve

BP = 1e-4  # one basis point in decimal


def _bond_price(curve: Curve, t: float, coupon_pct: float, freq: int = 2) -> float:
    """
    Prices a $100-face bond of maturity t (years) and annual coupon coupon_pct (%),
    off the given discount curve. For t <= 1Y uses the money-market convention
    (single payment), matching bootstrap_curve's own treatment.
    """
    if t <= 1.0:
        # Single terminal payment of principal + simple-interest coupon
        return (100 + coupon_pct * t) * curve.discount_factor(t)

    n_pmts = max(int(round(t * freq)), 1)
    times = np.array([i / freq for i in range(1, n_pmts + 1)])
    dfs = np.array([curve.discount_factor(ti) for ti in times])
    # coupon_pct is in percent (e.g. 4.05 = 4.05%); annual coupon $ on $100
    # face = coupon_pct, so each semiannual payment = coupon_pct / freq.
    coupon_pmt = coupon_pct / freq
    price = coupon_pmt * dfs[:-1].sum() + (100 + coupon_pmt) * dfs[-1]
    return float(price)


def price_par_bond(curve: Curve, tenor: str, tenor_years: dict[str, float],
                    par_yields_pct: dict[str, float]) -> float:
    """Prices the par bond for `tenor` (should equal ~100 by construction off its own curve)."""
    t = tenor_years[tenor]
    coupon = par_yields_pct[tenor]  # par bond's coupon = its own par yield
    return _bond_price(curve, t, coupon)


def dv01(tenor: str, tenor_years: dict[str, float], par_yields_pct: dict[str, float]) -> float:
    """
    Parallel DV01: fix the bond's coupon at its ORIGINAL par yield (real fixed
    cash flows, as a trader would hold), bump every curve node's yield by +1bp,
    reprice off the rebuilt (bumped) curve, compare to the base price.
    Returned as a positive number = price drop per +1bp (standard convention).
    """
    t = tenor_years[tenor]
    coupon = par_yields_pct[tenor]  # FIXED coupon -- do not let this shift with the bump

    base_curve = bootstrap_curve(par_yields_pct, tenor_years)
    base_price = _bond_price(base_curve, t, coupon)

    bumped_yields = {k: v + (BP * 100) for k, v in par_yields_pct.items()}  # +1bp in % terms
    bumped_curve = bootstrap_curve(bumped_yields, tenor_years)
    bumped_price = _bond_price(bumped_curve, t, coupon)

    return base_price - bumped_price


def convexity(tenor: str, tenor_years: dict[str, float], par_yields_pct: dict[str, float],
              bump_bp: float = 25.0) -> float:
    """
    Parallel convexity via central difference, same fixed-coupon convention as dv01():
        C ~= (P_up + P_down - 2*P_base) / (P_base * (dy)^2)
    dy in decimal (e.g. 25bp = 0.0025). Uses a larger bump than DV01
    (25bp) since convexity is a second-derivative estimate and more
    sensitive to numerical noise from a 1bp bump.
    """
    t = tenor_years[tenor]
    coupon = par_yields_pct[tenor]  # FIXED coupon
    dy_pct = bump_bp / 100  # convert bp to % (par_yields_pct is in % units)

    base_curve = bootstrap_curve(par_yields_pct, tenor_years)
    base_price = _bond_price(base_curve, t, coupon)

    up_yields = {k: v + dy_pct for k, v in par_yields_pct.items()}
    up_curve = bootstrap_curve(up_yields, tenor_years)
    up_price = _bond_price(up_curve, t, coupon)

    down_yields = {k: v - dy_pct for k, v in par_yields_pct.items()}
    down_curve = bootstrap_curve(down_yields, tenor_years)
    down_price = _bond_price(down_curve, t, coupon)

    dy_decimal = bump_bp * BP
    return (up_price + down_price - 2 * base_price) / (base_price * dy_decimal ** 2)


def key_rate_durations(tenor: str, tenor_years: dict[str, float],
                        par_yields_pct: dict[str, float]) -> dict[str, float]:
    """
    Same fixed-coupon convention as dv01(), but bumps ONE curve node at a time
    by +1bp instead of a parallel shift. Sum of all KRDs ~= parallel DV01
    (sanity check the caller can perform). Returned as positive = price drop
    per +1bp bump of that specific node.
    """
    t = tenor_years[tenor]
    coupon = par_yields_pct[tenor]  # FIXED coupon

    base_curve = bootstrap_curve(par_yields_pct, tenor_years)
    base_price = _bond_price(base_curve, t, coupon)

    krds = {}
    for node in par_yields_pct:
        bumped = dict(par_yields_pct)
        bumped[node] = bumped[node] + (BP * 100)
        bumped_curve = bootstrap_curve(bumped, tenor_years)
        bumped_price = _bond_price(bumped_curve, t, coupon)
        krds[node] = base_price - bumped_price

    return krds