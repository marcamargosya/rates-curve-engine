"""
Bootstraps a discount curve from Treasury par yields.

Method:
  - Tenors <= 1Y: money-market convention, single payment at maturity.
      D(t) = 1 / (1 + y * t)
  - Tenors > 1Y: par yield = coupon rate of a bond priced at 100, paying
    semiannual coupons. Solve iteratively for each discount factor given
    all shorter ones (standard bootstrap):
      100 = sum_{i=1}^{n-1} (c/2) * 100 * D(t_i)  +  (100 + c/2*100) * D(t_n)
    Rearranged to solve for D(t_n).

Interpolation between bootstrapped nodes: log-linear on discount factors,
equivalent to piecewise-linear on continuously compounded zero rates.
Chosen over cubic spline deliberately (per project scope: mechanical,
defensible, no overfitting between sparse nodes).
"""

import numpy as np
import pandas as pd


class Curve:
    """A bootstrapped discount curve, queryable at any maturity via interpolation."""

    def __init__(self, tenor_years: dict[str, float], discount_factors: dict[str, float]):
        # Sort nodes by maturity
        items = sorted(
            ((tenor_years[t], df) for t, df in discount_factors.items()),
            key=lambda x: x[0],
        )
        self.node_t = np.array([t for t, _ in items])
        self.node_df = np.array([df for _, df in items])
        # log(D(t)) is piecewise-linear in t -> zero rate is piecewise-linear
        self.node_log_df = np.log(self.node_df)

    def discount_factor(self, t: float) -> float:
        """Discount factor at maturity t (years), log-linear interpolation/flat extrapolation."""
        if t <= self.node_t[0]:
            # flat zero-rate extrapolation at the short end
            r0 = -self.node_log_df[0] / self.node_t[0]
            return float(np.exp(-r0 * t))
        if t >= self.node_t[-1]:
            r_last = -self.node_log_df[-1] / self.node_t[-1]
            return float(np.exp(-r_last * t))
        log_df = np.interp(t, self.node_t, self.node_log_df)
        return float(np.exp(log_df))

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero rate at maturity t (years)."""
        t = max(t, 1e-6)
        return -np.log(self.discount_factor(t)) / t

    def par_yield_check(self, t: float, freq: int = 2) -> float:
        """Recompute the par yield implied by this curve at maturity t (sanity check)."""
        if t <= 1.0:
            # Money-market convention, matches bootstrap: D(t) = 1/(1+y*t)
            df = self.discount_factor(t)
            return float((1.0 / df - 1.0) / t)
        n_pmts = max(int(round(t * freq)), 1)
        times = np.array([i / freq for i in range(1, n_pmts + 1)])
        dfs = np.array([self.discount_factor(ti) for ti in times])
        annuity = dfs.sum() / freq
        return float((1 - dfs[-1]) / annuity)


def bootstrap_curve(par_yields_pct: dict[str, float], tenor_years: dict[str, float]) -> Curve:
    """
    par_yields_pct: e.g. {"1M": 4.35, "3M": 4.28, ..., "30Y": 4.40}  (percent, not decimal)
    tenor_years: e.g. {"1M": 1/12, ..., "30Y": 30}
    """
    tenors_sorted = sorted(par_yields_pct.keys(), key=lambda t: tenor_years[t])
    discount_factors: dict[str, float] = {}
    freq = 2  # semiannual coupons for bootstrapped bonds

    for tenor in tenors_sorted:
        t = tenor_years[tenor]
        y = par_yields_pct[tenor] / 100.0

        if t <= 1.0:
            # Money-market convention: single terminal payment
            df = 1.0 / (1.0 + y * t)
            discount_factors[tenor] = df
            continue

        # Par bond bootstrap: need DFs at each semiannual coupon date up to t.
        known_t = np.array([tenor_years[k] for k in discount_factors])
        known_df = np.array([discount_factors[k] for k in discount_factors])
        known_log_df = np.log(known_df)

        n_pmts = max(int(round(t * freq)), 1)
        coupon_times = np.array([i / freq for i in range(1, n_pmts + 1)])
        c = y  # par yield = coupon rate

        def df_from_known(ti: float) -> float:
            if ti <= known_t[0]:
                r0 = -known_log_df[0] / known_t[0]
                return float(np.exp(-r0 * ti))
            if ti >= known_t[-1]:
                # flat zero-rate extrapolation, NOT flat log(DF) -- np.interp
                # would hold log(DF) constant past the last node, which
                # silently shrinks the implied zero rate for long coupon
                # dates. Must match Curve.discount_factor's convention.
                r_last = -known_log_df[-1] / known_t[-1]
                return float(np.exp(-r_last * ti))
            log_df = np.interp(ti, known_t, known_log_df)
            return float(np.exp(log_df))

        # Sum coupon PVs for all but the final payment using known/interpolated DFs
        pv_coupons_before_last = sum(
            (c / freq) * 100 * df_from_known(ti) for ti in coupon_times[:-1]
        )
        # Solve for the final discount factor:
        # 100 = pv_coupons_before_last + (100 + c/2*100) * D(t_n)
        final_pmt = 100 + (c / freq) * 100
        df_final = (100 - pv_coupons_before_last) / final_pmt
        discount_factors[tenor] = df_final

    return Curve(tenor_years, discount_factors)