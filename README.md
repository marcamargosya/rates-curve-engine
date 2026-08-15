# rates-curve-engine

A fixed-income rates/curve relative-value engine: bootstraps a discount curve
from Treasury par yields, computes DV01/convexity/key-rate durations, carry
and rolldown, and a curve-spread relative-value (z-score) signal.

Built as a companion to three equity-derivatives/microstructure engines
([vol-surface-engine], [rough-vol-engine], [market-making-engine]) to cover
rates -- the desk this repo's author is actually targeting.

## Architecture

data/ curve_fetcher.py -- pulls daily Treasury par yields from FRED
sample_curve.py -- offline sample curve for dev/testing
pricing/ bootstrap.py -- par yields -> discount curve (Curve class)
risk/ sensitivities.py -- DV01, convexity, key-rate durations
signals/ carry_rolldown.py -- carry, rolldown, curve-unchanged total return
relative_value.py -- z-score of curve spreads vs history, rich/cheap flags
tests/ one test file per module above, 28 tests total


## Method

**Bootstrapping** (`pricing/bootstrap.py`): tenors <= 1Y use money-market
convention (single terminal payment, simple interest). Tenors > 1Y are
bootstrapped as semiannual-coupon par bonds, solved recursively node by
node. Interpolation between nodes is log-linear on discount factors
(equivalent to piecewise-linear zero rates) -- chosen over cubic spline
deliberately: mechanical and defensible with sparse nodes, no overfitting
risk between widely-spaced quotes (e.g. 10Y to 20Y).

**Risk** (`risk/sensitivities.py`): DV01 and convexity via bump-and-reprice
on the curve's underlying par yield nodes, holding the bond's own coupon
FIXED at its original par yield (not re-par'd on every bump -- see "Bugs
found" below for why that distinction matters). Key-rate durations bump one
node at a time; their sum reconstructs the parallel DV01 as an internal
consistency check.

**Carry & rolldown** (`signals/carry_rolldown.py`): standard curve-unchanged
decomposition of expected bond return. Carry is running coupon yield only
(no repo/financing curve in this engine, so it's gross carry, not net of
funding cost). Rolldown ages the bond's actual cash-flow schedule and
strips accrued interest to isolate the pure curve-shape effect from coupon
income (see "Bugs found").

**Relative value** (`signals/relative_value.py`): z-scores a spread between
two tenors against its own historical distribution and flags dislocations
beyond a threshold (default |z| > 1.5). A curve-shape signal (steepener/
flattener), not an outright-yield signal.

## Known limitations

- **Long-end interpolation error**: with only 20Y and 30Y nodes past 10Y,
  the par-yield round-trip check shows ~6-7bp of error at 20Y (vs <1bp
  through the 2Y-10Y belly, where nodes are denser). This is a real
  consequence of log-linear interpolation over a wide gap, not a bug --
  documented and tested against explicitly (see
  `test_par_yield_round_trip_long_end_within_tolerance`).
- **Carry ignores financing cost**: no repo curve is modeled, so `carry()`
  is running yield, not carry net of funding -- a real simplification
  versus how a rates desk would actually compute it.
- **Key-rate durations show small artifacts** on nodes adjacent to (but not
  at) a bond's own maturity, from log-linear interpolation interacting with
  the recursive par-bond bootstrap. Small relative to the dominant
  own-maturity KRD; tested and bounded, not hidden.
- **One curve construction convention** (par-bond bootstrap + log-linear
  interpolation) -- no OIS-vs-Treasury dual-curve framework, no SOFR/repo
  basis. Single-curve scope was a deliberate choice to hit a 1-week build
  target without sacrificing correctness on the core pieces.

## Bugs found and fixed during development

Documented because they were genuine, non-obvious modeling errors caught by
testing against cases where the correct answer was independently known --
not just cosmetic fixes:

1. **Curve extrapolation bug**: the internal helper used during bootstrap
   held `log(discount factor)` flat beyond the last known node instead of
   holding the zero *rate* flat, silently shrinking long-end discount
   factors during the recursive bootstrap. Caught by a 20Y par-yield
   round-trip error of ~590bp before the fix (now ~7bp).
2. **DV01 sign/magnitude bug**: DV01 was initially computed by re-pricing a
   *new* par bond (coupon = bumped yield) at every bump, which by
   construction reprices back near par regardless of the bump -- measuring
   curve-shape noise, not real duration. Fixed by holding the bond's coupon
   fixed at its original par yield and only shifting the discounting curve.
3. **Dirty-price leakage into rolldown**: for holding periods shorter than
   one coupon period (e.g. 3M on a semiannual bond), rolldown was leaking
   accrued interest into the price-space rolldown figure, producing a
   result an order of magnitude too large and with the wrong scaling in h.
   Caught by testing on a flat curve, where true rolldown must be exactly
   zero at every holding period. Fixed by stripping accrued interest from
   the aged price.
4. **Near-zero-variance division**: a mathematically-constant historical
   spread series left `std` at ~1e-15 (floating-point noise) instead of
   exactly 0.0, so an exact `std == 0` guard didn't catch it, and dividing
   near-zero by near-zero produced a garbage z-score instead of the
   correct z=0. Fixed with an epsilon threshold.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

To pull live data, set a free FRED API key:
```bash
export FRED_API_KEY=your_key_here
python3 data/curve_fetcher.py
```