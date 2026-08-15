"""
Realistic sample Treasury par yield curve (%) for development/testing,
used in place of a live FRED pull while offline. Shape approximates a
mildly inverted-to-normal curve at the short end, upward-sloping long end.
"""

SAMPLE_PAR_CURVE = {
    "1M": 4.35,
    "3M": 4.28,
    "6M": 4.15,
    "1Y": 3.95,
    "2Y": 3.72,
    "3Y": 3.68,
    "5Y": 3.75,
    "7Y": 3.90,
    "10Y": 4.05,
    "20Y": 4.35,
    "30Y": 4.40,
}

TENOR_YEARS = {
    "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
    "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
}