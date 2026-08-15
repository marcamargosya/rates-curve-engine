"""
Fetches daily Treasury par yield curve rates from FRED.

Requires a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set it as an environment variable: export FRED_API_KEY=your_key_here
"""

import os
import requests
import pandas as pd

# FRED series IDs for Treasury par yield curve, by tenor
TENOR_SERIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

TENOR_YEARS = {
    "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
    "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
}

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, api_key: str, start_date: str, end_date: str) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    resp = requests.get(FRED_URL, params=params, timeout=10)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." = missing
    return df.set_index("date")["value"]


def fetch_curve_history(start_date: str, end_date: str, api_key: str | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date, columns = tenors (1M..30Y), values = par yields (%).
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("Set FRED_API_KEY env var or pass api_key explicitly.")

    series = {}
    for tenor, series_id in TENOR_SERIES.items():
        series[tenor] = fetch_series(series_id, api_key, start_date, end_date)

    df = pd.DataFrame(series)
    df = df.dropna(how="all")
    return df


def latest_curve(api_key: str | None = None) -> pd.Series:
    """Convenience: most recent available par yield curve as a Series (tenor -> yield %)."""
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=14)
    df = fetch_curve_history(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), api_key)
    return df.dropna(how="any").iloc[-1]


if __name__ == "__main__":
    curve = latest_curve()
    print(curve)