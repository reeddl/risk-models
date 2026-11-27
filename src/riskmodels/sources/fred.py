"""FRED (St. Louis Fed): risk-free rate and other macro or rate series.

Uses the public ``fredgraph.csv`` download, which needs no API key. The default risk-free
series is DTB3, the 3-month Treasury bill secondary-market rate (annualized percent, discount
basis). It is converted to a daily return so it can be subtracted from daily stock returns.

FRED values are revised for some macro series. ``fredgraph.csv`` returns the latest vintage,
so a macro series used as a predictor is not point in time. T-bill rates are not revised.
"""

from __future__ import annotations

import io

import pandas as pd

from .http import Fetcher

GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_fred_csv(text: str) -> pd.DataFrame:
    """fredgraph.csv -> DataFrame indexed by date, one float column per series ("." -> NaN)."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


def daily_rate_from_annual_percent(rate: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Annualized percent yield -> per-trading-day return, (1 + r)^(1/252) - 1."""
    return (1.0 + rate / 100.0) ** (1.0 / periods_per_year) - 1.0


class Fred:
    def __init__(self, client: Fetcher) -> None:
        self.client = client

    def series(self, *series_ids: str, start=None, end=None) -> pd.DataFrame:
        params = {"id": ",".join(series_ids)}
        if start is not None:
            params["cosd"] = pd.Timestamp(start).strftime("%Y-%m-%d")
        if end is not None:
            params["coed"] = pd.Timestamp(end).strftime("%Y-%m-%d")
        raw = self.client.get(GRAPH_CSV_URL, params=params, max_age_days=1)
        return parse_fred_csv(raw.decode("utf-8"))

    def risk_free(self, dates: pd.DatetimeIndex, series_id: str = "DTB3") -> pd.Series:
        """Daily risk-free return on ``dates``. The rate on each day is the last published
        value on or before it (carried over holidays and missing prints)."""
        dates = pd.DatetimeIndex(dates)
        start = dates.min() - pd.Timedelta(days=14)
        rate = self.series(series_id, start=start, end=dates.max())[series_id].dropna()
        rate = rate.reindex(rate.index.union(dates)).ffill().reindex(dates)
        return daily_rate_from_annual_percent(rate).rename("risk_free")
