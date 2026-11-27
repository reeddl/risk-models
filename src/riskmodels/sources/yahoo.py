"""Daily prices from Yahoo Finance's chart endpoint.

Yahoo returns a split-adjusted close, a split-and-dividend-adjusted close, split-adjusted
volume, and dividend and split events. Total returns come from the adjusted close.
Market cap needs an unadjusted price, which is rebuilt from the split events (see
:func:`split_factor`).

Known limits: Yahoo only serves symbols that trade today (delisted names are mostly
gone), symbols are today's tickers (a reused or changed ticker maps to the current
company), and history is occasionally revised. See the survivorship notes in
``riskmodels/sources/README.md``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .http import Fetcher

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def parse_chart(payload: dict) -> pd.DataFrame:
    """Chart JSON -> DataFrame indexed by exchange-local date.

    Columns: close (split-adjusted), adjclose (split and dividend adjusted), volume
    (split-adjusted), dividend (cash per share on the ex-date, 0 otherwise), split_ratio
    (new shares per old share on the split date, 1 otherwise).
    """
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=["close", "adjclose", "volume", "dividend", "split_ratio"])
    res = results[0]
    ts = res.get("timestamp") or []
    offset = int(res.get("meta", {}).get("gmtoffset", 0))
    dates = pd.to_datetime(np.asarray(ts, dtype="int64") + offset, unit="s").normalize()
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]
    adj = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    close = pd.Series(quote.get("close", [None] * len(ts)), dtype=float)
    df = pd.DataFrame({
        "close": close.to_numpy(),
        "adjclose": pd.Series(adj if adj is not None else close, dtype=float).to_numpy(),
        "volume": pd.Series(quote.get("volume", [None] * len(ts)), dtype=float).to_numpy(),
    }, index=pd.DatetimeIndex(dates, name="date"))
    df["dividend"] = 0.0
    df["split_ratio"] = 1.0

    events = res.get("events", {}) or {}
    for ev in (events.get("dividends") or {}).values():
        d = pd.Timestamp(int(ev["date"]) + offset, unit="s").normalize()
        if d in df.index:
            df.loc[d, "dividend"] += float(ev["amount"])
    for ev in (events.get("splits") or {}).values():
        d = pd.Timestamp(int(ev["date"]) + offset, unit="s").normalize()
        num, den = float(ev.get("numerator", 1)), float(ev.get("denominator", 1))
        if d in df.index and den > 0:
            df.loc[d, "split_ratio"] *= num / den

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close", "adjclose"], how="all")


def split_factor(split_ratio: pd.Series) -> pd.Series:
    """F(t) = product of split ratios strictly after t.

    unadjusted price_t = split-adjusted price_t * F(t). Shares reported on date d convert to
    today's share basis by multiplying by F(d).
    """
    r = split_ratio.fillna(1.0).to_numpy(dtype=float)
    after = np.cumprod(r[::-1])[::-1]               # product from t onward
    return pd.Series(np.r_[after[1:], 1.0], index=split_ratio.index)


def total_returns(prices: pd.DataFrame) -> pd.Series:
    """Daily total return from the adjusted close."""
    return prices["adjclose"].pct_change()


class YahooPrices:
    def __init__(self, client: Fetcher) -> None:
        self.client = client

    def history(self, symbol: str, start, end) -> pd.DataFrame:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        params = {
            "period1": int(start.timestamp()),
            "period2": int((end + pd.Timedelta(days=1)).timestamp()),
            "interval": "1d",
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
        raw = self.client.get(CHART_URL.format(symbol=symbol), params=params, max_age_days=1)
        df = parse_chart(json.loads(raw))
        return df.loc[(df.index >= start) & (df.index <= end)]
