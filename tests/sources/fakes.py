"""Hand-built responses in the shapes the public endpoints return, and a fake client."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from riskmodels.sources.http import HttpError


class FakeClient:
    def __init__(self, routes: dict[str, bytes | str | dict | list]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def _find(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self.routes:
            raise HttpError(url, 404)
        v = self.routes[url]
        if isinstance(v, (dict, list)):
            return json.dumps(v).encode()
        return v.encode() if isinstance(v, str) else v

    def get(self, url, params=None, max_age_days=None) -> bytes:
        return self._find(url)

    def post_json(self, url, body, max_age_days=None) -> bytes:
        return self._find(url)


def chart_payload(dates: pd.DatetimeIndex, close, volume, split_on=None, split_ratio=2.0,
                  dividends: dict | None = None) -> dict:
    """Yahoo chart JSON. ``close`` is the split-adjusted close; adjclose = close here unless
    dividends are given."""
    ts = [int((d + pd.Timedelta(hours=14, minutes=30)).timestamp()) for d in dates]
    events: dict = {}
    if split_on is not None:
        t = int((pd.Timestamp(split_on) + pd.Timedelta(hours=14, minutes=30)).timestamp())
        events["splits"] = {str(t): {"date": t, "numerator": split_ratio, "denominator": 1}}
    if dividends:
        events["dividends"] = {
            str(i): {"date": int((pd.Timestamp(d) + pd.Timedelta(hours=14, minutes=30)).timestamp()), "amount": a}
            for i, (d, a) in enumerate(dividends.items())
        }
    return {"chart": {"error": None, "result": [{
        "meta": {"gmtoffset": -18000},
        "timestamp": ts,
        "events": events,
        "indicators": {"quote": [{"close": list(map(float, close)), "volume": list(map(float, volume))}],
                       "adjclose": [{"adjclose": list(map(float, close))}]},
    }]}}


def _fact(start, end, val, accn, form, filed):
    d = {"end": end, "val": val, "accn": accn, "form": form, "filed": filed}
    if start:
        d["start"] = start
    return d


def companyfacts_payload(scale: float = 1.0) -> dict:
    """FY2022 10-K (A1), Q1 2023 10-Q (Q1) and FY2023 10-K (A2) restating FY2022 net income."""
    s = scale
    ni = [
        _fact("2021-01-01", "2021-12-31", 80 * s, "A1", "10-K", "2023-02-20"),
        _fact("2022-01-01", "2022-12-31", 100 * s, "A1", "10-K", "2023-02-20"),
        _fact("2022-01-01", "2022-03-31", 20 * s, "Q1", "10-Q", "2023-05-05"),
        _fact("2023-01-01", "2023-03-31", 30 * s, "Q1", "10-Q", "2023-05-05"),
        _fact("2022-01-01", "2022-12-31", 95 * s, "A2", "10-K", "2024-02-20"),
        _fact("2023-01-01", "2023-12-31", 120 * s, "A2", "10-K", "2024-02-20"),
    ]
    rev = [
        _fact("2022-01-01", "2022-12-31", 1000 * s, "A1", "10-K", "2023-02-20"),
        _fact("2022-01-01", "2022-03-31", 240 * s, "Q1", "10-Q", "2023-05-05"),
        _fact("2023-01-01", "2023-03-31", 260 * s, "Q1", "10-Q", "2023-05-05"),
    ]
    cost = [_fact("2022-01-01", "2022-12-31", 600 * s, "A1", "10-K", "2023-02-20")]
    equity = [
        _fact(None, "2022-12-31", 500 * s, "A1", "10-K", "2023-02-20"),
        _fact(None, "2023-03-31", 520 * s, "Q1", "10-Q", "2023-05-05"),
    ]
    assets = [_fact(None, "2022-12-31", 2000 * s, "A1", "10-K", "2023-02-20")]
    ltd = [_fact(None, "2022-12-31", 300 * s, "A1", "10-K", "2023-02-20")]
    stb = [_fact(None, "2022-12-31", 50 * s, "A1", "10-K", "2023-02-20")]
    other = [_fact("2022-01-01", "2022-12-31", 1.0, "X1", "8-K", "2023-01-15")]
    shares = [
        _fact(None, "2023-02-10", 1000.0, "A1", "10-K", "2023-02-20"),
        _fact(None, "2023-04-28", 1000.0, "Q1", "10-Q", "2023-05-05"),
    ]
    usd = lambda rows: {"units": {"USD": rows}}
    return {"facts": {
        "us-gaap": {
            "NetIncomeLoss": usd(ni + other),
            "Revenues": usd(rev),
            "CostOfRevenue": usd(cost),
            "StockholdersEquity": usd(equity),
            "Assets": usd(assets),
            "LongTermDebt": usd(ltd),
            "ShortTermBorrowings": usd(stb),
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}},
    }}


def submissions_payload(cik: int, sic: int, ticker: str, delisted_on: str | None = None) -> dict:
    acc = ["A1", "Q1", "A2"]
    forms = ["10-K", "10-Q", "10-K"]
    filed = ["2023-02-20", "2023-05-05", "2024-02-20"]
    accepted = ["2023-02-20T17:30:00.000Z", "2023-05-05T09:00:00.000Z", ""]
    if delisted_on:
        acc.append("D25")
        forms.append("25-NSE")
        filed.append(delisted_on)
        accepted.append(f"{delisted_on}T10:00:00.000Z")
    return {"cik": str(cik), "name": f"{ticker} corp", "sic": str(sic), "sicDescription": "x",
            "tickers": [ticker],
            "filings": {"recent": {"accessionNumber": acc, "form": forms, "filingDate": filed,
                                   "acceptanceDateTime": accepted}, "files": []}}


MASTER_IDX = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 2023

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
1|AAA CORP|10-K|2023-02-20|edgar/data/1/a1.txt
2|BBB INC|10-K|2023-02-21|edgar/data/2/a1.txt
3|DEAD CO|10-Q|2023-02-14|edgar/data/3/q1.txt
4|SOME FUND|N-CSR|2023-02-14|edgar/data/4/x.txt
"""


def price_path(n: int, seed: int, start: float = 50.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
