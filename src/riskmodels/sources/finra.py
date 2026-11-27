"""FINRA consolidated short interest (twice-monthly, all exchange-listed and OTC stocks).

Source: the FINRA Query API data set ``otcMarket / consolidatedShortInterest``. Short
interest is reported as of a settlement date and published about seven business days later,
so each value is stamped available ``publication_lag_bdays`` after settlement.

Not part of the factor model's default factors. It is loaded for the liquidity model and as
an optional extra descriptor (days to cover, short interest as a share of float).

FINRA's API was not reachable from the environment this was written in, so the request body
and field names follow FINRA's published API documentation and have not been run live. Some
FINRA data sets need API credentials; if this one does, pass a client that adds them.
"""

from __future__ import annotations

import json

import pandas as pd

from .http import Fetcher

SHORT_INTEREST_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FIELDS = {
    "symbolCode": "symbol",
    "settlementDate": "settlement_date",
    "currentShortPositionQuantity": "short_interest",
    "averageDailyVolumeQuantity": "avg_daily_volume",
    "daysToCoverQuantity": "days_to_cover",
}


def parse_short_interest(records: list[dict], publication_lag_bdays: int = 8) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[*FIELDS.values(), "available_date"])
    df = pd.DataFrame(records)
    df = df[[c for c in FIELDS if c in df.columns]].rename(columns=FIELDS)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    for c in ("short_interest", "avg_daily_volume", "days_to_cover"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["available_date"] = df["settlement_date"] + pd.offsets.BDay(publication_lag_bdays)
    return df.sort_values(["symbol", "settlement_date"]).reset_index(drop=True)


class FinraShortInterest:
    def __init__(self, client: Fetcher, page_size: int = 5000) -> None:
        self.client = client
        self.page_size = page_size

    def history(self, start, end, symbols: list[str] | None = None) -> pd.DataFrame:
        filters = [{
            "fieldName": "settlementDate", "startDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "endDate": pd.Timestamp(end).strftime("%Y-%m-%d"),
        }]
        body: dict = {"limit": self.page_size, "offset": 0, "dateRangeFilters": filters}
        if symbols:
            body["domainFilters"] = [{"fieldName": "symbolCode", "values": sorted(symbols)}]
        out: list[dict] = []
        while True:
            page = json.loads(self.client.post_json(SHORT_INTEREST_URL, body, max_age_days=7))
            out.extend(page)
            if len(page) < self.page_size:
                break
            body = {**body, "offset": body["offset"] + self.page_size}
        return parse_short_interest(out)
