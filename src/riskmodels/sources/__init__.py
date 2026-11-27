"""Loaders for free public data: Yahoo Finance, SEC EDGAR, FRED, AQR and FINRA.

See ``README.md`` in this folder for what each source covers and where it falls short of
the data contract in ``riskmodels.core.data``.
"""

from .aqr import AqrLibrary, parse_aqr_sheet
from .edgar import (
    Edgar,
    available_dates,
    companyfacts_to_fundamentals,
    parse_master_index,
    parse_submissions,
    sic_to_industry,
    ttm_from_durations,
)
from .finra import FinraShortInterest, parse_short_interest
from .fred import Fred, daily_rate_from_annual_percent, parse_fred_csv
from .http import DEFAULT_USER_AGENT, Fetcher, HttpClient, HttpError
from .loader import CoverageReport, PublicDataLoader, shares_on_today_basis
from .yahoo import YahooPrices, parse_chart, split_factor, total_returns

__all__ = [
    "AqrLibrary", "CoverageReport", "DEFAULT_USER_AGENT", "Edgar", "Fetcher", "FinraShortInterest",
    "Fred", "HttpClient", "HttpError", "PublicDataLoader", "YahooPrices", "available_dates",
    "companyfacts_to_fundamentals", "daily_rate_from_annual_percent", "parse_aqr_sheet",
    "parse_chart", "parse_fred_csv", "parse_master_index", "parse_short_interest",
    "parse_submissions", "shares_on_today_basis", "sic_to_industry", "split_factor",
    "total_returns", "ttm_from_durations",
]
