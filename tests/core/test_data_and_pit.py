import numpy as np
import pandas as pd
import pytest

from riskmodels.core.data import DataValidationError, MarketData, alive_mask
from riskmodels.core.pit import PointInTimeFundamentals, industry_panel


def test_snapshot_respects_available_date_and_restatements():
    fund = pd.DataFrame([
        ("A", "book_equity", 100.0, "2021-03-31", "2021-05-10"),
        ("A", "book_equity", 105.0, "2021-03-31", "2021-09-01"),   # restatement
        ("A", "book_equity", 110.0, "2021-06-30", "2021-08-10"),
        ("A", "book_equity", 90.0, "2020-06-30", "2020-08-10"),
    ], columns=["asset", "field", "value", "period_end", "available_date"])
    for c in ("period_end", "available_date"):
        fund[c] = pd.to_datetime(fund[c])
    pit = PointInTimeFundamentals(fund)

    assert pit.snapshot("2021-05-09").current.loc["A", "book_equity"] == 90.0
    assert pit.snapshot("2021-05-10").current.loc["A", "book_equity"] == 100.0
    s = pit.snapshot("2021-08-20")
    assert s.current.loc["A", "book_equity"] == 110.0
    assert s.prior_year.loc["A", "book_equity"] == 90.0
    s = pit.snapshot("2021-09-02")
    assert s.current.loc["A", "book_equity"] == 110.0


def test_snapshot_drops_stale_values():
    fund = pd.DataFrame([("A", "sales_ttm", 1.0, pd.Timestamp("2019-12-31"), pd.Timestamp("2020-02-01"))],
                        columns=["asset", "field", "value", "period_end", "available_date"])
    pit = PointInTimeFundamentals(fund, max_age_days=365)
    assert pit.snapshot("2020-06-01").current.loc["A", "sales_ttm"] == 1.0
    assert pit.snapshot("2021-06-01").current.empty


def test_industry_panel_is_effective_dated():
    dates = pd.bdate_range("2021-01-01", periods=10)
    ind = pd.DataFrame([
        ("A", "Tech", dates[0], dates[4]),
        ("A", "Health", dates[5], pd.NaT),
    ], columns=["asset", "industry", "effective_from", "effective_to"])
    panel = industry_panel(ind, dates, pd.Index(["A", "B"]))
    assert (panel["A"].iloc[:5] == "Tech").all()
    assert (panel["A"].iloc[5:] == "Health").all()
    assert panel["B"].isna().all()


def test_synthetic_data_is_survivorship_free(synthetic):
    data, _ = synthetic
    data.validate()
    sm = data.security_master
    dead = sm[sm["end_date"].notna()]
    assert len(dead) > 0
    a, end = dead.iloc[0]["asset"], dead.iloc[0]["end_date"]
    assert data.returns.loc[:end, a].notna().any()
    assert data.returns.loc[end + pd.Timedelta(days=1):, a].isna().all()
    alive = alive_mask(sm, data.dates, data.assets)
    assert alive.sum(axis=1).nunique() > 1   # the universe changes over time


def test_validate_catches_lookahead_and_backfilled_history(synthetic):
    data, _ = synthetic
    bad_fund = data.fundamentals.copy()
    bad_fund.loc[bad_fund.index[0], "available_date"] = (
        bad_fund.loc[bad_fund.index[0], "period_end"] - pd.Timedelta(days=1)
    )
    bad = MarketData(data.returns, data.market_cap, data.dollar_volume,
                     data.security_master, data.industries, bad_fund)
    with pytest.raises(DataValidationError, match="before their period end"):
        bad.validate()

    sm = data.security_master
    a = sm.loc[(sm["start_date"] > data.dates[0]).idxmax(), "asset"]
    rets = data.returns.copy()
    rets.loc[data.dates[0], a] = 0.01   # a return before the security listed
    bad = MarketData(rets, data.market_cap, data.dollar_volume, sm, data.industries,
                     data.fundamentals)
    with pytest.raises(DataValidationError, match="outside their security's listing window"):
        bad.validate()
