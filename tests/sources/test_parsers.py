import numpy as np
import pandas as pd
import pytest

from fakes import MASTER_IDX, chart_payload, companyfacts_payload, submissions_payload
from riskmodels.sources import (
    available_dates,
    companyfacts_to_fundamentals,
    daily_rate_from_annual_percent,
    parse_aqr_sheet,
    parse_chart,
    parse_fred_csv,
    parse_master_index,
    parse_short_interest,
    parse_submissions,
    shares_on_today_basis,
    sic_to_industry,
    split_factor,
    ttm_from_durations,
)
from riskmodels.sources.edgar import _merged


# ---------------------------------------------------------------- Yahoo
def test_parse_chart_dates_events_and_split_factor():
    dates = pd.bdate_range("2023-06-01", periods=6)
    p = chart_payload(dates, [10, 11, 12, 13, 14, 15], [100] * 6, split_on=dates[3],
                      dividends={dates[1]: 0.25})
    df = parse_chart(p)
    assert list(df.index) == list(dates)          # exchange-local dates, not UTC
    assert df.loc[dates[1], "dividend"] == 0.25
    assert df.loc[dates[3], "split_ratio"] == 2.0
    f = split_factor(df["split_ratio"])
    # days before the split carry factor 2, the split day and after carry 1
    assert list(f) == [2, 2, 2, 1, 1, 1]


def test_parse_chart_raises_on_error():
    with pytest.raises(ValueError):
        parse_chart({"chart": {"error": {"code": "Not Found"}, "result": None}})


def test_shares_rebased_to_split_adjusted_basis():
    dates = pd.bdate_range("2023-01-02", periods=10)
    ratio = pd.Series(1.0, index=dates)
    ratio.iloc[5] = 2.0                                   # 2-for-1 on day 5
    shares = pd.DataFrame({"value": [1000.0, 2000.0],
                           "period_end": [dates[1], dates[7]],
                           "available_date": [dates[2], dates[8]]})
    s = shares_on_today_basis(shares, split_factor(ratio))
    # pre-split count doubles to today's basis, post-split count stays
    assert list(s) == [2000.0, 2000.0]


# ---------------------------------------------------------------- EDGAR
def test_available_dates_use_acceptance_time_and_cutoff():
    sub = parse_submissions(submissions_payload(1, 3571, "AAA"))
    av = available_dates(sub.filings)
    assert av["A1"] == pd.Timestamp("2023-02-21")   # accepted 17:30 -> next business day
    assert av["Q1"] == pd.Timestamp("2023-05-05")   # accepted 09:00 -> same day
    assert av["A2"] == pd.Timestamp("2024-02-21")   # no acceptance time -> filed + 1 bday


def test_ttm_from_annual_and_year_to_date():
    facts = _merged(companyfacts_payload(), ("NetIncomeLoss",))
    t = ttm_from_durations(facts)
    q1 = t[(t["end"] == "2023-03-31") & (t["accn"] == "Q1")]
    assert q1["val"].item() == pytest.approx(100 + 30 - 20)
    # 8-K facts are ignored
    assert "X1" not in set(t["accn"])


def test_companyfacts_to_fundamentals_point_in_time():
    sub = parse_submissions(submissions_payload(1, 3571, "AAA"))
    f = companyfacts_to_fundamentals(companyfacts_payload(), "AAA", available_dates(sub.filings))
    ni22 = f[(f["field"] == "net_income_ttm") & (f["period_end"] == "2022-12-31")].sort_values("available_date")
    # original and restatement both kept, each from its own filing date
    assert list(ni22["value"]) == [100, 95]
    assert list(ni22["available_date"]) == [pd.Timestamp("2023-02-21"), pd.Timestamp("2024-02-21")]
    gp = f[(f["field"] == "gross_profit_ttm")]
    assert gp["value"].item() == pytest.approx(400)           # revenue - cost of revenue
    debt = f[f["field"] == "total_debt"]
    assert debt["value"].item() == pytest.approx(350)
    assert (f["available_date"] >= f["period_end"]).all()
    assert {"book_equity", "total_assets", "sales_ttm", "shares_outstanding"} <= set(f["field"])


@pytest.mark.parametrize("sic,expected", [(3571, "BusEq"), (6021, "Money"), (2834, "Hlth"),
                                          (1311, "Enrgy"), (9995, "Other"), (None, None)])
def test_sic_to_ff12(sic, expected):
    assert sic_to_industry(sic) == expected


def test_sic_other_schemes():
    assert sic_to_industry(3571, "sic2") == "SIC35"
    assert sic_to_industry(6021, "sic_division") == "Finance"


def test_parse_master_index():
    df = parse_master_index(MASTER_IDX)
    assert list(df["cik"]) == [1, 2, 3, 4]
    assert df.loc[2, "form"] == "10-Q"


def test_delisting_date_from_form_25():
    sub = parse_submissions(submissions_payload(2, 6021, "BBB", delisted_on="2023-09-20"))
    assert sub.delisting_date(after="2023-01-01") == pd.Timestamp("2023-09-20")


# ---------------------------------------------------------------- FRED, AQR, FINRA
def test_fred_csv_and_daily_rate():
    df = parse_fred_csv("observation_date,DTB3\n2023-01-03,4.30\n2023-01-04,.\n2023-01-05,4.40\n")
    assert np.isnan(df.loc["2023-01-04", "DTB3"])
    r = daily_rate_from_annual_percent(df["DTB3"].dropna())
    assert ((1 + r.iloc[0]) ** 252 - 1) == pytest.approx(0.043)


def test_aqr_sheet_finds_header_row():
    raw = pd.DataFrame([
        ["Quality Minus Junk: Factors, Monthly", None, None],
        ["Source: AQR", None, None],
        [None, None, None],
        ["DATE", "USA", "Global"],
        ["01/31/2023", 0.012, 0.008],
        ["02/28/2023", -0.004, None],
    ])
    df = parse_aqr_sheet(raw)
    assert list(df.columns) == ["USA", "Global"]
    assert df.index[1] == pd.Timestamp("2023-02-28")
    assert df.loc["2023-01-31", "USA"] == pytest.approx(0.012)


def test_finra_short_interest_available_after_publication_lag():
    df = parse_short_interest([{
        "symbolCode": "aaa", "settlementDate": "2023-03-15",
        "currentShortPositionQuantity": 1200, "averageDailyVolumeQuantity": 300, "daysToCoverQuantity": 4.0,
    }])
    assert df.loc[0, "symbol"] == "AAA"
    assert df.loc[0, "available_date"] == pd.Timestamp("2023-03-15") + pd.offsets.BDay(8)


def test_compare_factor_returns_recovers_relationship():
    from riskmodels.factor_model import compare_factor_returns

    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", "2023-12-29")
    ref_d = pd.Series(rng.normal(0, 0.004, len(days)), index=days)
    ours = pd.DataFrame({"quality": 0.5 * ref_d + rng.normal(0, 0.001, len(days))}, index=days)
    out = compare_factor_returns(ours, pd.DataFrame({"QMJ": ref_d}))
    assert out.loc["quality", "correlation"] > 0.8
    assert out.loc["quality", "slope"] == pytest.approx(0.5, abs=0.1)
