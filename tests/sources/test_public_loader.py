import numpy as np
import pandas as pd
import pytest

from fakes import MASTER_IDX, FakeClient, chart_payload, companyfacts_payload, price_path, submissions_payload
from riskmodels.sources import PublicDataLoader
from riskmodels.sources.edgar import FACTS_URL, MASTER_INDEX_URL, SUBMISSIONS_URL, TICKERS_URL
from riskmodels.sources.fred import GRAPH_CSV_URL
from riskmodels.sources.yahoo import CHART_URL

START, END = pd.Timestamp("2023-01-02"), pd.Timestamp("2023-12-29")
DATES = pd.bdate_range(START, END)
SPLIT_DAY = DATES[150]
BBB_LAST = DATES[DATES <= "2023-09-29"][-1]


def _routes():
    a_close = price_path(len(DATES), 1)
    a_close[150:] = a_close[149]                          # flat around the split
    b_dates = DATES[DATES <= BBB_LAST]
    routes = {
        TICKERS_URL: {"0": {"cik_str": 1, "ticker": "AAA", "title": "AAA CORP"},
                      "1": {"cik_str": 2, "ticker": "BBB", "title": "BBB INC"}},
        SUBMISSIONS_URL.format(cik=1): submissions_payload(1, 3571, "AAA"),
        SUBMISSIONS_URL.format(cik=2): submissions_payload(2, 6021, "BBB", delisted_on="2023-09-20"),
        FACTS_URL.format(cik=1): companyfacts_payload(),
        FACTS_URL.format(cik=2): companyfacts_payload(scale=3.0),
        CHART_URL.format(symbol="AAA"): chart_payload(DATES, a_close, np.full(len(DATES), 1e5),
                                                       split_on=SPLIT_DAY),
        CHART_URL.format(symbol="BBB"): chart_payload(b_dates, price_path(len(b_dates), 2),
                                                       np.full(len(b_dates), 5e4)),
        GRAPH_CSV_URL: "observation_date,DTB3\n2022-12-20,4.2\n2023-03-01,4.6\n2023-07-03,5.3\n",
        MASTER_INDEX_URL.format(year=2023, quarter=1): MASTER_IDX,
    }
    for q in (2, 3, 4):
        routes[MASTER_INDEX_URL.format(year=2023, quarter=q)] = MASTER_IDX.split("CIK|")[0]
    return routes


@pytest.fixture(scope="module")
def loaded():
    loader = PublicDataLoader(client=FakeClient(_routes()), delisting_return=-0.3, min_price_days=20)
    return loader, loader.load(START, END)


def test_loads_and_meets_data_contract(loaded):
    loader, data = loaded
    data.validate()
    assert set(data.returns.columns) == {"AAA", "BBB"}
    assert loader.report.filers == 3                       # the N-CSR filer is not a 10-K/10-Q filer
    reasons = loader.report.excluded.set_index("cik")["reason"]
    assert "no current ticker" in reasons[3]


def test_delisted_name_has_end_date_and_delisting_return(loaded):
    _, data = loaded
    sm = data.security_master.set_index("asset")
    assert sm.loc["BBB", "end_date"] == BBB_LAST
    assert pd.isna(sm.loc["AAA", "end_date"])
    assert data.returns.loc[BBB_LAST:, "BBB"].iloc[1:].isna().all()
    assert data.returns.loc[BBB_LAST, "BBB"] < -0.25


def test_market_cap_is_continuous_across_split(loaded):
    _, data = loaded
    cap = data.market_cap["AAA"]
    i = cap.index.get_loc(SPLIT_DAY)
    assert cap.iloc[i] == pytest.approx(cap.iloc[i - 1])
    # shares reported on the pre-split basis are doubled to Yahoo's split-adjusted basis
    close = cap.iloc[i] / 2000.0
    assert close > 0


def test_market_cap_waits_for_first_filing(loaded):
    _, data = loaded
    cap = data.market_cap["AAA"]
    assert cap.loc[: "2023-02-20"].isna().all()            # 10-K accepted after the close on Feb 20
    assert cap.loc["2023-02-21"] > 0


def test_industries_and_risk_free(loaded):
    _, data = loaded
    ind = data.industries.set_index("asset")["industry"]
    assert ind["AAA"] == "BusEq" and ind["BBB"] == "Money"
    assert data.risk_free.notna().all()
    assert data.risk_free.loc["2023-08-01"] > data.risk_free.loc["2023-01-03"]


def test_shares_not_in_fundamentals(loaded):
    _, data = loaded
    assert "shares_outstanding" not in set(data.fundamentals["field"])
