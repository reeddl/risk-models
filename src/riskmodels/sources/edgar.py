"""SEC EDGAR: point-in-time fundamentals, SIC industries and the historical filer list.

Endpoints (all free, no key; see ``http.py`` for fair-access settings)
    company_tickers.json            current ticker -> CIK map
    submissions/CIK##########.json  SIC code, filing history with acceptance times
    api/xbrl/companyfacts/...       every XBRL fact the company has filed, with the filing
    full-index/{year}/QTR{q}/master.idx  every filing in a quarter, including dead filers

Point in time
    Each XBRL fact carries the accession number of the filing it came from. A value is
    available from that filing's acceptance time: the same day if accepted before 16:00,
    otherwise the next business day. EDGAR labels ``acceptanceDateTime`` with a "Z" although
    it is generally Eastern time; either reading of the label gives a date on or after the
    true one under this rule, so the error is only ever to be late. When no acceptance time is
    known the filing date plus one business day is used.

    A later 10-K repeats prior-year figures as comparatives. Each repeat is kept as a new
    version, so a restated number replaces the original only once the restating filing is
    public.

Trailing twelve months
    Flow items (income, revenue, gross profit, dividends) are filed as quarter, half-year,
    nine-month and annual durations. TTM at a period end is the annual figure when one ends
    there, otherwise ``prior fiscal year + current year-to-date - prior year-to-date``, using
    only facts filed on or before the filing being processed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .http import Fetcher

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MASTER_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"

PERIODIC_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A")
DELISTING_FORMS = ("25", "25-NSE", "15-12B", "15-12G", "15-15D")

# Canonical flow fields -> XBRL concepts in priority order (merged, higher priority wins).
FLOW_CONCEPTS: dict[str, tuple[str, ...]] = {
    "net_income_ttm": ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    "sales_ttm": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "gross_profit_ttm": ("GrossProfit",),
    "cost_of_revenue_ttm": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "dividends_ttm": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
}
INSTANT_CONCEPTS: dict[str, tuple[str, ...]] = {
    "book_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "total_assets": ("Assets",),
}
DEBT_CONCEPTS = (
    "LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtCurrent",
    "ShortTermBorrowings", "CommercialPaper",
)


# --------------------------------------------------------------------------- industries
# Fama-French 12 industries from 4-digit SIC (Ken French data library definitions).
_FF12_RANGES: list[tuple[str, list[tuple[int, int]]]] = [
    ("NoDur", [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ("Durbl", [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714),
               (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),
    ("Manuf", [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569),
               (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
               (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)]),
    ("Enrgy", [(1200, 1399), (2900, 2999)]),
    ("Chems", [(2800, 2829), (2840, 2899)]),
    ("BusEq", [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ("Telcm", [(4800, 4899)]),
    ("Utils", [(4900, 4949)]),
    ("Shops", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Hlth", [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ("Money", [(6000, 6999)]),
]


def sic_to_industry(sic, scheme: str = "ff12") -> str | None:
    """Map a SIC code to an industry label. scheme: 'ff12', 'sic2' (2-digit major group)
    or 'sic_division' (the 10 SIC divisions)."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None
    if code <= 0:
        return None
    if scheme == "sic2":
        return f"SIC{code // 100:02d}"
    if scheme == "sic_division":
        for lo, hi, name in [(100, 999, "Agriculture"), (1000, 1499, "Mining"),
                             (1500, 1799, "Construction"), (2000, 3999, "Manufacturing"),
                             (4000, 4999, "Transport & Utilities"), (5000, 5199, "Wholesale"),
                             (5200, 5999, "Retail"), (6000, 6799, "Finance"),
                             (7000, 8999, "Services"), (9100, 9999, "Public Admin")]:
            if lo <= code <= hi:
                return name
        return "Other"
    if scheme == "ff12":
        for name, ranges in _FF12_RANGES:
            if any(lo <= code <= hi for lo, hi in ranges):
                return name
        return "Other"
    raise ValueError(f"unknown industry scheme {scheme!r}")


# --------------------------------------------------------------------------- parsing
def parse_ticker_map(payload: dict) -> pd.DataFrame:
    rows = [(int(v["cik_str"]), str(v["ticker"]).upper(), v.get("title")) for v in payload.values()]
    return pd.DataFrame(rows, columns=["cik", "ticker", "title"])


def _filings_frame(block: dict) -> pd.DataFrame:
    if not block:
        return pd.DataFrame(columns=["accession", "form", "filing_date", "acceptance"])
    n = len(block.get("accessionNumber", []))
    return pd.DataFrame({
        "accession": block.get("accessionNumber", []),
        "form": block.get("form", [None] * n),
        "filing_date": pd.to_datetime(block.get("filingDate", [None] * n)),
        "acceptance": pd.to_datetime(
            [str(a).replace("Z", "") if a else None for a in block.get("acceptanceDateTime", [None] * n)]
        ),
    })


@dataclass
class Submissions:
    cik: int
    name: str | None
    sic: int | None
    sic_description: str | None
    tickers: list[str]
    filings: pd.DataFrame = field(repr=False)
    older_files: list[str] = field(default_factory=list, repr=False)

    def delisting_date(self, after=None) -> pd.Timestamp | None:
        f = self.filings[self.filings["form"].isin(DELISTING_FORMS)]
        if after is not None:
            f = f[f["filing_date"] >= pd.Timestamp(after)]
        return None if f.empty else f["filing_date"].min()


def parse_submissions(payload: dict) -> Submissions:
    recent = payload.get("filings", {}).get("recent", {})
    files = [f["name"] for f in payload.get("filings", {}).get("files", []) if "name" in f]
    sic = payload.get("sic")
    return Submissions(
        cik=int(payload.get("cik", 0)),
        name=payload.get("name"),
        sic=int(sic) if sic not in (None, "") else None,
        sic_description=payload.get("sicDescription"),
        tickers=[str(t).upper() for t in payload.get("tickers", [])],
        filings=_filings_frame(recent),
        older_files=files,
    )


def available_dates(filings: pd.DataFrame, cutoff_hour: int = 16) -> pd.Series:
    """Accession -> first date the filing's data counts as known (see module docstring)."""
    if filings.empty:
        return pd.Series(dtype="datetime64[ns]")
    acc = filings["acceptance"]
    day = acc.dt.normalize()
    late = acc.dt.hour >= cutoff_hour
    from_accept = day.where(~late, day + pd.offsets.BDay(1))
    from_filed = filings["filing_date"] + pd.offsets.BDay(1)
    out = from_accept.where(acc.notna(), from_filed)
    return pd.Series(out.to_numpy(), index=filings["accession"]).groupby(level=0).min()


def _concept_facts(payload: dict, concept: str, taxonomy: str = "us-gaap",
                   unit: str | None = "USD") -> pd.DataFrame:
    node = payload.get("facts", {}).get(taxonomy, {}).get(concept)
    cols = ["start", "end", "val", "accn", "form", "filed"]
    if not node:
        return pd.DataFrame(columns=cols)
    units = node.get("units", {})
    rows = units.get(unit) if unit else next(iter(units.values()), [])
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["val"] = df["val"].astype(float)
    return df[df["form"].isin(PERIODIC_FORMS)]


def _merged(payload: dict, concepts: tuple[str, ...], unit: str = "USD") -> pd.DataFrame:
    frames = []
    for rank, c in enumerate(concepts):
        f = _concept_facts(payload, c, unit=unit)
        if len(f):
            frames.append(f.assign(rank=rank))
    if not frames:
        return pd.DataFrame(columns=["start", "end", "val", "accn", "form", "filed", "rank"])
    df = pd.concat(frames, ignore_index=True).sort_values("rank")
    return df.drop_duplicates(["start", "end", "accn"], keep="first").drop(columns="rank")


def _months(start: pd.Series, end: pd.Series) -> pd.Series:
    days = (end - start).dt.days + 1
    out = pd.Series(np.nan, index=start.index)
    for m, lo, hi in [(3, 80, 100), (6, 170, 195), (9, 260, 285), (12, 340, 380)]:
        out[(days >= lo) & (days <= hi)] = m
    return out


def ttm_from_durations(facts: pd.DataFrame, tolerance_days: int = 15) -> pd.DataFrame:
    """Trailing-twelve-month values, one row per (period end, filing).

    Returns columns end, val, accn, filed.
    """
    f = facts.dropna(subset=["start", "end", "val"]).copy()
    if f.empty:
        return pd.DataFrame(columns=["end", "val", "accn", "filed"])
    f["m"] = _months(f["start"], f["end"])
    f = f.dropna(subset=["m"])
    tol = pd.Timedelta(days=tolerance_days)
    annual = f[f["m"] == 12]
    rows = [(r.end, r.val, r.accn, r.filed) for r in annual.itertuples(index=False)]
    for r in f[f["m"] < 12].itertuples(index=False):
        known = f[f["filed"] <= r.filed]
        prev = known[(known["m"] == r.m)
                     & ((known["end"] - (r.end - pd.Timedelta(days=365))).abs() <= tol)]
        fy = known[(known["m"] == 12)
                   & ((known["end"] - (r.start - pd.Timedelta(days=1))).abs() <= tol)]
        if prev.empty or fy.empty:
            continue
        p = prev.sort_values("filed").iloc[-1]
        y = fy.sort_values("filed").iloc[-1]
        rows.append((r.end, y.val + r.val - p.val, r.accn, r.filed))
    out = pd.DataFrame(rows, columns=["end", "val", "accn", "filed"])
    return out.drop_duplicates(["end", "accn"], keep="first")


def _instants(facts: pd.DataFrame) -> pd.DataFrame:
    f = facts[facts["start"].isna()] if "start" in facts else facts
    return f.dropna(subset=["end", "val"])[["end", "val", "accn", "filed"]]


def _total_debt(payload: dict) -> pd.DataFrame:
    parts = []
    for c in DEBT_CONCEPTS:
        f = _instants(_concept_facts(payload, c))
        if len(f):
            parts.append(f.assign(concept=c))
    if not parts:
        return pd.DataFrame(columns=["end", "val", "accn", "filed"])
    df = pd.concat(parts)
    wide = df.pivot_table(index=["end", "accn", "filed"], columns="concept", values="val",
                          aggfunc="last")
    get = lambda c: wide[c] if c in wide else pd.Series(np.nan, index=wide.index)
    long_term = get("LongTermDebt").fillna(
        get("LongTermDebtNoncurrent").fillna(0) + get("LongTermDebtCurrent").fillna(0)
    )
    short = get("ShortTermBorrowings").fillna(0) + get("CommercialPaper").fillna(0)
    return (long_term + short).rename("val").reset_index()


def shares_outstanding(payload: dict) -> pd.DataFrame:
    """Common shares outstanding per filing: cover-page figure, else balance sheet."""
    dei = _concept_facts(payload, "EntityCommonStockSharesOutstanding", taxonomy="dei", unit="shares")
    if dei.empty:
        dei = _concept_facts(payload, "CommonStockSharesOutstanding", unit="shares")
    f = _instants(dei)
    return f.groupby(["end", "accn", "filed"], as_index=False)["val"].sum()


def companyfacts_to_fundamentals(
    payload: dict,
    asset: str,
    available: pd.Series | None = None,
) -> pd.DataFrame:
    """Company facts JSON -> long point-in-time fundamentals for one asset.

    Fields follow ``riskmodels.core.data.FUNDAMENTAL_FIELDS`` plus ``shares_outstanding``.
    ``available`` maps accession -> available date (from :func:`available_dates`).
    """
    pieces = []
    for fld, concepts in FLOW_CONCEPTS.items():
        t = ttm_from_durations(_merged(payload, concepts))
        if len(t):
            pieces.append(t.assign(field=fld))
    for fld, concepts in INSTANT_CONCEPTS.items():
        t = _instants(_merged(payload, concepts))
        if len(t):
            pieces.append(t.assign(field=fld))
    debt = _total_debt(payload)
    if len(debt):
        pieces.append(debt.assign(field="total_debt"))
    sh = shares_outstanding(payload)
    if len(sh):
        pieces.append(sh.assign(field="shares_outstanding"))
    if not pieces:
        return pd.DataFrame(columns=["asset", "field", "value", "period_end", "available_date"])

    df = pd.concat(pieces, ignore_index=True)
    fallback = df["filed"] + pd.offsets.BDay(1)
    if available is not None and len(available):
        df["available_date"] = df["accn"].map(available).fillna(fallback)
    else:
        df["available_date"] = fallback
    df["available_date"] = pd.to_datetime(df["available_date"])

    # Gross profit where only revenue and cost of revenue are filed.
    gp = df[df["field"] == "gross_profit_ttm"]
    rev = df[df["field"] == "sales_ttm"].set_index(["end", "accn"])["val"]
    cost = df[df["field"] == "cost_of_revenue_ttm"].set_index(["end", "accn"])
    derived = (rev - cost["val"]).dropna()
    derived = derived[~derived.index.isin(gp.set_index(["end", "accn"]).index)]
    if len(derived):
        extra = cost.loc[derived.index, ["filed", "available_date"]].assign(val=derived)
        extra = extra.reset_index().assign(field="gross_profit_ttm")
        df = pd.concat([df, extra], ignore_index=True)
    df = df[df["field"] != "cost_of_revenue_ttm"]

    # Never earlier than the period end; drop exact duplicates of a version.
    df["available_date"] = df[["available_date", "end"]].max(axis=1)
    out = pd.DataFrame({
        "asset": asset,
        "field": df["field"].to_numpy(),
        "value": df["val"].astype(float).to_numpy(),
        "period_end": df["end"].to_numpy(),
        "available_date": df["available_date"].to_numpy(),
    })
    return out.drop_duplicates(["asset", "field", "period_end", "available_date"], keep="first")


def parse_master_index(text: str) -> pd.DataFrame:
    """master.idx (pipe-delimited) -> cik, company, form, date_filed."""
    lines = text.splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if line.startswith("-----")), 0)
    rows = [line.split("|") for line in lines[start:] if line.count("|") == 4]
    df = pd.DataFrame(rows, columns=["cik", "company", "form", "date_filed", "path"])
    df["cik"] = df["cik"].astype(int)
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    return df.drop(columns="path")


# --------------------------------------------------------------------------- client
class Edgar:
    def __init__(self, client: Fetcher) -> None:
        self.client = client

    def ticker_map(self) -> pd.DataFrame:
        return parse_ticker_map(json.loads(self.client.get(TICKERS_URL, max_age_days=7)))

    def submissions(self, cik: int, include_older: bool = True) -> Submissions:
        sub = parse_submissions(json.loads(self.client.get(SUBMISSIONS_URL.format(cik=cik), max_age_days=7)))
        if include_older and sub.older_files:
            frames = [sub.filings]
            for name in sub.older_files:
                raw = json.loads(self.client.get(SUBMISSIONS_FILE_URL.format(name=name), max_age_days=30))
                frames.append(_filings_frame(raw))
            sub.filings = pd.concat(frames, ignore_index=True)
        return sub

    def fundamentals(self, cik: int, asset: str, submissions: Submissions | None = None) -> pd.DataFrame:
        sub = submissions or self.submissions(cik)
        facts = json.loads(self.client.get(FACTS_URL.format(cik=cik), max_age_days=7))
        return companyfacts_to_fundamentals(facts, asset, available_dates(sub.filings))

    def filers(self, start, end, forms=("10-K", "10-Q")) -> pd.DataFrame:
        """Every CIK that filed one of ``forms`` between start and end, alive or not."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        frames = []
        for year in range(start.year, end.year + 1):
            for q in range(1, 5):
                q_start = pd.Timestamp(year=year, month=3 * q - 2, day=1)
                if q_start > end or q_start + pd.offsets.QuarterEnd(0) < start:
                    continue
                raw = self.client.get(MASTER_INDEX_URL.format(year=year, quarter=q), max_age_days=30)
                df = parse_master_index(raw.decode("latin-1"))
                frames.append(df[df["form"].isin(forms)])
        if not frames:
            return pd.DataFrame(columns=["cik", "company", "first_filed", "last_filed", "n_filings"])
        df = pd.concat(frames)
        df = df[(df["date_filed"] >= start) & (df["date_filed"] <= end)]
        return (df.groupby("cik")
                  .agg(company=("company", "last"), first_filed=("date_filed", "min"),
                       last_filed=("date_filed", "max"), n_filings=("form", "size"))
                  .reset_index())
