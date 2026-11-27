"""PublicDataLoader: a ``DataLoader`` built only from free public sources.

    prices, returns, volume    Yahoo Finance chart API
    fundamentals, shares       SEC EDGAR XBRL company facts, stamped with filing acceptance dates
    industries                 SEC EDGAR SIC codes, mapped to Fama-French 12 (or SIC 2-digit)
    universe                   SEC EDGAR quarterly filing indexes (every 10-K/10-Q filer)
    risk-free rate             FRED (DTB3)
    short interest (extra)     FINRA consolidated short interest

How close this gets to the data contract in ``core/data.py``, and what stays biased, is in
``sources/README.md``. In short: the universe of *filers* is survivorship-free, but prices
are only available for names Yahoo still carries, so most dead companies drop out and are
listed in :attr:`PublicDataLoader.report`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.data import MarketData
from .edgar import Edgar, sic_to_industry
from .finra import FinraShortInterest
from .fred import Fred
from .http import Fetcher, HttpClient, HttpError
from .yahoo import YahooPrices, split_factor, total_returns


@dataclass
class CoverageReport:
    """What the loader found and what it had to leave out."""

    filers: int = 0
    with_ticker: int = 0
    loaded: int = 0
    excluded: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["cik", "name", "reason"]))
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"EDGAR 10-K/10-Q filers in window: {self.filers}",
            f"  with a current ticker:          {self.with_ticker}",
            f"  loaded with prices:             {self.loaded}",
        ]
        if len(self.excluded):
            for reason, n in self.excluded["reason"].value_counts().items():
                lines.append(f"  excluded, {reason}: {n}")
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def shares_on_today_basis(shares: pd.DataFrame, factor: pd.Series) -> pd.Series:
    """Reported share counts (one asset) -> series by available date, rebased to the split
    basis of Yahoo's split-adjusted prices: shares_d * F(d), d = the count's as-of date."""
    if shares.empty:
        return pd.Series(dtype=float)
    f = factor.sort_index()
    d = pd.DatetimeIndex(shares["period_end"])
    pos = f.index.searchsorted(d, side="right") - 1
    fd = np.where(pos >= 0, f.to_numpy()[np.clip(pos, 0, None)], f.iloc[0])
    out = pd.Series(shares["value"].to_numpy() * fd, index=pd.DatetimeIndex(shares["available_date"]))
    return out[out > 0].sort_index().groupby(level=0).last()


class PublicDataLoader:
    """Implements ``riskmodels.core.DataLoader`` from EDGAR, Yahoo and FRED.

    Parameters
    ----------
    client : Fetcher, optional
        Shared HTTP client (rate limits and cache). Defaults to ``HttpClient()``; set
        ``RISKMODELS_USER_AGENT`` to declare yourself to the SEC first.
    tickers : list of str, optional
        Fixed list of names. Default: every 10-K/10-Q filer in the window that has a ticker.
    industry_scheme : "ff12" | "sic2" | "sic_division"
    ticker_overrides : dict CIK -> ticker, optional
        Tickers for filers missing from EDGAR's current ticker file (for example a delisted
        name whose price history you can still get under some symbol).
    delisting_return : float, optional
        Return applied on the last day of a name whose SEC filings show a delisting (Form
        25/15). Default None: no delisting return. See README for the size of that bias.
    min_price_days : int
        Names with fewer priced days in the window are excluded.
    max_staleness_days : int
        Share counts older than this (by available date) are not used for market cap.
    limit : int, optional
        Load at most this many names (for trying the pipeline out).
    """

    def __init__(
        self,
        client: Fetcher | None = None,
        tickers: list[str] | None = None,
        industry_scheme: str = "ff12",
        ticker_overrides: dict[int, str] | None = None,
        delisting_return: float | None = None,
        min_price_days: int = 60,
        max_staleness_days: int = 400,
        risk_free_series: str = "DTB3",
        limit: int | None = None,
    ) -> None:
        self.client = client or HttpClient()
        self.edgar = Edgar(self.client)
        self.yahoo = YahooPrices(self.client)
        self.fred = Fred(self.client)
        self.tickers = [t.upper() for t in tickers] if tickers else None
        self.industry_scheme = industry_scheme
        self.ticker_overrides = dict(ticker_overrides or {})
        self.delisting_return = delisting_return
        self.min_price_days = min_price_days
        self.max_staleness_days = max_staleness_days
        self.risk_free_series = risk_free_series
        self.limit = limit
        self.report = CoverageReport()

    # ------------------------------------------------------------------ universe
    def universe(self, start, end) -> pd.DataFrame:
        """cik, ticker, name for every name to try. Records filers with no ticker."""
        tmap = self.edgar.ticker_map().drop_duplicates("cik", keep="first")
        excluded = []
        if self.tickers:
            u = tmap[tmap["ticker"].isin(self.tickers)].rename(columns={"title": "name"})
            missing = sorted(set(self.tickers) - set(u["ticker"]))
            excluded += [(None, t, "ticker not in EDGAR's ticker file") for t in missing]
            self.report.filers = len(self.tickers)
        else:
            filers = self.edgar.filers(start, end)
            self.report.filers = len(filers)
            u = filers.merge(tmap, on="cik", how="left")
            for cik, tkr in self.ticker_overrides.items():
                u.loc[u["cik"] == cik, "ticker"] = tkr.upper()
            no_tkr = u[u["ticker"].isna()]
            excluded += [(c, n, "no current ticker (delisted, merged or never listed)")
                         for c, n in zip(no_tkr["cik"], no_tkr["company"])]
            u = u.dropna(subset=["ticker"]).rename(columns={"company": "name"})
        self.report.with_ticker = len(u)
        self.report.excluded = pd.DataFrame(excluded, columns=["cik", "name", "reason"])
        u = u[["cik", "ticker", "name"]].drop_duplicates("ticker").reset_index(drop=True)
        return u.head(self.limit) if self.limit else u

    # ------------------------------------------------------------------ load
    def load(self, start, end) -> MarketData:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.report = CoverageReport()
        u = self.universe(start, end)

        rets, caps, dvol, master, inds, funds, excluded = {}, {}, {}, [], [], [], []
        for row in u.itertuples(index=False):
            asset = row.ticker
            try:
                px = self.yahoo.history(asset, start, end)
            except (HttpError, ValueError) as e:
                excluded.append((row.cik, row.name, f"no Yahoo prices ({type(e).__name__})"))
                continue
            px = px.dropna(subset=["adjclose"])
            if len(px) < self.min_price_days:
                excluded.append((row.cik, row.name, "too little price history"))
                continue
            try:
                sub = self.edgar.submissions(int(row.cik))
                fund = self.edgar.fundamentals(int(row.cik), asset, submissions=sub)
            except (HttpError, ValueError) as e:
                excluded.append((row.cik, row.name, f"no EDGAR data ({type(e).__name__})"))
                continue

            shares = fund[fund["field"] == "shares_outstanding"]
            funds.append(fund[fund["field"] != "shares_outstanding"])
            sh = shares_on_today_basis(shares, split_factor(px["split_ratio"]))
            if sh.empty:
                excluded.append((row.cik, row.name, "no shares outstanding in XBRL"))
                continue
            idx = sh.index.union(px.index)
            sh_d = sh.reindex(idx).ffill()
            age = pd.Series(sh.index, index=sh.index).reindex(idx).ffill()
            ok = (idx.to_series() - age).dt.days <= self.max_staleness_days
            sh_d = sh_d.where(ok).reindex(px.index)

            r = total_returns(px)
            first, last = px.index[0], px.index[-1]
            delist = sub.delisting_date(after=start)
            if self.delisting_return is not None and delist is not None and last < end - pd.Timedelta(days=7):
                r.iloc[-1] = (1 + r.iloc[-1]) * (1 + self.delisting_return) - 1 if pd.notna(r.iloc[-1]) else self.delisting_return

            rets[asset] = r
            caps[asset] = px["close"] * sh_d
            dvol[asset] = px["close"] * px["volume"]
            master.append((asset, first, last if last < end - pd.Timedelta(days=7) else pd.NaT))
            industry = sic_to_industry(sub.sic, self.industry_scheme) or "Unknown"
            inds.append((asset, industry, first, pd.NaT))

        self.report.loaded = len(rets)
        self.report.excluded = pd.concat(
            [self.report.excluded, pd.DataFrame(excluded, columns=["cik", "name", "reason"])],
            ignore_index=True,
        )
        if not rets:
            raise ValueError("no names could be loaded; see loader.report")

        returns = pd.DataFrame(rets).sort_index()
        dates = returns.index
        try:
            rf = self.fred.risk_free(dates, self.risk_free_series)
        except (HttpError, ValueError, KeyError) as e:
            rf = None
            self.report.notes.append(f"risk-free rate not loaded ({e}); returns are not in excess of cash")
            warnings.warn(self.report.notes[-1])

        fundamentals = pd.concat(funds, ignore_index=True) if funds else pd.DataFrame(
            columns=["asset", "field", "value", "period_end", "available_date"])
        self.report.notes.append("industries use each company's current SIC code for its whole history")
        return MarketData(
            returns=returns,
            market_cap=pd.DataFrame(caps).reindex(dates),
            dollar_volume=pd.DataFrame(dvol).reindex(dates),
            security_master=pd.DataFrame(master, columns=["asset", "start_date", "end_date"]),
            industries=pd.DataFrame(inds, columns=["asset", "industry", "effective_from", "effective_to"]),
            fundamentals=fundamentals,
            risk_free=rf,
        )

    def short_interest(self, start, end, symbols: list[str] | None = None) -> pd.DataFrame:
        """FINRA short interest, long format with available dates (not used by the factor model)."""
        return FinraShortInterest(self.client).history(start, end, symbols)
