"""Data contracts shared by every model.

Every model in this repo consumes a :class:`MarketData` bundle. A data source plugs in by
implementing :class:`DataLoader`. The engine never touches a vendor format directly, so the
point-in-time and survivorship guarantees below are the loader's job and are checked by
:meth:`MarketData.validate`.

Requirements for any loader
---------------------------
Survivorship-free universe
    ``security_master`` must list every security that traded at any time in the requested
    window, including those that later delisted, merged or went bankrupt. ``returns`` must
    include the delisting return on the final trading day (e.g. CRSP ``DLRET`` compounded
    into ``RET``). Dropping dead names biases factor returns and understates risk.

Point-in-time fundamentals
    ``fundamentals`` is long format, one row per (asset, field, period_end, version) with the
    date the number became public (``available_date``). Restatements are new rows with a later
    ``available_date``; the original row must be kept. The engine only uses a value on or
    after its ``available_date`` (Compustat point-in-time / "as first reported" plus
    restatement history, or a vendor feed with knowledge dates).

Point-in-time classifications
    ``industries`` is effective-dated so that a reclassification is only seen from the date
    it happened.

Units
    Market cap, dollar volume and the money-valued fundamentals must be in the same currency
    and units (e.g. USD). Returns are simple daily total returns as decimals (0.01 = 1%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

SECURITY_MASTER_COLUMNS = ("asset", "start_date", "end_date")
INDUSTRY_COLUMNS = ("asset", "industry", "effective_from", "effective_to")
FUNDAMENTAL_COLUMNS = ("asset", "field", "value", "period_end", "available_date")

# Canonical fundamental field names the factor model looks for. A loader maps vendor fields
# onto these. Missing fields are allowed: descriptors that need them are skipped.
FUNDAMENTAL_FIELDS = (
    "book_equity",        # common shareholders' equity
    "net_income_ttm",     # trailing twelve-month net income
    "sales_ttm",          # trailing twelve-month revenue
    "gross_profit_ttm",   # trailing twelve-month gross profit
    "total_assets",
    "total_debt",         # short + long term debt
    "dividends_ttm",      # trailing twelve-month common dividends paid (total, not per share)
)


class DataValidationError(ValueError):
    """Raised when a MarketData bundle breaks a point-in-time or survivorship rule."""


@dataclass
class MarketData:
    """Everything a model needs, as of the end of the loaded window.

    Attributes
    ----------
    returns : DataFrame (date x asset)
        Daily simple total returns including dividends and delisting returns. NaN when the
        security did not trade.
    market_cap : DataFrame (date x asset)
        End-of-day total market capitalisation.
    dollar_volume : DataFrame (date x asset)
        Daily traded value. Used for the liquidity factor.
    security_master : DataFrame
        Columns ``asset, start_date, end_date``; ``end_date`` is NaT while still listed.
    industries : DataFrame
        Columns ``asset, industry, effective_from, effective_to``; ``effective_to`` NaT while current.
    fundamentals : DataFrame
        Columns ``asset, field, value, period_end, available_date``.
    risk_free : Series (date), optional
        Daily risk-free return. Zero if omitted.
    """

    returns: pd.DataFrame
    market_cap: pd.DataFrame
    dollar_volume: pd.DataFrame
    security_master: pd.DataFrame
    industries: pd.DataFrame
    fundamentals: pd.DataFrame
    risk_free: pd.Series | None = None
    _validated: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        panels = {}
        for name in ("returns", "market_cap", "dollar_volume"):
            panel = getattr(self, name).copy()
            panel.index = pd.DatetimeIndex(panel.index)
            panel = panel.sort_index().astype(float)
            panels[name] = panel
        self.returns = panels["returns"]
        self.market_cap = panels["market_cap"].reindex(
            index=self.returns.index, columns=self.returns.columns
        )
        self.dollar_volume = panels["dollar_volume"].reindex(
            index=self.returns.index, columns=self.returns.columns
        )

        sm = self.security_master.copy()
        sm["start_date"] = pd.to_datetime(sm["start_date"])
        sm["end_date"] = pd.to_datetime(sm["end_date"])
        self.security_master = sm

        ind = self.industries.copy()
        ind["effective_from"] = pd.to_datetime(ind["effective_from"])
        ind["effective_to"] = pd.to_datetime(ind["effective_to"])
        self.industries = ind

        fund = self.fundamentals.copy()
        if len(fund):
            fund["period_end"] = pd.to_datetime(fund["period_end"])
            fund["available_date"] = pd.to_datetime(fund["available_date"])
            fund["value"] = fund["value"].astype(float)
        self.fundamentals = fund

        if self.risk_free is not None:
            rf = self.risk_free.copy()
            rf.index = pd.DatetimeIndex(rf.index)
            self.risk_free = rf.sort_index().reindex(self.returns.index).fillna(0.0)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index

    @property
    def assets(self) -> pd.Index:
        return self.returns.columns

    def validate(self) -> "MarketData":
        """Check the point-in-time and survivorship contract. Raises DataValidationError."""
        problems: list[str] = []

        for name, cols in (
            ("security_master", SECURITY_MASTER_COLUMNS),
            ("industries", INDUSTRY_COLUMNS),
            ("fundamentals", FUNDAMENTAL_COLUMNS),
        ):
            missing = set(cols) - set(getattr(self, name).columns)
            if missing:
                problems.append(f"{name} is missing columns {sorted(missing)}")
        if problems:
            raise DataValidationError("; ".join(problems))

        if not self.returns.index.is_unique:
            problems.append("returns has duplicate dates")
        if not self.returns.columns.is_unique:
            problems.append("returns has duplicate assets")

        r = self.returns.to_numpy()
        if np.any(r[np.isfinite(r)] <= -1.0):
            problems.append("returns contain values <= -100%")

        unknown = set(self.assets) - set(self.security_master["asset"])
        if unknown:
            problems.append(f"{len(unknown)} assets have returns but no security_master row")

        # No data outside a security's listing window: a sign that the panel was built from
        # today's universe or that dead names were back-filled.
        alive = alive_mask(self.security_master, self.dates, self.assets)
        outside = self.returns.notna().to_numpy() & ~alive.to_numpy()
        if outside.any():
            n = int(outside.sum())
            problems.append(f"{n} returns fall outside their security's listing window")

        fund = self.fundamentals
        if len(fund):
            early = fund["available_date"] < fund["period_end"]
            if early.any():
                problems.append(
                    f"{int(early.sum())} fundamental rows are available before their period end"
                )
            dupes = fund.duplicated(["asset", "field", "period_end", "available_date"])
            if dupes.any():
                problems.append(f"{int(dupes.sum())} duplicate fundamental versions")

        ind = self.industries.sort_values(["asset", "effective_from"])
        prev_to = ind.groupby("asset")["effective_to"].shift()
        overlap = prev_to.notna() & (ind["effective_from"] <= prev_to)
        if overlap.any():
            problems.append(f"{int(overlap.sum())} overlapping industry assignments")

        if problems:
            raise DataValidationError("; ".join(problems))
        self._validated = True
        return self


class DataLoader(Protocol):
    """Anything that can produce a :class:`MarketData` bundle for a date window."""

    def load(self, start: pd.Timestamp, end: pd.Timestamp) -> MarketData:  # pragma: no cover
        ...


def alive_mask(
    security_master: pd.DataFrame, dates: pd.DatetimeIndex, assets: pd.Index
) -> pd.DataFrame:
    """Boolean panel (date x asset): True while the security is listed."""
    sm = security_master.drop_duplicates("asset", keep="last").set_index("asset")
    sm = sm.reindex(assets)
    start = sm["start_date"].to_numpy(dtype="datetime64[ns]")
    end = sm["end_date"].fillna(pd.Timestamp.max).to_numpy(dtype="datetime64[ns]")
    d = dates.to_numpy(dtype="datetime64[ns]")[:, None]
    mask = (d >= start[None, :]) & (d <= end[None, :])
    mask &= ~pd.isna(sm["start_date"]).to_numpy()[None, :]
    return pd.DataFrame(mask, index=dates, columns=assets)
