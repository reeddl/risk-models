"""Point-in-time lookups for fundamentals and classifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FundamentalSnapshot:
    """What was knowable on ``as_of``.

    ``current`` holds the latest reported period per (asset, field), using the latest
    version (restatement) public by ``as_of``. ``prior_year`` holds the period about one year
    earlier, again as known on ``as_of``. Both are asset x field.
    """

    as_of: pd.Timestamp
    current: pd.DataFrame
    prior_year: pd.DataFrame


class PointInTimeFundamentals:
    """Fast repeated point-in-time snapshots of a long fundamentals table.

    A snapshot only changes when a new row becomes available, so snapshots are cached by the
    most recent ``available_date`` on or before the requested date. Staleness is applied per
    request: a value whose ``period_end`` is older than ``max_age_days`` counts as missing.
    """

    def __init__(
        self,
        fundamentals: pd.DataFrame,
        max_age_days: int = 548,
        prior_year_tolerance_days: int = 45,
    ) -> None:
        df = fundamentals.loc[:, ["asset", "field", "value", "period_end", "available_date"]]
        df = df.dropna(subset=["value", "period_end", "available_date"])
        self._df = df.sort_values("available_date").reset_index(drop=True)
        self._avail = self._df["available_date"].to_numpy(dtype="datetime64[ns]")
        self._keys = np.unique(self._avail)
        self.max_age = pd.Timedelta(days=max_age_days)
        self.tolerance = pd.Timedelta(days=prior_year_tolerance_days)
        self._cache: dict[np.datetime64, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _build(self, key: np.datetime64) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = int(np.searchsorted(self._avail, key, side="right"))
        avail = self._df.iloc[:n]
        ordered = avail.sort_values(["period_end", "available_date"], kind="mergesort")
        latest = ordered.groupby(["asset", "field"], sort=False).tail(1)

        anchor = latest[["asset", "field", "period_end"]].rename(columns={"period_end": "anchor"})
        cand = avail.merge(anchor, on=["asset", "field"])
        cand = cand[cand["period_end"] < cand["anchor"]]
        target = cand["anchor"] - pd.Timedelta(days=365)
        cand = cand.assign(gap=(cand["period_end"] - target).abs())
        cand = cand[cand["gap"] <= self.tolerance]
        cand = cand.sort_values(["gap", "available_date"], ascending=[True, False], kind="mergesort")
        prior = cand.groupby(["asset", "field"], sort=False).head(1)
        prior = prior.drop(columns=["gap"])
        return latest, prior

    def snapshot(self, as_of: pd.Timestamp) -> FundamentalSnapshot:
        as_of = pd.Timestamp(as_of)
        i = int(np.searchsorted(self._keys, np.datetime64(as_of, "ns"), side="right")) - 1
        empty = pd.DataFrame(dtype=float)
        if i < 0:
            return FundamentalSnapshot(as_of, empty, empty)
        key = self._keys[i]
        if key not in self._cache:
            self._cache[key] = self._build(key)
        latest, prior = self._cache[key]

        fresh = latest[latest["period_end"] >= as_of - self.max_age]
        current = fresh.pivot(index="asset", columns="field", values="value")
        prior = prior.merge(fresh[["asset", "field"]], on=["asset", "field"])
        prior_wide = prior.pivot(index="asset", columns="field", values="value")
        return FundamentalSnapshot(as_of, current, prior_wide)


def industry_panel(
    industries: pd.DataFrame, dates: pd.DatetimeIndex, assets: pd.Index
) -> pd.DataFrame:
    """Object panel (date x asset) holding each security's industry as known on that date."""
    out = np.full((len(dates), len(assets)), None, dtype=object)
    pos = {a: j for j, a in enumerate(assets)}
    d = dates.to_numpy(dtype="datetime64[ns]")
    ind = industries.sort_values("effective_from")
    for row in ind.itertuples(index=False):
        j = pos.get(row.asset)
        if j is None:
            continue
        lo = int(np.searchsorted(d, np.datetime64(row.effective_from, "ns"), side="left"))
        if pd.isna(row.effective_to):
            hi = len(d)
        else:
            hi = int(np.searchsorted(d, np.datetime64(row.effective_to, "ns"), side="right"))
        out[lo:hi, j] = row.industry
    return pd.DataFrame(out, index=dates, columns=assets)
