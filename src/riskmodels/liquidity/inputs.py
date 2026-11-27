"""Inputs the liquidity model needs, built from the shared data bundle and the risk model.

Nothing here is a new estimate of risk: volatilities come from the factor model and volume
from ``MarketData.dollar_volume``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.data import MarketData
from ..factor_model.risk import RiskModel
from .config import DEFAULT_SPREAD_TIERS, LiquidityConfig

TRADING_DAYS = 252


def _as_of_row(panel: pd.DataFrame, as_of) -> int:
    if as_of is None:
        return len(panel.index) - 1
    pos = int(panel.index.searchsorted(pd.Timestamp(as_of), side="right")) - 1
    if pos < 0:
        raise ValueError(f"no data on or before {pd.Timestamp(as_of).date()}")
    return pos


def adv_from_data(
    data: MarketData,
    as_of=None,
    window: int = 20,
    method: str = "median",
    min_obs: int = 5,
) -> pd.Series:
    """Average daily traded value per name over the ``window`` days ending on ``as_of``.

    Days a name did not trade are skipped. A name with fewer than ``min_obs`` traded days
    in the window gets NaN, which the model reports as "no ADV".
    """
    dv = data.dollar_volume
    end = _as_of_row(dv, as_of)
    block = dv.iloc[max(0, end - window + 1): end + 1]
    block = block.where(block > 0)
    adv = block.median() if method == "median" else block.mean()
    adv = adv.where(block.notna().sum() >= min_obs)
    adv.name = "adv"
    return adv


def market_cap_from_data(data: MarketData, as_of=None) -> pd.Series:
    """Market cap on ``as_of`` (last available value up to that day)."""
    cap = data.market_cap
    end = _as_of_row(cap, as_of)
    out = cap.iloc[: end + 1].ffill().iloc[-1]
    out.name = "market_cap"
    return out


def spreads_from_market_cap(
    market_cap: pd.Series,
    tiers: tuple[tuple[float, float], ...] = DEFAULT_SPREAD_TIERS,
    default_bps: float = 25.0,
) -> pd.Series:
    """Quoted spread in basis points from a market cap schedule (largest tier first)."""
    tiers = sorted(tiers, key=lambda t: -t[0])
    cap = market_cap.astype(float)
    out = pd.Series(default_bps, index=cap.index, dtype=float, name="spread_bps")
    known = cap.notna()
    assigned = pd.Series(False, index=cap.index)
    for floor, bps in tiers:
        hit = known & ~assigned & (cap >= floor)
        out[hit] = bps
        assigned |= hit
    return out


def daily_vol_from_risk_model(risk_model: RiskModel) -> pd.Series:
    """Total daily volatility per name: annualized total risk / sqrt(252)."""
    out = risk_model.total_risk() / np.sqrt(TRADING_DAYS)
    out.name = "daily_vol"
    return out


def resolve_inputs(
    names: pd.Index,
    adv: pd.Series,
    config: LiquidityConfig,
    market_cap: pd.Series | None = None,
    spread_bps: pd.Series | None = None,
    daily_vol: pd.Series | None = None,
    risk_model: RiskModel | None = None,
) -> pd.DataFrame:
    """One row per name: adv, spread_bps, daily_vol and where each came from."""
    out = pd.DataFrame(index=names)
    a = pd.Series(adv, dtype=float).reindex(names)
    out["adv"] = a.where(a > 0)

    if spread_bps is not None:
        s = pd.Series(spread_bps, dtype=float).reindex(names)
        src = np.where(s.notna(), "given", "")
    else:
        s = pd.Series(np.nan, index=names)
        src = np.full(len(names), "", dtype=object)
    if market_cap is not None:
        tier = spreads_from_market_cap(pd.Series(market_cap, dtype=float).reindex(names),
                                       config.spread_tiers, config.default_spread_bps)
        fill = s.isna() & pd.Series(market_cap, dtype=float).reindex(names).notna()
        s = s.where(~fill, tier)
        src = np.where(fill, "cap_tier", src)
    missing = s.isna()
    s = s.fillna(config.default_spread_bps)
    src = np.where(missing, "default", src)
    out["spread_bps"] = s
    out["spread_source"] = src

    if daily_vol is not None:
        v = pd.Series(daily_vol, dtype=float).reindex(names)
    elif risk_model is not None:
        v = daily_vol_from_risk_model(risk_model).reindex(names)
    else:
        v = pd.Series(np.nan, index=names)
    out["vol_source"] = np.where(v.notna(), "given" if daily_vol is not None else
                                 ("risk_model" if risk_model is not None else "default"), "default")
    out["daily_vol"] = v.fillna(config.default_daily_vol)
    return out
