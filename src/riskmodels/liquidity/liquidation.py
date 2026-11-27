"""Days to liquidate, buckets, the liquidation profile, trading cost and gross reduction.

Execution model: every name is traded at ``participation`` of its ADV per day, all names at
once. So a position of |w| x NAV in a name with ADV A takes

    days to liquidate = |w| x NAV / (participation x A)

and the part of the book exitable within t days is sum_i min(|w_i|, participation x A_i x t / NAV).
Longs (sell) and shorts (buy to cover) are treated the same.

Trading cost per unit traded, for an order of Q in a name with ADV A, daily volatility
sigma and quoted spread s:

    cost = s / 2 + Y x sigma x sqrt(Q / A)

the half spread plus the square-root market impact law. Stress multipliers scale A down,
s and sigma up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LiquidityConfig

NO_ADV = "no ADV"


def days_to_liquidate(weights: pd.Series, adv: pd.Series, nav: float,
                      participation: float, volume_multiplier: float = 1.0) -> pd.Series:
    """Days to exit each position; inf where there is no ADV."""
    w = weights.astype(float).abs()
    a = pd.Series(adv, dtype=float).reindex(w.index) * volume_multiplier
    d = w * nav / (participation * a)
    d = d.where(a > 0, np.inf)
    d[w == 0] = 0.0
    d.name = "days_to_liquidate"
    return d


def bucket(days: pd.Series, config: LiquidityConfig) -> pd.Series:
    """Bucket label per name ('<1d', '1-5d', '5-10d', '>10d', or 'no ADV')."""
    labels = config.bucket_labels
    edges = [-np.inf, *config.bucket_edges, np.inf]
    out = pd.cut(days.replace(np.inf, np.nan), bins=edges, labels=labels, right=True)
    out = out.astype(object).where(np.isfinite(days), NO_ADV)
    out.name = "bucket"
    return out


def bucket_table(weights: pd.Series, buckets: pd.Series, nav: float, config: LiquidityConfig) -> pd.DataFrame:
    """Long, short, gross and share of gross in each bucket, slowest last."""
    w = weights.astype(float)
    order = config.bucket_labels + [NO_ADV]
    g = pd.DataFrame({"w": w, "bucket": buckets.reindex(w.index)})
    gross = float(w.abs().sum())
    t = pd.DataFrame({
        "n_positions": g.groupby("bucket")["w"].size(),
        "long": g["w"].clip(lower=0).groupby(g["bucket"]).sum(),
        "short": g["w"].clip(upper=0).groupby(g["bucket"]).sum(),
    }).reindex(order).fillna(0.0)
    t["n_positions"] = t["n_positions"].astype(int)
    t["gross"] = t["long"] - t["short"]
    t["share_of_gross"] = t["gross"] / gross if gross > 0 else np.nan
    t["cumulative_share"] = t["share_of_gross"].cumsum()
    t["value"] = t["gross"] * nav
    t.index.name = "bucket"
    return t


def daily_capacity(adv: pd.Series, nav: float, participation: float,
                   volume_multiplier: float = 1.0) -> pd.Series:
    """Weight (fraction of NAV) each name can trade per day; 0 where there is no ADV."""
    a = pd.Series(adv, dtype=float) * volume_multiplier
    return (participation * a / nav).where(a > 0, 0.0).fillna(0.0)


def liquidatable(weights: pd.Series, capacity: pd.Series, days: float) -> pd.Series:
    """Weight exitable in each name within ``days``."""
    w = weights.astype(float).abs()
    return np.minimum(w, capacity.reindex(w.index).fillna(0.0) * days)


def liquidation_profile(weights: pd.Series, adv: pd.Series, nav: float, config: LiquidityConfig,
                        days: tuple[float, ...] | None = None,
                        volume_multiplier: float = 1.0) -> pd.DataFrame:
    """Share of the book exitable within each horizon (rows), as a share of gross and of NAV."""
    w = weights.astype(float)
    cap = daily_capacity(adv.reindex(w.index), nav, config.participation, volume_multiplier)
    gross = float(w.abs().sum())
    rows = []
    for t in days or config.profile_days:
        q = liquidatable(w, cap, t)
        rows.append({
            "days": t,
            "long": float(q[w > 0].sum()),
            "short": float(q[w < 0].sum()),
            "gross": float(q.sum()),
            "share_of_gross": float(q.sum()) / gross if gross > 0 else np.nan,
            "share_of_nav": float(q.sum()),
        })
    return pd.DataFrame(rows).set_index("days")


def days_to_liquidate_share(weights: pd.Series, adv: pd.Series, nav: float, share: float,
                            participation: float, volume_multiplier: float = 1.0) -> float:
    """Fewest days to exit ``share`` of gross trading every name at full participation.

    inf when the names with ADV hold less than ``share`` of gross.
    """
    w = weights.astype(float).abs()
    w = w[w > 0]
    if w.empty:
        return 0.0
    cap = daily_capacity(adv.reindex(w.index), nav, participation, volume_multiplier)
    target = share * float(w.sum())
    if float(w[cap > 0].sum()) < target - 1e-12:
        return float("inf")
    # liquidatable(t) is piecewise linear and increasing: its kinks are each name's DTL.
    dtl = (w / cap.where(cap > 0)).dropna().sort_values()
    for t in dtl.to_numpy():
        if float(np.minimum(w, cap * t).sum()) >= target - 1e-12:
            hi = t
            break
    else:  # pragma: no cover - guarded by the reachability check above
        return float("inf")
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(np.minimum(w, cap * mid).sum()) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def trade_cost(trade: pd.Series, inputs: pd.DataFrame, nav: float, config: LiquidityConfig,
               volume_multiplier: float = 1.0, spread_multiplier: float = 1.0,
               vol_multiplier: float = 1.0) -> pd.DataFrame:
    """Cost of trading |trade| (fraction of NAV) in each name.

    Returns spread, impact and total cost both per unit traded (``*_rate``) and as a
    fraction of NAV. Names with no ADV have NaN impact.
    """
    q = trade.astype(float).abs()
    inp = inputs.reindex(q.index)
    adv = inp["adv"] * volume_multiplier
    half_spread = 0.5 * inp["spread_bps"] * 1e-4 * spread_multiplier
    sigma = inp["daily_vol"] * vol_multiplier
    impact = config.impact_coefficient * sigma * np.sqrt(q * nav / adv)
    impact = impact.where(adv > 0)
    impact[q == 0] = 0.0
    rate = half_spread + impact
    out = pd.DataFrame({
        "traded": q,
        "spread_rate": half_spread.where(q > 0, 0.0),
        "impact_rate": impact,
        "cost_rate": rate.where(q > 0, 0.0),
    })
    out["spread_cost"] = out["traded"] * out["spread_rate"]
    out["impact_cost"] = out["traded"] * out["impact_rate"]
    out["cost"] = out["spread_cost"] + out["impact_cost"]
    return out


@dataclass
class GrossReduction:
    """Result of cutting gross by ``target``: trades, time and cost.

    ``summary`` holds gross before/after, days to complete, cost as a fraction of NAV and in
    basis points of value traded, and names that could not be traded (no ADV).
    """

    target: float
    method: str
    trades: pd.DataFrame
    summary: dict


def reduce_gross(weights: pd.Series, inputs: pd.DataFrame, nav: float, config: LiquidityConfig,
                 target: float | None = None, method: str | None = None,
                 volume_multiplier: float = 1.0, spread_multiplier: float = 1.0,
                 vol_multiplier: float = 1.0) -> GrossReduction:
    """Cut gross by ``target`` (default ``config.reduction_target``) and cost it.

    ``pro_rata`` trades target x |w_i| in every name; the book is done when its slowest name
    is. ``fastest`` trades every name at full participation until the target is reached.
    Names without ADV cannot be traded: ``pro_rata`` reports them and leaves them, so the
    cut falls short; ``fastest`` works around them where it can.
    """
    target = config.reduction_target if target is None else target
    method = config.reduction_method if method is None else method
    if method not in ("pro_rata", "fastest"):
        raise ValueError("method must be 'pro_rata' or 'fastest'")
    w = weights.astype(float)
    w = w[w != 0]
    gross = float(w.abs().sum())
    adv = inputs["adv"].reindex(w.index)
    cap = daily_capacity(adv, nav, config.participation, volume_multiplier)
    tradable = cap > 0

    if method == "pro_rata":
        cut = (target * w.abs()).where(tradable, 0.0)
        days = (cut / cap.where(tradable)).fillna(0.0)
        t_done = float(days.max()) if len(days) else 0.0
    else:
        t_done = days_to_liquidate_share(w, adv, nav, target, config.participation, volume_multiplier)
        if np.isfinite(t_done):
            cut = liquidatable(w, cap, t_done)
            # remove the tiny overshoot from bisection so the cut is exactly the target
            excess = float(cut.sum()) - target * gross
            if excess > 0 and cut.sum() > 0:
                cut = cut * (target * gross / float(cut.sum()))
        else:
            cut = w.abs().where(tradable, 0.0)
            t_done = float((cut / cap.where(tradable)).max()) if tradable.any() else float("inf")
        days = (cut / cap.where(tradable)).fillna(0.0)

    cost = trade_cost(cut, inputs, nav, config, volume_multiplier, spread_multiplier, vol_multiplier)
    after = w - np.sign(w) * cut
    trades = pd.DataFrame({
        "weight": w,
        "trade": -np.sign(w) * cut,
        "weight_after": after,
        "days": days,
        "cost_rate": cost["cost_rate"],
        "cost": cost["cost"],
        "tradable": tradable,
    })
    traded = float(cut.sum())
    total_cost = float(cost["cost"].sum(min_count=1)) if traded > 0 else 0.0
    achieved = traded / gross if gross > 0 else 0.0
    summary = {
        "method": method,
        "target_cut": target,
        "achieved_cut": achieved,
        "gross_before": gross,
        "gross_after": float(after.abs().sum()),
        "net_before": float(w.sum()),
        "net_after": float(after.sum()),
        "days_to_complete": float(t_done),
        "cost": total_cost,
        "cost_amount": total_cost * nav,
        "cost_bps_of_traded": total_cost / traded * 1e4 if traded > 0 else 0.0,
        "untradable_names": int((~tradable).sum()),
        "untradable_gross": float(w.abs()[~tradable].sum()),
        "volume_multiplier": volume_multiplier,
        "spread_multiplier": spread_multiplier,
        "vol_multiplier": vol_multiplier,
    }
    return GrossReduction(target=target, method=method, trades=trades, summary=summary)
